"""Three-dimensional bead-spring polymer for Homework 1, Question 2.

The model combines harmonic bonds between consecutive beads with Lennard-Jones
interactions between non-neighboring beads.  Langevin dynamics is integrated
with BAOAB.  Run this file directly to produce one figure for Question 2a and
one figure for Question 2b.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def initialize_self_avoiding_chain(
        num_beads: int,
        bond_length: float,
        sigma: float,
        rng: np.random.Generator,
) -> np.ndarray:
    """Build a three-dimensional chain without severe nonbonded overlaps."""
    minimum_separation = 0.85 * sigma

    for _ in range(100):
        positions = np.zeros((num_beads, 3))
        chain_completed = True

        for bead in range(1, num_beads):
            position_found = False
            for _ in range(1000):
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                candidate = positions[bead - 1] + bond_length * direction

                # The immediately preceding bead is bonded and is therefore
                # excluded from this nonbonded-overlap test.
                earlier_beads = positions[:max(0, bead - 1)]
                if (
                    len(earlier_beads) == 0
                    or np.all(
                        np.linalg.norm(earlier_beads - candidate, axis=1)
                        > minimum_separation
                    )
                ):
                    positions[bead] = candidate
                    position_found = True
                    break

            if not position_found:
                chain_completed = False
                break

        if chain_completed:
            return positions - positions.mean(axis=0)

    raise RuntimeError("Could not generate a non-overlapping initial chain")


def polymer_forces(
        positions: np.ndarray,
        box_size: float,
        sigma: float,
        epsilon: float,
        bond_length: float,
        spring_constant: float,
) -> np.ndarray:
    """Return the sum of nonbonded LJ forces and bonded harmonic forces."""
    num_beads = len(positions)

    # delta[i, j] = r_i - r_j, using the minimum-image convention.
    delta = positions[:, None, :] - positions[None, :, :]
    delta -= np.round(delta / box_size) * box_size
    distance_squared = np.sum(delta**2, axis=2)

    bead_indices = np.arange(num_beads)
    nonbonded = np.abs(
        bead_indices[:, None] - bead_indices[None, :]
    ) > 1
    safe_distance_squared = np.where(
        nonbonded,
        np.maximum(distance_squared, (1e-6 * sigma)**2),
        np.inf,
    )

    inverse_r2 = 1.0 / safe_distance_squared
    lj_prefactor = 24 * epsilon * (
        2 * sigma**12 * inverse_r2**7
        - sigma**6 * inverse_r2**4
    )
    forces = np.sum(lj_prefactor[:, :, None] * delta, axis=1)

    # Harmonic force for the N - 1 bonds.  LJ is already disabled for these
    # neighboring pairs by the nonbonded mask above.
    bond_vectors = positions[1:] - positions[:-1]
    bond_vectors -= np.round(bond_vectors / box_size) * box_size
    bond_distances = np.linalg.norm(bond_vectors, axis=1)
    bond_distances = np.maximum(bond_distances, 1e-12)
    bond_forces = (
        spring_constant
        * (bond_distances - bond_length)[:, None]
        * bond_vectors
        / bond_distances[:, None]
    )
    forces[:-1] += bond_forces
    forces[1:] -= bond_forces
    return forces


def radius_of_gyration_squared(positions: np.ndarray) -> float:
    """Calculate R_g^2 about the instantaneous center of mass."""
    centered_positions = positions - positions.mean(axis=0)
    return np.mean(np.sum(centered_positions**2, axis=1))


def simulate_polymer(
        num_beads: int,
        temperature_star: float,
        num_steps: int,
        *,
        sigma: float = 1.0,
        epsilon: float = 1.0,
        mass: float = 1.0,
        dt: float = 0.002,
        gamma: float = 1.0,
        equilibration_fraction: float = 0.6,
        record_stride: int = 20,
        seed: int = 0,
) -> dict:
    """Simulate one polymer and return its R_g history and final structure."""
    if num_beads < 2:
        raise ValueError("num_beads must be at least 2")
    if temperature_star <= 0 or num_steps <= 0:
        raise ValueError("temperature_star and num_steps must be positive")
    if not 0 <= equilibration_fraction < 1:
        raise ValueError("equilibration_fraction must be in [0, 1)")

    rng = np.random.default_rng(seed)
    kBT = temperature_star * epsilon
    bond_length = sigma
    spring_constant = 10 * kBT / sigma**2
    box_size = max(50 * sigma, 3 * num_beads * bond_length)

    positions = initialize_self_avoiding_chain(
        num_beads, bond_length, sigma, rng
    )
    velocities = rng.normal(
        scale=np.sqrt(kBT / mass), size=(num_beads, 3)
    )
    velocities -= velocities.mean(axis=0)
    forces = polymer_forces(
        positions, box_size, sigma, epsilon,
        bond_length, spring_constant,
    )

    times = [0.0]
    rg_squared_history = [radius_of_gyration_squared(positions)]
    thermostat_decay = np.exp(-gamma * dt)
    thermostat_noise = np.sqrt(
        kBT / mass * (1 - thermostat_decay**2)
    )

    for step in range(1, num_steps + 1):
        # B: half conservative-force kick.
        velocities += 0.5 * dt * forces / mass

        # A: half position drift.
        positions += 0.5 * dt * velocities

        # O: exact Ornstein-Uhlenbeck thermostat update.  Removing the mean
        # noise prevents the thermostat from translating the whole polymer.
        random_velocity = rng.normal(size=(num_beads, 3))
        random_velocity -= random_velocity.mean(axis=0)
        if num_beads > 1:
            random_velocity *= np.sqrt(num_beads / (num_beads - 1))
        velocities = (
            thermostat_decay * velocities
            + thermostat_noise * random_velocity
        )

        # A: second half position drift.  Keep the center of mass at the
        # origin so R_g remains straightforward under periodic boundaries.
        positions += 0.5 * dt * velocities
        positions -= positions.mean(axis=0)

        # B: second half conservative-force kick.
        forces = polymer_forces(
            positions, box_size, sigma, epsilon,
            bond_length, spring_constant,
        )
        velocities += 0.5 * dt * forces / mass
        velocities -= velocities.mean(axis=0)

        if step % record_stride == 0 or step == num_steps:
            times.append(step * dt)
            rg_squared_history.append(
                radius_of_gyration_squared(positions)
            )

    times = np.asarray(times)
    rg_squared_history = np.asarray(rg_squared_history)
    equilibration_time = equilibration_fraction * num_steps * dt
    equilibrated = times >= equilibration_time
    mean_rg = np.sqrt(np.mean(rg_squared_history[equilibrated]))
    instantaneous_rg = np.sqrt(rg_squared_history[equilibrated])

    return {
        "num_beads": num_beads,
        "temperature_star": temperature_star,
        "num_steps": num_steps,
        "box_size": box_size,
        "times": times,
        "rg_squared_history": rg_squared_history,
        "equilibration_time": equilibration_time,
        "mean_rg": mean_rg,
        "rg_fluctuation": np.std(instantaneous_rg),
        "final_positions": positions.copy(),
    }


def _set_equal_3d_limits(axis, positions: np.ndarray, sigma: float) -> None:
    """Use equal x/y/z scales for a polymer conformation plot."""
    extent = max(np.max(np.abs(positions)), sigma)
    limit = 1.12 * extent
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_zlim(-limit, limit)
    axis.set_box_aspect((1, 1, 1))


def run_question_a(
        *,
        num_beads: int = 50,
        num_steps: int = 50_000,
        dt: float = 0.002,
        seed: int = 100,
) -> tuple[plt.Figure, list[dict]]:
    """Question 2a: compare N=50 at T*=0.2, 2, and 5 in one figure."""
    temperatures = (0.2, 2.0, 5.0)
    colors = ("tab:blue", "tab:orange", "tab:red")
    results = []

    for index, temperature in enumerate(temperatures):
        print(
            f"Question 2a: N={num_beads}, T*={temperature}, "
            f"steps={num_steps}"
        )
        results.append(
            simulate_polymer(
                num_beads,
                temperature,
                num_steps,
                dt=dt,
                seed=seed + index,
            )
        )

    figure = plt.figure(figsize=(14, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(1.1, 1))

    for column, (result, color) in enumerate(zip(results, colors)):
        axis = figure.add_subplot(grid[0, column], projection="3d")
        positions = result["final_positions"]
        axis.plot(
            positions[:, 0], positions[:, 1], positions[:, 2],
            color=color, linewidth=1.2, alpha=0.8,
        )
        axis.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            color=color, s=18, depthshade=True,
        )
        _set_equal_3d_limits(axis, positions, sigma=1.0)
        axis.set(
            xlabel=r"$x/\sigma$",
            ylabel=r"$y/\sigma$",
            zlabel=r"$z/\sigma$",
            title=(
                rf"$T^*={result['temperature_star']}$, "
                rf"$R_g/\sigma={result['mean_rg']:.3f}$"
            ),
        )

    history_axis = figure.add_subplot(grid[1, :])
    for result, color in zip(results, colors):
        history_axis.plot(
            result["times"],
            np.sqrt(result["rg_squared_history"]),
            color=color,
            linewidth=1.2,
            label=(
                rf"$T^*={result['temperature_star']}$, "
                rf"$R_g={result['mean_rg']:.3f}\sigma$"
            ),
        )
    history_axis.axvline(
        results[0]["equilibration_time"],
        color="0.35",
        linestyle="--",
        linewidth=1,
        label="Measurement window begins",
    )
    history_axis.set(
        xlabel="Time",
        ylabel=r"Instantaneous $R_g/\sigma$",
        title="Radius-of-gyration equilibration",
    )
    history_axis.grid(alpha=0.25)
    history_axis.legend(ncol=2)
    figure.suptitle(
        "Question 2a: temperature dependence of a bead-spring polymer"
    )
    return figure, results


def run_question_b(
        *,
        bead_counts: tuple[int, ...] = (10, 20, 50, 100),
        low_temperature: float = 0.2,
        high_temperature: float = 5.0,
        base_steps: int = 1_000,
        dt: float = 0.002,
        seed: int = 200,
) -> tuple[plt.Figure, dict[str, list[dict]]]:
    """Question 2b: obtain R_g versus N at low and high temperature."""
    all_results = {"Low temperature": [], "High temperature": []}
    temperature_cases = (
        ("Low temperature", low_temperature, 5 / 3),
        ("High temperature", high_temperature, 2.2),
    )
    smallest_chain = min(bead_counts)

    for case_index, (label, temperature, rouse_exponent) in enumerate(
            temperature_cases
    ):
        for chain_index, num_beads in enumerate(bead_counts):
            # Longer chains require longer equilibration.  These exponents are
            # the globule (5/3) and self-avoiding coil (2*0.6 + 1) estimates.
            num_steps = max(
                100,
                int(
                    base_steps
                    * (num_beads / smallest_chain)**rouse_exponent
                ),
            )
            print(
                f"Question 2b: N={num_beads}, T*={temperature}, "
                f"steps={num_steps}"
            )
            all_results[label].append(
                simulate_polymer(
                    num_beads,
                    temperature,
                    num_steps,
                    dt=dt,
                    seed=seed + 10 * case_index + chain_index,
                )
            )

    figure, axis = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    bead_counts_array = np.asarray(bead_counts, dtype=float)
    plot_cases = (
        ("Low temperature", "tab:blue", "o", 1 / 3, "globule"),
        ("High temperature", "tab:red", "s", 0.6, "coil"),
    )

    for label, color, marker, theoretical_exponent, state_name in plot_cases:
        rg_values = np.asarray([
            result["mean_rg"] for result in all_results[label]
        ])
        fitted_exponent, log_prefactor = np.polyfit(
            np.log(bead_counts_array), np.log(rg_values), 1
        )
        fitted_rg = np.exp(log_prefactor) * bead_counts_array**fitted_exponent

        # Anchor the theoretical curve at the geometric-center data scale so
        # its slope can be compared without implying a known prefactor.
        theoretical_prefactor = np.exp(
            np.mean(
                np.log(rg_values)
                - theoretical_exponent * np.log(bead_counts_array)
            )
        )
        theoretical_rg = (
            theoretical_prefactor
            * bead_counts_array**theoretical_exponent
        )

        axis.loglog(
            bead_counts_array,
            rg_values,
            linestyle="none",
            marker=marker,
            markersize=7,
            color=color,
            label=(
                f"{label} data "
                rf"($T^*={all_results[label][0]['temperature_star']}$)"
            ),
        )
        axis.loglog(
            bead_counts_array,
            fitted_rg,
            color=color,
            linewidth=1.8,
            label=rf"{label} fit: $\nu={fitted_exponent:.3f}$",
        )
        axis.loglog(
            bead_counts_array,
            theoretical_rg,
            color=color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            label=(
                rf"{state_name} theory: $\nu="
                f"{theoretical_exponent:.3f}$"
            ),
        )

    axis.set(
        xlabel="Number of beads $N$",
        ylabel=r"Time-averaged $R_g/\sigma$",
        title="Question 2b: polymer-size scaling",
    )
    axis.grid(which="both", alpha=0.25)
    axis.legend(fontsize=9)
    return figure, all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Homework 1 bead-spring polymer simulations"
    )
    parser.add_argument(
        "--task",
        choices=("a", "b", "all"),
        default="all",
        help="which part of Question 2 to run (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use short runs for checking code and plot layout",
    )
    arguments = parser.parse_args()

    if arguments.task in ("a", "all"):
        run_question_a(
            num_steps=2_000 if arguments.quick else 50_000
        )
    if arguments.task in ("b", "all"):
        run_question_b(
            base_steps=20 if arguments.quick else 1_000
        )

    plt.show()


if __name__ == "__main__":
    main()
