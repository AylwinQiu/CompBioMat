from pygments.lexers.boa import BoaLexer
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def lennard_jones_fluid(
        task:str = "test", 
        T_star:float=0.5,
        rho_star:float=0.2,
        num_particle:int=50, 
        sigma:float=10.0, 
        epsilon:float=1.0, 
        box_size_times:float=10.0, 
        m=1.0, 
        kB=1.0,
        dt=0.1,
        tot_steps = 20000):
    box_size = box_size_times*sigma
    def init_r_array():
        # Make a lattice as close to square as possible
        n_x = int(np.ceil(np.sqrt(num_particle)))
        n_y = int(np.ceil(num_particle / n_x))
        dx = box_size / n_x
        dy = box_size / n_y
        x = (np.arange(n_x) + 0.5) * dx
        y = (np.arange(n_y) + 0.5) * dy
        xx, yy = np.meshgrid(x, y)
        r = np.column_stack((xx.ravel(), yy.ravel()))
        return r[:num_particle]
    def init_r_array_by_rho_star(rho_star:float):
        """Set the 2-D box size from reduced density and return a lattice.
        """
        nonlocal box_size
        if not np.isfinite(rho_star) or rho_star <= 0:
            raise ValueError("rho_star must be a positive finite number")

        box_size = sigma * np.sqrt(num_particle / rho_star)
        return init_r_array()
    def init_v_array_by_t_star(t_star:float):
        kBT = t_star*epsilon
        tmp = np.random.rand(num_particle)*2*np.pi
        ans = np.zeros((num_particle, 2))
        ans[:, 0] = np.cos(tmp)
        ans[:, 1] = np.sin(tmp)
        return ans*(kBT*2/m)**0.5
    def velocity_verlet(r_hist, v_hist, a_hist, f_func):
        r_hist = np.concatenate([r_hist, r_hist[[-1], :]+v_hist[[-1], :]*dt + 0.5*a_hist[[-1], :]*dt**2], axis=0)
        a_hist = np.concatenate([a_hist, f_func(r_hist[[-1], :], v_hist[[-1], :])/m], axis=0)
        v_hist = np.concatenate([v_hist, v_hist[[-1], :] + 0.5*(a_hist[[-2], :] +a_hist[[-1], :])*dt], axis=0)
        return (r_hist, v_hist, a_hist)
    def baoab(r_hist, v_hist, a_hist, gamma=1.0):
        r_old = r_hist[-1, :]
        v_old = v_hist[-1, :]
        a_old = a_hist[-1, :]
        # B
        v_025 = v_old + a_old*dt/2
        # A
        r_050 = r_old + v_025*dt/2
        # O
        c = np.exp(-gamma*dt/m)
        v_075 = (
            c*v_025
            + np.sqrt((1-c**2)*T_star*epsilon/m)
            *np.random.normal(size=(num_particle, 2))
        )
        # A
        r_new = r_050 + v_075 * dt/2
        # B
        a_new = force_lj(r_new[None, :, :])[0]/m
        v_new = v_075 + a_new*dt/2
        r_hist = np.concatenate([r_hist, np.expand_dims(r_new, 0)], axis=0)
        a_hist = np.concatenate([a_hist, np.expand_dims(a_new, 0)],axis=0)
        v_hist = np.concatenate([v_hist, np.expand_dims(v_new, 0)], axis=0)
        return (r_hist, v_hist, a_hist)
    def get_norm2_mat(position:np.ndarray):
        cat = np.concatenate
        x = position[:, [0]]
        y = position[:, [1]]
        delta_x = x - x.T
        delta_y = y - y.T
        delta_x_pbc = delta_x-np.round(delta_x/box_size)*box_size
        delta_y_pbc = delta_y-np.round(delta_y/box_size)*box_size
        return (delta_x_pbc**2+delta_y_pbc**2)**0.5
    def u_mat(r_mat:np.ndarray):
        return 4*epsilon*((t:=sigma/r_mat)**12 - t**6)
    def e_kin_avery(v:np.ndarray):
        """v should have shape [num_parameters, 2]"""
        x, y = v[:, 0], v[:, 1]
        return (0.5*(x**2+y**2)*m).sum()/num_particle
    def u_mat_avery(r_mat:np.ndarray):
        return np.nan_to_num(u_mat(r_mat), nan=0).sum()/num_particle/2
    def langevin_thermostat(r_hist_slice:np.ndarray, v_hist_slice:np.ndarray, gamma=1.0, ):
        # r_hist_slice should have shape like [1, N, 2]
        kBT=T_star*epsilon
        flj = force_lj(r_hist_slice)
        rt = lambda:np.sqrt(2*gamma*kBT/dt)*np.random.normal(size=(num_particle, 2))
        return flj - gamma*v_hist_slice+ rt()
    def force_lj(r_hist_slice:np.ndarray):
        # r_hist_slice should have shape like [1, N, 2]
        delta_r_hist_slice = r_hist_slice - r_hist_slice.transpose(1, 0, 2)
        delta_r_hist_slice = delta_r_hist_slice - np.round(delta_r_hist_slice/box_size)*box_size # Consider the PBC
        norm = get_norm2_mat(r_hist_slice[0, :, :])
        np.fill_diagonal(norm, np.inf)
        f_mat = 24*epsilon*delta_r_hist_slice*(-2*sigma**12/norm**14 + sigma**6/norm**8).reshape(num_particle, num_particle, 1) #Shape[N, N, 2]
        return f_mat.sum(axis=1).reshape(1, num_particle, 2)
    # script
    #print(distant := get_norm2_mat(r))
    #print(u_mat(distant))
    #print(f"potential for every particle:{u_mat_avery(r)}")
    #print(f"force:{f_func(r.reshape(1, num_particle, 2))}")
    if task=="a":
        r = init_r_array_by_rho_star(rho_star)
        r_hist = r.reshape(1, num_particle, 2)
        v_hist = init_v_array_by_t_star(T_star).reshape(1, num_particle, 2)
        a_hist =langevin_thermostat(r_hist[[-1], :], v_hist[[-1], :])/m
        E_pot = []
        E_kin = []
        E_tot = []
        for step in range(tot_steps):
            s = (r_hist, v_hist, a_hist) =baoab(r_hist, v_hist, a_hist)
            # velocity_verlet(r_hist, v_hist, a_hist, langevin_thermostat)
            if step%1==0:
                epot=u_mat_avery(get_norm2_mat(r_hist[-1, :]))
                ekin=e_kin_avery(v_hist[-1, :])
                #print(epot:=u_mat_avery(get_norm2_mat(r_hist[-1, :])),ekin:=e_kin_avery(v_hist[-1, :]))
                E_pot.append(epot)
                E_kin.append(ekin)
                E_tot.append(epot+ekin)
        #plt.plot(E_pot)
        #plt.plot(E_kin)
        plt.plot([dt*i for i in range(tot_steps)], E_tot)
        plt.xlabel("t")
        plt.ylabel(r"$E_{total}$")
        plt.title(rf"Total energy vs. time ($T^*={T_star}$, $\rho^*={rho_star}$)")
        def vid_r_hist(frame_stride=10, interval=30):
            """Animate the particle-position history stored in ``r_hist``.

            ``r_hist`` has shape ``(n_steps, num_particle, 2)``.  Positions
            are wrapped only for display, so the animation respects the
            periodic boundary condition used by the force calculation.
            """
            if r_hist.ndim != 3 or r_hist.shape[1:] != (num_particle, 2):
                raise ValueError("r_hist must have shape (n_steps, num_particle, 2)")
            if frame_stride <= 0:
                raise ValueError("frame_stride must be positive")

            frames = np.arange(0, len(r_hist), frame_stride)
            if frames[-1] != len(r_hist) - 1:
                frames = np.append(frames, len(r_hist) - 1)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set(xlim=(0, box_size), ylim=(0, box_size),
                   xlabel="x", ylabel="y", aspect="equal")
            ax.set_title("Lennard-Jones fluid")
            particles = ax.scatter([], [], s=35, color="tab:blue")
            time_label = ax.text(0.02, 0.97, "", transform=ax.transAxes,va="top")

            def update(frame):
                # modulo makes a particle crossing an edge re-enter opposite edge
                particles.set_offsets(r_hist[frame] % box_size)
                time_label.set_text(f"step = {frame},  t = {frame * dt:.2f}")
                return particles, time_label

            animation = FuncAnimation(
                fig, update, frames=frames, interval=interval,
                blit=True, repeat=True,
            )
            plt.show()
            return animation

        ani = vid_r_hist()
    if task=="b":
        # Run to equilibrium, then sample sufficiently separated late frames.
        r = init_r_array_by_rho_star(rho_star)
        r_hist = r.reshape(1, num_particle, 2)
        v_hist = init_v_array_by_t_star(T_star).reshape(1, num_particle, 2)
        a_hist = force_lj(r_hist) / m
        for step in range(tot_steps):
            r_hist, v_hist, a_hist = baoab(r_hist, v_hist, a_hist)
        equilibration_step = int(0.8 * tot_steps)
        sample_stride = max(1, int(round(1.0 / dt)))
        v_samples = v_hist[equilibration_step::sample_stride].reshape(-1, 2)
        kBT = T_star * epsilon
        sigma_v = np.sqrt(kBT / m)
        v_x = v_samples[:, 0]
        kinetic_energy = 0.5 * m * np.sum(v_samples**2, axis=1)
        kinetic_energy_per_dof = np.mean(0.5 * m * v_samples**2)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        # One Cartesian velocity component is Gaussian: N(0, kBT/m).
        v_limit = max(4 * sigma_v, np.max(np.abs(v_x)))
        v_reference = np.linspace(-v_limit, v_limit, 400)
        gaussian_reference = (
            np.exp(-v_reference**2 / (2 * sigma_v**2))
            / (np.sqrt(2 * np.pi) * sigma_v)
        )
        axes[0].hist(
            v_x, bins=40, density=True, alpha=0.65,
            label="Simulation",
        )
        axes[0].plot(
            v_reference, gaussian_reference, "--", linewidth=2,
            label=r"Theory: $\mathcal{N}(0,k_BT/m)$",
        )
        axes[0].set(
            xlabel=r"$v_x$",
            ylabel=r"Probability density $p(v_x)$",
            title="Velocity-component distribution",
        )
        axes[0].text(
            0.03, 0.97,
            f"sample mean = {np.mean(v_x):.3f}\n"
            f"sample variance = {np.var(v_x):.3f}\n"
            f"theory variance = {kBT / m:.3f}",
            transform=axes[0].transAxes, va="top",
        )
        axes[0].legend()
        axes[0].grid(alpha=0.25)
        # For two velocity degrees of freedom, total kinetic energy is
        # exponentially distributed: p(E) = exp(-E/kBT) / kBT.
        energy_limit = max(6 * kBT, np.max(kinetic_energy))
        energy_reference = np.linspace(0, energy_limit, 400)
        boltzmann_reference = np.exp(-energy_reference / kBT) / kBT
        axes[1].hist(
            kinetic_energy, bins=40, range=(0, energy_limit), density=True,
            alpha=0.65, label="Simulation",
        )
        axes[1].plot(
            energy_reference, boltzmann_reference, "--", linewidth=2,
            label=r"Theory: $e^{-E/(k_BT)}/(k_BT)$",
        )
        axes[1].set(
            xlabel=r"Kinetic energy $E=\frac{m}{2}(v_x^2+v_y^2)$",
            ylabel=r"Probability density $p(E)$",
            title="Boltzmann energy distribution",
        )
        axes[1].text(
            0.97, 0.97,
            f"<E/dof> = {kinetic_energy_per_dof:.3f}\n"
            f"theory = {0.5 * kBT:.3f}",
            transform=axes[1].transAxes, ha="right", va="top",
        )
        axes[1].legend()
        axes[1].grid(alpha=0.25)
        fig.suptitle(
            rf"Equilibrium distributions at $T^*={T_star}$ "
            f"({len(v_samples)} samples)"
        )
        plt.show()
    if task=="c":
        # Three points chosen away from the phase boundaries in Fig. 1.
        # Use num_particle=70 in the function call to match the assignment.
        phase_cases = [
            ("Gas", 0.5, 0.05),
            ("Liquid", 0.5, 0.70),
            ("Solid", 0.4, 0.95),
        ]
        phase_results = []
        upper_triangle = np.triu_indices(num_particle, k=1)

        for phase_name, phase_temperature, phase_density in phase_cases:
            # baoab reads T_star from this enclosing scope.
            T_star = phase_temperature
            rho_star = phase_density

            r = init_r_array_by_rho_star(rho_star)
            r_state = r.reshape(1, num_particle, 2)
            v_state = init_v_array_by_t_star(T_star).reshape(
                1, num_particle, 2
            )
            a_state = force_lj(r_state) / m

            # Preallocate the histories. Passing only the latest state to baoab
            # avoids repeatedly copying the entire growing trajectory.
            r_hist = np.empty((tot_steps + 1, num_particle, 2))
            v_hist = np.empty_like(r_hist)
            a_hist = np.empty_like(r_hist)
            r_hist[0] = r_state[0]
            v_hist[0] = v_state[0]
            a_hist[0] = a_state[0]

            for step in range(tot_steps):
                r_step, v_step, a_step = baoab(
                    r_state, v_state, a_state
                )
                r_state = r_step[[-1]]
                v_state = v_step[[-1]]
                a_state = a_step[[-1]]
                r_hist[step + 1] = r_state[0]
                v_hist[step + 1] = v_state[0]
                a_hist[step + 1] = a_state[0]

            # Discard the first 80% as equilibration and sample late frames.
            equilibration_step = int(0.8 * tot_steps)
            sample_stride = max(1, int(round(1.0 / dt)))
            sampled_positions = r_hist[equilibration_step::sample_stride]

            # Radial distribution function and potential energy per particle.
            # r_max <= L/2 keeps every circular shell inside the PBC cell.
            number_of_bins = 70
            r_max = min(3.5 * sigma, box_size / 2)
            rdf_edges = np.linspace(0, r_max, number_of_bins + 1)
            rdf_counts = np.zeros(number_of_bins)
            potential_energy_samples = []

            for positions in sampled_positions:
                pair_distances = get_norm2_mat(positions)[upper_triangle]
                rdf_counts += np.histogram(pair_distances, bins=rdf_edges)[0]

                reduced_distance = sigma / pair_distances
                pair_energies = 4 * epsilon * (
                    reduced_distance**12 - reduced_distance**6
                )
                potential_energy_samples.append(
                    np.sum(pair_energies) / num_particle
                )

            shell_areas = np.pi * (rdf_edges[1:]**2 - rdf_edges[:-1]**2)
            ideal_pairs_per_frame = (
                num_particle * (num_particle - 1)
                / (2 * box_size**2)
                * shell_areas
            )
            rdf = rdf_counts / (
                len(sampled_positions) * ideal_pairs_per_frame
            )
            rdf_r = 0.5 * (rdf_edges[:-1] + rdf_edges[1:])
            mean_potential_energy = np.mean(potential_energy_samples)

            # Time-averaged MSD using many time origins. r_hist is unwrapped,
            # which is essential when particles cross a periodic boundary.
            production_positions = r_hist[equilibration_step:]
            max_lag = max(1, (len(production_positions) - 1) // 2)
            lag_stride = max(1, int(round(1.0 / dt)))
            lags = np.arange(0, max_lag + 1, lag_stride, dtype=int)
            if lags[-1] != max_lag:
                lags = np.append(lags, max_lag)
            if len(lags) < 6:
                lags = np.arange(max_lag + 1, dtype=int)

            msd = np.zeros(len(lags))
            for lag_index, lag in enumerate(lags[1:], start=1):
                displacement = (
                    production_positions[lag:] - production_positions[:-lag]
                )
                msd[lag_index] = np.mean(
                    np.sum(displacement**2, axis=2)
                )

            msd_time = lags * dt
            fit_start = max(1, len(msd_time) // 2)
            if len(msd_time) - fit_start >= 2:
                msd_slope, msd_intercept = np.polyfit(
                    msd_time[fit_start:], msd[fit_start:], 1
                )
                diffusivity = msd_slope / 4
                fitted_msd = (
                    msd_slope * msd_time[fit_start:] + msd_intercept
                )
            else:
                diffusivity = np.nan
                fitted_msd = np.full(
                    len(msd_time) - fit_start, np.nan
                )

            phase_results.append({
                "name": phase_name,
                "temperature": phase_temperature,
                "density": phase_density,
                "box_size": box_size,
                "final_positions": r_hist[-1].copy(),
                "rdf_r": rdf_r,
                "rdf": rdf,
                "mean_potential_energy": mean_potential_energy,
                "msd_time": msd_time,
                "msd": msd,
                "fit_start": fit_start,
                "fitted_msd": fitted_msd,
                "diffusivity": diffusivity,
            })

        # Each row is one phase: configuration, g(r), and MSD.
        fig, axes = plt.subplots(
            3, 3, figsize=(14, 12), constrained_layout=True
        )
        colors = ("tab:blue", "tab:orange", "tab:green")

        for row, (result, color) in enumerate(zip(phase_results, colors)):
            positions = (
                result["final_positions"] % result["box_size"]
            ) / sigma
            reduced_box_size = result["box_size"] / sigma

            axes[row, 0].scatter(
                positions[:, 0], positions[:, 1], s=22,
                color=color, alpha=0.8,
            )
            axes[row, 0].set(
                xlim=(0, reduced_box_size),
                ylim=(0, reduced_box_size),
                xlabel=r"$x/\sigma$",
                ylabel=r"$y/\sigma$",
                aspect="equal",
                title=(
                    f"{result['name']}: equilibrium configuration\n"
                    rf"$T^*={result['temperature']}$, "
                    rf"$\rho^*={result['density']}$, "
                    rf"$\langle E^{{pot}}\rangle="
                    f"{result['mean_potential_energy']:.3f}$"
                ),
            )

            axes[row, 1].plot(
                result["rdf_r"] / sigma, result["rdf"],
                color=color, linewidth=1.8,
            )
            axes[row, 1].axhline(
                1, color="0.4", linestyle="--", linewidth=1,
                label="Ideal gas",
            )
            axes[row, 1].set(
                xlabel=r"$r/\sigma$",
                ylabel=r"$g(r)$",
                title=f"{result['name']}: radial distribution",
            )
            axes[row, 1].grid(alpha=0.25)
            if row == 0:
                axes[row, 1].legend()

            axes[row, 2].plot(
                result["msd_time"], result["msd"],
                color=color, linewidth=1.8, label="MSD",
            )
            axes[row, 2].plot(
                result["msd_time"][result["fit_start"]:],
                result["fitted_msd"],
                color="0.2", linestyle="--", linewidth=1.4,
                label="Long-time fit",
            )
            axes[row, 2].set(
                xlabel=r"Lag time $t$",
                ylabel=r"$\langle |\Delta\mathbf{r}(t)|^2\rangle$",
                title=(
                    f"{result['name']}: diffusion, "
                    rf"$D={result['diffusivity']:.4g}$"
                ),
            )
            axes[row, 2].grid(alpha=0.25)
            axes[row, 2].legend()

        fig.suptitle(
            "Lennard-Jones gas, liquid, and solid: structure and diffusion"
        )
        plt.show()
    return


# Task a
if True:
    lennard_jones_fluid(task="a", T_star=0.5)
# Task b
if False:
    lennard_jones_fluid(task="b")
# Task c
if False:
    lennard_jones_fluid(task="c")
