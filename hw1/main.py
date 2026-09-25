import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def lennard_jones_fluid(
        task:str = "test", 
        T_star:float=0.05,
        rho_star:float=0.5,
        num_particle:int=50, 
        sigma:float=10.0, 
        epsilon:float=1.0, 
        box_size_times:float=10.0, 
        m=1.0, 
        kB=1.0,
        dt=0.1):
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

        # TODO
        pass
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
        r = init_r_array()
        r_hist = r.reshape(1, num_particle, 2)
        v_hist = init_v_array_by_t_star(T_star).reshape(1, num_particle, 2)
        a_hist =langevin_thermostat(r_hist[[-1], :], v_hist[[-1], :])/m
        E_pot = []
        E_kin = []
        E_tot = []
        for step in range(10000):
            s = (r_hist, v_hist, a_hist) = velocity_verlet(r_hist, v_hist, a_hist, langevin_thermostat)
            if step%1==0:
                print(epot:=u_mat_avery(get_norm2_mat(r_hist[-1, :])),ekin:=e_kin_avery(v_hist[-1, :]))
                E_pot.append(epot)
                E_kin.append(ekin)
                E_tot.append(epot+ekin)
        plt.plot(E_pot)
        plt.plot(E_kin)
        plt.plot(E_tot)
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
            time_label = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                                 va="top")

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
        pass
    if task=="c":
        pass
    return


lennard_jones_fluid(task="a")
