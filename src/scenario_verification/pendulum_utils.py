import sys
import numpy as np
import tqdm
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.integrate import solve_ivp


def deriv(y, t, L1, L2, m1, m2, friction_coeff):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    g = 9.81
    theta1, z1, theta2, z2 = y

    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1) + friction_coeff*z1 + friction_coeff*z2*c ) / L1 / (m1 + m2*s**2)
    theta2dot = z2
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) +
             m2*L2*z2**2*s*c + friction_coeff*z1*c + friction_coeff*z2*(m1+m2)/m2 ) / L2 / (m1 + m2*s**2)
    return theta1dot, z1dot, theta2dot, z2dot

def calc_E(y, m1, m2, L1, L2):
    """Return the total energy of the system."""
    g = 9.81
    th1, th1d, th2, th2d = y.T
    V = -(m1+m2)*L1*g*np.cos(th1) - m2*L2*g*np.cos(th2)
    T = 0.5*m1*(L1*th1d)**2 + 0.5*m2*((L1*th1d)**2 + (L2*th2d)**2 +
            2*L1*L2*th1d*th2d*np.cos(th1-th2))
    return T + V


def generate_traj_doublepend(N_traj, t, time_steps, friction_flag=False):
    # Pendulum rod lengths (m), bob masses (kg).
    L1, L2 = 1, 1
    m1, m2 = 1, 1
    friction_coeff = 0.
    if friction_flag:
        friction_coeff = -5.
    init_angles = np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=(N_traj, 2))
    all_y = np.zeros((N_traj, time_steps, 4))
    for idx, thetas in tqdm.tqdm(enumerate(init_angles)):
        # y0 = np.array([3*np.pi/7, 0, 3*np.pi/4, 0])
        y0 = np.array([thetas[0], 0, thetas[1], 0])
        # Do the numerical integration of the equations of motion
        y = odeint(deriv, y0, t, args=(L1, L2, m1, m2, friction_coeff))
        all_y[idx, :, :] = y

    # return theta1 and theta2
    all_positions = np.zeros((all_y.shape[0], all_y.shape[1], 2))

    for idx, y in enumerate(all_y):
        # Unpack z and theta as a function of time
        theta1, theta2 = y[:, 0], y[:, 2]

        all_positions[idx, :, 0] = theta1
        all_positions[idx, :, 1] = theta2

        # Convert to Cartesian coordinates of the two bob positions.
        # x1 = L1 * np.sin(theta1)
        # y1 = -L1 * np.cos(theta1)
        # x2 = x1 + L2 * np.sin(theta2)
        # y2 = y1 - L2 * np.cos(theta2)

        # all_positions[idx, :, 0] = x1
        # all_positions[idx, :, 1] = y1
        # all_positions[idx, :, 2] = x2
        # all_positions[idx, :, 3] = y2

    return all_positions


def generate_vanderpol_traj(N_traj, t, time_steps, domain_bounds):

    def vdp(t, z, mu):
        x, y = z
        return [y, mu * (1 - x ** 2) * y - x]

    mu = 0.25
    domain_lb, domain_ub = domain_bounds

    init_pos = np.random.uniform(low=domain_lb, high=domain_ub, size=(N_traj, 2))
    all_y = np.zeros((N_traj, time_steps, 2))
    for idx, x0 in tqdm.tqdm(enumerate(init_pos)):
        sol = solve_ivp(vdp, t_span=[t[0], t[-1]], y0=x0, t_eval=t, args=(mu, ))
        all_y[idx, :, :] = sol.y.T
    #     plt.plot(sol.y[0], sol.y[1])
    # plt.show()

    return all_y

def plot_init_conditions(positions, ax):
    # Plot and save an image of the double pendulum configuration for time
    # point i.
    # The pendulum rods.
    r = 0.05
    for idx in range(positions.shape[0]):
        # x1, x2, y1, y2 = positions[idx, 0, 0], positions[idx, 0, 2], positions[idx, 0, 1], positions[idx, 0, 3]
        theta1, theta2 = positions[idx, 0, 0], positions[idx, 0, 1]
        # ax.plot([0, x1, x2], [0, y1, y2], lw=2, c='k')
        # Circles representing the anchor point of rod 1, and bobs 1 and 2.
        # c0 = Circle((0, 0), r/2, fc='k', zorder=10)
        c1 = Circle((theta1, theta2), r, fc='b', ec='b', zorder=10)
        # c2 = Circle((x2, y2), r, fc='r', ec='r', zorder=10)
        # ax.add_patch(c0)
        ax.add_patch(c1)
        # ax.add_patch(c2)


def partitions_sequences(all_positions, boundaries, N_traj, time_steps, parts_per_axis):

    parts_x_idx, parts_y_idx = boundaries

    partitions = []
    # find partitions of the trajectory
    for j in tqdm.tqdm(range(N_traj)):

        trajectory = all_positions[j, :, :]
        # angles between 0 and 2*pi
        trajectory = trajectory % (2*np.pi)
        trajectory_labels = []

        if trajectory.shape[1] == 4:
            x2 = trajectory[:, 1]
            y2 = trajectory[:, 3]
        elif trajectory.shape[1] == 2:
            x2 = trajectory[:, 0]
            y2 = trajectory[:, 1]

        for i in range(0, x2.shape[0]):
            x_part = np.argmax(x2[i] < parts_x_idx)
            y_part = np.argmax(y2[i] < parts_y_idx)
            trajectory_labels.append( (x_part-1, y_part-1) )

        partitions.append(trajectory_labels)

    # get the partition numbers
    all_trajectory_parts = np.zeros((N_traj, time_steps))
    for idx, list_of_parts in enumerate(partitions):
        partitions_labels = partition_idx(list_of_parts, parts_per_axis)
        all_trajectory_parts[idx, :] = partitions_labels

    # check if partitions do their job
    assert (all_trajectory_parts >= 0).all()

    return all_trajectory_parts


def partitions_1D(all_positions, boundaries, N_traj, time_steps):

    parts_x_idx = boundaries

    all_trajectory_parts = np.zeros((N_traj, time_steps))
    # find partitions of the trajectory
    for j in tqdm.tqdm(range(N_traj)):

        trajectory = all_positions[j, :]

        for i in range(0, trajectory.shape[0]):
            x_part = np.where( (trajectory[i] < parts_x_idx) == 1)[0][-1]
            all_trajectory_parts[j, i] = x_part

    # check if partitions do their job
    assert (all_trajectory_parts >= 0).all()

    return all_trajectory_parts


def partitions_sequences_theta(all_positions, boundaries, N_traj, time_steps, parts_per_axis):

    partitions = []
    # find partitions of the trajectory
    for j in tqdm.tqdm(range(N_traj)):

        trajectory = all_positions[j, :, :]
        trajectory_labels = []

        if trajectory.shape[1] == 4:
            x2 = trajectory[:, 2]
            y2 = trajectory[:, 3]
        elif trajectory.shape[1] == 2:
            x2 = trajectory[:, 0]
            y2 = trajectory[:, 1]

        theta = np.arctan2(y2, x2)

        for i in range(0, theta.shape[0]):
            part = np.argmax(theta[i] < boundaries)
            trajectory_labels.append( (part, ) )

        partitions.append(trajectory_labels)

    # get the partition numbers
    all_trajectory_parts = np.zeros((N_traj, time_steps))
    for idx, list_of_parts in enumerate(partitions):
        partitions_labels = partition_idx_theta(list_of_parts, parts_per_axis)
        all_trajectory_parts[idx, :] = partitions_labels

    # check if partitions do their job
    assert (all_trajectory_parts >= 0).all()

    return all_trajectory_parts


def partition_idx(series_of_tuples, parts_per_axis):

    partitions_single_idx = []
    for t in series_of_tuples:
        partitions_single_idx.append( int(t[1] * (parts_per_axis-1) + t[0]) )

    return partitions_single_idx


def partition_idx_theta(series_of_tuples, parts_per_axis):

    partitions_single_idx = []
    for t in series_of_tuples:
        partitions_single_idx.append( int(t[0]) )

    return partitions_single_idx


def get_sequences_stats(all_trajectory_parts, ell, time_steps):

    ell_seq_trajectory = set()
    ell_seq_init = set()
    ell_seq_rnd = set()
    for trajectory_parts in all_trajectory_parts:
        for idx in range(0, time_steps-ell):
            ell_seq_trajectory.add( tuple(trajectory_parts[idx:idx+ell]) )
        # find all ell-seq from INITIAL STATE
        ell_seq_init.add(tuple(trajectory_parts[0:ell]))
        # find ONE ell-seq from a trajectory at a random point
        if idx == 0:
            rand_idx = 0
        else:
            rand_idx = np.random.randint(0, idx)  # idx is at "max" point, so use it as upper bound of random generation
        ell_seq_rnd.add(tuple(trajectory_parts[rand_idx:rand_idx+ell]))

    return ell_seq_trajectory, ell_seq_init, ell_seq_rnd


# subplots
def distribution_subplots(all_trajectory_parts, time_steps, parts_per_axis, n_vars, plot_every=1, ell=1):

    plots_per_row = 5
    N_traj = all_trajectory_parts.shape[0]

    new_time_steps = time_steps // plot_every

    fig, axes = plt.subplots(np.maximum(1, np.ceil(new_time_steps/plots_per_row).astype(int)), plots_per_row)
    fig.suptitle(f'Distribution of {ell}-sequences')

    scale_bins_with_ell = np.flip(np.array([(parts_per_axis-1)**(n_vars*i) for i in range(ell)]))

    for t in tqdm.tqdm(range(new_time_steps)):
        x_at_step_k = all_trajectory_parts[:, plot_every*t:plot_every*t + ell]
        # augmented_x = np.zeros((x_at_step_k.shape[0]-ell+1, ))

        augmented_x = x_at_step_k @ scale_bins_with_ell

        if new_time_steps//plots_per_row > 0 and new_time_steps/plots_per_row > 1:
            count, bins, ignored = axes[t // plots_per_row, t % plots_per_row].hist(augmented_x,
                                                                                    bins=int((parts_per_axis - 1) ** (n_vars * ell)),
                                                                                    density=False)
            axes[t//plots_per_row, t%plots_per_row].set_title(f'Distribution at time {plot_every*t}')
            axes[t//plots_per_row, t%plots_per_row].set_ylim([0, N_traj//(parts_per_axis*n_vars**ell)])
        else:
            count, bins, ignored = axes[t % plots_per_row].hist(augmented_x,
                                                                bins=int((parts_per_axis - 1) ** (n_vars * ell)),
                                                                density=False)
            axes[t % plots_per_row].set_title(f'Distribution at time {plot_every*t}')
            axes[t % plots_per_row].set_ylim([0, N_traj // (parts_per_axis*n_vars**ell)])



