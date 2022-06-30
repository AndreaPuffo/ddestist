import sys
import numpy as np
import tqdm
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pendulum_utils import plot_init_conditions, partitions_1D, get_sequences_stats, \
    distribution_subplots, generate_vanderpol_traj


def contractive_system(x0, N_steps, epsilon):
    # generates N trajectories starting from x0
    traj = np.zeros((N_steps, ))
    traj[0] = x0
    for i in range(N_steps-1):
        xi = traj[i]
        if xi > epsilon:
            traj[i+1] = 0.5 * xi
        else:
            traj[i+1] = 0.5 * xi + 0.5

    return traj


# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 5, 1
t = np.arange(0, tmax+dt, dt)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
N_traj = 20000
time_steps = tmax + 1
n_vars = 1


traj_domain_bounds = [0, 1]
all_positions = np.zeros((N_traj, time_steps))
epsi = 1e-3
for i in range(N_traj):
    x0 = np.random.uniform(low=0., high=1., size=(1,))
    all_positions[i, :] = contractive_system(x0, time_steps, epsi)

# boundaries for system are [0,1]
x_bounds = [0., 1.]
parts_x_idx = [1., 0.5, 0.25, 0.125, 0.0625, 0.]
parts_per_axis = len(parts_x_idx)-1

all_trajectory_parts = partitions_1D(all_positions, parts_x_idx, N_traj, time_steps)

print(f'Partitions: {all_trajectory_parts}')

# find all ell-sequences from a trajectory
ell = 2

assert ell < time_steps

print(f'Total trajectories: {N_traj}. \nVisitable ell-sequences: {(parts_per_axis)**(n_vars*ell)}. ')

ell_seq_trajectory, ell_seq_init, ell_seq_rnd = get_sequences_stats(all_trajectory_parts, ell, time_steps)

if len(ell_seq_trajectory) > len(ell_seq_init) and len(ell_seq_trajectory) > len(ell_seq_rnd):
    print(f'No chance mate: visited {len(ell_seq_trajectory)}, initial: {len(ell_seq_init)}, '
          f'random: {len(ell_seq_rnd)}.')
    print(f'Visited: {ell_seq_trajectory}')
    print(f'Initial: {ell_seq_init}')
    print(f'Random: {ell_seq_rnd}')
elif len(ell_seq_trajectory) == len(ell_seq_rnd) and len(ell_seq_trajectory) > len(ell_seq_init):
    print(f'Random saves: visited {len(ell_seq_trajectory)}, initial: {len(ell_seq_init)}, '
          f'random: {len(ell_seq_rnd)}.')
else:
    print(f'Maybe, maybe, maybe....')



# new data
tmax = 39
t = np.arange(0, tmax+dt, dt)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
time_steps = tmax + 1
all_positions = np.zeros((N_traj, time_steps))
for i in range(N_traj):
    x0 = np.random.uniform(low=0., high=1., size=(1,))
    all_positions[i, :] = contractive_system(x0, time_steps, epsi)

# sequences of partitions
all_trajectory_parts = partitions_1D(all_positions, parts_x_idx, N_traj, time_steps)

# print(f'Sequences of partitions: {all_trajectory_parts}')
# find all ell-sequences from a trajectory
print(f'Total trajectories: {N_traj}. \nVisitable Partitions: {(parts_per_axis)**(n_vars*ell)}. ')

new_ell_seq_trajectory, new_ell_seq_init, new_ell_seq_rnd = get_sequences_stats(all_trajectory_parts, ell, time_steps)

tot_ell_visited = ell_seq_trajectory.union(new_ell_seq_trajectory)
tot_init_visited = ell_seq_init.union(new_ell_seq_init)
tot_rnd_visited = ell_seq_rnd.union(new_ell_seq_rnd)

print(f'New sequences seen: {len(tot_ell_visited)-len(ell_seq_trajectory)}')
print(f'New initial seen: {len(tot_init_visited)-len(ell_seq_init)}')
print(f'New random seen: {len(tot_rnd_visited)-len(ell_seq_rnd)}')

distribution_subplots(all_trajectory_parts, time_steps, parts_per_axis+1, n_vars, plot_every=5, ell=ell)


plt.show()
