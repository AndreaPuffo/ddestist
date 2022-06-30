import sys
import numpy as np
import tqdm
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pendulum_utils import plot_init_conditions, partitions_sequences, get_sequences_stats, \
    distribution_subplots, generate_vanderpol_traj


# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 5.8, 0.2
t = np.arange(0, tmax+dt, dt)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
N_traj = 20000
time_steps = round(tmax/dt) + 1

traj_domain_bounds = [-2.5, 2.5]
all_positions = generate_vanderpol_traj(N_traj, t, time_steps, traj_domain_bounds)


# boundaries for van der pol are [-3, -3], [3, 3]
parts_per_axis = 5+1
x_bounds, y_bounds = [-4, 4], [-4, 4]
parts_x_idx = np.linspace(x_bounds[0], x_bounds[1], parts_per_axis)
parts_y_idx = np.linspace(y_bounds[0], y_bounds[1], parts_per_axis)

all_trajectory_parts = partitions_sequences(all_positions, [parts_x_idx, parts_y_idx], N_traj,
                                            time_steps, parts_per_axis)

print(f'Partitions: {all_trajectory_parts}')

# find all ell-sequences from a trajectory
ell = 3
print(f'Total trajectories: {N_traj}. \nVisitable ell-sequences: {(parts_per_axis-1)**(2*ell)}. ')

ell_seq_trajectory, ell_seq_init, ell_seq_rnd = get_sequences_stats(all_trajectory_parts, ell, time_steps)

if len(ell_seq_trajectory) > len(ell_seq_init) and len(ell_seq_trajectory) > len(ell_seq_rnd):
    print(f'No chance mate: visited {len(ell_seq_trajectory)}, initial: {len(ell_seq_init)}, '
          f'random: {len(ell_seq_rnd)}.')
elif len(ell_seq_trajectory) == len(ell_seq_rnd) and len(ell_seq_trajectory) > len(ell_seq_init):
    print(f'Random saves: visited {len(ell_seq_trajectory)}, initial: {len(ell_seq_init)}, '
          f'random: {len(ell_seq_rnd)}.')
else:
    print(f'Maybe, maybe, maybe....')



# new data
tmax, dt = 24.8, 0.2
t = np.arange(0, tmax+dt, dt)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
time_steps = round(tmax/dt) + 1
all_positions = generate_vanderpol_traj(N_traj, t, time_steps, traj_domain_bounds)
# sequences of partitions
all_trajectory_parts = partitions_sequences(all_positions, [parts_x_idx, parts_y_idx], N_traj,
                                            time_steps, parts_per_axis)

print(f'Sequences of partitions: {all_trajectory_parts}')
# find all ell-sequences from a trajectory
print(f'Total trajectories: {N_traj}. \nVisitable Partitions: {(parts_per_axis-1)**(2*ell)}. ')

new_ell_seq_trajectory, new_ell_seq_init, new_ell_seq_rnd = get_sequences_stats(all_trajectory_parts, ell, time_steps)

tot_ell_visited = ell_seq_trajectory.union(new_ell_seq_trajectory)
tot_init_visited = ell_seq_init.union(new_ell_seq_init)
tot_rnd_visited = ell_seq_rnd.union(new_ell_seq_rnd)

print(f'New sequences seen: {len(tot_ell_visited)-len(ell_seq_trajectory)}')
print(f'New initial seen: {len(tot_init_visited)-len(ell_seq_init)}')
print(f'New random seen: {len(tot_rnd_visited)-len(ell_seq_rnd)}')

n_vars = 2
distribution_subplots(all_trajectory_parts, time_steps, parts_per_axis, n_vars, N_traj, plot_every=5)


plt.show()
