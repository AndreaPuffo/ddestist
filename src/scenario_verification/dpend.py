import sys
import numpy as np
import tqdm
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pendulum_utils import plot_init_conditions, partitions_sequences, get_sequences_stats, \
    distribution_subplots, generate_traj_doublepend, partitions_sequences_theta


# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 0.9, 0.1
t = np.arange(0, tmax+dt, dt)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
N_traj = 20000
time_steps = round(tmax/dt) + 1
friction = False
all_positions = generate_traj_doublepend(N_traj, t, time_steps, friction_flag=friction)

# plot one trajectory
# plt.plot(all_y[0, :, :])
# plt.show()

if N_traj <= 1000:
    # for t in range(time_steps//10):
    fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
    ax = fig.add_subplot(111)
    plot_init_conditions(all_positions, ax)


# boundaries for pendulum are [-2, -2], [2, 2] with top part actually empty
parts_per_axis = 3+1
x_bounds, y_bounds = [0., 2*np.pi], [0., 2*np.pi]
parts_x_idx = np.linspace(x_bounds[0], x_bounds[1], parts_per_axis)
parts_y_idx = np.linspace(y_bounds[0], y_bounds[1], parts_per_axis)

all_trajectory_parts = partitions_sequences(all_positions, [parts_x_idx, parts_y_idx], N_traj,
                                            time_steps, parts_per_axis)

n_vars = 2
# parts_theta = np.linspace(0., 2*np.pi, parts_per_axis)
# all_trajectory_parts = partitions_sequences_theta(all_positions, parts_theta, N_traj,
#                                             time_steps, parts_per_axis)


print(f'Partitions: {all_trajectory_parts}')

# find all ell-sequences from a trajectory
ell = 5
print(f'Total trajectories: {N_traj}. \nVisitable ell-sequences: {(parts_per_axis-1)**(n_vars*ell)}. ')

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
tmax = 4.9
t = np.arange(0, tmax+dt, dt)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
time_steps = round(tmax/dt) + 1
all_positions = generate_traj_doublepend(N_traj, t, time_steps, friction_flag=friction)
# sequences of partitions
# all_trajectory_parts = partitions_sequences_theta(all_positions, parts_theta, N_traj,
#                                             time_steps, parts_per_axis)
all_trajectory_parts = partitions_sequences(all_positions, [parts_x_idx, parts_y_idx], N_traj,
                                            time_steps, parts_per_axis)

print(f'Sequences of partitions: {all_trajectory_parts}')
# find all ell-sequences from a trajectory
print(f'Total trajectories: {N_traj}. \nVisitable Partitions: {(parts_per_axis-1)**(n_vars*ell)}. ')

new_ell_seq_trajectory, new_ell_seq_init, new_ell_seq_rnd = get_sequences_stats(all_trajectory_parts, ell, time_steps)

tot_ell_visited = ell_seq_trajectory.union(new_ell_seq_trajectory)
tot_init_visited = ell_seq_init.union(new_ell_seq_init)
tot_rnd_visited = ell_seq_rnd.union(new_ell_seq_rnd)

print(f'New sequences seen: {len(tot_ell_visited)-len(ell_seq_trajectory)}')
print(f'New initial seen: {len(tot_init_visited)-len(ell_seq_init)}')
print(f'New random seen: {len(tot_rnd_visited)-len(ell_seq_rnd)}')

distribution_subplots(all_trajectory_parts, time_steps, parts_per_axis, n_vars, plot_every=5, ell=ell)


plt.show()
