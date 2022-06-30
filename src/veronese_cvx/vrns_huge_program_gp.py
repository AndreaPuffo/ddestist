import timeit

import numpy as np
import tqdm
import matplotlib.pyplot as plt
from gurobipy import GRB

from utils import load_samples
from src.ovo_ovr_svm.l_complete_abs import divide_l_ists
import n_sphere as ns
from src.traffic_autom import TrafficAutomaton
from src.ovo_ovr_svm.ovo_utils import plot_subroutines, remove_blocking_states, compute_epsilons_homemade_clf_ovr, \
    create_ovr_clf, veronese_embedding
from src.ovo_ovr_svm.ovo_utils import scenario_eps

##########################
# DATA LOAD
##########################
tmax, delta_t = 2, 0.05
SEED = 42
np.random.seed(SEED)
N_traces = 20000
sigma = 0.1

# system definition
mode = 'linear'
noise = False
dims = 3

X, Y, len_sequences = load_samples(tmax, N_traces, sigma, mode, noise, dims)
Y = np.atleast_2d(Y).T
kmax = np.max(Y)+1

max_length = 50000

X0, Y0, l0 = X.copy(), Y.copy(), len_sequences.copy()[:max_length]
# len_sequences = len_sequences[:max_length]


x_plotting = ns.convert_spherical(X, digits=10)
cmap = plt.get_cmap('jet')
new_cmap = cmap(np.linspace(0, 1, int(np.max(Y)+1)))
for y in np.unique(Y):
    idx = np.where(Y == y)[0]
    plt.scatter(x_plotting[idx, -2], x_plotting[idx, -1], color=new_cmap[int(y)], cmap='jet', s=5, label=y)
plt.legend()
# plt.show()


print(f'Number of unique ISTs: {len(np.unique(Y))}')


nci = 3
print('-'*80)
print(f'Consecutive ISTs: {nci}')

number_of_consecutive_ist = nci
X, Y, len_sequences, aist_series, label_map_1, translation_map_1 = divide_l_ists(X0, Y0, l0,
                                                                                 l=number_of_consecutive_ist)

print(f'Number of unique l-tuples ISTs: {len(np.unique(Y))} (out of possible {kmax**number_of_consecutive_ist} labels)')


x_plotting = ns.convert_spherical(X, digits=10)
cmap = plt.get_cmap('jet')
new_cmap = cmap(np.linspace(0, 1, np.maximum(np.max(Y), np.max(Y))+1))
plt.figure()
for y in np.unique(Y):
    idx = np.where(Y == y)[0]
    plt.scatter(x_plotting[idx, -2], x_plotting[idx, -1], color=new_cmap[y], cmap='jet', s=5, label=y)
plt.legend()
# plt.show()

# VERONESE EMBEDDING
X = veronese_embedding(X, full_poly=False)

# split train and test data
np.random.seed(167)
d = X.shape[1]
N = int(X.shape[0] * 0.5)  # np.minimum(6000, int(X.shape[0] // 1.5))
TEST = X.shape[0] - N
L = np.max(Y) + 1

random_indx = np.random.permutation(X.shape[0])
Y = Y[:, 0]
X_tr, Y_tr = X[random_indx[:N], :], Y[random_indx[:N]]
X_test, Y_test = X[random_indx[N:], :], Y[random_indx[N:]]


# # create seprate Ys for each label
# separate_y_tr = -np.ones((len(np.unique(Y_tr)), Y_tr.shape[0]))
# for idx, label in enumerate(np.unique(Y_tr)):
#     # separate_y[idx, np.where(Y_tr != label)[0]] = -1
#     separate_y_tr[idx, np.where(Y_tr > label)[0]] = 1
#
# # create seprate Ys for each label
# separate_y_test = -np.ones((len(np.unique(Y_test)), Y_test.shape[0]))
# for idx, label in enumerate(np.unique(Y_test)):
#     # separate_y[idx, np.where(Y_tr != label)[0]] = -1
#     separate_y_test[idx, np.where(Y_test > label)[0]] = 1


# Compute a trade-off curve and record train and test error.
train_error = []
test_error = []  # np.zeros(TRIALS)
positive_xi = []
test_violated_constr = []

scenario_lows = []
scenario_ups = []
n_labels = len(np.unique(Y))

# Form SVM with L1 regularization problem.
import gurobipy as gp

# Create a new model
m = gp.Model("SVM_veronese")

m.setParam('outputFlag', 0)
cviol_tol = 1e-6
m.setParam('FeasibilityTol', cviol_tol)

allw = m.addVars(d*n_labels, name='w', lb=-GRB.INFINITY, ub=GRB.INFINITY)
allb = m.addVars(n_labels, name='b', lb=-GRB.INFINITY, ub=GRB.INFINITY)

start_optim = timeit.default_timer()
norm_w = gp.QuadExpr()
for i in range(d*n_labels):
    norm_w += allw[i] ** 2

# one constraint per sample
theta = m.addVars(N, lb=0)

sum_theta = gp.QuadExpr()
for i in range(N):
    sum_theta += theta[i]

for ist_level in tqdm.tqdm(range(n_labels)):

    print('-'*80)
    print(f'Hierarchy Level: {ist_level}')

    # Create variables
    w = [allw[i] for i in range(ist_level*d,(ist_level+1)*d)]
    b = allb[ist_level]
    # loss = cp.sum( beta**2 )

    # constraints
    constraints = []
    # for every sample
    for i in range(N):
        c = 0.
        # samples must obey W * X <= 0
        if Y_tr[i] == ist_level:
            for j in range(d):
                c += (w[j] * X_tr[i,j])
            const = c + b + theta[i] >= 0.5/L
        # samples must be W * X > 0
        else:
            for j in range(d):
                c += (w[j] * X_tr[i,j])
            const = 0.5/L + c + b <= theta[i]

        m.addConstr(const)


rho_vals = 1e12

objective = (1./rho_vals)*norm_w + rho_vals * sum_theta
m.setObjective(objective, GRB.MINIMIZE)

m.optimize()
if m.status >= 3:  # 'infeasible':
    print(f'Mate somethings wrong, problem not feasible. Status: {m.status}')
else:
    print(f'Problem {ist_level} solved. Status: {m.status}')
    m.printStats()

end_optim = timeit.default_timer()

print(f'time: {end_optim-start_optim}')


allW = np.array([allw[i].X for i in range(n_labels*d)]).reshape((n_labels, d))
allB = np.array([allb[i].X for i in range(n_labels)]).reshape((n_labels, 1))

for hierarchy_level in range(n_labels):
    W = np.array([allw[i].X for i in range(hierarchy_level*d,(hierarchy_level+1)*d)])

    print(W)

    # number_of_positive_xi = ( np.sum( (Y_tr<=hierarchy_level) * np.sign(W @ X_tr.T )>0 ) +
    #                        np.sum ( (Y_tr>hierarchy_level) * np.sign(W @ X_tr.T)<0 ))
    number_of_positive_xi = ( np.sum( (Y_tr == hierarchy_level) * (W @ X_tr.T < 1/L ) ) +
                           np.sum ( (Y_tr != hierarchy_level) * (W @ X_tr.T > cviol_tol ) ) )
    positive_xi.append(number_of_positive_xi / N)
    train_error.append(  ( np.sum( (Y_tr == hierarchy_level) * (W @ X_tr.T < -cviol_tol ) ) +
                           np.sum ( (Y_tr != hierarchy_level) * (W @ X_tr.T > cviol_tol ) ) )/ N )
    test_error.append( ( np.sum( (Y_test == hierarchy_level) * (W @ X_test.T < -cviol_tol ) ) +
                           np.sum( (Y_test != hierarchy_level) * (W @ X_test.T > cviol_tol ) ) ) / TEST )

    test_violated_constr.append( ( np.sum( (Y_test == hierarchy_level) * (W @ X_test.T < 1./L ) ) +
                           np.sum( (Y_test != hierarchy_level) * (W @ X_test.T > cviol_tol ) ) ) / TEST )

    e_low, e_up = scenario_eps(k=number_of_positive_xi, N=N, beta=1e-4)
    scenario_ups.append(e_up)
    scenario_lows.append(e_low)

# Plot the train and test error over the trade-off curve.
print(f'Train Errors: {train_error}')
print(f'Test Errors: {test_error}')
print(f'Train Violations: {positive_xi}')
print(f'Test Violations: {test_violated_constr}')

print(f'Scenario Low: {scenario_lows}')
print(f'Scenario Ups: {scenario_ups}')


plt.figure()
plt.plot(range(len(np.unique(Y))), train_error, label="Train misclass")
plt.plot(range(len(np.unique(Y))), test_error, label="Test misclass")
plt.plot(range(len(np.unique(Y))), positive_xi, label=r"Train Violat. Constr.")
plt.plot(range(len(np.unique(Y))), test_violated_constr, label=r"Test Violat. Constr.")

plt.plot(range(len(np.unique(Y))), scenario_lows, '--', label=r"$\epsilon_l$")
plt.plot(range(len(np.unique(Y))), scenario_ups, '--', label=r"$\epsilon_u$")
#
# plt.xscale('log')
plt.legend(loc='upper left')
plt.xlabel(r"Cones", fontsize=16)
# plt.ylim([0, 0.01])
plt.grid()

plt.show()