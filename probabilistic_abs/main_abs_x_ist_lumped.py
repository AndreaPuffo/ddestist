import timeit
from warnings import filterwarnings
import torch
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy as sc
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
from utils import my_ceil, my_floor, normalize_until, domain_splitting, compute_aist_x_ist_lumped

from sklearn.model_selection import train_test_split
from estimation import Estim_Net
from abstraction_partitions import Partition, Abstraction


warnings.simplefilter('always', ImportWarning)

filterwarnings("ignore")
sns.set_style("white")

"""
compute the ISt with the NN, 
partitions are based on color (the predicted IST) and the position in the state space.
in practice, splits colors in different regions of the state space
"""

##########################
# event triggered
##########################
noise = False

tmax = 10
SEED = 42
torch.manual_seed(SEED)
N_traces = 50
sigma = '01'  # '00038099999999999996'

folder_name = 'traces'
if noise:
    folder_name = folder_name + '_noise'

name = folder_name + '_lin/tmax'+str(tmax)+'_t'+str(N_traces)+'_s'+sigma
if noise:
    name = name + '_noise'

name_data = name +'.pkl'
name_lens = name + '_len' +'.pkl'
df = pd.read_pickle(name_data)

X = df[['norm_xi', 'angles']].to_numpy()
Y = df['ist'].to_numpy(dtype=int)

df_lens = pd.read_pickle(name_lens)
len_sequences = df_lens['len_sequences'].to_numpy()

n_output = int(np.max(Y))+1
n_input = X.shape[1]

########################################
# DATA PREP
########################################
Y_outputs = np.zeros((X.shape[0], n_output))
for idx in range(X.shape[0]):
    Y_outputs[idx, Y[idx]] = 1.0

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_outputs, test_size=0.6, shuffle=False)
X_tensor = torch.tensor(X_train).type(torch.FloatTensor)
Y_tensor = torch.tensor(Y_train).type(torch.FloatTensor)
X_cv_tensor = torch.tensor(X_test).type(torch.FloatTensor)
Y_cv_tensor = torch.tensor(Y_test).type(torch.FloatTensor)
# test_splits randomises the data, reconstruct Y
allX = torch.vstack([X_tensor, X_cv_tensor]).type(torch.FloatTensor)
allY = torch.vstack([Y_tensor, Y_cv_tensor])
all_ist = torch.argmax(allY, dim=1).type(torch.FloatTensor)

aist_series, traces = [], []
idx_so_far = 0
for l in range(len(len_sequences)):
    traces += [X_tensor[idx_so_far:idx_so_far+len_sequences[l], :]]
    aist_series += [all_ist[idx_so_far:idx_so_far+len_sequences[l]].mean()]
    idx_so_far += len_sequences[l]

print('AIST (train data): \n', aist_series)
print(f'Mean AIST (train data): {np.array(aist_series).mean()}')

idx_lowest_aist = torch.sort(torch.stack(aist_series)).indices[0]
idx_highest_aist = torch.sort(torch.stack(aist_series)).indices[-1]

########################################
# TORCH NET
########################################

n_hidden_neurons = 25
n_epochs = 1e4

print('-'*80)
print('Learning....')
start_lrn = timeit.default_timer()
nnet1 = Estim_Net(n_input, n_hidden_neurons, n_output,
                  n_epochs=int(n_epochs))
nnet1.learn(x_train=X_tensor, y=Y_tensor, batch_size=100)
end_lrn = timeit.default_timer()
print(f'Done. Elapsed Time: {end_lrn-start_lrn} sec')
yhat, _ = nnet1.forward(X_tensor)
mse = torch.norm(yhat-Y_tensor) / Y_tensor.shape[0]
print(f'MSE: {mse}')
spot_on_predictions = sum(torch.argmax(Y_tensor, dim=1) - torch.argmax(yhat, dim=1) == 0)
print(f'Accuracy: {spot_on_predictions/Y_tensor.shape[0] * 100} %')

yhat_cv, _ = nnet1.forward(X_cv_tensor)
mse = torch.norm(yhat_cv-Y_cv_tensor) / Y_cv_tensor.shape[0]
print(f'MSE (CV): {mse}')
spot_on_predictions = sum(torch.argmax(Y_cv_tensor, dim=1) - torch.argmax(yhat_cv, dim=1) == 0)
print(f'Accuracy (CV): {spot_on_predictions/Y_cv_tensor.shape[0] * 100} %')


#######################################
# ABSTRACTION
#######################################
"""
partitions classical by partitioning state space
the abstraction is generated counting the number of transitions between a partition and another,
then normalised to sum to 1.
"""

start_build_abstraction = timeit.default_timer()

n_parts = 100
lb = my_floor(np.min(X, axis=0), precision=1)
ub = my_ceil(np.max(X, axis=0), precision=1) + 0.1

parts_lb, parts_ub, parts_ctr = domain_splitting(lb, ub, n_parts)
# compute IST for each partition (assuming the center is informative)
probab_ist_ctr, _ = nnet1.forward(torch.tensor(parts_ctr).type(torch.FloatTensor))
probab_ist_lb, _ = nnet1.forward(torch.tensor(parts_lb).type(torch.FloatTensor))
probab_ist_ub, _ = nnet1.forward(torch.tensor(parts_ub).type(torch.FloatTensor))

probab_ist_traces, _ = nnet1.forward(allX)


predicted_ist_ctr = torch.argmax(probab_ist_ctr, dim=1).type(torch.FloatTensor)
predicted_ist_lb = torch.argmax(probab_ist_lb, dim=1).type(torch.FloatTensor)
predicted_ist_ub = torch.argmax(probab_ist_ub, dim=1).type(torch.FloatTensor)

predicted_ist_min = torch.min(torch.stack([predicted_ist_lb, predicted_ist_ctr, predicted_ist_ub]), dim=0).values
predicted_ist_max = torch.max(torch.stack([predicted_ist_lb, predicted_ist_ctr, predicted_ist_ub]), dim=0).values

predicted_ist_traces = torch.argmax(probab_ist_traces, dim=1).type(torch.FloatTensor)


def find_partitions(samples):
    prods = [n_parts**idx for idx in reversed(range(samples.shape[1]))]
    parts_number = np.sum(np.floor((samples - lb) / (ub - lb) * n_parts) * prods, axis=1)
    return (parts_number).astype(int)


color_plots = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'gold', 'grey',
               'seagreen', 'deeppink', 'maroon', 'salmon', 'teal', 'khaki', 'crimson']


# TRANSITION MATRIX
parts_samples = find_partitions(X)

# divide data into traces
out_traces, x_traces = [], []
idx_so_far = 0
for l in range(len(len_sequences)):
    out_traces += [predicted_ist_traces[idx_so_far:idx_so_far+len_sequences[l]]]
    x_traces += [X[idx_so_far:idx_so_far+len_sequences[l]]]
    idx_so_far += len_sequences[l]


abstr = Abstraction()
abstr.add_from_list(sample_list=[-1]*len(parts_lb), lb_list=parts_lb, ct_list=parts_ctr, ub_list=parts_ub,
                    ist_list=predicted_ist_ctr.numpy())

all_col_neig, col_neig_indexes = abstr.find_colored_partitions()

# #TODO: A POSTERIORI CHECK OF THE COLORED PARTITIONS TO AVOID DOUBLE DEFINITIONS

n_states = len(all_col_neig)
print(f'The abstraction has {n_states} states.')
print(col_neig_indexes)


# dim = n_output + 1 (uncertain)
overhead_trace = 0
# instatiate sparse matrix
transition_matrix = sc.sparse.dok_matrix((n_states, n_states))
for x_tr in x_traces:
    for idx in range(x_tr.shape[0]-1):
        start = np.where([parts_samples[idx+overhead_trace] in sublist for sublist in col_neig_indexes])
        targt = np.where([parts_samples[idx+overhead_trace+1] in sublist for sublist in col_neig_indexes])
        transition_matrix[start, targt] += 1
    overhead_trace += x_tr.shape[0]
norms = sc.sum(transition_matrix, axis=1)
# fix zeros in norms, replace with 1
idxs_zeros = np.where(norms == 0)
norms[idxs_zeros] = 1.
transition_matrix[idxs_zeros, idxs_zeros] = 1.

normalised_T = normalize_until(transition_matrix)

end_build_abstraction = timeit.default_timer()
print('-'*80)
print(f'Abstraction Built. Elapsed Time: {end_build_abstraction-start_build_abstraction}')

# plots
if n_parts < 151 and n_input < 5:
    fig, ax = plt.subplots()
    prdct_ist_np = predicted_ist_ctr.numpy()
    for g in np.unique(prdct_ist_np):
        ix = np.where(prdct_ist_np == g)
        ax.scatter(parts_ctr[ix][:, 0], parts_ctr[ix][:, 1], c=color_plots[int(g)],
                   label=g, s=7)
    plt.grid()
    plt.legend()
    plt.xlabel(r'Radius $r$')
    plt.ylabel(r'Angle $\phi$ [$-\pi$, $\pi$]')

# if n_parts < 5:
#     w, v = np.linalg.eig(normalised_T.T)
#     print('Transition Matrix: \n', normalised_T)
#     print('Eigenvals: \n', w)
# # print('Eigenvects: \n', v)
#     print('Eigenvector for 1: \n', v[:, np.where(w==1)]/sum(v[:, np.where(w==1)]))


predicted_ists = torch.stack([predicted_ist_ctr[col_part.pop()] for col_part in col_neig_indexes])

print('-'*80)
print(f'Start Computation of AIST.')
start_compute_aist = timeit.default_timer()

selected_trace = 0
compute_aist_x_ist_lumped(aist_series, out_traces, parts_samples, n_states,
                          selected_trace, len_sequences, predicted_ists,
                          col_neig_indexes, trans_matrix=normalised_T)

selected_trace = idx_lowest_aist
compute_aist_x_ist_lumped(aist_series, out_traces, parts_samples, n_states,
                          selected_trace, len_sequences, predicted_ists, col_neig_indexes,
                          trans_matrix=normalised_T, label='Lowest')

selected_trace = idx_highest_aist
compute_aist_x_ist_lumped(aist_series, out_traces, parts_samples, n_states,
                          selected_trace, len_sequences, predicted_ists, col_neig_indexes,
                          trans_matrix=normalised_T, label='Highest')

end_compute_aist = timeit.default_timer()
print('-'*80)
print(f'AIST Computed. Elapsed Time: {end_compute_aist-start_compute_aist}')

# print('='*80)
# print(predicted_ist_ctr)
# print('Centers')
# print(parts_ctr)

plt.show()
