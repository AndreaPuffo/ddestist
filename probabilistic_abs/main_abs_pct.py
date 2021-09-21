# taken from https://docs.pymc.io/notebooks/bayesian_neural_network_advi.html#Generating-data
import timeit
from warnings import filterwarnings
import torch
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pymc3 as pm
import seaborn as sns
import sklearn
import pandas as pd
import theano
from utils import create_partitions, compute_min_max_aist_with_pct, \
    compute_trans_matrix_unctn_state, compute_min_max_aist_unctn_state, \
    collapse_subparts_trmatrix

from sklearn.model_selection import train_test_split
from estimation import Estim_Net

from theano.sandbox.rng_mrg import MRG_RandomStream
from pymc3.theanof import set_tt_rng

set_tt_rng(MRG_RandomStream(42))
warnings.simplefilter('always', ImportWarning)

floatX = theano.config.floatX
filterwarnings("ignore")
sns.set_style("white")

##########################
# event triggered
##########################
tmax = 2
SEED = 42
torch.manual_seed(SEED)
N_traces = 50
sigma = '01'
name = 'traces_4D_lin/tmax'+str(tmax)+'_t'+str(N_traces)+'_s'+sigma
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

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y_outputs, test_size=0.6, shuffle=False)
# X_tensor = torch.tensor(X_train).type(torch.FloatTensor)
# Y_tensor = torch.tensor(Y_train).type(torch.FloatTensor)
# X_cv_tensor = torch.tensor(X_test).type(torch.FloatTensor)
# Y_cv_tensor = torch.tensor(Y_test).type(torch.FloatTensor)
# # test_splits randomises the data, reconstruct Y
# allY = torch.vstack([Y_tensor, Y_cv_tensor])
# all_ist = torch.argmax(allY, dim=1).type(torch.FloatTensor)

X_tensor, Y_tensor = torch.tensor(X).type(torch.FloatTensor), torch.tensor(Y_outputs).type(torch.FloatTensor)
all_ist = torch.tensor(Y).type(torch.FloatTensor)

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
n_epochs = 5*1e4

print('Classical Learning....')
start = timeit.default_timer()
nnet1 = Estim_Net(n_input, n_hidden_neurons, n_output,
                  n_epochs=int(n_epochs))
nnet1.learn(x_train=X_tensor, y=Y_tensor, batch_size=50)
end = timeit.default_timer()
print(f'Done. Elapsed Time: {end-start} sec')
yhat, _ = nnet1.forward(X_tensor)
mse = torch.norm(yhat-Y_tensor) / Y_tensor.shape[0]
print(f'MSE: {mse}')
spot_on_predictions = sum(torch.argmax(Y_tensor, dim=1) - torch.argmax(yhat, dim=1) == 0)
print(f'Accuracy: {spot_on_predictions/Y_tensor.shape[0] * 100} %')


#######################################
# ABSTRACTION
#######################################
"""
partitions are the ist values. e.g. if ist = 1,2,3,4 * h, there are 4 super-partitions.
the abstraction is generated counting the number of transitions between a partition and another,
then normalised to sum to 1.
each super-partition is then divided based on the percent of each IST
e.g. 
Super-partition 1*h
Partition 1: 90+% probabi of 1*h, 10% rest
Partition 2: 80+% probabi of 1*h, 20% rest
...
Partition 7: 30+% probabi of 1*h, 70% rest
and so on, for every possible IST
"""
allX = torch.vstack([X_tensor])
probab_ist, _ = nnet1.forward(allX)

subparts = 7
min_parts = 10 - subparts

unceert, uncert_avg, uncert_max, uncert_min, \
certain_features, idx_features, predicted_ist, prob_pct = \
    create_partitions(n_out=n_output, probab_ist=probab_ist, X=allX, min_parts=min_parts)


# BUILD PARTITIONS

if unceert != []:
    unceert = torch.stack(unceert)
    uncert_avg = uncert_avg.item()/unceert.shape[0]
    plt.scatter(unceert[:, 0], unceert[:, 1], c='k', label='uncertain')
    print(f'Uncertainty: \n',
          f'Max: {uncert_max}, Min: {uncert_min}, Avg: {uncert_avg}')
else:
    print('Uncertain is empty!')
for feature in range(n_output):
    if certain_features[feature] != []:
        certain_features[feature] = torch.stack(certain_features[feature])

# plt.figure()
for feature in range(n_output):
    if certain_features[feature] != []:
        plt.scatter(certain_features[feature][:, 0], certain_features[feature][:, 1],
                    label=str(feature))

plt.grid()
plt.legend()

plt.figure()
color_maps = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples', 'YlOrBr', 'YlOrRd', 'Greys',
              'Accent', 'Accent_r',  'Blues_r', 'BrBG',
              'BrBG_r', 'BuGn', 'BuGn_r',
              'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r',
              'Greens_r', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges_r',
              'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2',
              'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr',
              'PuOr_r', 'PuRd', 'PuRd_r', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy',
              'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r',
              'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral',
              'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r',
              'YlOrBr_r', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn',
              'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr',
              'bwr_r', 'cet_gray', 'cet_gray_r', 'cividis', 'cividis_r', 'cool', 'cool_r',
              'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix',
              'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r',
              'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r',
              'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg',
              'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r',
              'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r',
              'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral',
              'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r',
              'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic',
              'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r',
              'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain',
              'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted',
              'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter',
              'winter_r'
              ]

idx_color = 0
for feature in range(n_output):
    if certain_features[feature] != []:
        plt.scatter(certain_features[feature][:, 0], certain_features[feature][:, 1],
                    c=torch.max(probab_ist, dim=1).values[idx_features[feature]].detach().numpy(),
                    cmap=color_maps[idx_color], label=str(feature))
        # plt.colorbar()
        idx_color += 1
plt.grid()

# TRANSITION MATRIX

# divide data into traces
out_traces, pct_traces = [], []
idx_so_far = 0
for l in range(len(len_sequences)):
    out_traces += [predicted_ist[idx_so_far:idx_so_far+len_sequences[l]]]
    pct_traces += [prob_pct[0, idx_so_far:idx_so_far+len_sequences[l]]]
    idx_so_far += len_sequences[l]

# each super-partition is divided into 7 partitions (numbers from 3 to 9, included)
# uncertain has no sub-partitions
# dim = (n_output * 7)  + 1 (uncertain)
transition_matrix = np.zeros((n_output*subparts+1, n_output*subparts+1))
for out_tr, pct_tr in zip(out_traces, pct_traces):
    for idx in range(out_tr.shape[0]-1):
        start = int(out_tr[idx])*subparts + int(pct_tr[idx]) - min_parts
        end = int(out_tr[idx+1])*subparts + int(pct_tr[idx+1]) - min_parts
        transition_matrix[start, end] += 1
norms = np.sum(transition_matrix, axis=1)
# fix zeros in norms, replace with 1
idxs_zeros = np.where(norms == 0)
norms[idxs_zeros] = 1.

normalised_T = (transition_matrix.T/norms).T
normalised_T[idxs_zeros, idxs_zeros] = 1.

w, v = np.linalg.eig(normalised_T.T)

if n_output < 0:
    print('Transition Matrix: \n', normalised_T)
    print('Eigenvals: \n', w)
# print('Eigenvects: \n', v)
    print('Eigenvector for 1: \n', v[:, 0]/sum(v[:, 0]))

selected_trace = idx_lowest_aist
uncert_values = [uncert_min, uncert_avg, uncert_max]
compute_min_max_aist_with_pct(aist_series, out_traces, selected_trace,
                              subparts, min_parts,
                              uncert_values, len_sequences, normalised_T,
                              n_output, X, predicted_ist, prob_pct, label='Lowest with Pct')

# no sub-partitions
T_no_sub = compute_trans_matrix_unctn_state(n_output, out_traces)
compute_min_max_aist_unctn_state(aist_series, out_traces, selected_trace,
                                    uncert_values, len_sequences, T_no_sub,
                                    n_output, X, predicted_ist, label='Lowest without Pct')

small_t = collapse_subparts_trmatrix(transition_matrix, n_output, subparts)
assert (small_t == T_no_sub).all()

plt.show()
