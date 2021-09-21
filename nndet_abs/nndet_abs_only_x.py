# taken from https://docs.pymc.io/notebooks/bayesian_neural_network_advi.html#Generating-data
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
from utils import compute_aist_from_adj_matrix, \
    find_contraction_const, load_samples, data_split_and_prep, learning_net, build_ss_abstraction, save_results_pkl

warnings.simplefilter('always', ImportWarning)

filterwarnings("ignore")
sns.set_style("white")

"""
the partition is computed in a classical way, i.e. split the state space into hyper rectangles
the abstraction is non-deterministic, only 0/1 in the adjacency matrix
"""

##########################
# DATA LOAD
##########################
tmax, delta_t = 20, 0.05
SEED = 42
torch.manual_seed(SEED)
N_traces = 50
sigma = 0.5

X, Y, len_sequences = load_samples(tmax, N_traces, sigma)

n_output = int(np.max(Y))+1
n_input = X.shape[1]

########################################
# DATA PREP
########################################

X_tr, Y_tr, X_cv, Y_cv, allX, aist_series = data_split_and_prep(X, Y, len_sequences, n_output)

idx_lowest_aist = torch.sort(torch.stack(aist_series)).indices[0]
idx_highest_aist = torch.sort(torch.stack(aist_series)).indices[-1]


########################################
# TORCH NET
########################################

n_hidden_neurons = 25
n_epochs = 2*1e4

nnet, mse_train, accuracy_train, mse_cv, accuracy_cv, lrn_time = learning_net(n_input, n_hidden_neurons, n_output, n_epochs, X_tr, Y_tr, X_cv, Y_cv)


#######################################
# ABSTRACTION
#######################################
"""
partitions classical by partitioning state space
the abstraction is generated counting the number of transitions between a partition and another,
then normalised to sum to 1.
"""


n_parts = 200
tot_partitions = n_parts ** n_input
adj_matrix, predicted_ist_min, predicted_ist_ctr, predicted_ist_max, \
    out_traces, parts_samples, abs_time = build_ss_abstraction(n_parts, X, nnet, allX, len_sequences)


######################################
# COMPUTE AIST
######################################

predicted_ists = [predicted_ist_min, predicted_ist_ctr, predicted_ist_max]

print('-'*80)
print(f'Start Computation of AIST.')

selected_trace = 0
true_aist, estim_aist, \
min_aist_trsys, aist_trsys, max_aist_trsys, time_rnd = \
    compute_aist_from_adj_matrix(aist_series, out_traces, parts_samples, tot_partitions,
                             selected_trace, len_sequences, predicted_ists, adj_matrix,
                                 label='')

selected_trace = idx_lowest_aist
true_aist_low, estim_aist_low, \
min_aist_trsys_low, aist_trsys_low, max_aist_trsys_low, time_low = \
    compute_aist_from_adj_matrix(aist_series, out_traces, parts_samples, tot_partitions,
                                 selected_trace, len_sequences, predicted_ists, adj_matrix,
                                 label='Lowest')

selected_trace = idx_highest_aist
true_aist_high, estim_aist_high, \
min_aist_trsys_high, aist_trsys_high, max_aist_trsys_high, time_high = \
    compute_aist_from_adj_matrix(aist_series, out_traces, parts_samples, tot_partitions,
                                 selected_trace, len_sequences, predicted_ists, adj_matrix,
                                 label='Highest')

########################################
# DATA REC
########################################
"""
data saved look like this:
        data                        
tmax | n_traces | sigma | 
            training
n_hidden | mse_train | mse_cv | lrn_time |
            abstraction
tot_states | time_build | 
            AIST
true_aist | estimated_aist | m,~,M_AIST_from_partitions | avg_AIST_comp_time
            
"""
result_rec = False
if result_rec:

    perf = [
        true_aist, estim_aist, min_aist_trsys, aist_trsys, max_aist_trsys, time_rnd,
        true_aist_low, estim_aist_low, min_aist_trsys_low, aist_trsys_low, max_aist_trsys_low, time_low,
        true_aist_high, estim_aist_high, min_aist_trsys_high, aist_trsys_high, max_aist_trsys_high, time_high
        ]
    save_results_pkl(tmax, N_traces, sigma, n_hidden_neurons,
                     mse_train, accuracy_train, mse_cv, accuracy_cv, lrn_time,
                     tot_partitions, abs_time, perf)

plt.show()
