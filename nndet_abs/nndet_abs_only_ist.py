# taken from https://docs.pymc.io/notebooks/bayesian_neural_network_advi.html#Generating-data
import timeit
from warnings import filterwarnings
import torch
import warnings
from os import path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import sklearn
import pandas as pd
import theano
import theano.tensor as T

from sklearn.model_selection import train_test_split
from estimation import Estim_Net
from utils import load_samples, build_color_abstraction, compute_aist_from_color_adj, \
    data_split_and_prep, learning_net

warnings.simplefilter('always', ImportWarning)

floatX = theano.config.floatX
filterwarnings("ignore")
sns.set_style("white")

##########################
# event triggered
##########################
tmax, delta_t = 10, 0.05
SEED = 42
torch.manual_seed(SEED)
N_traces = 50

sigma = 0.4
sigma_str = ''.join(str(sigma).split('.'))

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
partitions are the ist values. e.g. if ist = 1,2,3,4 * h, there are 4 partitions.
the abstraction is generated counting the number of transitions between a partition and another,
then normalised to sum to 1.
"""

adj_matrix, out_traces, parts_samples = build_color_abstraction(allX, len_sequences, nnet)
selected_trace = 0
label = ''
compute_aist_from_color_adj(aist_series, out_traces, parts_samples,
                                n_output,
                                selected_trace, len_sequences,
                                adj_matrix, label)

selected_trace = idx_lowest_aist
label = 'Lowest'
compute_aist_from_color_adj(aist_series, out_traces, parts_samples,
                                n_output,
                                selected_trace, len_sequences,
                                adj_matrix, label)

selected_trace = idx_highest_aist
label = 'Highest'
compute_aist_from_color_adj(aist_series, out_traces, parts_samples,
                                n_output,
                                selected_trace, len_sequences,
                                adj_matrix, label)