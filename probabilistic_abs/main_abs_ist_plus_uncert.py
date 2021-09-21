# taken from https://docs.pymc.io/notebooks/bayesian_neural_network_advi.html#Generating-data
import timeit
from warnings import filterwarnings
import torch
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import sklearn
import pandas as pd
import theano

from sklearn.model_selection import train_test_split
from estimation import Estim_Net
from utils import compute_min_max_aist_unctn_state, compute_trans_matrix_unctn_state


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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_outputs,
                                                    test_size=0.7, shuffle=False)
X_tensor = torch.tensor(X_train).type(torch.FloatTensor)
Y_tensor = torch.tensor(Y_train).type(torch.FloatTensor)
X_cv_tensor = torch.tensor(X_test).type(torch.FloatTensor)
Y_cv_tensor = torch.tensor(Y_test).type(torch.FloatTensor)
# test_splits randomises the data, reconstruct Y
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

yhat_cv, _ = nnet1.forward(X_cv_tensor)
mse = torch.norm(yhat_cv-Y_cv_tensor) / Y_cv_tensor.shape[0]
print(f'MSE (CV): {mse}')
spot_on_predictions = sum(torch.argmax(Y_cv_tensor, dim=1) - torch.argmax(yhat_cv, dim=1) == 0)
print(f'Accuracy (CV): {spot_on_predictions/Y_cv_tensor.shape[0] * 100} %')

#######################################
# ABSTRACTION
#######################################
"""
partitions are the ist values. e.g. if ist = 1,2,3,4 * h, there are 4 partitions.
the abstraction is generated counting the number of transitions between a partition and another,
then normalised to sum to 1.
"""
allX = torch.vstack([X_tensor, X_cv_tensor])
probab_ist, _ = nnet1.forward(allX)
predicted_ist = torch.argmax(probab_ist, dim=1).type(torch.FloatTensor)

certain_features = [[] for i in range(n_output)]

# uncertain
unceert = []
uncert_avg, uncert_max, uncert_min = 0., -1, n_output+1
for sample in range(probab_ist.shape[0]):
    idx_max_ = torch.sort(probab_ist[sample, :], descending=True).indices[0]
    idx_second_max_ = torch.sort(probab_ist[sample, :], descending=True).indices[1]
    # uncertainty
    if torch.abs(probab_ist[sample, idx_max_]-probab_ist[sample, idx_second_max_]) < 1./(n_output+1):
        unceert.append(allX[sample])
        # if uncertain, set the partition to n_output + 1 (which is n_output starting from zero)
        predicted_ist[sample] = n_output
        most_prob_ist = idx_max_
        if most_prob_ist > uncert_max:
            uncert_max = most_prob_ist.item()
        if most_prob_ist < uncert_min:
            uncert_min = most_prob_ist.item()
        uncert_avg += most_prob_ist
    else:
        certain_features[idx_max_] += [allX[sample]]

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
plt.xlabel(r'Radius $r$')
plt.ylabel(r'Angle $\phi$')
plt.grid()
plt.legend()

# TRANSITION MATRIX

# divide data into traces
out_traces = []
idx_so_far = 0
for l in range(len(len_sequences)):
    out_traces += [predicted_ist[idx_so_far:idx_so_far+len_sequences[l]]]
    idx_so_far += len_sequences[l]

normalised_T = compute_trans_matrix_unctn_state(n_output, out_traces)

w, v = np.linalg.eig(normalised_T.T)

if n_output < 5:
    print('Transition Matrix: \n', normalised_T)
    print('Eigenvals: \n', w)
# print('Eigenvects: \n', v)
    print('Eigenvector for 1: \n', v[:, 0]/sum(v[:, 0]))

uncert_values = [uncert_min, uncert_avg, uncert_max]

selected_trace = 0
compute_min_max_aist_unctn_state(aist_series, out_traces, selected_trace,
                                    uncert_values, len_sequences, normalised_T,
                                    n_output, X, predicted_ist, label='')

selected_trace = idx_lowest_aist
compute_min_max_aist_unctn_state(aist_series, out_traces, selected_trace,
                                    uncert_values, len_sequences, normalised_T,
                                    n_output, X, predicted_ist, label='Lowest')

selected_trace = idx_highest_aist
compute_min_max_aist_unctn_state(aist_series, out_traces, selected_trace,
                                    uncert_values, len_sequences, normalised_T,
                                    n_output, X, predicted_ist, label='Highest')


plt.show()
