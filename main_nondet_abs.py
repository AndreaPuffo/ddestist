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
import theano.tensor as T

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
tmax, delta_t = 10, 0.05
SEED = 42
torch.manual_seed(SEED)
N_traces = 50
sigma = '03'
name = 'traces_lin/tmax'+str(tmax)+'_t'+str(N_traces)+'_s'+sigma
name_data = name +'.pkl'
name_lens = name + '_len' +'.pkl'
df = pd.read_pickle(name_data)

X = df[['norm_xi', 'angles']].to_numpy()
Y = df['ist'].to_numpy(dtype=int)

df_lens = pd.read_pickle(name_lens)
len_sequences = df_lens['len_sequences'].to_numpy()
# len_sequences = np.insert(len_sequences, 0, 0)

aist_series = []
idx_so_far = 0
for l in range(len(len_sequences)):
    aist_series += [Y[idx_so_far:idx_so_far+len_sequences[l]].mean()]
    idx_so_far += len_sequences[l]

print('AIST: \n', aist_series)
print(f'Mean AIST: {np.array(aist_series).mean()}')

n_output = int(np.max(Y))+1
n_input = X.shape[1]

########################################
# TORCH NET
########################################
Y_outputs = np.zeros((X.shape[0], n_output))
for idx in range(X.shape[0]):
    Y_outputs[idx, Y[idx]] = 1.0

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_outputs, test_size=0.6)
X_tensor = torch.tensor(X_train).type(torch.FloatTensor)
Y_tensor = torch.tensor(Y_train).type(torch.FloatTensor)

X_cv_tensor = torch.tensor(X_test).type(torch.FloatTensor)
Y_cv_tensor = torch.tensor(Y_test).type(torch.FloatTensor)

n_hidden_neurons = 15
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
partitions are the ist values. e.g. if ist = 1,2,3,4 * h, there are 4 partitions.
the abstraction is generated counting the number of transitions between a partition and another,
then normalised to sum to 1.
"""
allX = torch.vstack([X_tensor, X_cv_tensor])
probab_ist, _ = nnet1.forward(allX)
predicted_ist = torch.argmax(probab_ist, dim=1)

transition_matrix = np.zeros((n_output, n_output))

for idx in range(predicted_ist.shape[0]-1):
    transition_matrix[predicted_ist[idx], predicted_ist[idx+1]] += 1
norms = np.sum(transition_matrix, axis=1)
# fix zeros in norms, replace with 1
idxs_zeros = np.where(norms == 0)
norms[idxs_zeros] = 1.

normalised_T = (transition_matrix.T/norms).T

w, v = np.linalg.eig(normalised_T.T)

print('Transition Matrix: \n', normalised_T)
print('Eigenvals: \n', w)
print('Eigenvects: \n', v)
print('Eigenvector for 1: \n', v[:, 0]/sum(v[:, 0]))

first_x0 = X_tensor[0, :]
first_ist = torch.argmax(yhat, dim=1)[0].item()

pi = np.zeros((1, n_output))
pi[0, first_ist] = 1.

for i in range(len_sequences[0]):
    pi = pi @ normalised_T
print(f'Computed pi: {pi}')

aist_trsys = np.sum([i * pi[0, i] for i in range(pi.shape[1])])

print(f'AIST from transys: {aist_trsys}')

