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

from sklearn import datasets
from petc_generation import PETC_Generation
from baynet import construct_nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from plots import plot_all_ist, trace_plots, plot_prob_surface, plot_uncertainty
from estimation import Estim_Net
from utils import create_sampler_from_net
from abstraction_partitions import Abstraction, Partition

from theano.sandbox.rng_mrg import MRG_RandomStream
from pymc3.theanof import set_tt_rng

set_tt_rng(MRG_RandomStream(42))
warnings.simplefilter('always', ImportWarning)

print(f"Running on PyMC3 v{pm.__version__}")


floatX = theano.config.floatX
filterwarnings("ignore")
sns.set_style("white")

##########################
# event triggered
##########################
tmax, delta_t = 2.0, 0.05
SEED = 42
N_traces = 500
sigma = '01'
name = 'traces_lin/t'+str(N_traces)+'_s'+sigma+'.pkl'
df = pd.read_pickle(name)

X = df[['norm_xi', 'angles']].to_numpy()
Y = df['ist'].to_numpy(dtype=int)

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

n_hidden_neurons = 25

print('Torch Learning....')
start = timeit.default_timer()
nnet1 = Estim_Net(n_input, n_hidden_neurons, n_output,
                  n_epochs=int(5*1e1))
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

p1 = Partition(sample=X_train[0, :],
               lowers=[0.0, -np.pi], uppers=[np.max(X_train[:, 0])+0.01, np.pi],
               ist=np.argmax(Y_train[0, :]))

a = Abstraction()
a.add_p(p1)


print('Generating abstraction...')
start = timeit.default_timer()
for extra in range(2):
    for idx, sample in enumerate(X_train):
        point = sample
        new_ist = int(np.argmax(Y_train[idx, :]))
        candidate = (point, new_ist)
        # find partition
        p = a.belongs_to(point)
        if not p:
            print('Partition not found!')
            ValueError('Partition not found.')
        if p.get_ist() != new_ist:
            a.split_partition(p, candidate)


end = timeit.default_timer()
print(f'Done. Elapsed Time: {end-start} sec')


# plots
colors = ['b', 'r', 'g', 'y', 'k', 'm']
fig, ax = plt.subplots()
for partition in a.partitions:
    # print(f'Interval: ({partition.lowers}, {partition.uppers}), with color {colors[partition.get_ist()]}')
    # plt.scatter((partition.lowers[0]+partition.uppers[0])/2,
    #             (partition.lowers[1]+partition.uppers[1])/2,
    #             color=colors[partition.get_ist()])
    rect = patches.Rectangle(partition.lowers,
                             width=partition.uppers[0]-partition.lowers[0],
                             height=partition.uppers[1]-partition.lowers[1],
                             edgecolor=colors[partition.get_ist()],
                             facecolor=colors[partition.get_ist()])
    ax.add_patch(rect)
plt.xlim([-0.01, 1.01])
plt.ylim([-np.pi-0.01, np.pi+0.01])


for s in range(len(X_train)):
    plt.scatter(X_train[s, 0], X_train[s, 1], marker='x', c=colors[np.argmax(Y_train, axis=1)[s]])
plt.xlabel(r'|\xi|')
plt.ylabel(r'\phi')
# plt.xlim([0, 1.0])
# plt.ylim([-np.pi, np.pi])
#
# plt.figure()
# for s in range(len(X_train)):
#     plt.scatter(X_train[s, 0]*np.cos(X_train[s, 1]), X_train[s, 0]*np.sin(X_train[s, 1]),
#                 marker='x', c=colors[np.argmax(Y_train, axis=1)[s]])


plt.show()
