# taken from https://docs.pymc.io/notebooks/bayesian_neural_network_advi.html#Generating-data
import timeit
from warnings import filterwarnings
import torch

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import seaborn as sns
import sklearn
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

from theano.sandbox.rng_mrg import MRG_RandomStream
from pymc3.theanof import set_tt_rng

set_tt_rng(MRG_RandomStream(42))


print(f"Running on PyMC3 v{pm.__version__}")


floatX = theano.config.floatX
filterwarnings("ignore")
sns.set_style("white")

##########################
# event triggered
##########################
tmax, delta_t = 2.0, 0.05
SEED = 42
N_traces = 150
sigma_trigger = 0.3**2
desired_components = ['norm_xi', 'angles']

print('Generating traces_lin...')
start = timeit.default_timer()
pg = PETC_Generation(n_traces=N_traces, tmax=tmax, delta_t=delta_t,
                     sigma_tr=sigma_trigger, seed=SEED)
X, Y, len_sequences = pg.get_data(desired_components, stdize=False)
print(f'Done. Elapsed Time: {timeit.default_timer()-start} sec')

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
                  n_epochs=int(5*1e4))
nnet1.learn(x_train=X_tensor, y=Y_tensor, batch_size=50)
end = timeit.default_timer()
print(f'Done. Elapsed Time: {end-start} sec')


yhat, _ = nnet1.forward(X_tensor)
mse = torch.norm(yhat-Y_tensor) / Y_tensor.shape[0]
print(f'MSE: {mse}')
spot_on_predictions = sum(torch.argmax(Y_tensor, dim=1) - torch.argmax(yhat, dim=1) == 0)
print(f'Accuracy: {spot_on_predictions/Y_tensor.shape[0] * 100} %')


###############################################
# BYES NET
###############################################
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)
plot_all_ist(X, Y)

# mini batch advi
# minibatch_x = pm.Minibatch(X_train, batch_size=50)
# minibatch_y = pm.Minibatch(Y_train, batch_size=50)
neural_network_minibatch = construct_nn(nnet1, X_train, Y_train,
                                        n_input, n_hidden_neurons, n_output)
print('Learning....')
start = timeit.default_timer()
with neural_network_minibatch:
    approx = pm.fit(20000, method=pm.ADVI())
end = timeit.default_timer()
print(f'Done. Elapsed Time: {end-start} sec')

plt.figure()
plt.plot(approx.hist)
plt.ylabel("ELBO")
plt.xlabel("iteration")


trace = approx.sample(draws=10000)
trace_plots(trace, nnet1, varnames=['b1', 'w_in_1'])


# create a sampler from the baynet, i.e. use the net as a r.v.
sample_proba = create_sampler_from_net(approx, neural_network_minibatch, X_train)
# sample and use it to generate predictions
pred = np.argmax(sample_proba(X_test, 10000).mean(axis=0), axis=1)  # > 0.5
print("Accuracy = {} %".format((Y_test == pred).mean() * 100))

grid = pm.floatX(np.mgrid[0.0:3.5:100j, -3.5:3.5:100j])
grid_2d = grid.reshape(n_input, -1).T
# dummy_out = np.ones(grid.shape[1], dtype=np.int8)

ppc = sample_proba(grid_2d, 1000)

#  Probability surface
for tau in range(np.max(Y_test)):
    plot_prob_surface(ppc, grid, X_test, Y_test, tau_ist=tau)

# uncertainty
plot_uncertainty(ppc, grid, X_test, Y_test)

plt.show()
