import timeit
from warnings import filterwarnings

import pandas as pd
import torch
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nl_petc_generation import PETC_Generation_NL
from petc_generation import PETC_Generation
import theano

warnings.simplefilter('always', ImportWarning)

floatX = theano.config.floatX
filterwarnings("ignore")
sns.set_style("white")

##########################
# event triggered
##########################
tmax, delta_t = 20, 0.05
SEED = 42
torch.manual_seed(SEED)
N_traces = 50

sigma = 0.3  # 0.0127 * 0.3
sigma_squared = sigma**2

mode = 'linear'
noise = False

folder_name = 'traces'

if noise:
    folder_name = folder_name + '_noise'

if mode == 'linear':
    data_generator = PETC_Generation(n_traces=N_traces, tmax=tmax, delta_t=delta_t,
                                     sigma_tr=sigma_squared, seed=SEED, noise=noise)
    name = folder_name + '_lin/tmax' + str(tmax) + '_t' + str(N_traces) + '_s' + ''.join(str(sigma).split('.'))
elif mode == 'nonlinear':
    data_generator = PETC_Generation_NL(n_traces=N_traces, tmax=tmax, delta_t=delta_t,
                                        sigma_tr=sigma_squared, seed=SEED, noise=noise)
    name = folder_name + '_nonlin/tmax' + str(tmax) + '_t' + str(N_traces) + '_s' + ''.join(str(sigma).split('.'))

start = timeit.default_timer()
desired_components = ['norm_xi', 'angles']
X, Y, len_sequences = data_generator.get_data(desired_components)

if noise:
    name = name + '_noise'

data_df = pd.DataFrame(data=np.array([X[:, 0], X[:, 1], Y]).T, columns=['norm_xi', 'angles', 'ist'])
data_len_df = pd.DataFrame(data=np.array(len_sequences), columns=['len_sequences'])
end = timeit.default_timer()
print(f'Elapsed Time: {end-start} sec')

name_data = name+'.pkl'
name_lens = name+'_len'+'.pkl'
data_df.to_pickle(name_data)
data_len_df.to_pickle(name_lens)
