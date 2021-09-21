import timeit
from warnings import filterwarnings

import pandas as pd
import torch
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from petc_8d import PETC8D

warnings.simplefilter('always', ImportWarning)

filterwarnings("ignore")
sns.set_style("white")

##########################
# event triggered
##########################
tmax, delta_t = 2, 0.01
SEED = 42
torch.manual_seed(SEED)
N_traces = 500
sigma = 0.1
sigma_squared = sigma**2

noise = False

folder_name = 'traces_8D'

name = folder_name + '_lin/tmax' + str(tmax) + '_t' + str(N_traces) + '_s' + ''.join(str(sigma).split('.'))

data_generator = PETC8D(n_traces=N_traces, tmax=tmax, delta_t=delta_t,
                        sigma_tr=sigma_squared, seed=SEED, noise=noise)


start = timeit.default_timer()
desired_components = ['norm_xi', 'angles']
X, Y, len_sequences, all_data = data_generator.get_data(desired_components)
end = timeit.default_timer()

# for idx in range(len(all_data)):
#     plt.plot(all_data[idx].T)
# plt.grid()
# plt.show()

if noise:
    name = name + '_noise'

data_df = pd.DataFrame(data=np.array([X[0,:], X[1,:], X[2,:], X[3,:],
                                      X[4,:], X[5,:], X[6,:], X[7,:],
                                      Y]).T,
                       columns=['norm_xi',
                                'angles', 'angles', 'angles', 'angles',
                                'angles', 'angles', 'angles',
                                'ist'])
data_len_df = pd.DataFrame(data=np.array(len_sequences), columns=['len_sequences'])
end = timeit.default_timer()
print(f'Elapsed Time: {end-start} sec')

name_data = name+'.pkl'
name_lens = name+'_len'+'.pkl'
data_df.to_pickle(name_data)
data_len_df.to_pickle(name_lens)
