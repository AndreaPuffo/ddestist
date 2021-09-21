import timeit
from warnings import filterwarnings

import pandas as pd
import torch
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from petc_3d import PETC3D

warnings.simplefilter('always', ImportWarning)

filterwarnings("ignore")
sns.set_style("white")

##########################
# event triggered
##########################
tmax, delta_t = 60, 0.1
SEED = 42
torch.manual_seed(SEED)
N_traces = 150
# sigma in [0.1 ... 0.9]
# sigma = 0.3

for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    sigma_squared = sigma**2

    noise = False

    folder_name = 'traces_3D'

    name = folder_name + '_lin/tmax' + str(tmax) + '_t' + str(N_traces) + '_s' + ''.join(str(sigma).split('.'))

    data_generator = PETC3D(n_traces=N_traces, tmax=tmax, delta_t=delta_t,
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

    data_df = pd.DataFrame(data=np.array([X[0,:], X[1,:], X[2,:],
                                          Y]).T,
                           columns=['norm_xi', 'angles', 'angles', 'ist'])
    data_len_df = pd.DataFrame(data=np.array(len_sequences), columns=['len_sequences'])
    end = timeit.default_timer()
    print(f'Elapsed Time: {end-start} sec')

    name_data = name+'.pkl'
    name_lens = name+'_len'+'.pkl'
    data_df.to_pickle(name_data)
    data_len_df.to_pickle(name_lens)
