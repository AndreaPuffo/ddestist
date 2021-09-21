from os import path
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy as sc
import seaborn as sns
import pandas as pd


basepath = path.dirname(__file__)

tmax = 20
N_traces = 50
sigma = 0.5
sigma_str = ''.join(str(sigma).split('.'))

folder_name = 'results/'
name = 'tmax'+str(tmax)+'_t'+str(N_traces)+'_s'+sigma_str

all_dfs = []

for file in os.listdir(folder_name):
    if file.startswith(name):
        # print(file)
        df = pd.read_pickle(folder_name+file)
        all_dfs.append(df)
        # print(df.to_string())

final_df = pd.concat(all_dfs)
print(final_df.to_string())
