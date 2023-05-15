#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 16:52:14 2023

@author: bas
"""


import matplotlib.pyplot as plt
import dill as pickle
import numpy as np

atmosphere_type = 'A'
n_params=5

with open(f'timed_likelihood_reduced_{atmosphere_type}{n_params}.pkl', 'rb') as file:
     (ldict, delta_time_avg, reduction) = pickle.load(file)

fig, ax = plt.subplots()#2,1, sharex=True)

ax.semilogy(reduction, delta_time_avg, 'x')#, label=f'params={n_param}')
ax.set_xlabel('num. energies')
ax.set_ylabel('time (s)')
ax.set_title('avg. time to compute likelihood (s)')

# axes[1].semilogy(ne_list[n_param], delta_llh[n_param], 'x')
# axes[1].set_xlabel('num. energies')
# axes[1].set_ylabel(r'deviation ($\Delta$ln)')
# axes[1].set_title(r'Deviation from llh(max num. energies)')

# axes[0].legend()
plt.tight_layout()


#%%

fig, ax = plt.subplots()

diff = []
loglikelihood = []

for call in ldict[0]:
    diff.append(abs(ldict[0][call]['loglikelihood']-ldict[1][call]['loglikelihood']))
    loglikelihood.append(ldict[0][call]['loglikelihood'])

diff = np.asarray(diff)
loglikelihood = np.asarray(loglikelihood)
small = diff<1e85

ax.scatter(loglikelihood[small], diff[small])
ax.set_xlabel('loglikelihood')
ax.set_ylabel(r'$\Delta$loglikelihood')
    
plt.show()