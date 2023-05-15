#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 11:40:52 2023

@author: bas
"""

import matplotlib.pyplot as plt
import dill as pickle


atmosphere_types = ['A', 'N','A', 'N']
n_params=[4,4,5,5]#[4,5]

ldict = {}
delta_time_avg = {}
ne_list ={}
lastdict = {}
delta_llh = {}

fig, axes = plt.subplots(2,1, sharex=True)

# for n_param in n_params:
for atmosphere_type, n_param in zip(atmosphere_types, n_params):
    key = f'{atmosphere_type}{n_param}'
    print(key)
    with open(f'timed_likelihood_{atmosphere_type}{n_param}.pkl', 'rb') as file:
          (ldict[key], delta_time_avg[key], ne_list[key]) = pickle.load(file)


    lastdict[key] = ldict[key][ne_list[key][-1]]
    delta_llh[key] = []
     
    for num_energy in ne_list[key]:
        tmpdict = ldict[key][num_energy]
        # for call in tmpdict:
        #     print(tmpdict[call]['loglikelihood']-lastdict[call]['loglikelihood'])
        delta_llh[key].append(abs(tmpdict[0]['loglikelihood']-lastdict[key][0]['loglikelihood']))

    axes[0].semilogy(ne_list[key], delta_time_avg[key], 'x', label=f'type={key}')
    axes[0].set_xlabel('num. energies')
    axes[0].set_ylabel('time (s)')
    axes[0].set_title('avg. time to compute likelihood (s)')

    axes[1].semilogy(ne_list[key], delta_llh[key], 'x')
    axes[1].set_xlabel('num. energies')
    axes[1].set_ylabel(r'deviation ($\Delta$ln)')
    axes[1].set_title(r'Deviation from llh(max num. energies)')
    
axes[0].legend()
plt.tight_layout()

fig.savefig(f'plots/likelihood_timing_and_accuracy_{atmosphere_types}{n_params}.png', dpi=300)