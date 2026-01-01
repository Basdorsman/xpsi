#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 08:31:07 2025

@author: bas
"""

import os
import sys
this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory+'/../')

import numpy as np
from multiprocessing import Pool
from combine_kdes_STU import analysis


Analysis = analysis('test', 
                    'disk_NICER', 
                    sampler='multi', 
                    scenario='J1444_STU', 
                    eos_informed=False,
                    channel_min=100)
Analysis()



ndraws_total = 1e4

nproc=2
chunk_size = int(ndraws_total/nproc)
chunk_sizes = [chunk_size] * nproc   

def draw_wrapper(ndraws):
    samples = Analysis.prior.draw(ndraws=ndraws)[0]
    return samples

with Pool(processes=nproc) as pool:
    # inverse sampling test
    results_list = pool.map(draw_wrapper, chunk_sizes)
    # test=Analysis.prior.draw(ndraws=100)[0]

results = np.concatenate(results_list, axis=0)
    
# names_dictionary = Analysis.pv.names()
# labels_dictionary = Analysis.pv.labels()
# axis_labels = [labels_dictionary[key] for key in names_dictionary]
 


# np.savetxt(f'prior_draws={ndraws_total}.txt',results)

#%%


import corner
figure=corner.corner(results)#, labels=axis_labels[:19], label_kwargs={'fontsize': 12},)
figure.tight_layout()
