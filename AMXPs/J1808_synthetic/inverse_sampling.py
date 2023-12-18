#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 16:05:22 2023

@author: bas
"""

from main import likelihood
import numpy as np
import matplotlib.pyplot as plt

print('RUNNING TESTS..')
n_samples = 10000
# p_sampled = np.empty((n_samples, 12))
# for sample in range(n_samples):
    #p_sampled[sample] = np.asarray(likelihood.prior.inverse_sample())
p_sampled, acceptance_fraction = likelihood.prior.draw(n_samples)




n_parameters = 12
parameters = ['mass(solar)', 'radius(km)', 'distance(kpc)', 'cos_i(cosine)', 'phase', 'hs_colatitude (radian?)', 'hs_radius (radian)', 'hs_Tbb (data)', 'hs_Te (data)', 'hs_tau','else_T(log10 K)','nH (1e21, cm^-2)']
fig, axes = plt.subplots(12,1, figsize=(4,2*n_parameters))

for ax, parameter, index in zip(axes, parameters, range(n_parameters)):
    ax.hist(p_sampled[:,index])
    ax.set_title(parameter)
    
plt.tight_layout()