#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 10:42:43 2022

@author: bas
"""

import numpy as np

path_root='/home/bas/Documents/Projects/x-psi/model_datas/model_data/H-atmosphere_Spectra_fully_ionized/NSX_H-atmosphere_Spectra/'

path=path_root+'nsx_H_v171019.out'
size=(35, 11, 67, 166)
NSX = []
NSX = np.loadtxt(path, dtype=np.double)


#%%

# Now adding a dummy 5th variable, to modulate intensities by a factor 0.5 to 2. I also believe 10^-60 is the minimum, but that is ommitted for now.
modulator_size = 7
modulator = np.linspace(-0.3,0.3,num=modulator_size)


NSX_modulated = np.concatenate((NSX, np.zeros((NSX.shape[0],1))),axis=1)
NSX_combined = np.empty((0,NSX.shape[1]+1))
for i in range(modulator_size):
    NSX_modulated[:,2] = NSX[:,2]+modulator[i] #comment out +modulator[i] for a "no effect" variable
    NSX_modulated[:,5] = modulator[i]*np.ones(NSX.shape[0])
    NSX_combined = np.concatenate((NSX_combined, NSX_modulated))

np.savez_compressed(path_root+'nsx_H_v171019_5D_0dot5-2.npz', NSX_combined)
#np.savez_compressed(path_root+'nsx_H_v171019_5D_no_effect.npz', NSX_combined)

# to load this later, use:
# with np.load(path_root+'nsx_H_v171019_{...}.npz') as data:
#     mytest = data['arr_0.npy']
