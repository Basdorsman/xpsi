#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 14:16:07 2022

@author: bas
"""

from __future__ import print_function, division
import os
import numpy as np

import xpsi
from xpsi.global_imports import gravradius, _keV, _k_B

import sys
sys.path.append('../')
from custom_tools import CustomPhotosphere_4D, CustomPhotosphere_5D


n_params = os.environ['dimensionality']
print("n_params: ",n_params)

np.random.seed(xpsi._rank+10)

print('Rank reporting: %d' % xpsi._rank)

################################## SPACETIME ##################################

spacetime = xpsi.Spacetime.fixed_spin(300.0)

bounds = dict(distance = (0.1, 1.0),                     # (Earth) distance
                mass = (1.0, 3.0),                       # mass
                radius = (3.0 * gravradius(1.0), 16.0),  # equatorial radius
                cos_inclination = (0.0, 1.0))      # (Earth) inclination to rotation axis

spacetime = xpsi.Spacetime(bounds=bounds, values=dict(frequency=300.0))


################################## HOTREGIONS #################################

bounds = dict(super_colatitude = (None, None),
              super_radius = (None, None),
              phase_shift = (0.0, 0.1),
              super_temperature = (5.1, 6.8))

primary = xpsi.HotRegion(bounds=bounds,
	                    values={},
	                    symmetry=False, #call general integrator instead of for azimuthal invariance
	                    omit=False,
	                    cede=False,
	                    concentric=False,
	                    sqrt_num_cells=32,
	                    min_sqrt_num_cells=10,
	                    max_sqrt_num_cells=64,
	                    num_leaves=100,
	                    num_rays=200,
	                    prefix='p') 		                    

bounds['super_temperature'] = None # declare fixed/derived variable

class derive(xpsi.Derive):
    def __init__(self):
        """
        We can pass a reference to the primary here instead
        and store it as an attribute if there is risk of
        the global variable changing.

        This callable can for this simple case also be
        achieved merely with a function instead of a magic
        method associated with a class.
        """
        pass

    def __call__(self, boundto, caller = None):
        # one way to get the required reference
        #global primary # unnecessary, but for clarity
        return primary['super_temperature'] - 0.2

secondary = xpsi.HotRegion(bounds=bounds, # can otherwise use same bounds
	                      values={'super_temperature': derive()},
	                      symmetry=False, #call general integrator instead of for azimuthal invariance
	                      omit=False,
	                      cede=False, 
	                      concentric=False,
	                      sqrt_num_cells=32,
	                      min_sqrt_num_cells=10,
	                      max_sqrt_num_cells=100,
	                      num_leaves=100,
	                      num_rays=200,
	                      do_fast=False,
	                      is_antiphased=True,
	                      prefix='s') 


from xpsi import HotRegions
hot = HotRegions((primary, secondary))
h = hot.objects[0]
hot['p__super_temperature'] = 6.0 # equivalent to ``primary['super_temperature'] = 6.0``


################################ ATMOSPHERE ################################### 

if n_params == "4":   
    photosphere = CustomPhotosphere_4D(hot = hot, elsewhere = None,
                                    values=dict(mode_frequency = spacetime['frequency']))
    photosphere.hot_atmosphere = '/home/bas/Documents/Projects/x-psi/model_datas/model_data/H-atmosphere_Spectra_fully_ionized/NSX_H-atmosphere_Spectra/nsx_H_v171019.npz'

elif n_params== "5":
    photosphere = CustomPhotosphere_5D(hot = hot, elsewhere = None,
                                    values=dict(mode_frequency = spacetime['frequency']))
    #photosphere.hot_atmosphere = '/home/bas/Documents/Projects/x-psi/model_datas/model_data/H-atmosphere_Spectra_fully_ionized/NSX_H-atmosphere_Spectra/nsx_H_v171019_5D_no_effect.npz'
    photosphere.hot_atmosphere = '/home/bas/Documents/Projects/x-psi/model_datas/model_data/H-atmosphere_Spectra_fully_ionized/NSX_H-atmosphere_Spectra/nsx_H_v171019_modulated_0dot5_to_2.npz'

else:
    print("no dimensionality provided!")


atmosphere_num = photosphere._hot_atmosphere
#print(atmosphere_num)

from integrator_stripped import interpolate

def get_E_nsx(E_input, T):
    k_B_over_keV = _k_B / _keV
    E_nsx = np.log10(E_input/(k_B_over_keV*10**T))
    return E_nsx

def get_E_input(E_nsx, T):
    k_B_over_keV = _k_B / _keV
    E_input = k_B_over_keV*10**(E_nsx+T)
    return E_input

Threads = 1
E_nsx = -0.78
T = 5.1
cos_zenith=0.001
g = 13.8
expect_I_E = -18.0981

# 4D examples, use e.g. "add_dummy_dimension_to_NSX.py" for NSX values.
# example 1 (NSX[12928,:]):
# E_nsx = -0.78
# T = 5.1
# cos_zenith=0.001
# g = 13.8
# expect_I_E = -18.0981

# example 2 (NSX[2000000,:]):
# E_nsx = 1.42
# T = 5.9
# cos_zenith=0.33380686
# g = 14
# expect_I_E = -17.8141

E_input = get_E_input(E_nsx, T)
I_10_3T = interpolate(Threads, E_input, cos_zenith, g, T, atmosphere = atmosphere_num)
print("expecation for log10(I_E) = ", expect_I_E)
print("log10(I_E) =",np.log10(I_10_3T/(10**(3*T))))
