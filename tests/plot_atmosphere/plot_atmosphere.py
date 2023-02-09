#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:30:19 2023

@author: bas
"""

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import xpsi
from xpsi import HotRegions
from xpsi.global_imports import gravradius

import sys
sys.path.append('../')
from custom_tools import CustomPhotosphere_Bobrikova, CustomHotRegion_Accreting


# spacetime

# spacetime = xpsi.Spacetime.fixed_spin(300.0)


bounds = dict(distance = (0.1, 1.0),                     # (Earth) distance
                mass = (1.0, 3.0),                       # mass
                radius = (3.0 * gravradius(1.0), 16.0),  # equatorial radius
                cos_inclination = (0.0, 1.0))      # (Earth) inclination to rotation axis

spacetime = xpsi.Spacetime(bounds=bounds, values=dict(frequency=300.0))


# hotregions

bounds = dict(super_colatitude = (None, None),
              super_radius = (None, None),
              phase_shift = (0.0, 0.1),
              super_tbb = (0.00015, 0.003),
              super_te = (40., 200.),
              super_tau = (0.5, 3.5))

primary = CustomHotRegion_Accreting(bounds=bounds,
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


hot = HotRegions((primary,))

# photosphere
photosphere = CustomPhotosphere_Bobrikova(hot = hot, elsewhere = None,
                                values=dict(mode_frequency = spacetime['frequency']))
photosphere.hot_atmosphere = '/home/bas/Documents/Projects/x-psi/model_datas/bobrikova/Bobrikova_compton_slab.npz'

# length = 100
# E = np.linspace(0.001, 0.1, length)#np.ones(length) * 4.38815419e-03
# mu = np.ones(length) * 9.18016000e-01
# t_e = 132.1
# t_bb = 0.00271
# tau = 3.21
# local_vars = np.array([[t_e, t_bb, tau]]*length)

# hot_I = xpsi.surface_radiation_field.intensity_no_norm(E, mu, local_vars,
#                                                atmosphere=photosphere.hot_atmosphere,
#                                                extension='hot',
#                                                numTHREADS=2)

# hot_I = xpsi.surface_radiation_field.intensity(E, mu, local_vars,
#                                                atmosphere=photosphere.hot_atmosphere,
#                                                extension='hot',
#                                                numTHREADS=2)

# fig, ax = plt.subplots()
# ax.loglog(E, hot_I*E)

length = 100
E = np.linspace(1, 10, length) #keV
E_erg = E*1.602e-9 #ergs
#E_erest = E * 0.5109989e3 #electron rest energy
mu = 1 # cos(zenith angle)
mu_array = np.ones(length) * mu
t_e = 50 #keV
t_bb = 1 #keV
tau = 1 #unitless
local_vars = np.array([[t_e/0.511, t_bb/511, tau]]*length)

hot_I = xpsi.surface_radiation_field.intensity(E, mu_array, local_vars,
                                               atmosphere=photosphere.hot_atmosphere,
                                               extension='hot',
                                               numTHREADS=1) # photons/s/keV/cm^2
fig, ax = plt.subplots()
ax.loglog(E, hot_I*E_erg)
ax.set_title('mu={:0.2f}, te={:.2f} [keV], tbb={:.2f} [keV], tau={:.2f} [-]'.format(mu, t_e, t_bb, tau), loc='center') #unit conversion te and tbb is different due to a cluster leftover according to Anna B.
ax.set_xlabel('E (keV)')
ax.set_ylabel('E*I (ergs) * (photons/s/keV/cm^2)')
plt.show()


