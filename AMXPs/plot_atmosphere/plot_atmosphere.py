#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:30:19 2023

@author: bas
"""

# from __future__ import print_function, division
import os

import numpy as np
import matplotlib.pyplot as plt

import xpsi
from xpsi import HotRegions
from xpsi.global_imports import gravradius

import sys
sys.path.append('../')
from custom_tools import CustomHotRegion_Accreting, CustomHotRegion_Accreting_te_const, CustomPhotosphere_Accreting, CustomPhotosphere_Accreting_te_const


# parameters

try: #try to get parameters from shell input
    os.environ.get('atmosphere_type')    
    atmosphere_type = os.environ['atmosphere_type']
    os.environ.get('n_params')
    n_params = os.environ['n_params']
except:
    atmosphere_type = "A"
    atmosphere_type = "5"

print("atmosphere_type:", atmosphere_type)
print("n_params:", n_params)

# spacetime

# spacetime = xpsi.Spacetime.fixed_spin(300.0)


bounds = dict(distance = (0.1, 1.0),                     # (Earth) distance
                mass = (1.0, 3.0),                       # mass
                radius = (3.0 * gravradius(1.0), 16.0),  # equatorial radius
                cos_inclination = (0.0, 1.0))      # (Earth) inclination to rotation axis

spacetime = xpsi.Spacetime(bounds=bounds, values=dict(frequency=300.0))


# hotregions
if n_params=="4":
    bounds = dict(super_colatitude = (None, None),
                  super_radius = (None, None),
                  phase_shift = (0.0, 0.1),
                  super_tbb = (0.00015, 0.003),
                  super_tau = (0.5, 3.5))
    
    primary = CustomHotRegion_Accreting_te_const(bounds=bounds,
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

elif n_params=="5":
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

if n_params=="4":
    photosphere = CustomPhotosphere_Accreting_te_const(hot = hot, elsewhere = None,
                                    values=dict(mode_frequency = spacetime['frequency']))
elif n_params=="5":
    photosphere = CustomPhotosphere_Accreting(hot = hot, elsewhere = None,
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
# E = np.linspace(1, 10, length) #keV
# E_erg = E*1.602e-9 #ergs
# #E_erest = E * 0.5109989e3 #electron rest energy
mu = 1 # cos(zenith angle)
mu_array = np.ones(length) * mu
# t_e = 50 #keV
# t_bb = 1 #keV
# tau = 1 #unitless
# local_vars = np.array([[t_e/0.511, t_bb/511, tau]]*length)

# hot_I = xpsi.surface_radiation_field.intensity(E, mu_array, local_vars,
#                                                atmosphere=photosphere.hot_atmosphere,
#                                                extension='hot',
#                                                numTHREADS=1) # photons/s/keV/cm^2
# fig, ax = plt.subplots()
# ax.loglog(E, hot_I*E_erg)
# ax.set_title('mu={:0.2f}, te={:.2f} [keV], tbb={:.2f} [keV], tau={:.2f} [-]'.format(mu, t_e, t_bb, tau), loc='center') #unit conversion te and tbb is different due to a cluster leftover according to Anna B.
# ax.set_xlabel('E (keV)')
# ax.set_ylabel('E*I (ergs) * (photons/s/keV/cm^2)')
# plt.show()

# length = 100
E_value = 2
E = E_value * np.ones(length) #keV
E_erg = E*1.602e-9 #ergs
mu_value = 1 # cos(zenith angle)
mu = np.ones(length) * mu_value


t_e_value = 40*0.511 #keV #30
if n_params=="5":
    t_e = np.ones(length)*t_e_value #keV

t_bb_min = 0.511 # min = 0.511
t_bb_max = 0.7 # max = 1.5841
t_bb = np.linspace(t_bb_min,t_bb_max,length) #keV

tau_value=1
tau = np.ones(length)*tau_value #unitless


if n_params=="4":
    local_vars = np.ascontiguousarray(np.swapaxes(np.array((t_bb/511, tau)),0,1))
elif n_params=="5":
    local_vars = np.ascontiguousarray(np.swapaxes(np.array((t_e/0.511, t_bb/511, tau)),0,1))


# print("local_vars:", local_vars)
hot_I = xpsi.surface_radiation_field.intensity(E, mu_array, local_vars,
                                               atmosphere=photosphere.hot_atmosphere,
                                               extension='hot',
                                               numTHREADS=8) # photons/s/keV/cm^2

# hot_I2 = np.zeros((length))
# for ii in range(0,length):
#     local_varsX = np.array([[t_e[ii]/0.511, t_bb[ii]/511, tau[ii]]]*1)
# #     print("local_varsX: ", local_varsX)
#     hot_IX = xpsi.surface_radiation_field.intensity(np.array([E[ii]]),
#                                                     np.array([mu_array[ii]]),
#                                                     local_varsX, atmosphere=photosphere.hot_atmosphere,
#                                                     extension='hot',
#                                                     numTHREADS=8) 
#     hot_I2[ii] = hot_IX

# fig, axes = plt.subplots(1,2, figsize=(8,3))
# Is = (hot_I, hot_I2)

# for ax, I in zip(axes, Is):
#     ax.loglog(t_bb, I*E*E_erg)
#     ax.set_xlabel('t_bb (keV)')
#     ax.set_ylabel('E*I (ergs) * (photons/s/cm^2)')
# plt.title('mu={:0.2f}, tau={:.2f} [-], t_e={:.2f} [keV], E_value={:.2f} [keV]'.format(mu_value, tau_value, t_e_value, E_value), loc='center') #unit conversion te and tbb is different due to a cluster leftover according to Anna B.
# plt.show()

fig, ax = plt.subplots()
ax.loglog(t_bb, hot_I*E*E_erg)
ax.set_xlabel('t_bb (keV)')
ax.set_ylabel('E*I (ergs) * (photons/s/cm^2)')
plt.title('mu={:0.2f}, tau={:.2f} [-], t_e={:.2f} [keV], E_value={:.2f} [keV]'.format(mu_value, tau_value, t_e_value, E_value), loc='center') #unit conversion te and tbb is different due to a cluster leftover according to Anna B.
plt.show()

fig.savefig('plots/mu={:0.2f}_tau={:.2f}_t_e={:.2f}_E_value={:.2f}_{:}.png'.format(mu_value, tau_value, t_e_value, E_value, n_params))
print('fig saved in plots/mu={:0.2f}_tau={:.2f}_t_e={:.2f}_E_value={:.2f}_{:}.png'.format(mu_value, tau_value, t_e_value, E_value, n_params))