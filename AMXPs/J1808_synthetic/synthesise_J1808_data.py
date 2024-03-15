#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:11:44 2023

@author: bas
copied from: https://github.com/xpsi-group/xpsi/blob/main/examples/examples_fast/Synthetic_data.ipynb
"""



import os
import numpy as np
import math


from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import MultipleLocator, AutoLocator, AutoMinorLocator
from matplotlib import gridspec
from matplotlib import cm
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


import xpsi
from xpsi import Parameter, HotRegions, Elsewhere
from scipy.interpolate import Akima1DInterpolator
from xpsi.global_imports import _c, _G, _dpr, gravradius, _csq, _km, _2pi
from xpsi.tools.synthesise import synthesise_exposure_no_scaling as _synthesise # no scaling!

import sys
sys.path.append('../')
from custom_tools import SynthesiseData, plot_one_pulse

from CustomBackground import CustomBackground_DiskBB, k_disk_derive
from CustomPrior import CustomPrior
from CustomInstrument import CustomInstrument
from CustomPhotosphere import CustomPhotosphere
from CustomInterstellar import CustomInterstellar
from CustomSignal import CustomSignal
from CustomHotregion import CustomHotRegion_Accreting

from helper_functions import get_T_in_log10_Kelvin, plot_2D_pulse

this_directory = os.path.dirname(os.path.abspath(__file__))
print('this_directory: ', this_directory)

################################## SETTINGS ###################################

scenario = 'literature' # 'kajava'
second = False
te_index = 0

try: #try to get parameters from shell input
    os.environ.get('atmosphere_type')    
    atmosphere_type = os.environ['atmosphere_type']
    os.environ.get('n_params')
    n_params = os.environ['n_params']
    os.environ.get('machine')
    machine = os.environ['machine']
except:
    atmosphere_type = "A"
    n_params = "5"
    machine = "local"


if scenario == 'kajava' or scenario == 'literature':
    exposure_time=1.32366e5 ## is the same as Mason 2019

print("atmosphere_type:", atmosphere_type)
print("n_params:", n_params)

################################## INSTRUMENT #################################

min_input = 0 # 20 is used with 0.3 keV (channel_low=30). 0 is used with 0.2 keV (channel_low=20). 900 works with channel_low = 120 (1.2 keV). 
channel_low = 20 # 20 corresponds to 0.2 keV. # 30 corresponds to 0.3 keV
channel_hi = 300 # 300 corresponds to 3 keV. 600 corresponds to 6 keV (98.7% of total counts retained)
max_input = 1400 # 1400 works with channel-hi = 300. 2000 works with channel_hi = 600 (6 keV)

ARF_file=this_directory + '/../model_data/instrument_data/J1808_NICER_2019/merged_saxj1808_2019_arf_aeff.txt'
RMF_file=this_directory + '/../model_data/instrument_data/J1808_NICER_2019/merged_saxj1808_2019_rmf_matrix.txt'
channel_edges_file=this_directory + '/../model_data/instrument_data/J1808_NICER_2019/merged_saxj1808_2019_rmf_energymap.txt'

NICER = CustomInstrument.from_response_files(ARF = ARF_file,
            RMF = RMF_file,
            channel_edges = channel_edges_file,
            channel_low=channel_low,
            channel_hi=channel_hi,
            min_input=min_input,
            max_input=max_input)

############################### SPACETIME #####################################

bounds = dict(distance = (0.1, 10.0),                       # (Earth) distance
                mass = (1.0, 3.0),                          # mass
                radius = (3.0 * gravradius(1.0), 16.0),     # equatorial radius
                cos_inclination = (0.0, 1.0))               # (Earth) inclination to rotation axis

spacetime = xpsi.Spacetime(bounds=bounds, values=dict(frequency=401.0))# Fixing the spin

############################### SINGLE HOTREGION ##############################

num_leaves = 128  # 30
sqrt_num_cells = 128  # 50
num_energies = 128  # 40
num_rays = 512

kwargs = {'symmetry': 'azimuthal_invariance', #call general integrator instead of for azimuthal invariance
          'interpolator': 'split',
          'omit': False,
          'cede': False,
          'concentric': False,
          'sqrt_num_cells': sqrt_num_cells,
          'min_sqrt_num_cells': 10,
          'max_sqrt_num_cells': 128,
          'num_leaves': num_leaves,
          'num_rays': num_rays,
          'prefix': 'p'}
values = {}
bounds = dict(super_colatitude = (None, None),
              super_radius = (None, None),
              phase_shift = (0., 1.))
if atmosphere_type=='A':
    bounds['super_tbb'] = (0.001, 0.003)
    bounds['super_tau'] = (0.5, 3.5)
    if n_params=='5':
        bounds['super_te'] = (40., 200.)
        primary = CustomHotRegion_Accreting(bounds, values, **kwargs)

hot = HotRegions((primary,))


################################### ELSEWHERE ################################

elsewhere = Elsewhere(bounds=dict(elsewhere_temperature = (None,None)))

################################ ATMOSPHERE ################################### 
      

photosphere = CustomPhotosphere(hot = hot, elsewhere = elsewhere,
                                values=dict(mode_frequency = spacetime['frequency']))
# LOCAL
if machine=='local':
    photosphere.hot_atmosphere = '/home/bas/Documents/Projects/x-psi/model_datas/bobrikova/Bobrikova_compton_slab.npz'
# SNELLIUS
elif machine=='snellius':
    photosphere.hot_atmosphere = '/home/dorsman/xpsi-bas-fork/AMXPs/model_data/Bobrikova_compton_slab.npz'

    
################################### STAR ######################################

star = xpsi.Star(spacetime = spacetime, photospheres = photosphere)

#################################### PRIOR ####################################

prior = CustomPrior(scenario, 'model')

################################## INTERSTELLAR ###################################
if machine=='local':
    interstellar = CustomInterstellar.from_SWG("/home/bas/Documents/Projects/x-psi/xpsi-bas-fork/AMXPs/model_data/n_H/TBnew/tbnew0.14.txt", bounds=(None, None), value=None)
elif machine=='snellius':
    interstellar = CustomInterstellar.from_SWG("/home/dorsman/xpsi-bas-fork/AMXPs/model_data/interstellar/tbnew/tbnew0.14.txt", bounds=(None, None), value=None)


############################### BACKGROUND ####################################

bounds = dict(T_in = (None, None), R_in = (None, None), K_disk = None)
k_disk = k_disk_derive()
    
background = CustomBackground_DiskBB(bounds=bounds, values={'K_disk': k_disk}, interstellar = interstellar)
k_disk.star = star
k_disk.background = background


###################### SYNTHESISE DATA #################################

phases_space = np.linspace(0.0, 1.0, 33)
_data = SynthesiseData(np.arange(channel_low,channel_hi), phases_space, 0, channel_hi-channel_low-1)

################################## SIGNAL ###################################

signal = CustomSignal(data = _data,
                        instrument = NICER,  # Instrument
                        background = background,
                        interstellar = interstellar,
                        cache = True,
                        prefix='Instrument') # I can't change this?

################################# LIKELIHOOD ###############################

likelihood = xpsi.Likelihood(star = star, signals = signal,
                             num_energies=num_energies, 
                             threads=8, #1
                             externally_updated=False,
                             prior = prior)                             

for h in hot.objects:
    h.set_phases(num_leaves)


print("Prossecco ...")

if scenario == 'literature':
    mass = 1.4
    radius = 12.
    distance = 3.5
    inclination = 60
    cos_i = math.cos(inclination*math.pi/180)
    
    # Hotspot
    phase_shift = 0
    super_colatitude = 45*math.pi/180 # 20*math.pi/180 # 
    super_radius = 15.5*math.pi/180
    
    # Compton slab model parameters
    tbb=0.0012 # 0.0017 #0.001 -0.003 Tbb(data) = Tbb(keV)/511keV, 1 keV = 0.002 data
    te=100. # 50. # 40-200 corresponds to 20-100 keV (Te(data) = Te(keV)*1000/511keV), 50 keV = 100 data
    tau=1. #0.5 - 3.5 tau = ln(Fin/Fout)
    
    # elsewhere
    elsewhere_T_keV = 0.4 # 0.5 #  keV 
    
    # source background
    column_density = 1.17 #10^21 cm^-2
    diskbb_T_keV = 0.25 # 0.3  #  keV #0.3 keV for Kajava+ 2011
    R_in = 30 # 20 #  1 #  km #  for very small diskBB background

# SAX J1808-like v2
# mass = 1.706
# radius = 8.85
# distance = 3.4
# inclination = 8.11
# cos_i = math.cos(inclination*math.pi/180)

# # Hotspot
# phase_shift = 0.208
# super_colatitude = 137.7*math.pi/180 # 20*math.pi/180 # 
# super_radius = 89.9*math.pi/180

# # Compton slab model parameters
# tbb=0.868/511 # 0.0017 #0.001 -0.003 Tbb(data) = Tbb(keV)/511keV, 1 keV = 0.002 data
# te=101.9*1000/511. # 50. # 40-200 corresponds to 20-100 keV (Te(data) = Te(keV)*1000/511keV), 50 keV = 100 data
# tau=0.785 #0.5 - 3.5 tau = ln(Fin/Fout)

# # elsewhere
# elsewhere_T_keV = 0.453 # 0.5 #  keV 

# # source background
# column_density = 0.959 #10^21 cm^-2
# diskbb_T_keV = 0.116 # 0.3  #  keV #0.3 keV for Kajava+ 2011
# R_in = 64 # 20 #  1 #  km #  for very small diskBB background


if scenario == 'kajava':
    mass = 1.4
    radius = 11
    distance = 3.5
    inclination = 58
    cos_i = math.cos(inclination*math.pi/180)
    
    # Hotspot
    phase_shift = 0.20
    super_colatitude = 11*math.pi/180 # 20*math.pi/180 # 
    super_radius = 10*math.pi/180
    
    # Compton slab model parameters
    tbb=0.85/511 # 0.0017 #0.001 -0.003 Tbb(data) = Tbb(keV)/511keV, 1 keV = 0.002 data
    te=50*1000/511. # 50. # 40-200 corresponds to 20-100 keV (Te(data) = Te(keV)*1000/511keV), 50 keV = 100 data
    tau=1 #0.5 - 3.5 tau = ln(Fin/Fout)
    
    # elsewhere
    elsewhere_T_keV = 0.5 # 0.5 #  keV 
    
    # source background
    column_density = 1.13 #10^21 cm^-2
    diskbb_T_keV = 0.29 # 0.3  #  keV #0.3 keV for Kajava+ 2011
    R_in = 55 # 20 #  1 #  km #  for very small diskBB background

p = [mass, #1.4, #grav mass
      radius,#12.5, #coordinate equatorial radius
      distance, # earth distance kpc
      cos_i, #cosine of earth inclination
      phase_shift, #phase of hotregion
      super_colatitude, #colatitude of centre of superseding region
      super_radius,  #angular radius superceding region
      tbb,
      te,
      tau
      ]

elsewhere_T_log10_K = get_T_in_log10_Kelvin(elsewhere_T_keV)
p.append(elsewhere_T_log10_K) # 10^x Kelvin

diskbb_T_log10_K = get_T_in_log10_Kelvin(diskbb_T_keV)
p.append(diskbb_T_log10_K)


p.append(R_in)

if isinstance(interstellar, xpsi.Interstellar):
    p.append(column_density)


Instrument_kwargs = dict(exposure_time=exposure_time,
                         name=f'J1808_synthetic_{scenario}',
                         directory='./data/')

likelihood.synthesise(p, force=True, Instrument=Instrument_kwargs) 


np.savetxt(f'data/background_countrate_{scenario}.txt', np.sum(background.registered_background, axis=1))

print("Done !")

########## DATA PLOT ###############


my_data=np.loadtxt(f'./data/J1808_synthetic_{scenario}_realisation.dat')




figstring = f'J1808_synthetic_realisation_exp_time={exposure_time}.png'


fig1,ax1 = plot_one_pulse(my_data, phases_space, NICER.channel_edges, cm=cm.jet)

try:
    os.makedirs('./plots')
except OSError:
    if not os.path.isdir('./plots'):
        raise
fig1.savefig('./plots/'+figstring)
print('data plot saved in plots/{}'.format(figstring))

################ SIGNAL PLOT ###################################

num_rotations=1

fig2,ax2 = plot_2D_pulse((photosphere.signal[0][0],),
              x=signal.phases[0],
              shift=signal.shifts,
              y=signal.energies,
              ylabel=r'Energy (keV)',
              num_rotations=num_rotations,
              res=int(30*num_rotations))


ax2.set_title('atm={} params={} te={:.2e} [keV], tbb={:.2e} [keV], tau={:.2e} [-]'.format(atmosphere_type, n_params, te*0.511, tbb*511, tau), loc='center') #unit conversion te and tbb is different due to a cluster leftover according to Anna B.


fig2.savefig('./plots/J1808.png')
print('signal plot saved in plots/J1808.png')

print(np.sum(my_data))
