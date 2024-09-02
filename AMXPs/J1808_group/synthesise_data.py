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

import sys
this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory+'/../')

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
from xpsi.tools.synthesise import synthesise_exposure as _synthesise # no scaling!

from Disk import Disk, k_disk_derive
from CustomPrior import CustomPrior
from CustomInstrument import CustomInstrument
from CustomPhotosphereDisk import CustomPhotosphereDisk
from CustomInterstellar import CustomInterstellar
from CustomSignal import CustomSignal
from CustomHotregion import CustomHotRegion_Accreting

from helper_functions import get_T_in_log10_Kelvin, get_mids_from_edges, SynthesiseData
from parameter_values import parameter_values

################################## SETTINGS ###################################


bkg = 'model' #'model' 'fix'

second = False
te_index = 0

try: #try to get parameters from shell input
    os.environ.get('atmosphere_type')    
    atmosphere_type = os.environ['atmosphere_type']
    os.environ.get('n_params')
    n_params = os.environ['n_params']
    os.environ.get('machine')
    machine = os.environ['machine']
    os.environ.get('poisson_noise')
    poisson_noise = bool(os.environ['poisson_noise'])
    os.environ.get('poisson_seed')
    poisson_seed = int(os.environ['poisson_seed']) 
    os.environ.get('scenario')
    scenario = os.environ['scenario']

except:
    atmosphere_type = "A"
    n_params = "5"
    machine = "local"
    poisson_noise = True
    poisson_seed = 42
    scenario = 'large_r' # 'kajava', 'literature
  

pv = parameter_values(scenario, bkg)
p = pv.p()


if scenario == 'kajava' or scenario == 'literature' or scenario == '2019' or scenario == 'large_r' or scenario == 'small_r':
    exposure_time=1.32366e5 ## is the same as Mason 2019
    


################################## INSTRUMENT #################################

energy_range = 'large'

if energy_range == 'small':
    min_input = 0 # 20 is used with 0.3 keV (channel_low=30). 0 is used with 0.2 keV (channel_low=20). 900 works with channel_low = 120 (1.2 keV). 
    channel_low = 20 # 20 corresponds to 0.2 keV. # 30 corresponds to 0.3 keV
    channel_hi = 300 # 300 corresponds to 3 keV. 600 corresponds to 6 keV (98.7% of total counts retained)
    max_input = 1400 # 1400 works with channel-hi = 300. 2000 works with channel_hi = 600 (6 keV)

if energy_range == 'large':
    min_input = 20 # 20 is used with 0.3 keV (channel_low=30). 0 is used with 0.2 keV (channel_low=20). 900 works with channel_low = 120 (1.2 keV). 
    channel_low = 30 # 20 corresponds to 0.2 keV. # 30 corresponds to 0.3 keV
    channel_hi = 600 # 300 corresponds to 3 keV. 600 corresponds to 6 keV (98.7% of total counts retained)
    max_input = 2000 # 1400 works with channel-hi = 300. 2000 works with channel_hi = 600 (6 keV)



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

kwargs = {'symmetry': True, #call for azimuthal invariance
          'split': True,
          'atm_ext':'Num5D',
          'omit': False,
          'cede': False,
          'concentric': False,
          'sqrt_num_cells': sqrt_num_cells,
          'min_sqrt_num_cells': 10,
          'max_sqrt_num_cells': 128,
          'num_leaves': num_leaves,
          'num_rays': num_rays}#,
          #'prefix': 'p'}
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

# elsewhere = Elsewhere(bounds=dict(elsewhere_temperature = (None,None)))


############################### DISK ####################################

bounds = dict(T_in = (None, None), R_in = (None, None), K_disk = None)
k_disk = k_disk_derive()
    
disk = Disk(bounds=bounds, values={'K_disk': k_disk})
k_disk.disk = disk

################################ ATMOSPHERE ################################### 
      

photosphere = CustomPhotosphereDisk(hot = hot, elsewhere = None, custom=disk,
                                values=dict(mode_frequency = spacetime['frequency']))
# LOCAL
if machine=='local':
    photosphere.hot_atmosphere = '/home/bas/Documents/Projects/x-psi/model_datas/bobrikova/Bobrikova_compton_slab.npz'
# SNELLIUS
elif machine=='snellius':
    photosphere.hot_atmosphere = '/home/dorsman/xpsi-bas-fork/AMXPs/model_data/Bobrikova_compton_slab.npz'

    
################################### STAR ######################################

star = xpsi.Star(spacetime = spacetime, photospheres = photosphere)
k_disk.star = star


#################################### PRIOR ####################################

prior = CustomPrior(scenario, bkg)

################################## INTERSTELLAR ###################################
if machine=='local':
    interstellar = CustomInterstellar.from_SWG("/home/bas/Documents/Projects/x-psi/xpsi-bas-fork/AMXPs/model_data/n_H/TBnew/tbnew0.14.txt", bounds=(0, 3.), value=None)
elif machine=='snellius':
    interstellar = CustomInterstellar.from_SWG("/home/dorsman/xpsi-bas-fork/AMXPs/model_data/interstellar/tbnew/tbnew0.14.txt", bounds=(0, 3.), value=None)




###################### SYNTHESISE DATA #################################

phases_space = np.linspace(0.0, 1.0, 33)
_data = SynthesiseData(np.arange(channel_low,channel_hi), phases_space, 0, channel_hi-channel_low-1)

################################## SIGNAL ###################################

signal = CustomSignal(data = _data,
                        instrument = NICER,  # Instrument
                        background = None,
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


if poisson_noise:
    seed = poisson_seed

Instrument_kwargs = dict(exposure_time=exposure_time,
                         seed=seed, 
                         name=f'synthetic_{scenario}_seed={seed}',
                         directory='./data/')

likelihood.synthesise(p, force=True, Instrument=Instrument_kwargs) 

if __name__ == '__main__':

    #np.savetxt(f'data/background_countrate_{scenario}.txt', np.sum(background.registered_background, axis=1))
    #np.savetxt(f'data/J1808_synthetic_diskbb_{scenario}.txt', background.registered_background)
    
    print("Done !")
    
    ########## DATA PLOT ###############
    
    
    my_data=np.loadtxt(f'./data/synthetic_{scenario}_seed={poisson_seed}_realisation.dat')
    
    
    
    
    figstring = f'J1808_synthetic_{poisson_seed}_{scenario}'
    
    
    from helper_functions import custom_subplots
    
    
    fig, axes = custom_subplots(2,1, sharex=True, figsize=(5, 5))
    profile = axes[0].plot_2D_counts(my_data, phases_space, NICER.channel_edges, cm=cm.magma)
    cb = plt.colorbar(profile, ax=axes[0])
    cb.set_label(label='Counts', labelpad=10)
    cb.solids.set_edgecolor('face')
    axes[1].plot_bolometric_pulse(phases_space, my_data, normalized=True)
    cb2 = plt.colorbar(profile, ax=axes[1])
    cb2.remove()
    

    try:
        os.makedirs('./plots')
    except OSError:
        if not os.path.isdir('./plots'):
            raise
    
    
    fig.savefig(f'./plots/counts_and_bolometric_{figstring}.png')


    
    num_rotations=1
    
    fig2, ax2 = custom_subplots(figsize=(5, 3))
    profile = ax2.plot_2D_signal((photosphere.signal[0][0],),x=signal.phases[0],shift=signal.shifts,y=signal.energies,ylabel=r'Energy (keV)',num_rotations=num_rotations,res=int(30*num_rotations))
    cb = plt.colorbar(profile, ax=ax2)
    cb.set_label(label=r'Signal (arbitrary units)', labelpad=25)
    cb.solids.set_edgecolor('face')
    
    fig2.savefig(f'./plots/signal_{figstring}_.png')
    print('plot saved in plots/')
    

