#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:11:44 2023

@author: bas
copied from: https://github.com/xpsi-group/xpsi/blob/main/examples/examples_fast/Synthetic_data.ipynb
"""



import os
import numpy as np


import sys
this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory+'/../')

from matplotlib import pyplot as plt
from matplotlib import cm


import xpsi
from xpsi.global_imports import gravradius



from Disk import Disk, k_disk_derive
from CustomPrior import CustomPrior
from CustomInstrument import CustomInstrument
from CustomHotRegions import CustomHotRegions as HotRegions
from CustomHotRegion import CustomHotRegion
from CustomPhotosphere import CustomPhotosphere
from CustomInterstellar import CustomInterstellar
from CustomSignal import CustomSignal




from parameter_values import parameter_values

class SynthesiseData(xpsi.Data):
    """ Custom data container to enable synthesis. """

    def __init__(self, channels, phases, first, last):

        self.channels = channels
        # print(channels)
        # print(len(channels))
        # print(first)
        # print(last)
        self._phases = phases

        try:
            self._first = int(first)
            self._last = int(last)
        except TypeError:
            raise TypeError('The first and last channels must be integers.')
        if self._first >= self._last:
            raise ValueError('The first channel number must be lower than the '
                             'the last channel number.')

################################## SETTINGS ###################################


bkg = 'disk' #disk or fix if no disk
disk_blocking=True # use disk occultation or not

try:
    os.environ.get('machine')
    machine = os.environ['machine']
    os.environ.get('poisson_noise')
    poisson_noise = bool(os.environ['poisson_noise'])
    os.environ.get('poisson_seed')
    poisson_seed = int(os.environ['poisson_seed']) 
    os.environ.get('scenario')
    scenario = os.environ['scenario']

except:
    machine = "local"
    poisson_noise = True
    poisson_seed = 42
    scenario = 'molkov'
  

pv = parameter_values(scenario, bkg)
bounds = pv.bounds()
names = pv.names()
p = pv.p()


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

# sp_bounds = dict(distance = bounds['distance'],                       # (Earth) distance
#                 mass = bounds['mass'],                          # mass
#                 radius = bounds['radius'],     # equatorial radius
#                 cos_inclination = bounds['cos_inclination'])               # (Earth) inclination to rotation axis

spacetime = xpsi.Spacetime(bounds=bounds, values=dict(frequency=pv.frequency))# Fixing the spin

############################### First HOTREGION ##############################

num_leaves = 30 # 128
sqrt_num_cells = 50 # 128
num_energies = 40 # 128
num_rays = 512

p_kwargs = {'symmetry': True,
          'split': True,
          'disk_blocking': disk_blocking,
          'omit': False,
          'cede': False,
          'concentric': False,
          'sqrt_num_cells': sqrt_num_cells,
          'min_sqrt_num_cells': sqrt_num_cells,
          'max_sqrt_num_cells': sqrt_num_cells,
          'num_leaves': num_leaves,
          'num_rays': num_rays,
          'prefix': 'p'}

primary_bounds = {}
primary_bounds['super_radius'] = bounds['p__super_radius'] # I can't have the prefix so I remove it
primary_bounds['super_colatitude'] = bounds['p__super_colatitude']
primary_bounds['phase_shift'] = bounds['p__phase_shift']
primary_bounds['super_tbb'] = bounds['p__super_tbb']
primary_bounds['super_te'] = bounds['p__super_te']
primary_bounds['super_tau'] = bounds['p__super_tau']


s_kwargs = {'symmetry': True,
          'split': True,
          'disk_blocking': disk_blocking,
          'omit': False,
          'cede': False,
          'concentric': False,
          'sqrt_num_cells': sqrt_num_cells,
          'min_sqrt_num_cells': sqrt_num_cells,
          'max_sqrt_num_cells': sqrt_num_cells,
          'num_leaves': num_leaves,
          'num_rays': num_rays,
          'is_antiphased': True,
          'prefix': 's'}

values = {}
# bounds = {}

secondary_bounds = {}
secondary_bounds['super_radius'] = bounds['s__super_radius']
secondary_bounds['super_colatitude'] = bounds['s__super_colatitude']
secondary_bounds['phase_shift'] = bounds['s__phase_shift']
secondary_bounds['super_tbb'] = bounds['s__super_tbb']
secondary_bounds['super_te'] = bounds['s__super_te']
secondary_bounds['super_tau'] = bounds['s__super_tau']


primary = CustomHotRegion(primary_bounds, values, **p_kwargs)
secondary = CustomHotRegion(secondary_bounds, values, **s_kwargs)


hot = HotRegions((primary,secondary))


################################### ELSEWHERE ################################

# elsewhere = Elsewhere(bounds=dict(elsewhere_temperature = (None,None)))


############################### DISK ####################################

if 'disk' in bkg:
    k_disk = k_disk_derive()
    disk = Disk(bounds=bounds, values={'K_disk': k_disk})
    k_disk.spacetime = spacetime
    k_disk.disk = disk
elif bkg=='fix':
    disk=None

################################ ATMOSPHERE ################################### 
      



photosphere = CustomPhotosphere(hot = hot, 
                                        elsewhere = None, 
                                        disk=disk,
                                        disk_combined=True,
                                        disk_blocking=disk_blocking,
                                        values=dict(mode_frequency = spacetime['frequency']))
# LOCAL
if machine=='local':
    photosphere.hot_atmosphere = '/home/bas/Documents/Projects/x-psi/model_datas/bobrikova/Bobrikova_compton_slab.npz'
# SNELLIUS
elif machine=='snellius':
    photosphere.hot_atmosphere = '/home/bdorsman/xpsi-bas-fork/AMXPs/model_data/Bobrikova_compton_slab.npz'

    
################################### STAR ######################################

star = xpsi.Star(spacetime = spacetime, photospheres = photosphere)
if 'disk' in bkg:
    k_disk.star = star


#################################### PRIOR ####################################

prior = CustomPrior(scenario, bkg)

################################## INTERSTELLAR ###################################
if machine=='local':
    interstellar = CustomInterstellar.from_SWG("/home/bas/Documents/Projects/x-psi/xpsi-bas-fork/AMXPs/model_data/n_H/TBnew/tbnew0.14.txt", bounds=bounds['column_density'], value=None)
elif machine=='snellius':
    interstellar = CustomInterstellar.from_SWG("/home/bdorsman/xpsi-bas-fork/AMXPs/model_data/n_H/TBnew/tbnew0.14.txt", bounds=bounds['column_density'], value=None)




###################### SYNTHESISE DATA #################################

phases_space = np.linspace(0.0, 1.0, 33)
_data = SynthesiseData(np.arange(channel_low,channel_hi), phases_space, 0, channel_hi-channel_low-1)

################################## SIGNAL ###################################

signal = CustomSignal(data = _data,
                        instrument = NICER,  # Instrument
                        background = None,
                        interstellar = interstellar,
                        disk_combined = True,
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


print("Processing data now..")


if poisson_noise:
    seed = poisson_seed

Instrument_kwargs = dict(exposure_time=exposure_time,
                         seed=seed, 
                         name=f'synthetic_{scenario}_seed={seed}_bkg={bkg}_disk_blocking={disk_blocking}',
                         directory='./data/')


print(names)
print(p)

likelihood.synthesise(p, force=True, Instrument=Instrument_kwargs) 

if __name__ == '__main__':
    
    
    print("Done !")
    
    ########## DATA PLOT ###############
    
    
    my_data=np.loadtxt(f'./data/synthetic_{scenario}_seed={poisson_seed}_bkg={bkg}_disk_blocking={disk_blocking}_realisation.dat')
    
    
    
    
    figstring = f'J1808_synthetic_{poisson_seed}_{scenario}'
    
    
    from helper_functions import custom_subplots
    
    
    fig, axes = custom_subplots(2,1, sharex=True, figsize=(5, 5))
    profile = axes[0].plot_2D_counts(my_data, phases_space, NICER.channel_edges, cm=cm.magma)
    cb = plt.colorbar(profile, ax=axes[0])
    cb.set_label(label='Counts', labelpad=10)
    cb.solids.set_edgecolor('face')
    axes[1].plot_bolometric_pulse(phases_space, my_data, normalized=True)
    # cb2 = plt.colorbar(profile, ax=axes[1])
    # cb2.remove()
    

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
    

