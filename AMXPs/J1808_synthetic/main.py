#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:35:05 2022

@author: bas
"""
# from __future__ import print_function, division

import os
import numpy as np
import math

from matplotlib import rcParams
import matplotlib.pyplot as plt

import xpsi
from xpsi import HotRegions, Elsewhere
from xpsi.global_imports import gravradius


################################ OPTIONS ###############################
machine = os.environ.get('machine')
run_type=os.environ.get('run_type')

try:
    num_energies = int(os.environ.get('num_energies'))
    num_leaves = int(os.environ.get('num_leaves'))
    sqrt_num_cells = int(os.environ.get('sqrt_num_cells'))
    live_points = int(os.environ.get('live_points'))
    max_iter = int(os.environ.get('max_iter'))
except:
    pass


# default options if os environment not provided
if  isinstance(os.environ.get('machine'),type(None)) or isinstance(os.environ.get('num_energies'),type(None)) or isinstance(os.environ.get('live_points'),type(None)) or isinstance(os.environ.get('max_iter'),type(None)): # if that fails input them here.
    print('E: failed to import some OS environment variables, using defaults.')    
    machine = 'local' # local, helios, snellius
    num_energies = 40
    live_points = 64
    sqrt_num_cells = 50
    num_leaves = 30
    max_iter = 1
    run_type = 'test' #test, sample
    
try:
    analysis_name = os.environ.get('LABEL')
except:
    print('cannot import analysis name, using default name')
    analysis_name = 'analysis_name'

integrator = 'azimuthal_invariance'
interpolator = 'split'


print('machine: ', machine)
print('num_energies: ', num_energies)
print('integrator:', integrator)
print('interpolator:', interpolator)

this_directory = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append(this_directory+'/../')

from custom_tools import CustomLikelihood

from CustomPrior import CustomPrior
from CustomInstrument import CustomInstrument
from CustomPhotosphere import CustomPhotosphere
from CustomInterstellar import CustomInterstellar
from CustomSignal import CustomSignal
from CustomHotregion import CustomHotRegion_Accreting

from helper_functions import get_T_in_log10_Kelvin, plot_2D_pulse


np.random.seed(xpsi._rank+10)
print('Rank reporting: %d' % xpsi._rank)

##################################### DATA ####################################

exposure_time = 1.32366e5 #Mason's 2019 data cut
channel_low = 20
channel_hi = 300
max_input = 1400
phases_space = np.linspace(0.0, 1.0, 33)

datastring = this_directory + '/data/J1808_synthetic_realisation.dat' 
settings = dict(counts = np.loadtxt(datastring, dtype=np.double),
                channels=np.arange(channel_low,channel_hi),
                phases=phases_space,
                first=0, last=channel_hi-channel_low-1,
                exposure_time=exposure_time)

data = xpsi.Data(**settings)

################################## INSTRUMENT #################################

ARF_file=this_directory + '/../model_data/instrument_data/J1808_NICER_2019/merged_saxj1808_2019_arf_aeff.txt'
RMF_file=this_directory + '/../model_data/instrument_data/J1808_NICER_2019/merged_saxj1808_2019_rmf_matrix.txt'
channel_edges_file=this_directory + '/../model_data/instrument_data/J1808_NICER_2019/merged_saxj1808_2019_rmf_energymap.txt'


NICER = CustomInstrument.from_response_files(ARF = ARF_file,
        RMF = RMF_file,
        channel_edges = channel_edges_file,       
        channel_low=channel_low,
        channel_hi=channel_hi,
        max_input=max_input)


################################## SPACETIME ##################################

bounds = dict(distance = (3.4, 3.6),                       # (Earth) distance
                mass = (1.0, 3.0),                          # mass
                radius = (3.0 * gravradius(1.0), 16.0),     # equatorial radius
                cos_inclination = (0.0, 1.0))               # (Earth) inclination to rotation axis

spacetime = xpsi.Spacetime(bounds=bounds, values=dict(frequency=401.0))# Fixing the spin


################################## HOTREGIONS #################################

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
              phase_shift = (-0.25, 0.75))

bounds['super_tbb'] = (0.001, 0.003)
bounds['super_tau'] = (0.5, 3.5)
bounds['super_te'] = (40., 200.)
primary = CustomHotRegion_Accreting(bounds, values, **kwargs)


hot = HotRegions((primary,))

################################### ELSEWHERE ################################

elsewhere = Elsewhere(bounds=dict(elsewhere_temperature = (5.0,7.0)))

################################ ATMOSPHERE ################################### 
      

photosphere = CustomPhotosphere(hot = hot, elsewhere = elsewhere,
                                values=dict(mode_frequency = spacetime['frequency']))
if machine == 'local':
    photosphere.hot_atmosphere = '/home/bas/Documents/Projects/x-psi/model_datas/bobrikova/Bobrikova_compton_slab.npz'
elif machine == 'helios' or machine == 'snellius':
    photosphere.hot_atmosphere = this_directory + '/../' + 'model_data/Bobrikova_compton_slab.npz'


################################### STAR ######################################

star = xpsi.Star(spacetime = spacetime, photospheres = photosphere)

################################## INTERSTELLAR ###################################

if machine=='local':
    interstellar_file = "/home/bas/Documents/Projects/x-psi/xpsi-bas-fork/AMXPs/model_data/n_H/TBnew/tbnew0.14.txt"
elif machine=='snellius':
    interstellar_file = "/home/dorsman/xpsi-bas-fork/AMXPs/model_data/interstellar/tbnew/tbnew0.14.txt"
interstellar=CustomInterstellar.from_SWG(interstellar_file, bounds=(0., 3.), value=None)

#################################### SIGNAL ###################################

signal = CustomSignal(data = data,
                        instrument = NICER,
                        background = None,
                        interstellar = interstellar,
                        cache = False,
                        epsrel = 1.0e-8,
                        epsilon = 1.0e-3,
                        sigmas = 10.0,
                        support = None)

################ parameter vector ########################

# SAX J1808-like 
mass = 1.4
radius = 12.
distance = 3.5
inclination = 60
cos_i = math.cos(inclination*math.pi/180)
phase_shift = 0
super_colatitude = 45*math.pi/180
super_radius = 15.5*math.pi/180


# Compton slab model parameters
tbb=0.0012 #0.001 -0.003 Tbb(data) = Tbb(keV)/511keV, 1 keV = 0.002 data
te=100. #40-200 corresponds to 20-100 keV (Te(data) = Te(keV)*1000/511keV), 50 keV = 25.55 data
tau=1 #0.5 - 3.5 tau = ln(Fin/Fout)


# elsewhere
elsewhere_T_keV = 0.4 # keV  for Kajava+ 2011  0.4  # 0.45 # 
elsewhere_T_log10_K = get_T_in_log10_Kelvin(elsewhere_T_keV)

# source background
column_density = 1.17 #10^21 cm^-2

p = [mass, #1.4, #grav mass
      radius,#12.5, #coordinate equatorial radius
      distance, # earth distance kpc
      cos_i, #cosine of earth inclination
      phase_shift, #phase of hotregion
      super_colatitude, #colatitude of centre of superseding region
      super_radius,  #angular radius superceding region
      tbb,
      te,
      tau,
      elsewhere_T_log10_K,
      column_density]

##################################### PRIOR ##############################

prior = CustomPrior()

##################################### LIKELIHOOD ##############################


likelihood = CustomLikelihood(star = star, signals = signal,
                              num_energies=num_energies, #128
                              threads=1,
                              prior=prior,
                              externally_updated=True)


########## likelihood check
true_logl = -4.6402898384e+04 

# likelihood(p)

likelihood.check(None, [true_logl], 1.0e-4, physical_points=[p], force_update=True)

wrapped_params = [0]*len(likelihood)
wrapped_params[likelihood.index('p__phase_shift')] = 1

if machine == 'local':
    folderstring = f'local_runs/{analysis_name}'
elif machine == 'helios':
    folderstring = f'helios_runs/{analysis_name}'
elif machine == 'snellius':
    folderstring = f'{analysis_name}'

try: 
    os.makedirs(folderstring)
except OSError:
    if not os.path.isdir(folderstring):
        raise


sampling_efficiency = 0.3
max_iter = max_iter
    
outputfiles_basename = f'./{folderstring}/run_ST_'
runtime_params = {'resume': False,
                  'importance_nested_sampling': False,
                  'multimodal': False,
                  'n_clustering_params': None,
                  'outputfiles_basename': outputfiles_basename,
                  'n_iter_before_update': 100,
                  'n_live_points': live_points,
                  'sampling_efficiency': sampling_efficiency,
                  'const_efficiency_mode': False,
                  'wrapped_params': wrapped_params,
                  'evidence_tolerance': 0.5,
                  'seed': 7,
                  'max_iter': max_iter, # manual termination condition for short test
                  'verbose': True}

if __name__ == '__main__':
    
    # ################################### PLOTS #####################################
    
    print('plotting...')
    
    rcParams['text.usetex'] = False
    rcParams['font.size'] = 14.0
    
    # Likelihood check and plot
    from matplotlib import cm
    fig, ax = plot_2D_pulse((photosphere.signal[0][0],),
                  x=signal.phases[0],
                  shift=signal.shifts,
                  y=signal.energies,
                  ylabel=r'Energy (keV)',
                  cm=cm.jet)
    

    ax.set_title('te={:.2e} [keV], tbb={:.2e} [keV], tau={:.2e} [-]'.format(te*0.511, tbb*511, tau), loc='center') #unit conversion te and tbb is different due to a cluster leftover according to Anna B.
    figstring = '{}/te={:.2e}_tbb={:.2e}_tau={:.2e}.png'.format(folderstring, te, tbb, tau)

    
    plt.savefig(figstring)
    print('figure saved in {}'.format(figstring))

    
    ##################### DO SAMPLING ################
    
    if run_type=='sample':
        print("sampling starts ...")
        xpsi.Sample.nested(likelihood, prior,**runtime_params)
        print("... sampling done")
        
    