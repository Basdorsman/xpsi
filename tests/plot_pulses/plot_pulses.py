#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:35:05 2022

@author: bas
"""
from __future__ import print_function, division

import os
import numpy as np
import math

from matplotlib import rcParams
import matplotlib.pyplot as plt

import xpsi
from xpsi.global_imports import gravradius


import sys
sys.path.append('../')
from custom_tools import CustomInstrument, CustomHotRegion, CustomHotRegion_Accreting, CustomPhotosphere_BB, CustomPhotosphere_4D, CustomPhotosphere_5D, CustomPhotosphere_Bobrikova, CustomSignal, CustomPrior, CustomPrior_NoSecondary, plot_2D_pulse

import time

np.random.seed(xpsi._rank+10)
print('Rank reporting: %d' % xpsi._rank)

################################ OPTIONS ###############################
second = False
atmosphere = 'accreting' #'blackbody', 'numerical'
n_params_numerical = "4"

try: #try to get parameters from shell input
    os.environ.get('n_params')
    n_params = os.environ['n_params']
    if n_params=='B': atmosphere = 'blackbody'
    elif n_params=='4' or n_params=='5': atmosphere = 'numerical'
    elif n_params=='A': atmosphere = 'accreting'
except:
    if atmosphere=='accreting': n_params = "A"
    elif atmosphere=='numerical': n_params = n_params_numerical
    elif atmosphere=='blackbody': n_params = "B"
    
print('atmosphere:', atmosphere)
print('n_params:', n_params)

##################################### DATA ####################################
settings = dict(counts = np.loadtxt('../model_data/example_synthetic_realisation.dat', dtype=np.double),
        channels=np.arange(20,201), #201
        phases=np.linspace(0.0, 1.0, 33),
        first=0, last=180,
        exposure_time=984307.6661)
data = xpsi.Data(**settings)

################################## INSTRUMENT #################################
try:
    NICER = CustomInstrument.from_response_files(ARF = '../model_data/nicer_v1.01_arf.txt',
                                             RMF = '../model_data/nicer_v1.01_rmf_matrix.txt',
                                             max_input = 500, #500
                                             min_input = 0,
                                             channel_edges = '../model_data/nicer_v1.01_rmf_energymap.txt')
except:
    print("ERROR: You might miss one of the following files (check Modeling tutorial or the link below how to find them): \n model_data/nicer_v1.01_arf.tx, model_data/nicer_v1.01_rmf_matrix.txt, model_data/nicer_v1.01_rmf_energymap.txt")
    print("https://github.com/ThomasEdwardRiley/xpsi_workshop.git")
    exit()


################################## SPACETIME ##################################

# spacetime = xpsi.Spacetime.fixed_spin(300.0)


bounds = dict(distance = (0.1, 1.0),                     # (Earth) distance
                mass = (1.0, 3.0),                       # mass
                radius = (3.0 * gravradius(1.0), 16.0),  # equatorial radius
                cos_inclination = (0.0, 1.0))      # (Earth) inclination to rotation axis

spacetime = xpsi.Spacetime(bounds=bounds, values=dict(frequency=300.0))


################################## HOTREGIONS #################################
print("hotregions")
################################## PRIMARY ####################################
from xpsi import HotRegions

if atmosphere=='accreting':
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
elif atmosphere=='numerical':
    if n_params=='4':
        bounds = dict(super_colatitude = (None, None),
                      super_radius = (None, None),
                      phase_shift = (0.0, 0.1),
                      super_temperature = (5.1, 6.8))#,
                      #super_modulator = (-0.3, 0.3))
    
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
                                #modulated = True, #modulation flag
           	                    prefix='p')
    elif n_params=='5':
        bounds = dict(super_colatitude = (None, None),
                      super_radius = (None, None),
                      phase_shift = (0.0, 0.1),
                      super_temperature = (5.1, 6.8),
                      super_modulator = (-0.3, 0.3))
    
        primary = CustomHotRegion(bounds=bounds,
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
                                modulated = True, #modulation flag
           	                    prefix='p')

elif atmosphere=='blackbody':
    bounds = dict(super_colatitude = (None, None),
                  super_radius = (None, None),
                  phase_shift = (0.0, 0.1),
                  super_temperature = (5.1, 6.8))
    
    primary = xpsi.HotRegion(bounds=bounds,
    	                    values={},
    	                    symmetry=True,
    	                    omit=False,
    	                    cede=False,
    	                    concentric=False,
    	                    sqrt_num_cells=32,
    	                    min_sqrt_num_cells=10,
    	                    max_sqrt_num_cells=64,
    	                    num_leaves=100,
    	                    num_rays=200,
    	                    prefix='p') 	
    


###################################### SECONDARY ##############################
if second: # 4 doesn't work for secondary
    # If you derive bounds for a secondary hotspots, you cannot also define bounds
    # (above). You must set them to "None" to avoid some conflict. 

    if atmosphere=='accreting':
        bounds['super_tbb'] = None
        bounds['super_te'] = None
        bounds['super_tau'] = None
    if atmosphere=='numerical':
        bounds['super_temperature'] = None # declare fixed/derived variable
        bounds['super_modulator'] = None
    
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
            global primary # unnecessary, but for clarity
            return primary['super_temperature'] - 0.2
        
    class derive_modulator(xpsi.Derive):
        def __init__(self):
            pass
    
        def __call__(self, boundto, caller = None):
            global primary
            return primary['super_modulator']  
    
    class derive_tbb(xpsi.Derive):
        def __init__(self):
            pass
    
        def __call__(self, boundto, caller = None):
            global primary
            # print("super_tbb derive")
            # print(primary['super_tbb'])
            return primary['super_tbb']  
        
    class derive_te(xpsi.Derive):
        def __init__(self):
            pass
    
        def __call__(self, boundto, caller = None):
            global primary
            return primary['super_te']  
    
    class derive_tau(xpsi.Derive):
        def __init__(self):
            pass
    
        def __call__(self, boundto, caller = None):
            global primary
            return primary['super_tau']  
        
    if atmosphere=='accreting':
        secondary = CustomHotRegion_Accreting(bounds=bounds, # can otherwise use same bounds
        	                      values={'super_tbb': derive_tbb(), 'super_te': derive_te(), 'super_tau': derive_tau()},
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
    elif atmosphere=='numerical':
        secondary = CustomHotRegion(bounds=bounds, # can otherwise use same bounds
         	                      values={'super_temperature': derive(), 'super_modulator': derive_modulator()},
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
                                  modulated=True,
         	                      prefix='s')
    
    
    
    
    hot = HotRegions((primary, secondary))
    # h = hot.objects[0]
    # hot['p__super_temperature'] = 6.0 # equivalent to ``primary['super_temperature'] = 6.0``
    # print("printing hot:",hot)
elif not second:
    hot = HotRegions((primary,))
    
    
    
################################ ATMOSPHERE ################################### 
      

# print("n_params: ",n_params)

if n_params == "4":   
    photosphere = CustomPhotosphere_4D(hot = hot, elsewhere = None,
                                    values=dict(mode_frequency = spacetime['frequency']))
    photosphere.hot_atmosphere = '/home/bas/Documents/Projects/x-psi/model_datas/model_data/H-atmosphere_Spectra_fully_ionized/NSX_H-atmosphere_Spectra/nsx_H_v171019.npz'

elif n_params== "5":
    photosphere = CustomPhotosphere_5D(hot = hot, elsewhere = None,
                                    values=dict(mode_frequency = spacetime['frequency']))
    # photosphere.hot_atmosphere = '/home/bas/Documents/Projects/x-psi/model_datas/model_data/H-atmosphere_Spectra_fully_ionized/NSX_H-atmosphere_Spectra/nsx_H_v171019_5D_no_effect.npz'
    photosphere.hot_atmosphere = '/home/bas/Documents/Projects/x-psi/model_datas/model_data/H-atmosphere_Spectra_fully_ionized/NSX_H-atmosphere_Spectra/nsx_H_v171019_modulated_0dot5_to_2.npz'
    # photosphere.hot_atmosphere = '/home/bas/Documents/Projects/x-psi/model_datas/Bobrikova_compton_slab.npz'

elif n_params== "A":
    photosphere = CustomPhotosphere_Bobrikova(hot = hot, elsewhere = None,
                                    values=dict(mode_frequency = spacetime['frequency']))
    photosphere.hot_atmosphere = '/home/bas/Documents/Projects/x-psi/model_datas/bobrikova/Bobrikova_compton_slab.npz'

elif n_params=="B":
    photosphere = CustomPhotosphere_BB(hot = hot, elsewhere = None, 
                                       values=dict(mode_frequency = spacetime['frequency']))


else:
    print("no dimensionality provided!")

################################### STAR ######################################

star = xpsi.Star(spacetime = spacetime, photospheres = photosphere)
# star['mass'] = 1.6
# star['radius'] = 14.0
# star['distance'] = 0.2
# star['cos_inclination'] = math.cos(1.25)
# star['p__phase_shift'] = 0.0
# star['p__super_colatitude'] = 1.0
# star['p__super_radius'] = 0.075
# star['p__super_temperature'] = 6.2 
# star['p__super_modulator'] = 0.0

# star['s__phase_shift'] = 0.025 #0.2 gives problems!
# star['s__super_colatitude'] = math.pi - 1.0
# star['s__super_radius'] = 0.025


# with modulator


if atmosphere=='accreting':
    # Compton slab model parameters
    tbb=0.001
    te=200.
    tau=0.5

    if second:
        p = [1.6, #1.4, #grav mass
              14.0,#12.5, #coordinate equatorial radius
              0.2, # earth distance kpc
              math.cos(1.25), #cosine of earth inclination
              0.0, #phase of hotregion
              1.0, #colatitude of centre of superseding region
              0.075,  #angular radius superceding region
              tbb,
              te,
              tau,
              #6.2, #primary temperature
              #modulator, #modulator
              0.025,
              math.pi - 1.0,
              0.075
              ]
    elif not second:
        p = [1.6, #1.4, #grav mass
              14.0,#12.5, #coordinate equatorial radius
              0.2, # earth distance kpc
              math.cos(1.25), #cosine of earth inclination
              0.0, #phase of hotregion
              1.0, #colatitude of centre of superseding region
              0.075,  #angular radius superceding region
              tbb,
              te,
              tau,
              #6.2, #primary temperature
              #modulator, #modulator
              ]
elif atmosphere=='numerical':   
    p_temperature=6.2
    modulator = 0 

    if second:
        p = [1.6, #1.4, #grav mass
              14.0,#12.5, #coordinate equatorial radius
              0.2, # earth distance kpc
              math.cos(1.25), #cosine of earth inclination
              0.0, #phase of hotregoin
              1.0, #colatitude of centre of superseding region
              0.075,  #angular radius superceding region
              p_temperature, #primary temperature
              modulator, #modulator
              0.025,
              math.pi - 1.0,
              0.025
              ]
    elif not second:
        if n_params=='5':
            p = [1.6, #1.4, #grav mass
                  14.0,#12.5, #coordinate equatorial radius
                  0.2, # earth distance kpc
                  math.cos(1.25), #cosine of earth inclination
                  0.0, #phase of hotregoin
                  1.0, #colatitude of centre of superseding region
                  0.075,  #angular radius superceding region
                  p_temperature, #primary temperature
                  modulator #modulator
                  ]
        elif n_params=='4':
            p = [1.6, #1.4, #grav mass
                  14.0,#12.5, #coordinate equatorial radius
                  0.2, # earth distance kpc
                  math.cos(1.25), #cosine of earth inclination
                  0.0, #phase of hotregoin
                  1.0, #colatitude of centre of superseding region
                  0.075,  #angular radius superceding region
                  p_temperature, #primary temperature
                  #modulator #modulator
                  ]
        
elif atmosphere=='blackbody':
    p_temperature=6.2
    
    if second:
        p = [1.6, #1.4, #grav mass
              14.0,#12.5, #coordinate equatorial radius
              0.2, # earth distance kpc
              math.cos(1.25), #cosine of earth inclination
              0.0, #phase of hotregoin
              1.0, #colatitude of centre of superseding region
              0.075,  #angular radius superceding region
              p_temperature, #primary temperature
              0.025,
              math.pi - 1.0,
              0.025
              ]
    elif not second:
        p = [1.6, #1.4, #grav mass
              14.0,#12.5, #coordinate equatorial radius
              0.2, # earth distance kpc
              math.cos(1.25), #cosine of earth inclination
              0.0, #phase of hotregoin
              1.0, #colatitude of centre of superseding region
              0.075,  #angular radius superceding region
              p_temperature #primary temperature
              ]
        
        
    
star(p)
star.update() 

print("printing Parameters of the star:")
print(star.params)



#################################### SIGNAL ###################################

# signal = CustomSignal(data = data,
#                         instrument = NICER,
#                         background = None,
#                         interstellar = None,
#                         workspace_intervals = 1000,
#                         cache = True,
#                         epsrel = 1.0e-8,
#                         epsilon = 1.0e-3,
#                         sigmas = 10.0,
#                         support = None)



##################################### LIKELIHOOD ##############################

# if second:
#     prior = CustomPrior()
# elif not second:
#     prior = CustomPrior_NoSecondary()

# # print('define likelihood')
# likelihood = xpsi.Likelihood(star = star, signals = signal,
#                              num_energies=128,
#                              threads=1,
#                              prior=prior,
#                              externally_updated=True)



# # print('define wrapped_params')
# wrapped_params = [0]*len(likelihood)
# wrapped_params[likelihood.index('p__phase_shift')] = 1
# if second:
#     wrapped_params[likelihood.index('s__phase_shift')] = 1

# try: 
#     os.makedirs("run")
# except OSError:
#     if not os.path.isdir("run"):
#         raise

# runtime_params = {'resume': False,
#                   'importance_nested_sampling': False,
#                   'multimodal': False,
#                   'n_clustering_params': None,
#                   'outputfiles_basename': './run/run_Num', 
#                   'n_iter_before_update': 50,
#                   'n_live_points': 100,
#                   'sampling_efficiency': 0.8,
#                   'const_efficiency_mode': False,
#                   'wrapped_params': wrapped_params,
#                   'evidence_tolerance': 0.5,
#                   'seed': 7,
#                   'max_iter': 1000, # manual termination condition for short test
#                   'verbose': True}
                  
# print("likelihood check but no sampling")

# try:
#     true_logl = -7.94188579e+89 #-1.15566075e+05
#     # print('about to evaluate likelihood')
#     print("Compute likelikihood(p) once so the check passes. Likelihood = ", likelihood(p)) 

#     likelihood.check(None, [true_logl], 1.0e-6,physical_points=[p])
#     #print(likelihood(p))
# except:
#     print("Likelihood check did not pass. Checking if wrong atmosphere model installed.")
#     true_logl = -3.27536126e+04
#     print(likelihood(p))
#     try:
#         likelihood.check(None, [true_logl], 1.0e-6,physical_points=[p])
#         print("Seems that blacbkody atmosphere extension was used instead of numerical.")
#         print("Please re-install X-PSI using numerical atmosphere extension if want to use this test run.")
#     except:
#         print("Seems that neither of the likelihood checks passed, so something must be wrong.")


######### PHOTOSPHERE INTEGRATE #####################
print('photosphere integrating')
energies=np.logspace(-1.0,np.log10(3.0), 128, base=10.0)

st = time.time()
photosphere.integrate(energies, threads=8)
et = time.time()
elapsed_time = et - st
print('Photosphere integration time:', elapsed_time,'seconds')

# ################################### PLOTS #####################################



print('plotting...')


rcParams['text.usetex'] = False
rcParams['font.size'] = 14.0

# if second:
#     ax = plot_2D_pulse((photosphere.signal[0][0], photosphere.signal[1][0]),
#                   x=signal.phases[0],
#                   shift=signal.shifts,
#                   y=signal.energies,
#                   ylabel=r'Energy (keV)')
# if not second:
#     ax = plot_2D_pulse((photosphere.signal[0][0],),
#                   x=signal.phases[0],
#                   shift=signal.shifts,
#                   y=signal.energies,
#                   ylabel=r'Energy (keV)')

# Photosphere.intergrate and plot



if not second:    
    #plt.plot(hot.phases_in_cycles[0],photosphere.signal[0][0][87])
    ax = plot_2D_pulse((photosphere.signal[0][0],),
                      x=hot.phases_in_cycles[0],
                      shift=[hot['p__phase_shift'],],
                      y=energies,
                      ylabel=r'Energy (keV)')

if atmosphere=='accreting':
    ax.set_title('atmosphere={} te={:.2e} [keV], tbb={:.2e} [keV], tau={:.2e} [-]'.format(n_params, te*0.511, tbb*511, tau), loc='center') #unit conversion te and tbb is different due to a cluster leftover according to Anna B.
    figstring = 'plots/pulses_atm={}_sec={}_te={:.2e}_tbb={:.2e}_tau={:.2e}.png'.format(atmosphere, second, te, tbb, tau)
elif atmosphere=='numerical':
    if n_params=="5":
        ax.set_title('atmosphere={} p_temperature={} modulator={}'.format(n_params, p_temperature, modulator))
        figstring='plots/5D_pulses_atm={}_sec={}_p_temperature={}_modulator={}.png'.format(atmosphere, second, p_temperature, modulator)
    elif n_params=="4":
        ax.set_title('atmosphere={} p_temperature={}'.format(n_params, p_temperature))
        figstring='plots/4D_pulses_atm={}_sec={}_p_temperature={}.png'.format(atmosphere, second, p_temperature)
elif atmosphere=='blackbody':
    ax.set_title('atmosphere={} p_temperature={}'.format(n_params, p_temperature))
    figstring='plots/pulses_atm={}_sec={}_p_temperature={}.png'.format(atmosphere, second, p_temperature)


plt.savefig(figstring)
print('figure saved in {}'.format(figstring))