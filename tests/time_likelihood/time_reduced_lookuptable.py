#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:51:13 2023

@author: bas
"""

import os
import numpy as np
import math

from matplotlib import rcParams
import matplotlib.pyplot as plt

import xpsi
from xpsi.global_imports import gravradius


import sys
sys.path.append('../')
from custom_tools import CustomInstrument, CustomHotRegion, CustomHotRegion_Accreting, CustomHotRegion_Accreting_te_const
from custom_tools import CustomPhotosphere_BB, CustomPhotosphere_N4, CustomPhotosphere_N5, CustomPhotosphere_A5, CustomPhotosphere_A4
from custom_tools import CustomSignal, CustomPrior, CustomPrior_NoSecondary, plot_2D_pulse, CustomLikelihood

import time

np.random.seed(xpsi._rank+10)
print('Rank reporting: %d' % xpsi._rank)

################################ OPTIONS ###############################
second = False
te_index=0 # t__e = np.arange(40.0, 202.0, 4.0), there are 40.5 values (I expect that means 40)
reduction = [1,2]

try: #try to get parameters from shell input
    os.environ.get('atmosphere_type')    
    atmosphere_type = os.environ['atmosphere_type']
    os.environ.get('n_params')
    n_params = os.environ['n_params']
except:
    atmosphere_type = "A"
    n_params = "5"

print("atmosphere_type:", atmosphere_type)
print("n_params:", n_params)

##################################### DATA ####################################
settings = dict(counts = np.loadtxt('../synthesise_pulse_data/data/{}{}_synthetic_realisation.dat'.format(atmosphere_type, n_params), dtype=np.double),  # np.loadtxt('../model_data/example_synthetic_realisation.dat', dtype=np.double),
        channels=np.arange(20,201), #201
        phases=np.linspace(0.0, 1.0, 33),
        first=0, last=180,
        exposure_time=1000)#984307.6661)
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



bounds = dict(distance = (0.1, 1.0),                     # (Earth) distance
                mass = (1.0, 3.0),                       # mass
                radius = (3.0 * gravradius(1.0), 16.0),  # equatorial radius
                cos_inclination = (0.0, 1.0))      # (Earth) inclination to rotation axis

spacetime = xpsi.Spacetime(bounds=bounds, values=dict(frequency=300.0))


################################## HOTREGIONS #################################
print("hotregions")
################################## PRIMARY ####################################
from xpsi import HotRegions

if atmosphere_type=='A':
    if n_params=='5':
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
    elif n_params=='4':
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
elif atmosphere_type=='N':
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

elif atmosphere_type=='B':
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

    if atmosphere_type=='A':
        if n_params=='5':
            bounds['super_tbb'] = None
            bounds['super_te'] = None
            bounds['super_tau'] = None
        if n_params=='4':
            bounds['super_tbb'] = None
            bounds['super_tau'] = None
    if atmosphere_type=='N':
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
        
    if atmosphere_type=='A':
        if n_params=='5':
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
    elif atmosphere_type=='N':
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
    
    
    
#################################### SIGNAL ###################################

signal = CustomSignal(data = data,
                        instrument = NICER,
                        background = None,
                        interstellar = None,
                        workspace_intervals = 1000,
                        cache = True,
                        epsrel = 1.0e-8,
                        epsilon = 1.0e-3,
                        sigmas = 10.0,
                        support = None)

    
################################ ATMOSPHERES ################################### 
atmospherestrings = ['/home/bas/Documents/Projects/x-psi/model_datas/bobrikova/Bobrikova_compton_slab.npz', '/home/bas/Documents/Projects/x-psi/model_datas/bobrikova/Bobrikova_compton_slab_reduced=2.npz']  
photospheres = []

print('helllooooooooo')

for atmospherestring in atmospherestrings:
    if atmosphere_type=='A':
        if n_params== "5":
            photosphere = CustomPhotosphere_A5(hot = hot, elsewhere = None,
                                            values=dict(mode_frequency = spacetime['frequency']))
            photosphere.hot_atmosphere = atmospherestring
            
    
        elif n_params== "4":
            photosphere = CustomPhotosphere_A4(hot = hot, elsewhere = None,
                                            values=dict(mode_frequency = spacetime['frequency']))
            photosphere.te_index = te_index
            photosphere.hot_atmosphere = atmospherestring
    elif atmosphere_type=='N':
        if n_params == "4":   
            photosphere = CustomPhotosphere_N4(hot = hot, elsewhere = None,
                                            values=dict(mode_frequency = spacetime['frequency']))
            photosphere.hot_atmosphere = '/home/bas/Documents/Projects/x-psi/model_datas/model_data/H-atmosphere_Spectra_fully_ionized/NSX_H-atmosphere_Spectra/nsx_H_v171019.npz'
        
        elif n_params== "5":
            photosphere = CustomPhotosphere_N5(hot = hot, elsewhere = None,
                                            values=dict(mode_frequency = spacetime['frequency']))
            # photosphere.hot_atmosphere = '/home/bas/Documents/Projects/x-psi/model_datas/model_data/H-atmosphere_Spectra_fully_ionized/NSX_H-atmosphere_Spectra/nsx_H_v171019_5D_no_effect.npz'
            photosphere.hot_atmosphere = '/home/bas/Documents/Projects/x-psi/model_datas/model_data/H-atmosphere_Spectra_fully_ionized/NSX_H-atmosphere_Spectra/nsx_H_v171019_modulated_0dot5_to_2.npz'
            # photosphere.hot_atmosphere = '/home/bas/Documents/Projects/x-psi/model_datas/Bobrikova_compton_slab.npz'
    
    elif atmosphere_type=="B":
        photosphere = CustomPhotosphere_BB(hot = hot, elsewhere = None, 
                                           values=dict(mode_frequency = spacetime['frequency']))
    else:
        print("no atmosphere_type provided!")
    photospheres.append(photosphere)


    
################################### Parameter ######################################
  
if atmosphere_type=='A':
    if n_params=='5':
        # Compton slab model parameters
        tbb=0.001 #0.00015-  0.003
        te=40 #200. #40 - 200
        tau=0.5 #0.5 - 3.5
    
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
    elif n_params=='4':
        # Compton slab model parameters
        tbb=0.001
        #te=200.
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
                  #te,
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
                  #te,
                  tau,
                  #6.2, #primary temperature
                  #modulator, #modulator
                  ]
elif atmosphere_type=='N':   
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
        
elif atmosphere_type=='B':
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

################################### STARS ######################################

stars = []
for photosphere, i in zip(photospheres, range(len(photospheres))):
    stars.append(xpsi.Star(spacetime = spacetime, photospheres = photosphere))
    stars[i](p)
    stars[i].update() 






#####################################  PRIOR ##############################

if second:
    prior = CustomPrior()
elif not second:
    prior = CustomPrior_NoSecondary()




################################### LIKELIHOOD ################################

likelihoods = []

for star in stars:
    likelihoods.append(CustomLikelihood(star = star, signals = signal,
                                        num_energies=32,
                                        threads=1,
                                        prior=prior,
                                        externally_updated=False))

################################## INVERSE SAMPLE #############################
sample_size = 5
p_samples = [p,] #first sample is my sample above
fig, ax = plt.subplots()
for sample in range(sample_size):
    p_samples.append(likelihoods[0].prior.inverse_sample())

################################# SAMPLING ########################
ldict = []
delta_time_avg = []

for likelihood in likelihoods:
    for sample in p_samples:
        likelihood(sample)
    
    delta_time_avg.append(np.mean([likelihood.ldict[call]['deltatime'] for call in likelihood.ldict]))
    ldict.append(likelihood.ldict)

#################################### SAVING ##################################

# import dill as pickle

# with open(f'timed_likelihood_reduced_{atmosphere_type}{n_params}.pkl', 'wb') as file:
#       #file.write(pickle.dumps(likelihood.ldict)) # use `pickle.loads` to do the reverse
#       pickle.dump((ldict, delta_time_avg, reduction), file)