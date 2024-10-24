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
from xpsi import Parameter, HotRegions
from scipy.interpolate import Akima1DInterpolator
from xpsi.global_imports import _c, _G, _dpr, gravradius, _csq, _km, _2pi

import sys
sys.path.append('../')
from custom_tools import CustomInstrument, CustomHotRegion, CustomHotRegion_Accreting, CustomHotRegion_Accreting_te_const, CustomPhotosphere_BB, CustomPhotosphere_N4, CustomPhotosphere_N5, CustomPhotosphere_A5, CustomPhotosphere_A4, CustomSignal, CustomPrior, CustomPrior_NoSecondary, plot_2D_pulse, CustomBackground, SynthesiseData

################################## SETTINGS ###################################

second = False

try: #try to get parameters from shell input
    os.environ.get('atmosphere_type')    
    atmosphere_type = os.environ['atmosphere_type']
    os.environ.get('n_params')
    n_params = os.environ['n_params']
except:
    atmosphere_type = "N"
    n_params = "5"

print("atmosphere_type:", atmosphere_type)
print("n_params:", n_params)

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


############################### SPACETIME #####################################

bounds = dict(distance = (0.1, 10.0),                       # (Earth) distance
                mass = (1.0, 3.0),                          # mass
                radius = (3.0 * gravradius(1.0), 16.0),     # equatorial radius
                cos_inclination = (0.0, 1.0))               # (Earth) inclination to rotation axis

spacetime = xpsi.Spacetime(bounds=bounds, values=dict(frequency=300.0))# Fixing the spin

############################### SINGLE HOTREGION ##############################

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

hot = HotRegions((primary,))
    
################################ ATMOSPHERE ################################### 
      
if atmosphere_type=='A':
    if n_params== "5":
        photosphere = CustomPhotosphere_A5(hot = hot, elsewhere = None,
                                        values=dict(mode_frequency = spacetime['frequency']))
        photosphere.hot_atmosphere = '/home/bas/Documents/Projects/x-psi/model_datas/bobrikova/Bobrikova_compton_slab.npz'

    elif n_params== "4":
        photosphere = CustomPhotosphere_A4(hot = hot, elsewhere = None,
                                        values=dict(mode_frequency = spacetime['frequency']))
        photosphere.hot_atmosphere = '/home/bas/Documents/Projects/x-psi/model_datas/bobrikova/Bobrikova_compton_slab.npz'

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
    
################################### STAR ######################################

star = xpsi.Star(spacetime = spacetime, photospheres = photosphere)

#################################### PRIOR ####################################

prior = CustomPrior_NoSecondary()

############################### BACKGROUND ####################################

background = CustomBackground(bounds=(None, None))

###################### SYNTHESISE DATA #################################



#_data = SynthesiseData(np.arange(10,301), np.linspace(0.0, 1.0, 33), 0, 290 )
_data = SynthesiseData(np.arange(20,201), np.linspace(0.0, 1.0, 33), 0, 180 ) #Apparently some hardcoded stuff for NICER


from xpsi.tools.synthesise import synthesise_exposure as _synthesise

def synthesise(self,
               exposure_time,
               expected_background_counts,
               name='synthetic',
               directory='./',
               **kwargs):
    
        """ Synthesise data set.

        """
        self._expected_counts, synthetic, bkg= _synthesise(exposure_time,
                                                             self._data.phases,
                                                             self._signals,
                                                             self._phases,
                                                             self._shifts,
                                                             expected_background_counts,
                                                             self._background.registered_background,
                                                             gsl_seed=42)
        
        try:
            if not os.path.isdir(directory):
                os.mkdir(directory)
        except OSError:
            print('Cannot create write directory.')
            raise

        np.savetxt(os.path.join(directory, name+'_realisation.dat'),
                   synthetic,
                   fmt = '%u')

        self._write(self.expected_counts,
                    filename = os.path.join(directory, name+'_expected_hreadable.dat'),
                    fmt = '%.8e')

        self._write(synthetic,
                    filename = os.path.join(directory, name+'_realisation_hreadable.dat'),
                    fmt = '%u')

def _write(self, counts, filename, fmt):
        """ Write to file in human readable format. """

        rows = len(self._data.phases) - 1
        rows *= len(self._data.channels)

        phases = self._data.phases[:-1]
        array = np.zeros((rows, 3))

        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                array[i*len(phases) + j,:] = self._data.channels[i], phases[j], counts[i,j]

            np.savetxt(filename, array, fmt=['%u', '%.6f'] + [fmt])

CustomSignal.synthesise = synthesise
CustomSignal._write = _write

################################## SIGNAL ###################################
signal = CustomSignal(data = _data,
                        instrument = NICER, #Instrument
                        background = background,
                        interstellar = None,
                        cache = True,
                        prefix='Instrument')


################################# LIKELIHOOD ###############################
likelihood = xpsi.Likelihood(star = star, signals = signal,
                             num_energies=128, #384
                             threads=8, #1
                             externally_updated=False,
                             prior = prior)                             


for h in hot.objects:
    h.set_phases(num_leaves = 100)


print("Prossecco ...")

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

p.append(-2)        # Background sprectral index : gamma (E^gamma) 

Instrument_kwargs = dict(exposure_time=50000.0,              
                         expected_background_counts=0., #10000.0,
                         name='{}{}_synthetic'.format(atmosphere_type, n_params),
                         directory='./data/')

likelihood.synthesise(p, force=True, Instrument=Instrument_kwargs) 

print("Done !")

print('signals:', np.sum(signal.signals))
print('expected counts:', np.sum(signal.expected_counts))

########## PLOT ###############

my_data=np.loadtxt('./data/{}{}_synthetic_realisation.dat'.format(atmosphere_type, n_params))

fig,ax=plt.subplots(1,1,figsize=(10,5))

#xpsi_d=ax[0].imshow(good_xspi_data,cmap=cm.magma,origin="lower", aspect="auto",extent=[0,1,10,300])
my_d=ax.imshow(my_data,cmap=cm.magma,origin="lower", aspect="auto",extent=[0,1,10,300])
#res=ax[2].imshow(residual,cmap=cm.magma,origin="lower", aspect="auto",extent=[0,1,10,300])

# anchored_text1 = AnchoredText("xpsi_good_installation",loc=1)
anchored_text = AnchoredText("Your installation",loc=1)
# anchored_text3 = AnchoredText("Residual",loc=1)

ax.set_ylabel("Channels")
ax.set_xlabel("Phases")
#ax[1].set_xlabel("Phases")
#ax[2].set_xlabel("Phases")
ax.add_artist(anchored_text)
#ax[1].add_artist(anchored_text2)
#ax[2].add_artist(anchored_text3)
plt.colorbar(my_d,ax=ax)
#plt.colorbar(you_d,ax=ax[1])
#plt.colorbar(res,ax=ax[2])

figstring = './plots/{}{}_synthetic_realisation.png'.format(atmosphere_type, n_params)
fig.savefig(figstring)
print('plot saved in {}'.format(figstring))
