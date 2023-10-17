#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:48:58 2023

@author: bas, much copied from postprocessing tutorial
"""

# Importing relevant modules

# %matplotlib inline

# from __future__ import division

import sys
import os
import matplotlib.pyplot as plt

from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator, AutoLocator, AutoMinorLocator
from matplotlib import gridspec
from matplotlib import cm
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import math

from collections import OrderedDict

import xpsi

from xpsi import PostProcessing

# choose a seed for the notebook if you want caching to be useful
# and the notebook exactly reproducible
PostProcessing.set_random_seed(42)

from xpsi.global_imports import gravradius


path="../"
sys.path.append(path)


atmosphere_type = 'A'
n_params = 5
sampling_params = 8
likelihood = 'custom'
num_energies=16
sampling_efficiency=0.8
live_points=64
max_iterations=-1
job_id =327461 # 325177 # 325354 #327461 #325177 # 324879 #

run_ID = f'{atmosphere_type}{n_params}-{sampling_params}-interpolation=3+2'

os.environ['atmosphere_type']=atmosphere_type
os.environ['n_params']=str(n_params)
os.environ['likelihood']=likelihood
os.environ['machine']='local'
os.environ['num_energies']=str(num_energies)
os.environ['sampling_params'] = str(sampling_params)
import sample as ST

# We will use the same bounds used during sampling
if atmosphere_type=='A':
    # Settings names, bounds and labels
    ST.names=['mass','radius','distance','cos_inclination','p__phase_shift','p__super_colatitude','p__super_radius','p__super_tbb']
    #ST.names=['mass','radius','distance','cos_inclination','p__phase_shift','p__super_colatitude','p__super_radius','p__super_tbb','p__super_tau']

    ST.bounds = {'mass':(1.0,3.0),
                 'radius':(3.0 * gravradius(1.0), 16.0),
                 'distance':(0.1, 1.0),
                 'cos_inclination':(0.,1.),
                 'p__phase_shift':(0.0, 0.1),
                 'p__super_colatitude':(0.001, math.pi/2 - 0.001),
                 'p__super_radius':(0.001, math.pi/2.0 - 0.001),
                 'p__super_tbb':(0.00015, 0.003)} 
                 #'p__super_te': (40., 200.),


    if sampling_params == 9:
        ST.names.append('p__super_tau')
        ST.bounds['p__super_tau'] = (0.5, 3.5)
    
    # Now the labels
    ST.labels = {'mass': r"M\;\mathrm{[M}_{\odot}\mathrm{]}",
                  'radius': r"R_{\mathrm{eq}}\;\mathrm{[km]}",
                  'distance': r"D \;\mathrm{[kpc]}",
                  'cos_inclination': r"\cos(i)",
                  'p__phase_shift': r"\phi_{p}\;\mathrm{[cycles]}",
                  'p__super_colatitude': r"\Theta_{spot}\;\mathrm{[rad]}",
                  'p__super_radius': r"\zeta_{spot}\;\mathrm{[rad]}",
                  'p__super_tbb': r"t_{bb}\;\mathrm{[data units]}"}
                  #'p__super_te': r"t_e data units",
    
    ST.truths={'mass': 1.6,                               # Mass in solar Mass
              'radius': 14.,                              # Equatorial radius in km
              'distance': 0.2,                            # Distance in kpc
              'cos_inclination': math.cos(1.25),          # Cosine of Earth inclination to rotation axis
              'p__phase_shift': 0.0,                    # Phase shift
              'p__super_colatitude': 1.,                # Colatitude of the centre of the superseding region
              'p__super_radius': 0.075,                 # Angular radius of the (circular) superseding region
              'p__super_tbb': 0.001}#,                      # Blackbody temperature
              #'p__super_te': 40,                          # Electron temperature

    if sampling_params == 9:
        ST.bounds['p__super_tau'] = (0.5, 3.5) # Optical depth
        ST.labels['p__super_tau'] = r"\tau\;\mathrm{[data units]}"
        ST.truths['p__super_tau'] = 0.5

if atmosphere_type=='N':
    ST.names=['mass','radius','distance','cos_inclination','p__phase_shift',
              'p__super_colatitude','p__super_radius','p__super_temperature']
    
    ST.bounds = {'mass':(1.0,3.0),
                 'radius':(3.0 * gravradius(1.0), 16.0),
                 'distance':(0.1, 1.0),
                 'cos_inclination':(0.,1.),
                 'p__phase_shift':(0.0, 0.1),
                 'p__super_colatitude':(0.001, math.pi/2 - 0.001),
                 'p__super_radius':(0.001, math.pi/2.0 - 0.001),
                 'p__super_temperature':(5.1, 6.8)}#, 

    # Now the labels
    ST.labels = {'mass': r"M\;\mathrm{[M}_{\odot}\mathrm{]}",
                  'radius': r"R_{\mathrm{eq}}\;\mathrm{[km]}",
                  'distance': r"D \;\mathrm{[kpc]}",
                  'cos_inclination': r"\cos(i)",
                  'p__phase_shift': r"\phi_{p}\;\mathrm{[cycles]}",
                  'p__super_colatitude': r"\Theta_{spot}\;\mathrm{[rad]}",
                  'p__super_radius': r"\zeta_{spot}\;\mathrm{[rad]}",
                  'p__super_temperature': r"log_{10}(T)\;mathrm{[K]}"}#,

    
    ST.truths={'mass': 1.6,                               # Mass in solar Mass
              'radius': 14.,                              # Equatorial radius in km
              'distance': 0.2,                            # Distance in kpc
              'cos_inclination': math.cos(1.25),          # Cosine of Earth inclination to rotation axis
              'p__phase_shift': 0.0,                    # Phase shift
              'p__super_colatitude': 1.,                # Colatitude of the centre of the superseding region
              'p__super_radius': 0.075,                 # Angular radius of the (circular) superseding region
              'p__super_temperature': 6.764}
                

ST.truths['compactness']=gravradius(ST.truths['mass'])/ST.truths['radius']

#### SOMETHING SOMETHING COMPACTNESS

ST.names +=['compactness']
ST.bounds['compactness']=(gravradius(1.0)/16.0, 1.0/3.0)
ST.labels['compactness']= r"M/R_{\mathrm{eq}}"

getdist_kde_settings = {'ignore_rows': 0,
                        'min_weight_ratio': 1.0e-10,
                        'contours': [0.683, 0.954, 0.997],
                        'credible_interval_threshold': 0.001,
                        'range_ND_contour': 0,
                        'range_confidence': 0.001,
                        'fine_bins': 1024,
                        'smooth_scale_1D': 0.4,
                        'num_bins': 100,
                        'boundary_correction_order': 1,
                        'mult_bias_correction_order': 1,
                        'smooth_scale_2D': 0.4,
                        'max_corr_2D': 0.99,
                        'fine_bins_2D': 512,
                        'num_bins_2D': 40}

print('names',ST.names)
print('likelihood:', ST.likelihood)



ST.runs = xpsi.Runs.load_runs(ID='ST',
                               run_IDs=[run_ID],
                               roots=[f'run_se={sampling_efficiency}_lp={live_points}_atm={atmosphere_type}{n_params}_ne={num_energies}_mi={max_iterations}'],
                               base_dirs=[f'/home/bas/Documents/Projects/x-psi/xpsi-bas-fork/AMXPs/inference_run/helios_runs/run_{atmosphere_type}{n_params}custom/{job_id}'],
                               # base_dirs=['/home/bas/Documents/Projects/x-psi/xpsi-bas-fork/AMXPs/inference_run/local_runs/run_A4/'],
                               # base_dirs=['/home/bas/Documents/Projects/x-psi/xpsi-bas-fork/tests/inference_run/helios_runs/run_A4/304400/run_A4'],
                               # base_dirs=['/home/bas/Documents/Projects/x-psi/xpsi-bas-fork/tests/inference_run/helios_runs/run_A5/304399/run_A5'],
                               use_nestcheck=[False],
                               kde_settings=getdist_kde_settings,
                               likelihood=ST.likelihood,
                               names=ST.names,
                               bounds=ST.bounds,
                               labels=ST.labels,
                               truths=ST.truths,
                               implementation='multinest',
                               overwrite_transformed=True)


pp = xpsi.PostProcessing.CornerPlotter([ST.runs])
fig = pp.plot(
     params=ST.names,
     IDs=OrderedDict([('ST', [run_ID,]),]),
     prior_density=False,
     KL_divergence=True,
     ndraws=5e4,
     combine=False, combine_all=True, only_combined=False, overwrite_combined=True,
     param_plot_lims={},
     bootstrap_estimators=False,
     bootstrap_density=False,
     n_simulate=200,
     crosshairs=True,
     write=True,
     ext='.png',
     maxdots=3000,
     root_filename=f'run',
     credible_interval_1d=True,
     annotate_credible_interval=True,
     compute_all_intervals=False,
     sixtyeight=True,
     axis_tick_x_rotation=45.0,
     num_plot_contours=3,
     subplot_size=4.0,
     legend_corner_coords=(0.675,0.8),
     legend_frameon=False,
     scale_attrs=OrderedDict([('legend_fontsize', 2.0),
                              ('axes_labelsize', 1.35),
                              ('axes_fontsize', 'axes_labelsize'),
                             ]
                            ),
     colormap='Reds',
     shaded=True,
     shade_root_index=-1,
     rasterized_shade=True,
     no_ylabel=True,
     no_ytick=True,
     lw=1.0,
     lw_1d=1.0,
     filled=False,
     normalize=True,
     veneer=True,
     #contour_colors=['orange'],
     tqdm_kwargs={'disable': False},
     lengthen=2.0,
     embolden=1.0,
     nx=500)

pp = xpsi.SignalPlotter([ST.runs])
pp.plot(IDs=OrderedDict([('ST', [run_ID]),
                        ]),
        combine=False, # use these controls if more than one run for a posterior
        combine_all=False,
        force_combine=False,
        only_combined=False,
        force_cache=True,
        nsamples=3,
        plots = {'ST': xpsi.ResidualPlot()})

pp.plots["ST"].fig


#%%

# plt.savefig(f'./corner_run_se={sampling_efficiency}_lp={live_points}_atm={atmosphere_type}{n_params}_ne={num_energies}_mi={max_iterations}.png')

