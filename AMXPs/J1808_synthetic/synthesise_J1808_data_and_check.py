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

from CustomBackground import CustomBackground_DiskBB
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



exposure_time=1.32366e5 ## is the same as Mason 2019

print("atmosphere_type:", atmosphere_type)
print("n_params:", n_params)

################################## INSTRUMENT #################################
channel_low = 20
channel_hi = 300 #600
max_input = 1400 #2000

ARF_file=this_directory + '/../model_data/instrument_data/J1808_NICER_2019/merged_saxj1808_2019_arf_aeff.txt'
RMF_file=this_directory + '/../model_data/instrument_data/J1808_NICER_2019/merged_saxj1808_2019_rmf_matrix.txt'
channel_edges_file=this_directory + '/../model_data/instrument_data/J1808_NICER_2019/merged_saxj1808_2019_rmf_energymap.txt'

try:   
    NICER = CustomInstrument.from_response_files(ARF = ARF_file,
            RMF = RMF_file,
            channel_edges = channel_edges_file,
            channel_low=channel_low,
            channel_hi=channel_hi,
            max_input=max_input)

except:
    print('error! No instrument file!')

############################### SPACETIME #####################################

bounds = dict(distance = (3.4, 3.6),                       # (Earth) distance
                mass = (1.0, 3.0),                          # mass
                radius = (3.0 * gravradius(1.0), 16.0),     # equatorial radius
                cos_inclination = (0.0, 1.0))               # (Earth) inclination to rotation axis

spacetime = xpsi.Spacetime(bounds=bounds, values=dict(frequency=401.0))# Fixing the spin

############################### SINGLE HOTREGION ##############################

num_leaves = 128
sqrt_num_cells = 128
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
if atmosphere_type=='A':
    bounds['super_tbb'] = (0.001, 0.003)
    bounds['super_tau'] = (0.5, 3.5)
    if n_params=='5':
        bounds['super_te'] = (40., 200.)
        primary = CustomHotRegion_Accreting(bounds, values, **kwargs)

hot = HotRegions((primary,))


################################### ELSEWHERE ################################

elsewhere = Elsewhere(bounds=dict(elsewhere_temperature = (5.0,7.0)))

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

prior = CustomPrior()

################################## INTERSTELLAR ###################################
if machine=='local':
    interstellar = CustomInterstellar.from_SWG("/home/bas/Documents/Projects/x-psi/xpsi-bas-fork/AMXPs/model_data/n_H/TBnew/tbnew0.14.txt", bounds=(None, None), value=None)
elif machine=='snellius':
    interstellar = CustomInterstellar.from_SWG("/home/dorsman/xpsi-bas-fork/AMXPs/model_data/interstellar/tbnew/tbnew0.14.txt", bounds=(None, None), value=None)


############################### BACKGROUND ####################################

background = CustomBackground_DiskBB(bounds=(None, None), values={}, interstellar = interstellar)

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
                             num_energies=128, #384
                             threads=8, #1
                             externally_updated=False,
                             prior = prior)                             

for h in hot.objects:
    h.set_phases(num_leaves)


print("Prossecco ...")

# SAX J1808-like 
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
r_in = 30 # 20 #  1 #  km #  for very small diskBB background

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

K_disk = cos_i*(r_in/(distance/10))**2  # (km / 10 kpc)^2
# K_disk = 0
p.append(K_disk)

if isinstance(interstellar, xpsi.Interstellar):

    p.append(column_density)

Instrument_kwargs = dict(exposure_time=exposure_time,
                         name='J1808_synthetic',
                         directory='./data/')

likelihood.synthesise(p, force=True, Instrument=Instrument_kwargs) 

print("Done !")



########## DATA PLOT ###############

my_data=np.loadtxt('./data/J1808_synthetic_realisation.dat'.format(atmosphere_type, n_params))


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


################ likelihood check ###############################
datastring = this_directory + '/data/J1808_synthetic_realisation.dat' 
settings = dict(counts = np.loadtxt(datastring, dtype=np.double),
                channels=np.arange(channel_low,channel_hi),
                phases=phases_space,
                first=0, last=channel_hi-channel_low-1,
                exposure_time=exposure_time)

data_for_check = xpsi.Data(**settings)

signal_for_check = CustomSignal(data = data_for_check,
                        instrument = NICER,  # Instrument
                        background = None,
                        interstellar = interstellar,
                        cache = True,
                        prefix='Instrument')

# signal_for_check = CustomSignal(data = data_for_check,
#                         instrument = NICER,
#                         background = background,
#                         interstellar = interstellar,
#                         cache = False,
#                         epsrel = 1.0e-8,
#                         epsilon = 1.0e-3,
#                         sigmas = 10.0,
#                         support = None)

likelihood_for_check = xpsi.Likelihood(star = star, signals = signal_for_check,
                             num_energies=128, 
                             threads=8, #1
                             externally_updated=True,
                             prior = prior)

p_cut = p[:-3]
# p_cut.append(6.5)
# p_cut.append(0.0)
p_cut.append(column_density)

print('True Likelihood: ', likelihood_for_check(p_cut, reinitialise=True))
print('Summed difference: ', np.sum((likelihood_for_check.signal.expected_counts-data_for_check.counts)**2))



##### max likelihood from sampling

P_MaxL_192 = [0.166089488067012603E+01,
0.880002192119270887E+01,
0.346449329423984453E+01,
0.146451450087112731E-01,
-0.824004344790990606E-02,
0.113710559990825955E+01,
0.334581240206683672E+00,
0.147006720245396834E-02,
0.738928142100897958E+02,
0.119092736886722195E+01,
0.670045739653590822E+01,
# 6.5,
# 0.,
0.121213084628677747E+01]


print('Max Likelihood parameter vector from run with 192 live points: ', likelihood_for_check(P_MaxL_192, reinitialise=True))
print('Summed difference: ', np.sum((likelihood_for_check.signal.expected_counts-data_for_check.counts)**2))

P_MaxL_1000 = [0.143487922741846097E+01,
0.105670546531530594E+02,
0.359702712157563642E+01,
0.567626751460999612E+00,
-0.962955296681333728E-03,
0.947700302807278794E+00,
0.333738393572487135E+00,
0.120907353807114988E-02,
0.541298122468860612E+02,
0.134618148088802281E+01,
0.665965052985182204E+01,
# 6.5,
# 0.,
0.896414323034762095E+00]

print('Max Likelihood parameter vector from run with 1000 live points: ', likelihood_for_check(P_MaxL_1000, reinitialise=True))
print('Summed difference: ', np.sum((likelihood_for_check.signal.expected_counts-data_for_check.counts)**2))
