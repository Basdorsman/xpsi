#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:32:15 2024

@author: bas
"""
import os
import sys
this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory+'/../')

import math
from helper_functions import get_T_in_log10_Kelvin
from xpsi.global_imports import gravradius
import numpy as np

class parameter_values(object):
    def __init__(self, scenario, bkg):
        self.scenario = scenario
        self.bkg = bkg


        if self.scenario == 'molkov':
            self.mass = 1.4
            self.radius = 12
            self.distance = 8 # lower range of what Molkov found
            self.inclination = 58
            self.cos_i = math.cos(self.inclination*math.pi/180)
            
            # Primary Hotspot
            self.p_phase_shift = 0
            self.p_colatitude = 14*math.pi/180 
            self.p_radius = 33*math.pi/180
            self.p_tbb=1/511 # Tbb(data) = Tbb(keV)/511keV, 1 keV = 0.002 data
            self.p_te=50*1000/511. # Te(data) = Te(keV)*1000/511keV, 50 keV = 100 data
            self.p_tau=1 #0.5 - 3.5 tau = ln(Fin/Fout)
            
            # Secondary Hotspot
            self.s_phase_shift = 0 # assuming antiphased is True in the hotregion
            self.s_colatitude = (180-14)*math.pi/180 
            self.s_radius = 33*math.pi/180
            self.s_tbb=1/511 # Tbb(data) = Tbb(keV)/511keV, 1 keV = 0.002 data
            self.s_te=50*1000/511. # Te(data) = Te(keV)*1000/511keV, 50 keV = 100 data
            self.s_tau=1 #0.5 - 3.5 tau = ln(Fin/Fout)

            if 'disk' in self.bkg:
            # source background
                self.diskbb_T_keV = 0.37 # corresponds 1.48e-10 Msol/yr  (in line with range found for SAX J1808 by Casten+ 2023)
                self.diskbb_T_log10_K = get_T_in_log10_Kelvin(self.diskbb_T_keV)
                self.R_in = 24.5 # km
            self.column_density = 29. #10^21 cm^-2 #corresponds to value found by Papitto et al. 2024
            
            self.frequency=447.8718 #hz
        
        
    def p(self):
        self.p = [
        self.mass,  
        self.radius,  
        self.distance,  
        self.cos_i,  
        self.p_phase_shift,  
        self.p_colatitude, 
        self.p_radius,  
        self.p_tbb,
        self.p_te,
        self.p_tau,
        self.s_phase_shift,
        self.s_colatitude, 
        self.s_radius, 
        self.s_tbb,
        self.s_te,
        self.s_tau,
        self.diskbb_T_log10_K if 'disk' in self.bkg else None,
        self.R_in if 'disk' in self.bkg else None,
        self.column_density
        ]

        # Remove any None values (e.g., mass if fix_mass is True, or optional elements)
        self.p = [x for x in self.p if x is not None]
        return self.p
        
      
    
    def names(self):
	# Base parameter names
        self.names = [
            'mass',
            'radius',
      		'distance',
      		'cos_inclination',
     		'p__phase_shift',
    		'p__super_colatitude',
    		'p__super_radius',
    		'p__super_tbb',
    		'p__super_te',
    		'p__super_tau',
    		's__phase_shift',
    		's__super_colatitude',
    		's__super_radius',
    		's__super_tbb',
    		's__super_te',
    		's__super_tau',
      		'T_in' if 'disk' in self.bkg else None,
      		'R_in' if 'disk' in self.bkg else None,
            'column_density', 
            'compactness',  # derived parameters below (new units)
            'inclination_deg',
            'p__colatitude_deg', 
            'p__radius_deg',
            'p__tbb_keV', 
            'p__te_keV', 
            's__colatitude_deg', 
            's__radius_deg',
            's__tbb_keV', 
            's__te_keV', 
            'T_in_keV' if 'disk' in self.bkg else None
            ]

        # Filter out None values
        self.names = [name for name in self.names if name is not None]
        return self.names
    

    def bounds(self):
        cos_i_low = 0.
        cos_i_high = 1.
        dist_low = 5.
        dist_high = 15. 
        nh_high = 100.
        
        bounds = {
                  'mass': (1.0, 3.0),
                  'radius':(3.0 * gravradius(1.0), 16.0),
                  'distance': (dist_low, dist_high),
                  'cos_inclination':(cos_i_low, cos_i_high), 
                  'p__phase_shift':(-0.25, 0.75),
                  'p__super_colatitude':(0.001, math.pi - 0.001),
                  'p__super_radius':(0.001, math.pi/2.0),
                  'p__super_tbb':(0.001, 0.003),
                  'p__super_te':(40.,200.),
                  'p__super_tau': (0.5, 3.5),
                  's__phase_shift':(-0.25, 0.75),
                  's__super_colatitude':(0.001, math.pi - 0.001),
                  's__super_radius':(0.001, math.pi/2.0),
                  's__super_tbb':(0.001, 0.003),
                  's__super_te':(40.,200.),
                  's__super_tau': (0.5, 3.5),
                  'column_density': (0., nh_high), 
                  'compactness': (0., 10.), # derived parameters bellow
                  'inclination_deg': (np.arccos(cos_i_high)*180/np.pi, 
                                      np.arccos(cos_i_low)*180/np.pi),
                  'p__colatitude_deg': (0.001, 180-0.001),
                  'p__radius_deg': (0.001, 90),
                  'p__tbb_keV': (0.511, 1.533),
                  'p__te_keV': (40*511/1000, 200*511/1000), 
                  's__colatitude_deg': (0.001, 180-0.001),
                  's__radius_deg': (0.001, 90),
                  's__tbb_keV': (0.511, 1.533),
                  's__te_keV': (40*511/1000, 200*511/1000)
                  }
        

        if 'disk' in self.bkg:
            bounds['T_in'] = (5.06, 6.84) # log10
            bounds['R_in'] = (5, 60) # from star radius to around corotation radius for the heaviest saxJ1808 possible # (27, 33)  # (20, 200) # km
            bounds['T_in_keV'] = (0.01, 0.6)
        
        return bounds

    def truths(self):
        truths={
            'mass':self.mass, # solar masses
            'radius': self.radius,                              # Equatorial radius in km
            'distance': self.distance,                            # Distance in kpc
            'cos_inclination': self.cos_i,          # Cosine of Earth inclination to rotation axis
            'p__phase_shift': self.p_phase_shift,                    # Phase shift
            'p__super_colatitude': self.p_colatitude,                # Colatitude of the centre of the superseding region
            'p__super_radius': self.p_radius,                 # Angular radius of the (circular) superseding region
            'p__super_tbb': self.p_tbb,                      # Blackbody temperature
            'p__super_te': self.s_te,                          # Electron temperature
            'p__super_tau': self.s_tau,
            's__phase_shift': self.s_phase_shift,                    # Phase shift
            's__super_colatitude': self.s_colatitude,                # Colatitude of the centre of the superseding region
            's__super_radius': self.s_radius,                 # Angular radius of the (circular) superseding region
            's__super_tbb': self.s_tbb,                      # Blackbody temperature
            's__super_te': self.s_te,                          # Electron temperature
            's__super_tau': self.s_tau,
            'column_density': self.column_density,
            'compactness': gravradius(self.mass/self.radius), #derived parameters
            'inclination_deg':self.inclination,
            'p__colatitude_deg': self.p_colatitude*180/np.pi,
            'p__radius_deg': self.p_radius*180/np.pi,
            'p__tbb_keV': self.p_tbb*511,
            'p__te_keV': self.p_te*511/1000, 
            's__colatitude_deg': self.s_colatitude*180/np.pi,
            's__radius_deg': self.s_radius*180/np.pi,
            's__tbb_keV': self.s_tbb*511,
            's__te_keV': self.s_te*511/1000
            }


        if 'disk' in self.bkg:
            truths['T_in'] = self.diskbb_T_log10_K
            truths['T_in_keV'] = self.diskbb_T_keV
            truths['R_in'] = self.R_in

        return truths
    
    def labels(self):
        # labels = {'mass': r"M\;\mathrm{[M}_{\odot}\mathrm{]}",
        #           'radius': r"R_{\mathrm{eq}}\;\mathrm{[km]}",
        #           'distance': r"D \;\mathrm{[kpc]}",
        #           'cos_inclination': r"\mathrm{cos}(i)",
        #           'p__phase_shift': r"\phi_\mathrm{p}\;\mathrm{[cycles]}",
        #           'p__super_colatitude': r"\theta_\mathrm{p}\;\mathrm{[rad]}",
        #           'p__super_radius': r"\zeta_\mathrm{p}\;\mathrm{[rad]}",
        #           'p__super_tbb': r"T_\{bb,p}\;\mathrm{[data units]}",
        #           'p__super_te': r"T_\mathrm{e,p}\;\mathrm{[data units]}",
        #           'p__super_tau': r"\tau_\mathrm{p}\;[-]",
        #           's__phase_shift': r"\phi_\mathrm{s}\;\mathrm{[cycles]}",
        #           's__super_colatitude': r"\theta_\mathrm{s}\;\mathrm{[rad]}",
        #           's__super_radius': r"\zeta_\mathrm{s}\;\mathrm{[rad]}",
        #           's__super_tbb': r"T_\{bb,s}\;\mathrm{[data units]}",
        #           's__super_te': r"T_\mathrm{e,s}\;\mathrm{[data units]}",
        #           's__super_tau': r"\tau_\mathrm{s}\;[-]",
        #           'column_density': r"N_\mathrm{H}\;[10^{21} \mathrm{cm}^{-2}]",
        #           'compactness': r"M/R_{\mathrm{eq}}",
        #           'inclination_deg': r'i\;\mathrm{[deg]}',
        #           'p__colatitude_deg': r'\theta_\mathrm{p}\;\mathrm{[deg]}',
        #           'p__radius_deg': r'\zeta_\mathrm{p}\;\mathrm{[deg]}',
        #           'p__tbb_keV': r"T_\mathrm{bb,p}\;\mathrm{[keV]}",
        #           'p__te_keV': r"T_\mathrm{e,p}\;\mathrm{[keV]}",
        #           's__colatitude_deg': r'\theta_\mathrm{s}\;\mathrm{[deg]}',
        #           's__radius_deg': r'\zeta_\mathrm{s}\;\mathrm{[deg]}',
        #           's__tbb_keV': r"T_\mathrm{bb,s}\;\mathrm{[keV]}",
        #           's__te_keV': r"T_\mathrm{e,s}\;\mathrm{[keV]}"
        #           }
        
        labels = {
            'mass': r"$M\;\mathrm{[M}_\odot\mathrm{]}$",
            'radius': r"$R_{\mathrm{eq}}\;\mathrm{[km]}$",
            'distance': r"$D\;\mathrm{[kpc]}$",
            'cos_inclination': r"$\cos(i)$",
            'p__phase_shift': r"$\phi_\mathrm{p}\;\mathrm{[cycles]}$",
            'p__super_colatitude': r"$\theta_\mathrm{p}\;\mathrm{[rad]}$",
            'p__super_radius': r"$\zeta_\mathrm{p}\;\mathrm{[rad]}$",
            'p__super_tbb': r"$T_{\mathrm{bb,p}}\;\mathrm{[data\ units]}$",
            'p__super_te': r"$T_{\mathrm{e,p}}\;\mathrm{[data\ units]}$",
            'p__super_tau': r"$\tau_\mathrm{p}\;[-]$",
            's__phase_shift': r"$\phi_\mathrm{s}\;\mathrm{[cycles]}$",
            's__super_colatitude': r"$\theta_\mathrm{s}\;\mathrm{[rad]}$",
            's__super_radius': r"$\zeta_\mathrm{s}\;\mathrm{[rad]}$",
            's__super_tbb': r"$T_{\mathrm{bb,s}}\;\mathrm{[data\ units]}$",
            's__super_te': r"$T_{\mathrm{e,s}}\;\mathrm{[data\ units]}$",
            's__super_tau': r"$\tau_\mathrm{s}\;[-]$",
            'column_density': r"$N_\mathrm{H}\;[10^{21}\ \mathrm{cm}^{-2}]$",
            'compactness': r"$M/R_{\mathrm{eq}}$",
            'inclination_deg': r"$i\;\mathrm{[deg]}$",
            'p__colatitude_deg': r"$\theta_\mathrm{p}\;\mathrm{[deg]}$",
            'p__radius_deg': r"$\zeta_\mathrm{p}\;\mathrm{[deg]}$",
            'p__tbb_keV': r"$T_\mathrm{bb,p}\;\mathrm{[keV]}$",
            'p__te_keV': r"$T_\mathrm{e,p}\;\mathrm{[keV]}$",
            's__colatitude_deg': r"$\theta_\mathrm{s}\;\mathrm{[deg]}$",
            's__radius_deg': r"$\zeta_\mathrm{s}\;\mathrm{[deg]}$",
            's__tbb_keV': r"$T_\mathrm{bb,s}\;\mathrm{[keV]}$",
            's__te_keV': r"$T_\mathrm{e,s}\;\mathrm{[keV]}$"
            }
             
        if 'disk' in self.bkg:
            labels['T_in'] = r"T_{in} log10 of Kelvin"
            labels['T_in_keV'] = r"T_\mathrm{in}\;\mathrm{[keV]}"
            labels['R_in'] =  r"R_\mathrm{in}\;\mathrm{[km]}"
            

        
        return labels
