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
    def __init__(self, 
                 scenario, 
                 bkg, 
                 fix_mass=False, 
                 ew=False, 
                 polarization=False,
                 secondary=False):
        self.scenario = scenario
        self.bkg = bkg
        self.fix_mass = fix_mass
        self.ew = ew
        self.polarization = polarization
        self.secondary = secondary

                
        if self.scenario in ('J1444','J1444s'):
            self.mass = 1.4 
            self.radius = 11.
            self.distance = 8. # Assumed in Papitto+ 2024 and Malacaria+ 2025
            self.inclination = 74.1 # best fit papitto+ 2024
            self.cos_i = math.cos(self.inclination*math.pi/180) #
            
            # Hotspot
            self.phase_shift = 0.0
            self.super_colatitude = 0.175993450466385537E+00 #0.21642082724729686 # 45*math.pi/180 # 20*math.pi/180 # 
            self.super_radius = 30.*math.pi/180
            
            # Compton slab model parameters
            self.tbb=0.0025#0.0025 #0.001 -0.003 Tbb(data) = Tbb(keV)/511keV, 1 keV = 0.002 data
            self.te=100. #  #37*1000/511 # 50. # 40-200 corresponds to 20-100 keV (Te(data) = Te(keV)*1000/511keV), 50 keV = 100 data
            self.tau=2.0 #0.5 - 3.5 tau = ln(Fin/Fout)
            
            
            if self.secondary:
                # secondary
                self.phase_shift_s = self.phase_shift + 0.5
                self.super_colatitude_s = np.pi - self.super_colatitude 
                self.super_radius_s = self.super_radius
                self.tbb_s = self.tbb
                self.te_s = self.te
                self.tau_s = self.tau
                
    
            if 'disk' in self.bkg:
            # source background
                #self.diskbb_T_keV = 0.16845756373108872# 0.17#  # # 0.3  #  keV #0.3 keV for Kajava+ 2011
                #self.diskbb_T_log10_K = get_T_in_log10_Kelvin(self.diskbb_T_keV)
                self.T_in_keV = 0.16845756373108872
                self.R_in = 24. #   # 20 #  1 #  km #  for very small diskBB background
            self.column_density = 29. #10^21 cm^-2
            if self.polarization:
                self.spin_axis_angle = 0.0
                
            self.frequency=447.8715611
            
        
        
    def p(self):
        self.p = [
        self.mass if not self.fix_mass else None,  # gravitational mass
        self.radius,  # coordinate equatorial radius
        self.distance,  # earth distance in kpc.
        self.cos_i,  # cosine of earth inclination
        self.spin_axis_angle if self.polarization else None, #Spin axis position angle measured from the north counterclock- wise to the projection of the rotation axis on the plane of the sky [in radians],
        self.phase_shift,  # phase of hot region
        self.super_colatitude,  # colatitude of center of superseding region
        self.super_radius,  # angular radius of superseding region
        self.tbb,
        self.te,
        self.tau,
        self.phase_shift_s if self.secondary else None,
        self.super_colatitude_s if self.secondary else None,
        self.super_radius_s if self.secondary else None,
        self.tbb_s if self.secondary else None,
        self.te_s if self.secondary else None,
        self.tau_s if self.secondary else None,
        self.T_in_keV if 'disk' in self.bkg else None,
        self.R_in if 'disk' in self.bkg else None,
        self.mu if 'line' in self.bkg else None,
        self.sigma if 'line' in self.bkg else None,
        self.N if 'line' in self.bkg else None,
        self.column_density
        ]

        # Remove any None values (e.g., mass if fix_mass is True, or optional elements)
        self.p = [x for x in self.p if x is not None]
        return self.p
        
    def names(self):
        self.names = [
            'mass' if not self.fix_mass else None, 
            'radius', 
            'distance', 
            'cos_inclination',
            'spin_axis_position_angle' if self.polarization else None,
            'phase_shift' if not self.secondary else None, 
            'super_colatitude' if not self.secondary else None, 
            'super_radius' if not self.secondary else None,
            'super_tbb' if not self.secondary else None,
            'super_te' if not self.secondary else None, 
            'super_tau' if not self.secondary else None, 
            'p__phase_shift' if self.secondary else None, 
            'p__super_colatitude' if self.secondary else None, 
            'p__super_radius' if self.secondary else None, 
            'p__super_tbb' if self.secondary else None, 
            'p__super_te' if self.secondary else None, 
            'p__super_tau' if self.secondary else None, 
            's__phase_shift' if self.secondary else None, 
            's__super_colatitude' if self.secondary else None, 
            's__super_radius' if self.secondary else None, 
            's__super_tbb' if self.secondary else None, 
            's__super_te' if self.secondary else None, 
            's__super_tau' if self.secondary else None, 
            # 'T_in' if self.bkg in ['disk', 'diskline'] else None,
            'T_in_keV' if self.bkg in ['disk', 'diskline'] else None,
            'R_in' if self.bkg in ['disk', 'diskline'] else None,
            'mu' if self.bkg == 'diskline' else None,
            'sigma' if self.bkg == 'diskline' else None,
            'N' if self.bkg == 'diskline' else None,
            'column_density', 
            'compactness',
            'inclination_deg',
            'tbb_keV' if not self.secondary else None, 
            'te_keV' if not self.secondary else None, 
            'colatitude_deg' if not self.secondary else None, 
            'radius_deg' if not self.secondary else None,
            'p__tbb_keV' if self.secondary else None, 
            'p__te_keV' if self.secondary else None, 
            'p__colatitude_deg' if self.secondary else None, 
            'p__radius_deg' if self.secondary else None,
            's__tbb_keV' if self.secondary else None, 
            's__te_keV' if self.secondary else None, 
            's__colatitude_deg' if self.secondary else None, 
            's__radius_deg' if self.secondary else None,
            'N_norm' if self.bkg == 'diskline' else None
        ]
        
        # Filter out None values
        self.names = [name for name in self.names if name is not None]
        return self.names

    def bounds(self):
        
        cos_i_constr = False
        
        if cos_i_constr:
            cos_i_low = np.cos((74.1+5.8)*np.pi/180) # papitto2024 limit here for j1444 
            cos_i_high = np.cos((74.1-6.3)*np.pi/180)
        elif not cos_i_constr:
            i_low_deg = 50
            i_high_deg = 90
            cos_i_low = np.cos(i_high_deg*np.pi/180)
            cos_i_high = np.cos(i_low_deg*np.pi/180)

        
        bounds = {'radius':(3.0 * gravradius(1.0), 16.0),
                  'distance': (2.5, 10.6),
                  'cos_inclination':(cos_i_low, cos_i_high), 
                  'phase_shift':(-0.5, 0.5),
                  'super_colatitude':(0.001, math.pi - 0.001),
                  'super_radius':(0.001, math.pi/2.0),
                  'super_tbb':(0.001, 0.003),
                  'tbb_keV': (0.511, 1.533),
                  'super_te': (40., 200.),
                  'te_keV': (40*511/1000, 200*511/1000),
                  'super_tau': (0.5, 3.5),
                  'column_density': (19., 29.),
                  'compactness': (0., 10.),
                  'inclination_deg': (i_low_deg, i_high_deg),
                  'colatitude_deg': (0.001, 180-0.001),
                  'radius_deg': (0.001, 90)              
                  }
        
        if not self.fix_mass:
            bounds['mass'] = (1.0, 3.0)
            
        if self.ew:
            bounds['elsewhere_temperature'] = (None, None)

        if 'disk' in self.bkg:
            bounds['T_in_keV'] = (0.01, 0.6) # (0.225, 0.275 )  # (0.01, 0.6) # keV
            bounds['R_in'] = (5, 60) # from star radius to around corotation radius for the heaviest saxJ1808 possible # (27, 33)  # (20, 200) # km
            # bounds['T_in_keV'] = (None, None)
            # bounds['T_in_keV'] = (0.01, 0.6)
            
            
        if 'line' in self.bkg:
            bounds['mu'] = (0.8,1.1)
            bounds['sigma'] = (1e-2,5e-1)
            bounds['N'] = (1e35,1e38)
            bounds['N_norm'] = (1e-2,1e1)
        
        if self.polarization:
            bounds['spin_axis_position_angle']=(-math.pi/2.0, math.pi/2.0)
        
        return bounds

    def truths(self):
        truths={'radius': self.radius,                              # Equatorial radius in km
          'compactness': gravradius(self.mass/self.radius),
          'distance': self.distance,                            # Distance in kpc
          'cos_inclination': self.cos_i,          # Cosine of Earth inclination to rotation axis
          'phase_shift': self.phase_shift,                    # Phase shift
          'super_colatitude': self.super_colatitude,                # Colatitude of the centre of the superseding region
          'super_radius': self.super_radius,                 # Angular radius of the (circular) superseding region
          'super_tbb': self.tbb,                      # Blackbody temperature
          'tbb_keV': self.tbb*511,
          'super_te': self.te,                          # Electron temperature
          'te_keV': self.te*511/1000,
          'super_tau': self.tau,
          'column_density': self.column_density,
          'inclination_deg':self.inclination,
          'colatitude_deg': self.super_colatitude*180/np.pi,
          'radius_deg': self.super_radius*180/np.pi}
    
        if not self.fix_mass:
            truths['mass'] = self.mass

        if 'disk' in self.bkg:
            truths['T_in'] = self.diskbb_T_log10_K
            truths['T_in_keV'] = self.diskbb_T_keV
            # truths['T_in_keV'] = self.T_in_keV
            truths['R_in'] = self.R_in
        
        if 'line' in self.bkg:
            truths['mu'] = self.mu
            truths['sigma'] = self.sigma
            truths['N'] = self.N
            truths['N_norm'] = self.N*1e-37
        
        if self.polarization:
            truths['spin_axis_position_angle']=self.spin_axis_angle
        
        return truths
    
    def labels(self):
        # labels = {'radius': r"R_{\mathrm{eq}}\;\mathrm{[km]}",
        #       'compactness': r"M/R_{\mathrm{eq}}",
        #       'distance': r"D \;\mathrm{[kpc]}",
        #       'cos_inclination': r"\mathrm{cos}(i)",
        #       'phase_shift': r"\phi\;\mathrm{[cycles]}",
        #       'super_colatitude': r"\Theta_{spot}\;\mathrm{[rad]}",
        #       'super_radius': r"\zeta_{spot}\;\mathrm{[rad]}",
        #       'super_tbb': r"T_\{seed}\;\mathrm{[data units]}",
        #       'tbb_keV': r"T_\mathrm{seed}\;\mathrm{[keV]}",
        #       'super_te': r"T_\mathrm{electrons}\;\mathrm{[data units]}",
        #       'te_keV': r"T_\mathrm{e}\;\mathrm{[keV]}",
        #       'super_tau': r"\tau\;[-]",
        #       'column_density': r"N_\mathrm{H}\;[10^{21} \mathrm{cm}^{-2}]",
        #       'inclination_deg': r'i\;\mathrm{[deg]}',
        #       'colatitude_deg': r'\theta\;\mathrm{[deg]}',
        #       'radius_deg': r'\zeta\;\mathrm{[deg]}'}
        
        # if not self.fix_mass:
        #     labels['mass'] =  r"M\;\mathrm{[M}_{\odot}\mathrm{]}"
        
        # if 'disk' in self.bkg:
        #     labels['T_in'] = r"T_{in} log10 of Kelvin"
        #     # labels['T_in_keV'] = r"T_\mathrm{in}\;\mathrm{[keV]}"
        #     labels['R_in'] =  r"R_\mathrm{in}\;\mathrm{[km]}"
            
        # if 'line' in self.bkg:
        #     labels['mu'] = r"\mu\;\mathrm{[keV]}"
        #     labels['sigma'] = r"\sigma\;\mathrm{[keV]}"
        #     labels['N'] =  r"N\;\mathrm{[photons/cm^2/s]}"
        #     labels['N_norm'] =  r"N_\mathrm{norm}\;\mathrm{[photons/cm^2/s]}"

        # if self.polarization:
        #     labels['spin_axis_position_angle']=r"Chi\;\mathrm{[rad]}"
        
        labels = {'radius': r"$R_{\mathrm{eq}}\;\mathrm{[km]}$",
              'compactness': r"$M/R_{\mathrm{eq}}$",
              'distance': r"$D \;\mathrm{[kpc]}$",
              'cos_inclination': r"$\mathrm{cos}(i)$",           
              'column_density': r"$N_\mathrm{H}\;[10^{21} \mathrm{cm}^{-2}]$",
              'inclination_deg': r"$i\;\mathrm{[deg]}$"}
        
        
        if not self.secondary:
            labels['phase_shift']       = r"$\phi\;\mathrm{[cycles]}$"
            labels['super_colatitude']  = r"$\Theta_{spot}\;\mathrm{[rad]}$"
            labels['colatitude_deg']    = r"$\theta\;\mathrm{[deg]}$",
            labels['super_radius']      = r"$\zeta_{spot}\;\mathrm{[rad]}$"
            labels['radius_deg']        = r"$\zeta\;\mathrm{[deg]}$"
            labels['super_tbb']         = r"$T_\mathrm{seed}\;\mathrm{[data units]}$"
            labels['tbb_keV']           = r"$T_\mathrm{seed}\;\mathrm{[keV]}$"
            labels['super_te']          = r"$T_\mathrm{electrons}\;\mathrm{[data units]}$"
            labels['te_keV']            = r"$T_\mathrm{e}\;\mathrm{[keV]}$"
            labels['super_tau']         = r"$\tau\;[-]$"    
        elif self.secondary:
            labels['p__phase_shift']       = r"$\phi_\mathrm{p}\;\mathrm{[cycles]}$"
            labels['p__super_colatitude']  = r"$\Theta_\mathrm{p}\;\mathrm{[rad]}$"
            labels['p__colatitude_deg']    = r"$\theta_\mathrm{p}\;\mathrm{[deg]}$",
            labels['p__super_radius']      = r"$\zeta_\mathrm{p}\;\mathrm{[rad]}$"
            labels['p__radius_deg']        = r"$\zeta_\mathrm{p}\;\mathrm{[deg]}$"
            labels['p__super_tbb']         = r"$T_\mathrm{seed,p}\;\mathrm{[data units]}$"
            labels['p__tbb_keV']           = r"$T_\mathrm{seed,p}\;\mathrm{[keV]}$"
            labels['p__super_te']          = r"$T_\mathrm{e,p}\;\mathrm{[data units]}$"
            labels['p__te_keV']            = r"$T_\mathrm{e,p}\;\mathrm{[keV]}$"
            labels['p__super_tau']         = r"$\tau_\mathrm{p}\;[-]$"
            
            labels['s__phase_shift']       = r"$\phi_\mathrm{s}\;\mathrm{[cycles]}$"
            labels['s__super_colatitude']  = r"$\Theta_\mathrm{s}\;\mathrm{[rad]}$"
            labels['s__colatitude_deg']    = r"$\theta_\mathrm{s}\;\mathrm{[deg]}$",
            labels['s__super_radius']      = r"$\zeta_\mathrm{s}\;\mathrm{[rad]}$"
            labels['s__radius_deg']        = r"$\zeta_\mathrm{s}\;\mathrm{[deg]}$"
            labels['s__super_tbb']         = r"$T_\mathrm{seed,s}\;\mathrm{[data units]}$"
            labels['s__tbb_keV']           = r"$T_\mathrm{seed,s}\;\mathrm{[keV]}$"
            labels['s__super_te']          = r"$T_\mathrm{e,s}\;\mathrm{[data units]}$"
            labels['s__te_keV']            = r"$T_\mathrm{e,s}\;\mathrm{[keV]}$"
            labels['s__super_tau']         = r"$\tau_\mathrm{s}\;[-]$"
            
        
        if not self.fix_mass:
            labels['mass'] =  r"$M\;\mathrm{[M}_{\odot}\mathrm{]}$"
        
        if 'disk' in self.bkg:
            labels['T_in'] = r"$T_{in} log10 of Kelvin$"
            labels['T_in_keV'] = r"$T_\mathrm{in}\;\mathrm{[keV]}$"
            labels['R_in'] =  r"$R_\mathrm{in}\;\mathrm{[km]}$"
            
        if 'line' in self.bkg:
            labels['mu'] = r"$\mu\;\mathrm{[keV]}$"
            labels['sigma'] = r"$\sigma\;\mathrm{[keV]}$"
            labels['N'] =  r"$N\;\mathrm{[photons/cm^2/s]}$"
            labels['N_norm'] =  r"$N_\mathrm{norm}\;\mathrm{[photons/cm^2/s]}$"

        if self.polarization:
            labels['spin_axis_position_angle']=r"$Chi\;\mathrm{[rad]}$"

        
        return labels
