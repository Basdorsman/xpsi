#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:32:15 2024

@author: bas
"""
import math
from helper_functions import get_T_in_log10_Kelvin
from xpsi.global_imports import gravradius

class parameter_values(object):
    def __init__(self, scenario, bkg):
        self.scenario = scenario
        self.bkg = bkg


        if self.scenario == 'kajava':
            self.mass = 1.4
            self.radius = 11
            self.distance = 3.5
            self.inclination = 58
            self.cos_i = math.cos(self.inclination*math.pi/180)
            
            # Hotspot
            self.phase_shift = 0.20
            self.super_colatitude = 11*math.pi/180 # 20*math.pi/180 # 
            self.super_radius = 10*math.pi/180
            
            # Compton slab model parameters
            self.tbb=0.85/511 # 0.0017 #0.001 -0.003 Tbb(data) = Tbb(keV)/511keV, 1 keV = 0.002 data
            self.te=50*1000/511. # 50. # 40-200 corresponds to 20-100 keV (Te(data) = Te(keV)*1000/511keV), 50 keV = 100 data
            self.tau=1 #0.5 - 3.5 tau = ln(Fin/Fout)
            
            # elsewhere
            self.elsewhere_T_keV = 0.5 # 0.5 #  keV 
            self.elsewhere_T_log10_K = get_T_in_log10_Kelvin(self.elsewhere_T_keV)
            if self.bkg == 'model':
            # source background
                self.diskbb_T_keV = 0.29 # 0.3  #  keV #0.3 keV for Kajava+ 2011
                self.diskbb_T_log10_K = get_T_in_log10_Kelvin(self.diskbb_T_keV)
                self.R_in = 55 # 20 #  1 #  km #  for very small diskBB background
            self.column_density = 1.13 #10^21 cm^-2

        if self.scenario =='literature' or self.scenario == '2019' or self.scenario == '2022':
            self.mass = 1.4
            self.radius = 12.
            self.distance = 3.5
            self.inclination = 60
            self.cos_i = math.cos(self.inclination*math.pi/180)
            
            # Hotspot
            self.phase_shift = 0
            self.super_colatitude = 45*math.pi/180 # 20*math.pi/180 # 
            self.super_radius = 15.5*math.pi/180
            
            # Compton slab model parameters
            self.tbb=0.0012 # 0.0017 #0.001 -0.003 Tbb(data) = Tbb(keV)/511keV, 1 keV = 0.002 data
            self.te=100. # 50. # 40-200 corresponds to 20-100 keV (Te(data) = Te(keV)*1000/511keV), 50 keV = 100 data
            self.tau=1. #0.5 - 3.5 tau = ln(Fin/Fout)
            
            # elsewhere
            self.elsewhere_T_keV = 0.4 # 0.5 #  keV 
            self.elsewhere_T_log10_K = get_T_in_log10_Kelvin(self.elsewhere_T_keV)
    
            if self.bkg == 'model':
            # source background
                self.diskbb_T_keV = 0.25 # 0.3  #  keV #0.3 keV for Kajava+ 2011
                self.diskbb_T_log10_K = get_T_in_log10_Kelvin(self.diskbb_T_keV)
                self.R_in = 30 # 20 #  1 #  km #  for very small diskBB background
            self.column_density = 1.17 #10^21 cm^-2
            
        # if self.scenario =='2019':
        #     self.mass = 1.706
        #     self.radius = 8.85
        #     self.distance = 3.4
        #     self.inclination = 8.11
        #     self.cos_i = math.cos(self.inclination*math.pi/180)
            
        #     # Hotspot
        #     self.phase_shift = 0.208
        #     self.super_colatitude = 137.7*math.pi/180 # 20*math.pi/180 # 
        #     self.super_radius = 90*math.pi/180
            
        #     # Compton slab model parameters
        #     self.tbb=0.868/511 # 0.0017 #0.001 -0.003 Tbb(data) = Tbb(keV)/511keV, 1 keV = 0.002 data
        #     self.te=101.9*1000/511 # 50. # 40-200 corresponds to 20-100 keV (Te(data) = Te(keV)*1000/511keV), 50 keV = 100 data
        #     self.tau=0.785 #0.5 - 3.5 tau = ln(Fin/Fout)
            
        #     # elsewhere
        #     self.elsewhere_T_keV = 0.453 # 0.5 #  keV 
        #     self.elsewhere_T_log10_K = get_T_in_log10_Kelvin(self.elsewhere_T_keV)
    
        #     if self.bkg == 'model':
        #     # source background
        #         self.diskbb_T_keV = 0.116 # 0.3  #  keV #0.3 keV for Kajava+ 2011
        #         self.diskbb_T_log10_K = get_T_in_log10_Kelvin(self.diskbb_T_keV)
        #         self.R_in = 64.0 # 20 #  1 #  km #  for very small diskBB background
        #     self.column_density = 1.17 #10^21 cm^-2
        
    def p(self):
        self.p = [#self.mass, #1.4, #grav mass
              #self.radius,#12.5, #coordinate equatorial radius
              # self.distance, # earth distance kpc
              self.cos_i, #cosine of earth inclination
              self.phase_shift, #phase of hotregion
              self.super_colatitude, #colatitude of centre of superseding region
              self.super_radius,  #angular radius superceding region
              self.tbb,
              self.te,
              self.tau]#,
              #self.elsewhere_T_log10_K]

        if self.bkg == 'model':
            self.p.append(self.diskbb_T_log10_K)
            self.p.append(self.R_in)

        # self.p.append(self.column_density)
        return self.p
        
    def names(self):
        if self.bkg == 'marginalise' or self.bkg == 'fix':
            self.names=['mass','radius','distance','cos_inclination',
                        'phase_shift','super_colatitude','super_radius',
                        'super_tbb','super_te','super_tau','elsewhere_temperature',
                        'column_density', 'compactness', 'T_else_keV', 'tbb_keV','te_keV' ]
        elif self.bkg =='model':
            self.names=['mass','radius','distance','cos_inclination',
                        'phase_shift','super_colatitude','super_radius',
                        'super_tbb','super_te','super_tau','elsewhere_temperature',
                        'T_in', 'R_in', 'column_density', 'compactness', 'T_else_keV',
                        'T_in_keV', 'tbb_keV','te_keV' ]
        return self.names

    def bounds(self):
        bounds = {'mass':(1.0,3.0),
              'radius':(3.0 * gravradius(1.0), 16.0),
              'compactness': (0., 10.),
              'distance': (1.2, 4.2), #5 sigma around 2.7   #(3.4, 3.6),  # (2.5, 3.6), #(3.4, 3.6),
              'cos_inclination':(0.15, 1.),
              'phase_shift':(-0.25, 0.75),
              'super_colatitude':(0.001, math.pi - 0.001),
              'super_radius':(0.001, math.pi/2.0 - 0.001),
              'super_tbb':(0.001, 0.003),
              'tbb_keV': (0.511, 1.533),
              'super_te': (40., 200.),
              'te_keV': (40*511/1000, 200*511/1000),
              'super_tau': (0.5, 3.5),
              'elsewhere_temperature': (5., 7.), #(6.5, 7.),
              'T_else_keV': (None, None),
              'column_density': (0., 3.)}
        bounds['T_else_keV'] = (None, None)
        if self.bkg == 'model':
            bounds['T_in'] = (0.01, 0.6) # (0.225, 0.275 )  # (0.01, 0.6) # keV
            bounds['R_in'] = (10, 50) # from star radius to around corotation radius for the heaviest saxJ1808 possible # (27, 33)  # (20, 200) # km
            bounds['T_in_keV'] = (None, None)
        
        return bounds

    def truths(self):
        truths={'mass': self.mass,                               # Mass in solar Mass
          'radius': self.radius,                              # Equatorial radius in km
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
          'elsewhere_temperature': self.elsewhere_T_log10_K,
          'T_else_keV': self.elsewhere_T_keV,
          'column_density': self.column_density}
    
        if self.bkg == 'model':
            truths['T_in'] = self.diskbb_T_log10_K
            truths['T_in_keV'] = self.diskbb_T_keV
            truths['R_in'] = self.R_in
        
        return truths
    
    def labels(self):
        labels = {'mass': r"M\;\mathrm{[M}_{\odot}\mathrm{]}",
              'radius': r"R_{\mathrm{eq}}\;\mathrm{[km]}",
              'compactness': r"M/R_{\mathrm{eq}}",
              'distance': r"D \;\mathrm{[kpc]}",
              'cos_inclination': r"\cos(i)",
              'phase_shift': r"\phi_{p}\;\mathrm{[cycles]}",
              'super_colatitude': r"\Theta_{spot}\;\mathrm{[rad]}",
              'super_radius': r"\zeta_{spot}\;\mathrm{[rad]}",
              'super_tbb': r"T_{seed}\;\mathrm{[data units]}",
              'tbb_keV': r"T_{seed} (keV)",
              'super_te': r"T_\mathrm{electrons}\;\mathrm{[data units]}",
              'te_keV': r"T_\mathrm{electrons} (keV)",
              'super_tau': r"\tau [-]",
              'elsewhere_temperature': r"T_e log10 of Kelvin",
              'T_else_keV': r"T_{\mathrm{else}} (keV)",
              'column_density': r"n_H 10\^21 cm\^{-2}"}
        
        if self.bkg == 'model':
            labels['T_in'] = r"T_{in} log10 of Kelvin"
            labels['T_in_keV'] = r"T_{\mathrm{in}} (keV)"
            labels['R_in'] =  r"R_\mathrm{in} (km)"

        
        return labels
