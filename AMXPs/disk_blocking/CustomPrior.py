#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:44:07 2023

@author: bas
"""
import numpy as np
import xpsi
import math
from scipy.stats import truncnorm
from xpsi.global_imports import gravradius, _2pi
from helper_functions import get_keV_from_log10_Kelvin
from scipy.interpolate import Akima1DInterpolator

import os
this_directory = os.path.dirname(os.path.abspath(__file__))

class CustomPrior_STU(xpsi.Prior):
    """ A custom (joint) prior distribution.

    Model variant: ST-U
        Two single temperature hotspots, unshared parameters

   
    p[0] = 1 to 3 solar mass
    p[1] = 3G to 16 km (and also there are compactness restrictions)
    p[2] = distance
    p[3] = cos inclination 0 to 1
    p[4] = primary phase shift 0 to 2pi
    p[5] = primary colatitude 0 to pi (/2? From inverse sampling I see it is not divided by two.)
    p[6] = primary angular radius 0 to pi/2
    p[7] = primary hotspot seed temperature 0.5 - 1.5 keV
    p[8] = primary hotspot electron temperature 20 - 100 keV
    p[9] = primary tau 0.5 - 3.5
    p[10] = secondary phase shift 0 to 2pi
    p[11] = secondary colatitude 0 to pi (/2? From inverse sampling I see it is not divided by two.)
    p[12] = secondary angular radius 0 to pi/2
    p[13] = secondary hotspot seed temperature 0.5 - 1.5 keV
    p[14] = secondary hotspot electron temperature 20 - 100 keV
    p[15] = secondary tau 0.5 - 3.5
    p[16] = disk temperature log10
    p[17] = disk inner radius 20 to 64 km
    p[18] = nH gaussian 1.17 += 0.2 x 10^21 cm^-2
    

    """
    __derived_names__ = [
        'compactness',  # derived parameters below (new units)
        'inclination_deg',
        'p_colatitude_deg', 
        'p_radius_deg',
        'p_tbb_keV', 
        'p_te_keV', 
        's_colatitude_deg', 
        's_radius_deg',
        's_tbb_keV', 
        's_te_keV', 
        'T_in_keV',
        ]

    __draws_from_support__ = 4 #10^x
    
    
    def __init__(self, scenario, bkg, *args, **kwargs):
        self.scenario = scenario
        self.bkg = bkg
        
        super(CustomPrior_STU, self).__init__(*args, **kwargs)


    def __call__(self, p = None):

        """ Evaluate distribution at ``p``.

        :param list p: Model parameter values.

        :returns: Logarithm of the distribution evaluated at ``p``.

        """
        temp = super(CustomPrior_STU, self).__call__(p)
        if not np.isfinite(temp):
            return temp

        ref = self.parameters.star.spacetime # shortcut

        # based on contemporary EOS theory
        if not ref['radius'] <= 16.0:
            return -np.inf
      
        # causality limit for compactness
        R_p = 1.0 + ref.epsilon * (-0.788 + 1.030 * ref.zeta)
        if R_p < 1.45 / ref.R_r_s:
            return -np.inf

        mu = math.sqrt(-1.0 / (3.0 * ref.epsilon * (-0.788 + 1.030 * ref.zeta)))

        # 2-surface cross-section have a single maximum in |z|
        # i.e., an elliptical surface; minor effect on support, if any,
        # for high spin frequenies
        if mu < 1.0:
            return -np.inf
        
        if 'disk' in  self.bkg:
        
            # inner disk must be smaller than corotation radius, otherwise we enter (weak) propeller regime
           if not self.parameters['R_in'] < 1.49790e3*ref['mass']**(1/3)*ref['frequency']**(-2/3): # 1.49790e3 = (G*M_sol/4pi^2)^(1/3) in km
               return -np.inf
    
            # inner disk must be larger than neutron star equatorial radius
           if not self.parameters['R_in'] > ref['radius']:
               return -np.inf
        
        ref = self.parameters # redefine shortcut
        
        # enforce order in hot region colatitude
        if ref['p__super_colatitude'] > ref['s__super_colatitude']:
            # print('no order in hotregions')
            return -np.inf
 
        phi = (ref['p__phase_shift'] - 0.5 - ref['s__phase_shift']) * _2pi
 
        ang_sep = xpsi.HotRegion.psi(ref['s__super_colatitude'],
                                     phi,
                                     ref['p__super_colatitude'])
 
        # hot regions cannot overlap
        if ang_sep < ref['p__super_radius'] + ref['s__super_radius']:
            # print('overlapping hotregions')
            return -np.inf

        return 0.0

    def inverse_sample(self, hypercube=None):
        """ Draw sample uniformly from the distribution via inverse sampling. """

        to_cache = self.parameters.vector

        if hypercube is None:
            hypercube = np.random.rand(len(self))

        # the base method is useful, so to avoid writing that code again:
        _ = super(CustomPrior_STU, self).inverse_sample(hypercube)

        ref = self.parameters # shortcut
        
        idx = ref.index('column_density')
        temporary = truncnorm.ppf(hypercube[idx], -5.0, 5.0, loc=29., scale=4.)  # I think this is reasonably wide for SRGA J1444 based on the values I find in the literature. (Malacaria et al. 2025, Li et al. 2025, Papitto et al. 2025)
        if temporary < 0: temporary = 0
        ref['column_density'] = temporary
    
        idx = ref.index('distance')
        temporary = truncnorm.ppf(hypercube[idx], -5.0, 5.0, loc=8, scale=1.) # 8 is based on Molkov et al. 2024. Scale is sort of based on nothing, but it at least allows values found in the literature (Ng. et al 2025, Fu et al. 2025)
        if temporary < 0: temporary = 0
        ref['distance'] = temporary

        # flat priors in cosine of hot region centre colatitudes (isotropy)
        # support modified by no-overlap rejection condition
        idx = ref.index('p__super_colatitude')
        a, b = ref.get_param('p__super_colatitude').bounds
        a = math.cos(a); b = math.cos(b)
        ref['p__super_colatitude'] = math.acos(b + (a - b) * hypercube[idx])
        
        idx = ref.index('s__super_colatitude')
        a, b = ref.get_param('s__super_colatitude').bounds
        a = math.cos(a); b = math.cos(b)
        ref['s__super_colatitude'] = math.acos(b + (a - b) * hypercube[idx])

        # restore proper cache
        for parameter, cache in zip(ref, to_cache):
            parameter.cached = cache

        # it is important that we return the desired vector because it is
        # automatically written to disk by MultiNest and only by MultiNest
        return self.parameters.vector

    def transform(self, p, **kwargs):
        """ Add compactness. """

        p = list(p) # copy

        # used ordered names and values
        ref = dict(zip(self.parameters.names, p))


        for phase_shift in ['p_phase_shift', 's_phase_shift']:
            if ref[phase_shift] > 0.5:
                p += [ref[phase_shift] - 1.0]
            else:
                p += [ref[phase_shift]]

    
        p += [gravradius(ref['mass']) / ref['radius']]     # compactness ratio M/R_eq
        p += [np.arccos(ref['cos_inclination'])*180/np.pi]
        p += [ref['p_super_colatitude']*180/np.pi]
        p += [ref['p_super_radius']*180/np.pi]
        p += [ref['p_super_tbb']*511]
        p += [ref['p_super_te']*511/1000]
        p += [ref['s_super_tbb']*511]
        p += [ref['s_super_te']*511/1000]
        p += [ref['s_super_colatitude']*180/np.pi]
        p += [ref['s_super_radius']*180/np.pi]
        if 'disk' in self.bkg:
            p += [get_keV_from_log10_Kelvin(ref['T_in'])] # T_in keV
        return p
