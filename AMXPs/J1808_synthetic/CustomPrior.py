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
from xpsi.global_imports import _c, _G, _dpr, gravradius, _csq, _km, _2pi, _keV, _k_B, _c_cgs
from helper_functions import get_keV_from_log10_Kelvin

class CustomPrior(xpsi.Prior):
    """ A custom (joint) prior distribution.

    Source: SAX-J1808.4-3658
    Model variant: ST
        One single-temperature

   
    p[0] = 1 to 3 solar mass
    p[1] = 3G to 16 km (and also there are compactness restrictions)
    p[2] = distance with a uniform prior from 3.4 to 4.6 (Galloway & Cumming 2006)
    p[3] = cos inclination 0 to 1
    p[3] = phase shift 0 to 2pi
    p[4] = colatitude 0 to pi (/2? From inverse sampling I see it is not divided by two.)
    p[5] = angular radius 0 to pi/2
    p[6] = hotspot seed temperature 0.5 - 1.5 keV
    p[7] = hotspot electron temperature 20 - 100 keV
    p[8] = tau 0.5 - 3.5
    p[9] = elsewhere temperature 0.01 - 0.6 keV
    p[10] = disk temperature 0.01 - 0.6 keV
    p[11] = disk inner radius 20 to 64 km
    p[12] = nH gaussian 1.17 += 0.2 x 10^21 cm^-2
    

    """

    __derived_names__ = ['compactness', 'T_else_keV', 'T_in_keV', 'tbb_keV', 'te_keV']#, 'phase_separation',]
    __draws_from_support__ = 4 #10^x
    
    def __init__(self, scenario, bkg, *args, **kwargs):
        self.scenario = scenario
        self.bkg = bkg
        super(CustomPrior, self).__init__(*args, **kwargs)

    def __call__(self, p = None):
        """ Evaluate distribution at ``p``.

        :param list p: Model parameter values.

        :returns: Logarithm of the distribution evaluated at ``p``.

        """
        temp = super(CustomPrior, self).__call__(p)
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

        # polar radius at photon sphere for ~static star (static ambient spacetime)
        #if R_p < 1.5 / ref.R_r_s:
        #    return -np.inf

        mu = math.sqrt(-1.0 / (3.0 * ref.epsilon * (-0.788 + 1.030 * ref.zeta)))

        # 2-surface cross-section have a single maximum in |z|
        # i.e., an elliptical surface; minor effect on support, if any,
        # for high spin frequenies
        if mu < 1.0:
            return -np.inf
        
        
        # inner disk must be smaller than corotation radius, otherwise we enter (weak) propeller regime
        if not self.parameters['R_in'] < 1.49790e3*ref['mass']**(1/3)*ref['frequency']**(-2/3): # 1.49790e3 = (G*M_sol/4pi^2)^(1/3) in km
            return -np.inf
        
        # ref = self.parameters # redefine shortcut

        return 0.0

    def inverse_sample(self, hypercube=None):
        """ Draw sample uniformly from the distribution via inverse sampling. """

        to_cache = self.parameters.vector

        if hypercube is None:
            hypercube = np.random.rand(len(self))

        # the base method is useful, so to avoid writing that code again:
        _ = super(CustomPrior, self).inverse_sample(hypercube)

        ref = self.parameters # shortcut
        
        if self.scenario == 'literature' or self.scenario == '2019' or self.scenario == '2022':
            idx = ref.index('column_density')
            temporary = truncnorm.ppf(hypercube[idx], -5.0, 5.0, loc=1.17, scale=0.2)
            if temporary < 0: temporary = 0
            ref['column_density'] = temporary

        if self.scenario == 'kajava':
            idx = ref.index('column_density')
            temporary = truncnorm.ppf(hypercube[idx], -5.0, 5.0, loc=1.13, scale=0.2)
            if temporary < 0: temporary = 0
            ref['column_density'] = temporary
    
        idx = ref.index('distance')
        temporary = truncnorm.ppf(hypercube[idx], -5.0, 5.0, loc=2.7, scale=0.3)
        if temporary < 0: temporary = 0
        ref['distance'] = temporary

        # flat priors in cosine of hot region centre colatitudes (isotropy)
        # support modified by no-overlap rejection condition
        idx = ref.index('super_colatitude')
        a, b = ref.get_param('super_colatitude').bounds
        a = math.cos(a); b = math.cos(b)
        ref['super_colatitude'] = math.acos(b + (a - b) * hypercube[idx])




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

        # compactness ratio M/R_eq
        # p += [gravradius(ref['mass']) / ref['radius']]
        # p += [get_keV_from_log10_Kelvin(ref['elsewhere_temperature'])]
        if self.bkg == 'model':
            p += [get_keV_from_log10_Kelvin(ref['T_in'])]
        p += [ref['super_tbb']*511]
        p += [ref['super_te']*511/1000]
        

        return p
