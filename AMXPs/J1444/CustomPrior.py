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

import sys
this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory+'/data/EoS_prior/')
from load_and_sample_eos_nf import NormalizingFlow


class CustomPrior(xpsi.Prior):
    """ A custom (joint) prior distribution.

    Source: SAX-J1808.4-3658
    Model variant: ST
        One single-temperature

   
    p[0] = 1 to 3 solar mass
    p[1] = 3G to 16 km (and also there are compactness restrictions)
    p[2] = distance with a uniform prior from 3.4 to 4.6 (Galloway & Cumming 2006)
    p[3] = cos inclination 0 to 1
    p[3] = phase shift -0.5 to 0.5
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

    __derived_names__ = ['compactness', 'tbb_keV', 'te_keV', 'inclination_deg', 'colatitude_deg', 'radius_deg', 'N_norm']
    __draws_from_support__ = 3 #10^x
    
    def __init__(self, scenario, bkg, *args, **kwargs):
        self.scenario = scenario
        self.bkg = bkg
        self.fix_mass = kwargs.pop('fix_mass', None)
        self.eos_informed = kwargs.pop('eos_informed', None)
        
        
        if self.eos_informed:        
            self.nf_eos_mr_prior = NormalizingFlow(this_directory+'/data/EoS_prior/flow_and_scaler_PP.pth')
        
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

        mu = math.sqrt(-1.0 / (3.0 * ref.epsilon * (-0.788 + 1.030 * ref.zeta)))

        # 2-surface cross-section have a single maximum in |z|
        # i.e., an elliptical surface; minor effect on support, if any,
        # for high spin frequenies
        if mu < 1.0:
            return -np.inf
        
        if 'disk' in self.bkg:
        
            # inner disk must be smaller than corotation radius, otherwise we enter (weak) propeller regime
            if not self.parameters['R_in'] < 1.49790e3*ref['mass']**(1/3)*ref['frequency']**(-2/3): # 1.49790e3 = (G*M_sol/4pi^2)^(1/3) in km
                return -np.inf
    
            # inner disk must be larger than neutron star equatorial radius
            if not self.parameters['R_in'] > ref['radius']:
                return -np.inf

        return 0.0

    def inverse_sample(self, hypercube=None):
        """ Draw sample uniformly from the distribution via inverse sampling. """

        to_cache = self.parameters.vector

        if hypercube is None:
            hypercube = np.random.rand(len(self))

        # the base method is useful, so to avoid writing that code again:
        _ = super(CustomPrior, self).inverse_sample(hypercube)

        ref = self.parameters # shortcut
    
        idx = ref.index('distance')
        temporary = truncnorm.ppf(hypercube[idx], -3.0, 3.0, loc=8.5, scale=2.)
        ref['distance'] = temporary

        # flat priors in cosine of hot region centre colatitudes (isotropy)
        # support modified by no-overlap rejection condition
        idx = ref.index('super_colatitude')
        a, b = ref.get_param('super_colatitude').bounds
        a = math.cos(a); b = math.cos(b)
        ref['super_colatitude'] = math.acos(b + (a - b) * hypercube[idx])

        if self.eos_informed:
            ref['mass'], ref['radius'] = self.nf_eos_mr_prior.sample_mr_from_nf()

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
        # print('ref', ref)

        # compactness ratio M/R_eq
        if not self.fix_mass:
            p += [gravradius(ref['mass']) / ref['radius']]
        elif self.fix_mass and self.scenario == '2019':
            p += [gravradius(1.4) / ref['radius']]
        else:
            raise(NotImplementedError)

        p += [ref['super_tbb']*511] # tbb in keV
        p += [ref['super_te']*511/1000] # te in keV
        p += [np.arccos(ref['cos_inclination'])*180/np.pi] # inclination in deg
        p += [ref['super_colatitude']*180/np.pi] # colatitude in deg
        p += [ref['super_radius']*180/np.pi] # ang radius in deg

        if 'line' in self.bkg:
            p+=[ref['N']*1e-37]
        return p


class CustomPrior_twohotspots(xpsi.Prior):
    """ A custom (joint) prior distribution.

    Source: SRGA J144459.2-604207
    Model variant: ST-U, ST-S
        Two single temperature hotspots
   
    p[0] = 1 to 3 solar mass (and EoS constraints)
    p[1] = 3G to 16 km (and EoS constraints, compactness restrictions)
    p[2] = distance normal distributed prior
    p[3] = cos inclination 0 to 0.64
    p[3] = phase shift -0.5 to 0.5
    p[4] = colatitude 0 to pi
    p[5] = angular radius 0 to pi/2
    p[6] = hotspot seed temperature 0.5 - 1.5 keV
    p[7] = hotspot electron temperature 20 - 100 keV
    p[8] = tau 0.5 - 3.5
    p[9] = phase shift -0.5 to 0.5
    p[10] = colatitude 0 to pi
    p[11] = angular radius 0 to pi/2
    p[12] = hotspot seed temperature 0.5 - 1.5 keV
    p[13] = hotspot electron temperature 20 - 100 keV
    p[14] = tau 0.5 - 3.5
    p[15] = disk temperature 0.01 - 0.6 keV
    p[16] = disk inner radius 20 to 64 km
    p[17] = nH gaussian 19 to 29 x 10^21 cm^-2

    """

    __derived_names__ = ['compactness', 'inclination_deg', 'p__tbb_keV', 'p__te_keV','p__colatitude_deg', 'p__radius_deg', 's__tbb_keV', 's__te_keV','s__colatitude_deg', 's__radius_deg' ]
  
    __draws_from_support__ = 3 #10^x
    
    
    def __init__(self, scenario, bkg, *args, **kwargs):
        self.scenario = scenario
        self.bkg = bkg
        self.fix_mass = kwargs.pop('fix_mass', None)
        self.eos_informed = kwargs.pop('eos_informed', None)
        self.variable_params = kwargs.pop('variable_params', None)
        self.posterior_combiner = kwargs.pop('posterior_combiner', None)

        
        if self.eos_informed:        
            self.nf_eos_mr_prior = NormalizingFlow(this_directory+'/data/EoS_prior/flow_and_scaler_PP.pth')
        
        super(CustomPrior_twohotspots, self).__init__(*args, **kwargs)

    def __call__(self, p = None):

        """ Evaluate distribution at ``p``.

        :param list p: Model parameter values.

        :returns: Logarithm of the distribution evaluated at ``p``.

        """
        temp = super(CustomPrior_twohotspots, self).__call__(p)
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
        
       
        if self.variable_params or self.posterior_combiner:  
            if 'disk' in  self.bkg:  
                # inner disk must be smaller than corotation radius, otherwise we enter (weak) propeller regime
                if not self.parameters['NICER__R_in'] < 1.49790e3*ref['mass']**(1/3)*ref['frequency']**(-2/3): # 1.49790e3 = (G*M_sol/4pi^2)^(1/3) in km
                    return -np.inf
        
                # inner disk must be larger than neutron star equatorial radius
                if not self.parameters['NICER__R_in'] > ref['radius']:
                    return -np.inf
              
                # inner disk must be smaller than corotation radius, otherwise we enter (weak) propeller regime
                if not self.parameters['IXPE__R_in'] < 1.49790e3*ref['mass']**(1/3)*ref['frequency']**(-2/3): # 1.49790e3 = (G*M_sol/4pi^2)^(1/3) in km
                    return -np.inf
        
                # inner disk must be larger than neutron star equatorial radius
                if not self.parameters['IXPE__R_in'] > ref['radius']:
                    return -np.inf

        elif not self.variable_params and not self.posterior_combiner:
            if 'disk' in  self.bkg:  
                # inner disk must be smaller than corotation radius, otherwise we enter (weak) propeller regime
                if not self.parameters['R_in'] < 1.49790e3*ref['mass']**(1/3)*ref['frequency']**(-2/3): # 1.49790e3 = (G*M_sol/4pi^2)^(1/3) in km
                   return -np.inf
        
                # inner disk must be larger than neutron star equatorial radius
                if not self.parameters['R_in'] > ref['radius']:
                   return -np.inf         
            ref = self.parameters # redefine shortcut
            
            if self.scenario == 'J1444_STU':
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
        _ = super(CustomPrior_twohotspots, self).inverse_sample(hypercube)

        ref = self.parameters # shortcut
    
        idx = ref.index('distance')
        temporary = truncnorm.ppf(hypercube[idx], -3.0, 3.0, loc=8.5, scale=2.)
        ref['distance'] = temporary

        if self.eos_informed:
            ref['mass'], ref['radius'] = self.nf_eos_mr_prior.sample_mr_from_nf()
            
            

        # flat priors in cosine of hot region centre colatitudes (isotropy)
        # support modified by no-overlap rejection condition
        if not self.posterior_combiner:
            if self.scenario == 'J1444_STS':
                idx = ref.index('NICER__p__super_colatitude')
                a, b = ref.get_param('NICER__p__super_colatitude').bounds
                a = math.cos(a); b = math.cos(b)
                ref['NICER__p__super_colatitude'] = math.acos(b + (a - b) * hypercube[idx])
                
                idx = ref.index('IXPE__p__super_colatitude')
                a, b = ref.get_param('IXPE__p__super_colatitude').bounds
                a = math.cos(a); b = math.cos(b)
                ref['IXPE__p__super_colatitude'] = math.acos(b + (a - b) * hypercube[idx])
            
            elif self.scenario == 'J1444_STU':
                idx = ref.index('p__super_colatitude')
                a, b = ref.get_param('p__super_colatitude').bounds
                a = math.cos(a); b = math.cos(b)
                ref['p__super_colatitude'] = math.acos(b + (a - b) * hypercube[idx])
        
                idx = ref.index('s__super_colatitude')
                a, b = ref.get_param('s__super_colatitude').bounds
                a = math.cos(a); b = math.cos(b)
                ref['s__super_colatitude'] = math.acos(b + (a - b) * hypercube[idx])
            else:
                idx = ref.index('p__super_colatitude')
                a, b = ref.get_param('p__super_colatitude').bounds
                a = math.cos(a); b = math.cos(b)
                ref['p__super_colatitude'] = math.acos(b + (a - b) * hypercube[idx])

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

        p += [gravradius(ref['mass']) / ref['radius']]
        p += [np.arccos(ref['cos_inclination'])*180/np.pi]


        if self.scenario=='J1444':
            p += [ref['p__super_tbb']*511]
            p += [ref['p__super_te']*511/1000]
            p += [ref['p__super_colatitude']*180/np.pi]
            p += [ref['p__super_radius']*180/np.pi]
    
            p += [ref['s__super_tbb']*511]
            p += [ref['s__super_te']*511/1000]
            p += [ref['s__super_colatitude']*180/np.pi]
            p += [ref['s__super_radius']*180/np.pi]
        elif self.scenario=='J1444_STS':
            p += [ref['NICER__p__super_tbb']*511]
            p += [ref['NICER__p__super_te']*511/1000]
            p += [ref['NICER__p__super_colatitude']*180/np.pi]
            p += [ref['NICER__p__super_radius']*180/np.pi]
            
            p += [ref['IXPE__p__super_tbb']*511]
            p += [ref['IXPE__p__super_te']*511/1000]
            p += [ref['IXPE__p__super_colatitude']*180/np.pi]
            p += [ref['IXPE__p__super_radius']*180/np.pi]
            

        return p
