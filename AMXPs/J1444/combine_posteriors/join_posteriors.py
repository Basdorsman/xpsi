#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 19:55:06 2025

@author: bas
"""

import os
import sys
this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory+'/../')

import numpy as np
from scipy.stats import gaussian_kde
import pymultinest
from posterior_combiner_STU import analysis

# [root].txt. Compatable with getdist with 2+nPar columns. Columns have sample 
# probability, -2*loglikehood, parameter values. Sample probability is the 
# sample prior mass multiplied by its likelihood & normalized by the evidence.

# [root]post_equal_weights.dat. Contains the equally weighted posterior 
# samples. Columns have parameter values followed by loglike value.

#0 [Gravitational mass [solar masses] = 1.400e+00,
#1  Coordinate equatorial radius [km] = 1.100e+01,
#2  Earth distance [kpc] = 8.000e+00,
#3  Cosine of Earth inclination to rotation axis = 2.740e-01,
#4  The phase of the hot region, a periodic parameter [cycles] = 0.000e+00,
#5  The colatitude of the centre of the superseding region [radians] = 1.760e-01,
#6  The angular radius of the (circular) superseding region [radians] = 5.236e-01,
#7  tbb = 2.500e-03,
#8  te = 1.000e+02,
#9  tau = 2.000e+00,
#10  The phase of the hot region, a periodic parameter [cycles] = 5.000e-01,
#11  The colatitude of the centre of the superseding region [radians] = 2.966e+00,
#12  The angular radius of the (circular) superseding region [radians] = 5.236e-01,
#13  tbb = 2.500e-03,
#14  te = 1.000e+02,
#15  tau = 2.000e+00,
#16  Temperature at inner disk radius in keV = 1.685e-01,
#17  Disk R_in in kilometers = 2.400e+01,
#18  Units of 10^21 cm^-2 = 2.900e+01]

params_NICER = [0,1,2,3,17,18]
post_NICER_eqw=np.loadtxt(this_directory+'/../data/J1444_STU_flatmr_lp1000/run_ST_post_equal_weights.dat')
post_NICER_eqw_shared = post_NICER_eqw[:,params_NICER].T
kde_post_NICER=gaussian_kde(post_NICER_eqw_shared)


#0 [mass, #grav mass
#1 radius, #coordinate equatorial radius
#2 distance, # earth distance kpc
#3 cos_i, #cosine of earth inclination
#4 chi0, #spin axis position angle
#5 phase_shift, #phase of hotregion
#6 super_colatitude, #colatitude of centre of superseding region
#7 super_radius,  #angular radius superceding region
#8 tbb,
#9 te,
#10 tau,
#11 phase_shift2,
#12 super_colatitude2,
#13 super_radius2,
#14 tbb2,
#15 te2,
#16 tau2,
#17 t_in,
#18 r_in
#19 p.append(du1_alpha)
#20 p.append(column_density)
#21 p.append(du2_alpha)
#22 p.append(du3_alpha)


params_IXPE = [0,1,2,3,18,20]
# post_IXPE_eqw=np.loadtxt(this_directory+'/run1_QU_DiskF_EOS_lp10k/run_rdata_QUpost_equal_weights.dat')
post_IXPE_eqw=np.loadtxt(this_directory+'/../data/run1_IQU/run_rdata_IQUpost_equal_weights.dat')
post_IXPE_eqw_shared = post_IXPE_eqw[:,params_IXPE].T
kde_post_IXPE=gaussian_kde(post_IXPE_eqw_shared)


# inverse sample from prior with e.g. 10^4 points to get a prior kde. I think there are no prior weights
ndraws='10000.0'
prior_draws = np.loadtxt(this_directory+f'/../data/flat_prior/prior_draws={ndraws}.txt')
prior_shared = prior_draws[:,params_NICER].T
kde_prior=gaussian_kde(prior_shared)
# kde_prior.logpdf(list(prior_shared[:,0])) #try out log probability 

# here I calculate that 97 percent of the samples are preserved with the flat M-R prior.
# i=0
# for mass,radius in zip(prior_draws[:,0],prior_draws[:,1]):    
#     if radius>8 and radius<14 and mass<2.2:
#         i+=1
# print(i)


def loglike_NICER(params):
    loglike=kde_post_NICER.logpdf(params)-kde_prior.logpdf(params)
    return loglike

def loglike_combined_without_prior_constraints(params_combined):
    #0 mass: Gravitational mass [solar masses].
    #1 radius: Coordinate equatorial radius [km].
    #2 distance: Earth distance [kpc].
    #3 cos_inclination: Cosine of Earth inclination to rotation axis.
    #4 NICER__R_in: Disk R_in in kilometers.
    #5 IXPE__R_in: Disk R_in in kilometers.
    #6 column_density: Units of 10^21 cm^-2.
    
    # select only NICER__R_in or IXPE__R_IN
    params_NICER = params_combined[[0,1,2,3,4,6]]
    params_IXPE = params_combined[[0,1,2,3,5,6]]
    
    loglike_NICER = kde_post_NICER.logpdf(params_NICER)-kde_prior.logpdf(params_NICER)   
    loglike_IXPE = kde_post_IXPE.logpdf(params_IXPE)-kde_prior.logpdf(params_IXPE)
    return loglike_NICER+loglike_IXPE


def loglike_combined(params):
    # unpack for clarity
    mass = params[0]
    radius = params[1]
    distance = params[2]
    cosi = params[3]
    Rin_NICER = params[4]
    Rin_IXPE  = params[5]
    NH = params[6]
    frequency=447.8715611

    # corotation radius [km]
    R_co = 1.49790e3 * mass**(1/3) * frequency**(-2/3)

    # ---- hard physical constraints ----
    if Rin_NICER <= radius:
        return -10**89
    if Rin_IXPE <= radius:
        return -10**89
    if Rin_NICER >= R_co:
        return -10**89
    if Rin_IXPE >= R_co:
        return -10**89
    if radius>=16:
        return -10**89
    
    #I cannot add the causality limit here so we need to watch out for small R and high M

    # ---- KDE likelihoods ----
    params_NICER = np.array([mass, radius, distance, cosi, Rin_NICER, NH])
    params_IXPE  = np.array([mass, radius, distance, cosi, Rin_IXPE,  NH])

    ll_NICER = kde_post_NICER.logpdf(params_NICER) - kde_prior.logpdf(params_NICER)
    ll_IXPE  = kde_post_IXPE.logpdf(params_IXPE)  - kde_prior.logpdf(params_IXPE)

    return ll_NICER + ll_IXPE

Analysis = analysis('test', 
                    'disk', 
                    sampler='multi', 
                    scenario='J1444_STU', 
                    eos_informed=False, 
                    channel_min=100,
                    posterior_combiner=True)
Analysis()


analysis_name = os.environ.get('LABEL')
if not isinstance(analysis_name, str):
        print('cannot import analysis name, using test_analysis')
        analysis_name = 'test_analysis'
print(f'analysis_name: {analysis_name}')

folderstring = f'{analysis_name}'

try:
    live_points = int(os.environ.get('live_points'))
except:
    print('live_points from environment variables failed, proceeding with default.')
    live_points = 1000
    pass
print(f'live_points: {live_points}')

try: 
    os.makedirs(folderstring)
except OSError:
    if not os.path.isdir(folderstring):
        raise

prior=Analysis.prior.inverse_sample
outputfiles_basename = f'./{folderstring}/run_'
runtime_params = {'resume': False,
                  'importance_nested_sampling': False,
                  'multimodal': False,
                  'n_clustering_params': None,
                  'outputfiles_basename': outputfiles_basename,
                  'n_iter_before_update': 100,
                  'n_live_points': live_points,
                  'sampling_efficiency': 0.1,
                  'const_efficiency_mode': False,
                  # 'wrapped_params': wrapped_params,
                  'evidence_tolerance': 0.5,
                  'seed': 7,
                  'verbose': True}

if __name__ == '__main__':
    _ = pymultinest.solve(LogLikelihood=loglike_combined, Prior=prior, n_dims=len(params_NICER)+1,
                          **runtime_params)
