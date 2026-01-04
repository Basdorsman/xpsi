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


# [root].txt. Compatable with getdist with 2+nPar columns. Columns have sample 
# probability, -2*loglikehood, parameter values. Sample probability is the 
# sample prior mass multiplied by its likelihood & normalized by the evidence.

# [root]post_equal_weights.dat. Contains the equally weighted posterior 
# samples. Columns have parameter values followed by loglike value.


######################### FILE NAMES ########################################
# NICER_path = '/../data/J1444_STU_flatmr_lp1000/run_ST_post_equal_weights.dat'
NICER_path = '/../data/NICER_runs/J1444_STU_lp2000/run_ST_post_equal_weights.dat'
IXPE_path = '/../data/IXPE_runs/run1_IQUf_Disk/run_rdata_IQUpost_equal_weights.dat'
IXPE_nd_path = '/../data/IXPE_runs/run1_IQUf_lp4k0/run_rdata_IQUpost_equal_weights.dat'
ndraws='10000.0'
prior_path = '/../data/prior_files/flat_prior_draws=1e4.txt'

############################ SETTINGS #######################################
mode = 'sample'
bkg = 'disk_NICER'
nh_shared=False

###########################  NICER POSTERIOR ###########################################
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

if nh_shared:
    params_NICER = [0,1,2,3,17,18]
elif not nh_shared:
    params_NICER = [0,1,2,3,17]
post_NICER_eqw=np.loadtxt(this_directory+NICER_path)
post_NICER_eqw_shared = post_NICER_eqw[:,params_NICER].T
kde_post_NICER=gaussian_kde(post_NICER_eqw_shared)

################################ IXPE POSTERIOR ###############################

### WITH DISK
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

if nh_shared:
    params_IXPE = [0,1,2,3,18,20]
elif not nh_shared:
    params_IXPE = [0,1,2,3,18]

post_IXPE_eqw=np.loadtxt(this_directory+IXPE_path)
# post_IXPE_eqw=np.loadtxt(this_directory+'/../data/run1_IQU/run_rdata_IQUpost_equal_weights.dat')
post_IXPE_eqw_shared = post_IXPE_eqw[:,params_IXPE].T
kde_post_IXPE=gaussian_kde(post_IXPE_eqw_shared)

##### NO DISK 
#0 ['mass',
#1  'radius',
#2  'distance',
#3  'cos_inclination',
#4  'spin_axis_position_angle',
#5  'p__phase_shift',
#6  'p__super_colatitude',
#7  'p__super_radius',
#8  'p__super_tbb',
#9  'p__super_te',
#10  'p__super_tau',
#11  's__phase_shift',
#12  's__super_colatitude',
#13  's__super_radius',
#14  's__super_tbb',
#15  's__super_te',
#16  's__super_tau',
#17  'du1__alpha',
#18  'column_density',
#19  'du2__alpha',
#20  'du3__alpha']

if nh_shared:
    params_IXPE_nd = [0,1,2,3,18]
elif not nh_shared:
    params_IXPE_nd = [0,1,2,3]


post_IXPE_nd_eqw=np.loadtxt(this_directory+IXPE_nd_path)
post_IXPE_nd_eqw_shared = post_IXPE_nd_eqw[:,params_IXPE_nd].T
kde_post_IXPE_nd=gaussian_kde(post_IXPE_nd_eqw_shared)

################################## PRIOR ######################################


# inverse sample from prior with e.g. 10^4 points to get a prior kde. I think there are no prior weights
prior_draws = np.loadtxt(this_directory+prior_path)
prior_disk = prior_draws[:,params_NICER].T
kde_prior_disk = gaussian_kde(prior_disk)

if nh_shared:
    params_NICER_nd = [0,1,2,3,18]
elif not nh_shared:
    params_NICER_nd = [0,1,2,3]
prior_nd = prior_draws[:,params_NICER_nd].T
kde_prior_nd = gaussian_kde(prior_nd)

####################### LOGLIKELIHOODS ################################
# def loglike_NICER(params):
#     loglike=kde_post_NICER.logpdf(params)-kde_prior.logpdf(params)
#     return loglike

# def loglike_combined_without_prior_constraints(params_combined):
#     #0 mass: Gravitational mass [solar masses].
#     #1 radius: Coordinate equatorial radius [km].
#     #2 distance: Earth distance [kpc].
#     #3 cos_inclination: Cosine of Earth inclination to rotation axis.
#     #4 NICER__R_in: Disk R_in in kilometers.
#     #5 IXPE__R_in: Disk R_in in kilometers.
#     #6 column_density: Units of 10^21 cm^-2.
    
#     # select only NICER__R_in or IXPE__R_IN
#     params_NICER = params_combined[[0,1,2,3,4,6]]
#     params_IXPE = params_combined[[0,1,2,3,5,6]]
    
#     loglike_NICER = kde_post_NICER.logpdf(params_NICER)-kde_prior.logpdf(params_NICER)   
#     loglike_IXPE = kde_post_IXPE.logpdf(params_IXPE)-kde_prior.logpdf(params_IXPE)
#     return loglike_NICER+loglike_IXPE

def Rin_constraint_breached(R_in, radius, mass, frequency):
    # corotation radius [km]
    R_co = 1.49790e3 * mass**(1/3) * frequency**(-2/3)
    # ---- hard physical constraints ----
    if R_in <= radius:
        return True
    elif R_in >= R_co:
        return True
    else:
        return False
    


def loglike_combined_IXPE_disk(params):
    # unpack for clarity
    mass = params[0]
    radius = params[1]
    distance = params[2]
    cosi = params[3]
    Rin_NICER = params[4]
    Rin_IXPE = params[5]
    NH = params[6]

    # I cannot add the causality limit here so we need to watch out for small R
    # and high M
    # if Rin_constraint_breached(Rin_NICER, radius, mass, frequency) or Rin_constraint_breached(Rin_IXPE, radius, mass, frequency):
    #     return-1e89

    # ---- KDE likelihoods ----
    params_NICER = np.array([mass, radius, distance, cosi, Rin_NICER, NH])
    params_IXPE  = np.array([mass, radius, distance, cosi, Rin_IXPE,  NH])

    ll_NICER = kde_post_NICER.logpdf(params_NICER) - kde_prior_disk.logpdf(params_NICER)
    ll_IXPE  = kde_post_IXPE.logpdf(params_IXPE)  - kde_prior_disk.logpdf(params_IXPE)
    return ll_NICER + ll_IXPE

def loglike_combined_IXPE_nd(params):
    # unpack for clarity
    mass = params[0]
    radius = params[1]
    distance = params[2]
    cosi = params[3]
    Rin_NICER = params[4]
    NH = params[5]
    frequency=447.8715611
    
    # I cannot add the causality limit here so we need to watch out for small R
    # and high M
    if Rin_constraint_breached(Rin_NICER, radius, mass, frequency):
        return -1e89

    # ---- KDE likelihoods ----
    params_NICER = np.array([mass, radius, distance, cosi, Rin_NICER, NH])
    params_IXPE_nd  = np.array([mass, radius, distance, cosi,  NH])

    ll_NICER = kde_post_NICER.logpdf(params_NICER) - kde_prior_disk.logpdf(params_NICER)
    ll_IXPE_nd  = kde_post_IXPE_nd.logpdf(params_IXPE_nd)  - kde_prior_nd.logpdf(params_IXPE_nd)
    return ll_NICER + ll_IXPE_nd

def loglike_combined_IXPE_ndnh(params):
    # unpack for clarity
    mass = params[0]
    radius = params[1]
    distance = params[2]
    cosi = params[3]
    Rin_NICER = params[4]
    # NH = params[5]
    frequency=447.8715611
    
    # I cannot add the causality limit here so we need to watch out for small R
    # and high M
    if Rin_constraint_breached(Rin_NICER, radius, mass, frequency):
        return -1e89

    # ---- KDE likelihoods ----
    params_NICER = np.array([mass, radius, distance, cosi, Rin_NICER])
    params_IXPE_nd  = np.array([mass, radius, distance, cosi])

    ll_NICER = kde_post_NICER.logpdf(params_NICER) - kde_prior_disk.logpdf(params_NICER)
    ll_IXPE_nd  = kde_post_IXPE_nd.logpdf(params_IXPE_nd)  - kde_prior_nd.logpdf(params_IXPE_nd)
    return ll_NICER + ll_IXPE_nd


from combine_kdes_STU import analysis
Analysis = analysis('test', 
                    bkg, 
                    sampler='multi', 
                    scenario='J1444_STU', 
                    eos_informed=False, 
                    channel_min=100,
                    combine_kdes=True,
                    nh_shared=nh_shared)
Analysis()



if __name__ == '__main__':
    if mode == 'sample':
    
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
                          'max_iter': -1,
                          'verbose': True}
        
        if nh_shared:
            if bkg == 'disk':
                _ = pymultinest.solve(LogLikelihood=loglike_combined_IXPE_disk,
                                      Prior=prior, 
                                      n_dims=len(params_NICER)+1,
                                      **runtime_params)
        
            elif bkg == 'disk_NICER':
                _ = pymultinest.solve(LogLikelihood=loglike_combined_IXPE_nd, 
                                      Prior=prior, 
                                      n_dims=len(params_NICER),
                                      **runtime_params)
            
        elif not nh_shared:
            _ = pymultinest.solve(LogLikelihood=loglike_combined_IXPE_ndnh, 
                                  Prior=prior, 
                                  n_dims=len(params_NICER),
                                  **runtime_params)
    if mode=='draw_prior_samples':
        ndraws=1e4
        prior_samples = Analysis.prior.draw(ndraws=int(ndraws))[0]


