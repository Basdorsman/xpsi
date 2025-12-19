#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 15:49:38 2025

@author: bas
"""
import os
import sys
this_directory = os.path.dirname(os.path.abspath(__file__))
import numpy as np

# [root].txt. Compatable with getdist with 2+nPar columns. Columns have sample 
# probability, -2*loglikehood, parameter values. Sample probability is the 
# sample prior mass multiplied by its likelihood & normalized by the evidence.

NICER_samples = np.loadtxt('/home/bas/Documents/Projects/x-psi/xpsi-bas-fork/AMXPs/outputs/J1444_STU_lp1000/17178314/J1444_STU_lp1000/run_ST_.txt')
#0 prob
#1 -2 loglike    
#2 [Gravitational mass [solar masses] = 1.400e+00,
#3  Coordinate equatorial radius [km] = 1.100e+01,
#4  Earth distance [kpc] = 8.000e+00,
#5  Cosine of Earth inclination to rotation axis = 2.740e-01,

#6  The phase of the hot region, a periodic parameter [cycles] = 0.000e+00,
#7  The colatitude of the centre of the superseding region [radians] = 1.760e-01,
#8  The angular radius of the (circular) superseding region [radians] = 5.236e-01,
#9  tbb = 2.500e-03,
#10  te = 1.000e+02,
#11  tau = 2.000e+00,
#12  The phase of the hot region, a periodic parameter [cycles] = 5.000e-01,
#13  The colatitude of the centre of the superseding region [radians] = 2.966e+00,
#14  The angular radius of the (circular) superseding region [radians] = 5.236e-01,
#15  tbb = 2.500e-03,
#16  te = 1.000e+02,
#17  tau = 2.000e+00,
#18  Temperature at inner disk radius in keV = 1.685e-01,
#19  Disk R_in in kilometers = 2.400e+01,
#20  Units of 10^21 cm^-2 = 2.900e+01]


comb_samples = np.loadtxt(this_directory+'/../../outputs/combine_posteriors_lp1000/17640043/combine_posteriors_lp1000/run_.txt')
#0 prob
#1 -2 loglike    
#2 mass: Gravitational mass [solar masses].
#3 radius: Coordinate equatorial radius [km].
#4 distance: Earth distance [kpc].
#5 cos_inclination: Cosine of Earth inclination to rotation axis.
#6 NICER__R_in: Disk R_in in kilometers.
#7 IXPE__R_in: Disk R_in in kilometers.
#8 column_density: Units of 10^21 cm^-2.



comb_samples_padded = np.zeros((comb_samples.shape[0],NICER_samples.shape[1]))
comb_samples_padded[:,[0,1,2,3,4,5]]=comb_samples[:,[0,1,2,3,4,5]]
comb_samples_padded[:,[20]]=comb_samples[:,[8]]

np.savetxt(this_directory+'/run_IQU_.txt', comb_samples_padded)
