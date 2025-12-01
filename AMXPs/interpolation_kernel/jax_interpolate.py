#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 11:56:51 2025

@author: bas
"""


import jax.numpy as jnp
import numpy as np
from interpax import interp1d, interp2d
import xpsi
from time import time




system='local'
n_repeats=100
method = 'cubic'

#%% interpax test

xp = jnp.linspace(0, 2 * np.pi, 100)
xq = jnp.linspace(0, 2 * np.pi, 3200000)
f = lambda x: jnp.sin(x)
fp = f(xp)




start_1d = time()
for i in range(n_repeats):
    fq = interp1d(xq, xp, fp, method=method)
    #Iq_jax = interp2d(Eq_prime, Muq, Ep_prime, Mup, Ip, method=method)
time_1d = time()-start_1d
print(f'jax timing, repeats n={n_repeats}, t_average={time_1d/n_repeats:.6f}s')

# np.testing.assert_allclose(fq, f(xq), rtol=1e-6, atol=1e-5)



#%% load 2D and 5D atmosphere

def preload_atmosphere_A5(path):
    """ A photosphere extension to preload the numerical atmosphere NSX. """

    with np.load(path, allow_pickle=True) as data_dictionary:
        NSX = data_dictionary['NSX.npy']
        size_reorderme = data_dictionary['size.npy']
        # print(size_reorderme)
    
    #size = (150, 9, 31, 11, 41)
    size = [size_reorderme[3], size_reorderme[4], size_reorderme[2], size_reorderme[1], size_reorderme[0]]

    Energy = np.ascontiguousarray(NSX[0:size[0],0])
    cos_zenith = np.ascontiguousarray([NSX[i*size[0],1] for i in range(size[1])])
    tau = np.ascontiguousarray([NSX[i*size[0]*size[1],2] for i in range(size[2])])
    t_bb = np.ascontiguousarray([NSX[i*size[0]*size[1]*size[2],3] for i in range(size[3])])
    t_e = np.ascontiguousarray([NSX[i*size[0]*size[1]*size[2]*size[3],4] for i in range(size[4])])
    intensities = np.ascontiguousarray(NSX[:,5])

    atmosphere = (t_e, t_bb, tau, cos_zenith, Energy, intensities)
    return atmosphere

if system=='snellius':
    root = '/home/dorsman/xpsi-bas-fork/AMXPs/model_data/'
elif system=='local':
    root = '/home/bas/Documents/Projects/x-psi/model_datas/bobrikova/'


atmosphere = preload_atmosphere_A5(root+'Bobrikova_compton_slab.npz')

te = 101 #40 - 200 te[150*9*31*11*15]
tbb = 0.0015 #0.001 - 0.0031 tbb[150*9*31*2 (or 3)]
tau = 1.01  #0.5 - 3.55 te[150*9*5]
local_vars = np.asarray([[te, tbb, tau]])

atmosphere_2D = xpsi.surface_radiation_field.produce_atmosphere_2D(local_vars,
                                                                atmosphere=atmosphere,
                                                                region_extension='hot',
                                                                atmos_extension = 'Num5D',
                                                                numTHREADS=1)





#%% setting up vectors for atmosphere interpolation

# ENERGY AND MU VECTOR ARE MADE ANALYTICALLY LIKE THIS
x_l, x_u = -3.7, .3 # lower and upper bounds of the log_10 energy span
NEnergy = 150 # 50# 101 # number of energy points (x)
IntEnergy = np.logspace(x_l,x_u,NEnergy), np.log(1e1)*(x_u-x_l)/(NEnergy-1.) # sample points and weights for integrations over the spectrum computing sorce function
E_vector,x_weight=IntEnergy
E_prime = np.log10(E_vector)

from numpy.polynomial.legendre import leggauss

def init_mu(n = 3):
        NMu = n # number of propagation zenith angle cosines (\mu) [0,1]
        NZenith = 2*NMu # number of propagation zenith angles (z) [0,pi]
        mu = np.empty(NZenith)
        #mu = Array{Float64}(undef,NZenith)
        #mu_weight = Array{Float64}(undef,NZenith)
        m2,mw = leggauss(NMu)
        mu[:NMu] = (m2 - 1.)/2
        mu[NMu:NZenith] = (m2 + 1.)/2
        
        #mu_weight[1:NMu] = (mw)./2
        #mu_weight[NMu+1:2NMu] = (mw)./2
        #global μ_grid = n, 2n, mu, mu_weight
        
        return mu[NMu:NZenith]

mu_vector = init_mu(9)


size = 60000  # Number of interpolation points

# grid_x_test = torch.rand(size, device=device) * 2 - 1  # Random values between -1 and 1
# grid_y_test = torch.rand(size, device=device) * 2 - 1  # Random values between -1 and 1


def random_with_bounds(lower_limit, upper_limit, size):
    return (upper_limit - lower_limit) * np.random.random(size = size) + lower_limit


E_prime_random = random_with_bounds(x_l, x_u, size)
E_random = 10 ** E_prime_random

Mu_random = random_with_bounds(min(mu_vector), max(mu_vector), size)







#%% test jax interpolate
Mup = jnp.asarray(atmosphere_2D[0])
Ep = jnp.asarray(atmosphere_2D[1])
Ep_prime = jnp.asarray(np.log10(atmosphere_2D[1]))
Ip = jnp.asarray(atmosphere_2D[2].reshape(len(Ep),len(Mup), order='F'))

Muq = jnp.asarray(Mu_random)
Eq = jnp.asarray(E_random)
Eq_prime = jnp.asarray(np.log10(E_random))



start_split = time()
for i in range(n_repeats):
    Iq_jax = interp2d(Eq, Muq, Ep, Mup, Ip, method=method)
    #Iq_jax = interp2d(Eq_prime, Muq, Ep_prime, Mup, Ip, method=method)
time_split = time()-start_split
print(f'jax timing, repeats n={n_repeats}, t_average={time_split/n_repeats:.6f}s')


#%% interpolate with split

# setting up contiguous arrays for split interpolation

nT=1


E_contiguous = np.ascontiguousarray(E_random, dtype = np.double)
mu_contiguous = np.ascontiguousarray(Mu_random, dtype = np.double)


Iq_split = np.empty(size)

start_split = time()
for i in range(n_repeats):
    Iq_split = xpsi.surface_radiation_field.interpolate_2D(E_contiguous, 
                                                          mu_contiguous, 
                                                          atmosphere_2D=atmosphere_2D,
                                                          numTHREADS=nT)
time_split = time()-start_split
print(f'split timing, repeats n={n_repeats}, t_average={time_split/n_repeats:.6f}s')


#%% relative error plot
import matplotlib.pyplot as plt

err_rel = (Iq_split - Iq_jax)/Iq_split
err_abs = Iq_split - Iq_jax

fig, axes = plt.subplots(4, figsize=(5,10))
# axes[0].hist(err_rel, bins=100)


# #Select values from Iq_split where relative error is below -0.05
# Iq_split_off = Iq_split[err_rel < -0.05]


# axes[1].plot(Iq_split, err_rel, 'x')


# axes[2].semilogy(E_contiguous, Iq_jax, 'x')

# Histogram of relative error
axes[0].hist(err_rel, bins=500)
axes[0].set_title(f"Histogram of Relative Errors. method={method}")
axes[0].set_xlabel("Relative Error")
axes[0].set_ylabel("Frequency")
axes[0].set_xlim([-0.01,0.01])

# Scatter plot of Iq_split vs. relative error
axes[1].plot(Iq_split, err_rel, 'x')
axes[1].set_title("Relative Error vs. Intensity")
axes[1].set_xlabel("Intensity (split)")
axes[1].set_ylabel("Relative Error")

# Select values from Iq_split where relative error is below -0.05
Iq_split_off = Iq_split[err_rel < -0.05]

# Semilog plot of Iq_jax vs. E_contiguous
axes[2].semilogy(E_contiguous, err_rel, 'x')
axes[2].set_title("Semilog Plot of Iq_jax vs. E_contiguous")
axes[2].set_xlabel("E")
axes[2].set_ylabel("Relative error")


# Semilog plot of Iq_jax vs. E_contiguous
axes[3].semilogy(Mu_random, err_rel, 'x')
axes[3].set_title("Semilog Plot of Iq_jax vs. E_contiguous")
axes[3].set_xlabel("mu_equidistant")
axes[3].set_ylabel("Relative error")

fig.tight_layout()

print(method)
print('np.std(err_rel)',np.std(err_rel))
print('np.std(err_abs)',np.std(err_abs))