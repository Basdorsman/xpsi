#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:02:00 2025

@author: bas
"""

import torch
import torch.nn.functional as F
from time import time
from timeit import timeit
import numpy as np
import xpsi

np.random.seed(42)

equidistant=True
system='local'
interpolation_mode='bilinear'

# Move tensors to GPU if available
device_preference = 'cuda'
device_choice = device_preference if torch.cuda.is_available() else "cpu"
print(device_choice)
device = torch.device(device_choice)

#%% interpolations with a test grid

# # Create a small 2D tensor (1x1x3x3 for a single-channel 3x3 image)
# input_tensor = torch.tensor([[[[1.0, 2.0, 3.0],
#                                [4.0, 5.0, 6.0],
#                                [7.0, 8.0, 9.0]]]], device=device)  # Shape: (1,1,3,3)

# Define a large number of grid coordinates for interpolation (normalized from -1 to 1)
# N = 60000  # Number of interpolation points
# grid_x = torch.rand(N, device=device) * 2 - 1  # Random values between -1 and 1
# grid_y = torch.rand(N, device=device) * 2 - 1  # Random values between -1 and 1
# grid = torch.stack((grid_x, grid_y), dim=-1).view(1, N, 1, 2)  # Shape: (1, N, 1, 2)

# # Measure execution time
# start_time = time.time()

# # Use grid_sample to interpolate at multiple points
# output_tensor = F.grid_sample(input_tensor, grid, mode='bilinear', align_corners=True)

# # Extract interpolated values
# interpolated_values = output_tensor[0, 0, :, 0].tolist()

# # End timing
# end_time = time.time()
# elapsed_time = end_time - start_time

# print("Input Tensor:")
# print(input_tensor.squeeze().cpu())  # Remove batch and channel dims for readability

# print("\nInterpolated Values at first 10 points:", interpolated_values[:10])  # Print only first 10 values for readability
# print(f"\nExecution Time: {elapsed_time:.6f} seconds")


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

mu_equidistant_vector = np.linspace(0.01, 0.98, 9)

fake_I = np.empty(len(mu_equidistant_vector)*len(E_vector))
for i,mu in enumerate(mu_equidistant_vector):
    for j,E in enumerate(E_vector):
        #fake_I[i*len(E_vector)+j] = mu*E**2
        fake_I[i * len(E_vector) + j] = mu * E**2 * (1 + 0.2 * np.sin(5 * E))


fake_I = np.tile(fake_I, 13981)

#%% load 5D atmosphere

def preload_atmosphere_A5(path, energy=None, mu=None, fake_I=None):
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
    
    if not energy is None:
        Energy = np.ascontiguousarray(energy)
    if not mu is None:
        cos_zenith = np.ascontiguousarray(mu)
    if not fake_I is None:
        intensities = np.ascontiguousarray(fake_I)

    atmosphere = (t_e, t_bb, tau, cos_zenith, Energy, intensities)
    return atmosphere

if system=='snellius':
    root = '/home/dorsman/xpsi-bas-fork/AMXPs/model_data/'
elif system=='local':
    root = '/home/bas/Documents/Projects/x-psi/model_datas/bobrikova/'

if equidistant:
    atmosphere = preload_atmosphere_A5(root+'Bobrikova_compton_slab.npz', energy=E_prime, mu=mu_equidistant_vector, fake_I=fake_I)
else:
    atmosphere = preload_atmosphere_A5(root+'Bobrikova_compton_slab.npz', energy=E_prime)



#%% produce E and mu vectors for Torch

def normalize_coordinates(coords: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Normalize coordinates to the range [-1, 1].

    Parameters:
    - coords (np.ndarray): Input array of coordinates.
    - min_val (float): Minimum value in the original range.
    - max_val (float): Maximum value in the original range.

    Returns:
    - np.ndarray: Normalized coordinates in the range [-1, 1].
    """
    return 2 * (coords - min_val) / (max_val - min_val) - 1

E_norm = normalize_coordinates(E_vector, min(E_vector), max(E_vector))
E_prime_norm = normalize_coordinates(E_prime, min(E_prime), max(E_prime))
mu_norm = normalize_coordinates(mu_vector, min(mu_vector), max(mu_vector))
mu_equidistant_norm = normalize_coordinates(mu_equidistant_vector, min(mu_equidistant_vector), max(mu_equidistant_vector))

#%% produce random E and mu points
size = 60000  # Number of interpolation points

# grid_x_test = torch.rand(size, device=device) * 2 - 1  # Random values between -1 and 1
# grid_y_test = torch.rand(size, device=device) * 2 - 1  # Random values between -1 and 1


def random_with_bounds(lower_limit, upper_limit, size):
    return (upper_limit - lower_limit) * np.random.random(size = size) + lower_limit


E_prime_random = random_with_bounds(x_l, x_u, size)
E_random = 10 ** E_prime_random
mu_random = random_with_bounds(min(mu_vector), max(mu_vector), size)
mu_equidistant_random = random_with_bounds(min(mu_equidistant_vector), max(mu_equidistant_vector), size)


grid_x = torch.tensor(normalize_coordinates(mu_random, min(mu_vector), max(mu_vector)), device=device)
grid_x_equidistant = torch.tensor(normalize_coordinates(mu_equidistant_random, min(mu_equidistant_vector), max(mu_equidistant_vector)), device=device)

grid_y = torch.tensor(normalize_coordinates(E_random, min(E_vector), max(E_vector)), device=device) 
grid_y_prime = torch.tensor(normalize_coordinates(E_prime_random, min(E_prime), max(E_prime)), device=device) #I am using here the exponent because it is spaced linearly. I want my grid points to be spaced linearly for accuracy in torch grid sample.

grid = torch.stack((grid_y_prime, grid_x), dim=-1).view(1, size, 1, 2)  # Shape: (1, N, 1, 2)
grid_equidistant = torch.stack((grid_y_prime, grid_x_equidistant), dim=-1).view(1, size, 1, 2)  # Shape: (1, N, 1, 2)


#%% Interpolation with Torch

te = 101 #40 - 200 te[150*9*31*11*15]
tbb = 0.0015 #0.001 - 0.0031 tbb[150*9*31*2 (or 3)]
tau = 1.01  #0.5 - 3.55 te[150*9*5]
local_vars = np.asarray([[te, tbb, tau]])

n_repeats=100
t__e = np.arange(40.0, 202.0, 4.0) #actual range is 40-200 imaginaty units, ~20-100 keV (Te(keV)*1000/511keV is here)
t__bb = np.arange(0.001, 0.0031, 0.0002) #this one is non-physical, we went for way_to_low Tbbs here, I will most probably delete results from too small Tbbs. This is Tbb(keV)/511keV, so these correspond to 0.07 - 1.5 keV, but our calculations don't work correctly for Tbb<<0.5 keV
tau__t = np.arange(0.5, 3.55, 0.1) 
te_random = random_with_bounds(min(t__e), max(t__e), n_repeats)
tbb_random = random_with_bounds(min(t__bb), max(t__bb), n_repeats)
tau_random = random_with_bounds(min(tau__t), max(tau__t), n_repeats)
random_local_vars = np.asarray([te_random, tbb_random, tau_random])


# def interp_torch(local_vars, grid, mode):

#     atmosphere_2D = xpsi.surface_radiation_field.produce_atmosphere_2D(local_vars,
#                                                                     atmosphere=atmosphere,
#                                                                     region_extension='hot',
#                                                                     atmos_extension = 'Num5D',
#                                                                     numTHREADS=1)


#     intensities_vector = atmosphere_2D[2]

#     I_tensor = torch.tensor(intensities_vector, dtype=torch.float64, device=device).view(1,1,len(mu_norm), len(E_norm))
#     output_tensor = F.grid_sample(I_tensor, grid, mode=mode, align_corners=True)
#     intensity_t = np.asarray(output_tensor[0, 0, :, 0].cpu())
#     return intensity_t


# intensity_t = interp_torch(local_vars, grid_equidistant, interpolation_mode)

# start_torch = time()
# for i in range(n_repeats): 
#     random_local_var = np.asarray([random_local_vars[:,i]])
#     intensity_t = interp_torch(random_local_var, grid_equidistant, interpolation_mode)
# time_torch = time()-start_torch

# print(f'torch timing, repeats n={n_repeats}, t/n={time_torch/n_repeats:.6f}s')

def interp_torch(local_vars, grid, mode, n_repeats):
    atmosphere_time = 0.0
    grid_sample_time = 0.0

    for _ in range(n_repeats):
        random_local_var = np.asarray([random_local_vars[:, _]])

        # Time atmosphere production
        start_atmosphere = time()
        atmosphere_2D = xpsi.surface_radiation_field.produce_atmosphere_2D(
            random_local_var,
            atmosphere=atmosphere,
            region_extension='hot',
            atmos_extension='Num5D',
            numTHREADS=1
        )
        atmosphere_time += time() - start_atmosphere

        intensities_vector = atmosphere_2D[2]

        # Time grid sampling
        start_grid = time()
        I_tensor = torch.tensor(intensities_vector, dtype=torch.float64, device=device).view(1, 1, len(mu_norm), len(E_norm))
        output_tensor = F.grid_sample(I_tensor, grid, mode=mode, align_corners=True)
        grid_sample_time += time() - start_grid

        intensity_t = np.asarray(output_tensor[0, 0, :, 0].cpu())

    # Print timing results
    print(f'Atmosphere production timing, repeats n={n_repeats}, t/n={atmosphere_time/n_repeats:.6f}s')
    print(f'Grid sampling timing, repeats n={n_repeats}, t/n={grid_sample_time/n_repeats:.6f}s')
    print(f'Total execution timing, repeats n={n_repeats}, t/n={(atmosphere_time + grid_sample_time)/n_repeats:.6f}s')


    return intensity_t

# Run the function with the desired number of repeats
intensity_t = interp_torch(local_vars, grid_equidistant, interpolation_mode, n_repeats)


#%% setting up contiguous arrays for split interpolation

nT=1

te_constant = np.ascontiguousarray(te*np.ones(size), dtype = np.double)
tbb_constant = np.ascontiguousarray(tbb*np.ones(size), dtype = np.double)
tau_constant = np.ascontiguousarray(tau*np.ones(size), dtype = np.double,)

constant_local_vars = np.ascontiguousarray([te_constant, tbb_constant, tau_constant])
local_variables = np.ascontiguousarray(constant_local_vars.T)


E_contiguous = np.ascontiguousarray(E_random, dtype = np.double)
E_prime_contiguous = np.ascontiguousarray(E_prime_random, dtype = np.double)

mu_contiguous = np.ascontiguousarray(mu_random, dtype = np.double)
mu_equidistant_contiguous = np.ascontiguousarray(mu_equidistant_random, dtype = np.double)




#%% interpolate with split
intensity_s = np.empty(size)
intensity_s = xpsi.surface_radiation_field.intensity_split_interpolation(E_prime_contiguous, mu_equidistant_contiguous, local_variables,
                                                        atmosphere=atmosphere,
                                                        region_extension='hot',
                                                        atmos_extension ='Num5D',
                                                        numTHREADS=nT)
start_split = time()
for i in range(n_repeats):
    random_local_var = np.asarray([random_local_vars[:,i]])
    intensity_s = xpsi.surface_radiation_field.intensity_split_interpolation(E_prime_contiguous, mu_equidistant_contiguous, random_local_var,
                                                            atmosphere=atmosphere,
                                                            region_extension='hot',
                                                            atmos_extension ='Num5D',
                                                            numTHREADS=nT)
time_split = time()-start_split
print(f'split timing, repeats n={n_repeats}, t/n={time_split/n_repeats:.6f}s')


#%% relative error plot
import matplotlib.pyplot as plt

err_rel = (intensity_s - intensity_t)/intensity_s
err_abs = intensity_s - intensity_t

fig, axes = plt.subplots(4, figsize=(5,10))
# axes[0].hist(err_rel, bins=100)


# #Select values from intensity_s where relative error is below -0.05
# intensity_s_off = intensity_s[err_rel < -0.05]


# axes[1].plot(intensity_s, err_rel, 'x')


# axes[2].semilogy(E_contiguous, intensity_t, 'x')

# Histogram of relative error
axes[0].hist(err_rel, bins=100)
axes[0].set_title("Histogram of Relative Errors")
axes[0].set_xlabel("Relative Error")
axes[0].set_ylabel("Frequency")

# Scatter plot of intensity_s vs. relative error
axes[1].plot(intensity_s, err_rel, 'x')
axes[1].set_title("Relative Error vs. Intensity")
axes[1].set_xlabel("Intensity (split)")
axes[1].set_ylabel("Relative Error")

# Select values from intensity_s where relative error is below -0.05
intensity_s_off = intensity_s[err_rel < -0.05]

# Semilog plot of intensity_t vs. E_contiguous
axes[2].semilogy(E_contiguous, err_rel, 'x')
axes[2].set_title("Semilog Plot of Intensity_t vs. E_contiguous")
axes[2].set_xlabel("E")
axes[2].set_ylabel("Relative error")


# Semilog plot of intensity_t vs. E_contiguous
axes[3].semilogy(mu_equidistant_random, err_rel, 'x')
axes[3].set_title("Semilog Plot of Intensity_t vs. E_contiguous")
axes[3].set_xlabel("mu_equidistant")
axes[3].set_ylabel("Relative error")

fig.tight_layout()

#%% accuracy comparison



E_single = np.asarray([0.1]) # Energy[101]
E_single_prime = np.log10(E_single)

mu_single = mu_single_equidistant = np.asarray([0.5]) # cos(emission angle) mu[150*4]


mu_single_norm = normalize_coordinates(mu_single, min(mu_vector), max(mu_vector))
mu_single_norm_equidistant = normalize_coordinates(mu_single_equidistant, min(mu_equidistant_vector), max(mu_equidistant_vector))

E_single_norm = normalize_coordinates(E_single, min(E_vector), max(E_vector))
E_single_norm_prime = normalize_coordinates(E_single_prime, min(E_prime), max(E_prime))

# TORCH

te = 101 #40 - 200 te[150*9*31*11*15]
tbb = 0.0015 #0.001 - 0.0031 tbb[150*9*31*2 (or 3)]
tau = 1.01  #0.5 - 3.55 te[150*9*5]
local_vars = np.asarray([[te, tbb, tau]])
atmosphere_2D = xpsi.surface_radiation_field.produce_atmosphere_2D(local_vars,
                                                                atmosphere=atmosphere,
                                                                region_extension='hot',
                                                                atmos_extension = 'Num5D',
                                                                numTHREADS=1)


intensities_vector = atmosphere_2D[2]
I_tensor = torch.tensor(intensities_vector, dtype=torch.float64, device=device).view(1,1,len(mu_norm), len(E_norm))
single_point_grid = torch.tensor([[[[E_single_norm_prime[0],mu_single_norm_equidistant[0]]]]])  # Shape (1,1,1,2)
output_tensor = F.grid_sample(I_tensor, single_point_grid, mode=interpolation_mode, align_corners=True)
print('accuracy comparison')
print(output_tensor.squeeze().cpu().item())

#LAGRANGE
intensity_single = xpsi.surface_radiation_field.intensity_split_interpolation(E_single_prime, mu_single_equidistant, local_variables,
                                                        atmosphere=atmosphere,
                                                        region_extension='hot',
                                                        atmos_extension ='Num5D',
                                                        numTHREADS=nT)

print(intensity_single[0])


#%%% printing the values in the equidistant vector here

# print([mu_vector[i+1]-mu_vector[i] for i in range(len(mu_vector)-1)])
# print([mu_equidistant_vector[i+1]-mu_equidistant_vector[i] for i in range(len(mu_equidistant_vector)-1)])
# print([E_prime[i+1]-E_prime[i] for i in range(len(E_prime)-1)])



