import numpy as np
import xpsi

# machine = 'local' #'local', 'helios'


# import os
# import sys
# if machine == 'local':
#     sys.path.append('/home/bas/Documents/Projects/x-psi/xpsi-bas-fork/tests/')
    
# def preload_atmosphere_A4(path):
#     with np.load(path, allow_pickle=True) as data_dictionary:
#         NSX = data_dictionary['NSX.npy']
#         size_reorderme = data_dictionary['size.npy']
#         # size = (150, 9, 31, 11)
#         size = [size_reorderme[3], size_reorderme[4], size_reorderme[2], size_reorderme[1]]
    
#     te_index = 0 # I believe this can be an integer up to 40. Given that t__e = np.arange(40.0, 202.0, 4.0), there are 40.5 values (I expect that means 40). 
#     Energy = np.ascontiguousarray(NSX[0:size[0],0])
#     cos_zenith = np.ascontiguousarray([NSX[i*size[0],1] for i in range(size[1])])
#     tau = np.ascontiguousarray([NSX[i*size[0]*size[1],2] for i in range(size[2])])
#     t_bb = np.ascontiguousarray([NSX[i*size[0]*size[1]*size[2],3] for i in range(size[3])])
    
#     te_step_size = size[0]*size[1]*size[2]*size[3]

#     intensities = np.ascontiguousarray(NSX[te_step_size*te_index:te_step_size*(te_index+1),5]) #change the value of te here, it is now 40
#     atmosphere = (t_bb, tau, cos_zenith, Energy, intensities)  
#     return atmosphere

# def preload_atmosphere_A5(path):
#     """ A photosphere extension to preload the numerical atmosphere NSX. """

#     with np.load(path, allow_pickle=True) as data_dictionary:
#         NSX = data_dictionary['NSX.npy']
#         size_reorderme = data_dictionary['size.npy']
#         print(size_reorderme)
    
#     #size = (150, 9, 31, 11, 41)
#     size = [size_reorderme[3], size_reorderme[4], size_reorderme[2], size_reorderme[1], size_reorderme[0]]

#     Energy = np.ascontiguousarray(NSX[0:size[0],0])
#     cos_zenith = np.ascontiguousarray([NSX[i*size[0],1] for i in range(size[1])])
#     tau = np.ascontiguousarray([NSX[i*size[0]*size[1],2] for i in range(size[2])])
#     t_bb = np.ascontiguousarray([NSX[i*size[0]*size[1]*size[2],3] for i in range(size[3])])
#     t_e = np.ascontiguousarray([NSX[i*size[0]*size[1]*size[2]*size[3],4] for i in range(size[4])])
#     intensities = np.ascontiguousarray(NSX[:,5])

#     atmosphere = (t_e, t_bb, tau, cos_zenith, Energy, intensities)
#     return atmosphere


# # try to get parameters from shell input
# atmosphere_type = os.environ.get('atmosphere_type')
# n_params = os.environ.get('n_params')

# if isinstance(os.environ.get('atmosphere_type'),type(None)) or isinstance(os.environ.get('n_params'),type(None)): # if that fails input them here.
#     print('E: failed to import OS environment variables, using defaults.')    
#     atmosphere_type = 'A' #A, N, B
#     n_params = '4' #4, 5

# if atmosphere_type == 'A': atmosphere = 'accreting'
# elif atmosphere_type == 'N': atmosphere = 'numerical'
# elif atmosphere_type == 'B': atmosphere = 'blackbody'

# print('atmosphere:', atmosphere)
# print('n_params:', n_params)

# if atmosphere_type == 'A':

#     # photon parameters
#     E = np.asarray([0.1]) # Energy, need to look up units still
#     mu = np.asarray([0.5]) # cos(emission angle)

#     if n_params == '4':
#         if machine == 'local':
#             atmosphere = preload_atmosphere_A4('/home/bas/Documents/Projects/x-psi/model_datas/bobrikova/Bobrikova_compton_slab.npz')
#         elif machine == 'helios':
#             atmosphere = preload_atmosphere_A4('model_data/Bobrikova_compton_slab.npz')
            

#         # atmosphere parameters
#         #te = 101 #40 - 200
#         tbb = 0.0015 #0.001 - 0.0031
#         tau = 1.01  #0.5 - 3.55
#         local_vars = np.asarray([[tbb, tau]])

#     if n_params == '5':
#         if machine == 'local':
#             atmosphere = preload_atmosphere_A5('/home/bas/Documents/Projects/x-psi/model_datas/bobrikova/Bobrikova_compton_slab.npz')
#         elif machine == 'helios':
#             atmosphere = preload_atmosphere_A5('model_data/Bobrikova_compton_slab.npz')
            

#         # atmosphere parameters
#         te = 101 #40 - 200
#         tbb = 0.0015 #0.001 - 0.0031
#         tau = 1.01  #0.5 - 3.55
#         local_vars = np.asarray([[te, tbb, tau]])


#%%


#import:
from numpy import linspace, logspace, empty, zeros, ones, array, fromfile
#import matplotlib
#matplotlib.use('agg')
from numpy import pi, exp, log, sqrt, sin, cos, arccos, arctan2
from numpy import absolute, sign, floor, ceil, argmin
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import interp1d#,CubicSpline 
from scipy.interpolate import CubicSpline 
from scipy.interpolate import interp2d
from scipy.special import kn
#from matplotlib.pyplot import *
from bisect import bisect
import numpy as np
from astropy.io import fits
from numpy import ones, zeros
from scipy import interpolate

t__e = np.arange(40.0, 202.0, 4.0) #actual range is 40-200 imaginaty units, ~20-100 keV (Te(keV)*1000/511keV is here)
t__bb = np.arange(0.001, 0.0031, 0.0002) #this one is non-physical, we went for way_to_low Tbbs here, I will most probably delete results from too small Tbbs. This is Tbb(keV)/511keV, so these correspond to 0.07 - 1.5 keV, but our calculations don't work correctly for Tbb<<0.5 keV
tau__t = np.arange(0.5, 3.55, 0.1) 

NEnergy_i = 150
NZenith_i = 9

I_mighty = ones((len(t__e), len(t__bb), len(tau__t), NEnergy_i, NZenith_i)) #5D Intensity tensor[Te,Tbb,tau_t, Energy, Mu]
#Q_mighty = ones((len(t__e), len(t__bb), len(tau__t), NEnergy_i, NZenith_i)) #5D Q-parameter tensor[Te,Tbb,tau_t, Energy, Mu]

def reading_table(): #routine that just reads fits tables to I_mighty and Q_mighty
    path='/home/bas/Documents/Projects/x-psi/pulse_profiling_bobrikova_2/'
    global I_mighty
    global Q_mighty
    p=0 #as we will not go over lenght_Te but just over values within t__e array, we'll need a counter for index corr. to Te
    for i in t__e:
        hdu = fits.open(path+'CompSlab_%s.fits' %(i), mode='update')
        p#rint('With CompSlab_%s.fits still works' %(i)) #just to see sth on the screen while the files are being opened
        for ii in range(0,len(t__bb)): #Tbb
            for iii in range(0,len(tau__t)): #that's tau_t
                science_data = hdu[(iii)*len(t__bb)+ii+1].data #this messy index leads to the correct "list" within FITS file
                data = np.transpose(science_data.field(1)) #we need to transpose them, because tables were generated by julia and read in python, and there languages have different "majoring" in rows or columns, afaik
                #data2 = np.transpose(science_data.field(0))
                for kk in range(0, NEnergy_i): #energy
                    for kkk in range(0,NZenith_i): #zenith angles
                        I_mighty[p, ii, iii, kk, kkk] = data[kk, kkk]#no +1 anymore
                        #Q_mighty[p, ii, iii, kk, kkk] = data2[kk, kkk]
        p +=1
          
reading_table()

#%%

# ENERGY AND MU VECTOR ARE MADE ANALYTICALLY LIKE THIS
x_l, x_u = -3.7, .3 # lower and upper bounds of the log_10 energy span
NEnergy = 150 # 50# 101 # number of energy points (x)
IntEnergy = np.logspace(x_l,x_u,NEnergy), np.log(1e1)*(x_u-x_l)/(NEnergy-1.) # sample points and weights for integrations over the spectrum computing sorce function
E_vector,x_weight=IntEnergy

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

# photon parameters
E = np.asarray([0.1]) # Energy, need to look up units still
mu = np.asarray([0.5]) # cos(emission angle)

# atmosphere parameters
te = 101 #40 - 200
tbb = 0.0015 #0.001 - 0.0031
tau = 1.01  #0.5 - 3.55

# E = 9.65080984e-02
# mu = 5.00000000e-01
# tau = 1.00000000e+00
# tbb = 1.40000000e-03
# te = 1.04000000e+02

variable_arrays = (t__e, t__bb, tau__t, E_vector, mu_vector)
variable_point = (te, tbb, tau, E, mu)

Intensity = interpolate.interpn(variable_arrays, I_mighty, variable_point)
print(Intensity)

#%% MAKE RANDOM VARIABLES

def random_with_bounds(lower_limit, upper_limit, size):
    return (upper_limit - lower_limit) * np.random.random(size = size) + lower_limit

size = 10000

te_random = random_with_bounds(min(t__e), max(t__e), size)
tbb_random = random_with_bounds(min(t__bb), max(t__bb), size)
tau_random = random_with_bounds(min(tau__t), max(tau__t), size)
E_random_exponent = random_with_bounds(x_l, x_u, size)
E_random = 10 ** E_random_exponent
mu_random = random_with_bounds(min(mu_vector), max(mu_vector), size)

random_variable_draws = np.asarray((te_random, tbb_random, tau_random, E_random, mu_random))


    

#%%
from time import time
repetitions = size
Intensity = np.empty(repetitions)

time_start = time()
for i in range(repetitions):
    Intensity[i] = interpolate.interpn(variable_arrays, I_mighty, random_variable_draws[:,i])
    
time_elapsed = time() - time_start
print(time_elapsed)  