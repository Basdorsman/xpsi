import numpy as np
import xpsi

machine = 'local' #'local', 'helios'


import os
import sys
if machine == 'local':
    sys.path.append('/home/bas/Documents/Projects/x-psi/xpsi-bas-fork/tests/')
    
def preload_atmosphere_A4(path):
    with np.load(path, allow_pickle=True) as data_dictionary:
        NSX = data_dictionary['NSX.npy']
        size_reorderme = data_dictionary['size.npy']
        # size = (150, 9, 31, 11)
        size = [size_reorderme[3], size_reorderme[4], size_reorderme[2], size_reorderme[1]]
    
    te_index = 0 # I believe this can be an integer up to 40. Given that t__e = np.arange(40.0, 202.0, 4.0), there are 40.5 values (I expect that means 40). 
    Energy = np.ascontiguousarray(NSX[0:size[0],0])
    cos_zenith = np.ascontiguousarray([NSX[i*size[0],1] for i in range(size[1])])
    tau = np.ascontiguousarray([NSX[i*size[0]*size[1],2] for i in range(size[2])])
    t_bb = np.ascontiguousarray([NSX[i*size[0]*size[1]*size[2],3] for i in range(size[3])])
    
    te_step_size = size[0]*size[1]*size[2]*size[3]

    intensities = np.ascontiguousarray(NSX[te_step_size*te_index:te_step_size*(te_index+1),5]) #change the value of te here, it is now 40
    atmosphere = (t_bb, tau, cos_zenith, Energy, intensities)  
    return atmosphere

def preload_atmosphere_A5(path):
    """ A photosphere extension to preload the numerical atmosphere NSX. """

    with np.load(path, allow_pickle=True) as data_dictionary:
        NSX = data_dictionary['NSX.npy']
        size_reorderme = data_dictionary['size.npy']
        print(size_reorderme)
    
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


# try to get parameters from shell input
atmosphere_type = os.environ.get('atmosphere_type')
n_params = os.environ.get('n_params')

if isinstance(os.environ.get('atmosphere_type'),type(None)) or isinstance(os.environ.get('n_params'),type(None)): # if that fails input them here.
    print('E: failed to import OS environment variables, using defaults.')    
    atmosphere_type = 'A' #A, N, B
    n_params = '5' #4, 5

if atmosphere_type == 'A': atmosphere = 'accreting'
elif atmosphere_type == 'N': atmosphere = 'numerical'
elif atmosphere_type == 'B': atmosphere = 'blackbody'

print('atmosphere:', atmosphere)
print('n_params:', n_params)

if atmosphere_type == 'A':

    # photon parameters
    E = np.asarray([0.1]) # Energy, need to look up units still
    mu = np.asarray([0.5]) # cos(emission angle)

    if n_params == '4':
        if machine == 'local':
            atmosphere = preload_atmosphere_A4('/home/bas/Documents/Projects/x-psi/model_datas/bobrikova/Bobrikova_compton_slab.npz')
        elif machine == 'helios':
            atmosphere = preload_atmosphere_A4('model_data/Bobrikova_compton_slab.npz')
            

        # atmosphere parameters
        #te = 101 #40 - 200
        tbb = 0.0015 #0.001 - 0.0031
        tau = 1.01  #0.5 - 3.55
        local_vars = np.asarray([[tbb, tau]])

    if n_params == '5':
        if machine == 'local':
            atmosphere = preload_atmosphere_A5('/home/bas/Documents/Projects/x-psi/model_datas/bobrikova/Bobrikova_compton_slab.npz')
        elif machine == 'helios':
            atmosphere = preload_atmosphere_A5('model_data/Bobrikova_compton_slab.npz')
            

        # atmosphere parameters
        te = 101 #40 - 200
        tbb = 0.0015 #0.001 - 0.0031
        tau = 1.01  #0.5 - 3.55
        local_vars = np.asarray([[te, tbb, tau]])

    intensity = xpsi.surface_radiation_field.intensity(E, mu, local_vars,
                                                       atmosphere=atmosphere,
                                                       extension='hot',
                                                       numTHREADS=2)
    

print(intensity)