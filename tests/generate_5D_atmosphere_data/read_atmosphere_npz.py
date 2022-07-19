#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 14:52:21 2022

@author: bas
"""

import numpy as np

def hot_atmosphere_4D(path):
    size = (35,11,67,166)
    
    # NSX = np.loadtxt(path, dtype=np.double)
    with np.load(path) as data_dictionary:
        NSX = data_dictionary['arr_0.npy']

    _mu_opt = np.ascontiguousarray(NSX[0:size[2],1][::-1])
    logE_opt = np.ascontiguousarray([NSX[i*size[2],0] for i in range(size[3])])
    logT_opt = np.ascontiguousarray([NSX[i*size[1]*size[2]*size[3],3] for i in range(size[0])])
    logg_opt = np.ascontiguousarray([NSX[i*size[2]*size[3],4] for i in range(size[1])])

    def reorder_23(array, size):
        new_array=np.zeros(size)
        index=0
        for i in range(size[3]):
            for j in range(size[2]):
                    new_array[:,:,j,i]=array[:,:,index]
                    index+=1
        return new_array

    reorder_buf_opt=reorder_23(10**NSX[:,2].reshape(size[0],size[1],int(np.prod(size)/(size[0]*size[1]))),size)
    buf_opt=np.ravel(np.flip(reorder_buf_opt,2))
    return buf_opt

def hot_atmosphere_5D(path):
    size = (7,35,11,67,166)

    # Loading LARGE npz file which is hopefully quite fast. 
    with np.load(path) as data_dictionary:
        NSX = data_dictionary['arr_0.npy']
    
    _mu_opt = np.ascontiguousarray(NSX[0:size[3],1][::-1])
    logE_opt = np.ascontiguousarray([NSX[i*size[3],0] for i in range(size[4])])
    logT_opt = np.ascontiguousarray([NSX[i*size[2]*size[3]*size[4],3] for i in range(size[1])])
    logg_opt = np.ascontiguousarray([NSX[i*size[3]*size[4],4] for i in range(size[2])])
    modulator = np.ascontiguousarray([NSX[i*size[1]*size[2]*size[3]*size[4],5] for i in range(size[0])])

    def reorder_last_two(array, size):
        new_array=np.zeros(size)
        index=0
        for i in range(size[4]):
            for j in range(size[3]):
                    new_array[:,:,:,j,i]=array[:,:,:,index]
                    index+=1
        return new_array

    reorder_buf_opt=reorder_last_two(10**NSX[:,2].reshape(size[0],size[1],size[2],int(np.prod(size)/(size[0]*size[1]*size[2]))),size)
    buf_opt=np.ravel(np.flip(reorder_buf_opt,3))

    return buf_opt

path4D = '/home/bas/Documents/Projects/x-psi/model_datas/model_data/H-atmosphere_Spectra_fully_ionized/NSX_H-atmosphere_Spectra/nsx_H_v171019.npz'
path5D = '/home/bas/Documents/Projects/x-psi/model_datas/model_data/H-atmosphere_Spectra_fully_ionized/NSX_H-atmosphere_Spectra/nsx_H_v171019_5D_no_effect.npz'

buf4D=hot_atmosphere_4D(path4D)
buf5D=hot_atmosphere_5D(path5D)

print(buf4D[1993630])
print(buf5D[1993630])


# size = (7,35,11,67,166)
# with np.load(path5D) as data_dictionary:
#     NSX = data_dictionary['arr_0.npy']

# def reorder_last_two(array, size):
#     new_array=np.zeros(size)
#     index=0
#     for i in range(size[4]):
#         for j in range(size[3]):
#                 new_array[:,:,:,j,i]=array[:,:,:,index]
#                 index+=1
#     return new_array

# reorder_buf_opt=reorder_last_two(10**NSX[:,2].reshape(size[0],size[1],size[2],int(np.prod(size)/(size[0]*size[1]*size[2]))),size)