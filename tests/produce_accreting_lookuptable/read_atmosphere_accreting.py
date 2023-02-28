#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 16:21:43 2023

@author: bas
"""
import numpy as np

# def hot_atmosphere_accreting(path):
#     size = (150, 9, 31, 11, 41)

#     # Loading LARGE npz file which is hopefully quite fast. 
#     with np.load(path) as data_dictionary:
#         NSX = data_dictionary['arr_0.npy']
    
#     Energy = np.ascontiguousarray(NSX[0:size[0],0])
#     cos_zenith = np.ascontiguousarray([NSX[i*size[0],1] for i in range(size[1])])
#     tau = np.ascontiguousarray([NSX[i*size[0]*size[1],2] for i in range(size[2])])
#     t_bb = np.ascontiguousarray([NSX[i*size[0]*size[1]*size[2],3] for i in range(size[3])])
#     t_e = np.ascontiguousarray([NSX[i*size[0]*size[1]*size[2]*size[3],4] for i in range(size[4])])
#     intensities = np.ascontiguousarray(NSX[:,5])

#     def reorder_last_two(array, size):
#         new_array=np.zeros(size)
#         index=0
#         for i in range(size[4]):
#             for j in range(size[3]):
#                     new_array[:,:,:,j,i]=array[:,:,:,index]
#                     index+=1
#         return new_array

#     reorder_buf_opt=reorder_last_two(10**NSX[:,2].reshape(size[0],size[1],size[2],int(np.prod(size)/(size[0]*size[1]*size[2]))),size)
#     buf_opt=np.ravel(np.flip(reorder_buf_opt,3))

#     return buf_opt


path_accreting = '/home/bas/Documents/Projects/x-psi/model_datas/bobrikova/Bobrikova_compton_slab.npz'

# buf_accreting=hot_atmosphere_accreting(path_accreting)
# print(buf_accreting[1993630])

with np.load(path_accreting) as data:
    atmosphere = data['arr_0.npy']

