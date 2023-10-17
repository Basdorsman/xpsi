#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:36:27 2023

@author: bas
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
sys.path.append('../')
from custom_tools import veneer, CustomInstrument, CustomInstrumentJ1808 

# fig = plt.figure(figsize = (10,7))
# axes = fig.subplots(1,2)

# RMF_j1808 = '../model_data/J1808/ni2584010103mpu7_rmf_matrix.txt'
# RMF_j1808_array = np.loadtxt(RMF_j1808, dtype=np.double)
# axes[0].imshow(RMF_j1808_array, cmap = cm.viridis, rasterized = True)

# RMF_example = '../model_data/nicer_v1.01_rmf_matrix.txt'
# RMF_example_array = np.loadtxt(RMF_example, dtype=np.double)
# axes[1].imshow(RMF_example_array, cmap = cm.viridis, rasterized = True)

def from_response_files(ARF, RMF, max_input, min_input=0,
                        channel_edges=None):
    """ Constructor which converts response files into :class:`numpy.ndarray`s.
    :param str ARF: Path to ARF which is compatible with
                            :func:`numpy.loadtxt`.
    :param str RMF: Path to RMF which is compatible with
                            :func:`numpy.loadtxt`.
    :param str channel_edges: Optional path to edges which is compatible with
                              :func:`numpy.loadtxt`.
    """
    skiprows = 2
    specresp_index = 2
    energy_hi_index = 1
    energy_low_index = 0
    channel_low = 20
    channel_hi = 600#201

    if min_input != 0:
        min_input = int(min_input)

    max_input = int(max_input)

    try:
        ARF = np.loadtxt(ARF, dtype=np.double, skiprows=skiprows)
        RMF = np.loadtxt(RMF, dtype=np.double)
        if channel_edges:
            channel_edges = np.loadtxt(channel_edges, dtype=np.double, skiprows=skiprows)[:,1:]
    except:
        print('A file could not be loaded.')
        raise

    matrix = np.ascontiguousarray(RMF[min_input:max_input,channel_low:channel_hi].T, dtype=np.double)

    edges = np.zeros(ARF[min_input:max_input,specresp_index].shape[0]+1, dtype=np.double)

    edges[0] = ARF[min_input,energy_low_index]; edges[1:] = ARF[min_input:max_input,energy_hi_index]

    for i in range(matrix.shape[0]):
        # print('before')
        # print(np.sum(matrix[i,:]))
        matrix[i,:] *= ARF[min_input:max_input,specresp_index]
        # print('after')
        # print(np.sum(matrix[i,:]))
    channels = np.arange(channel_low,channel_hi)
    # print(channel_edges)
    channel_edges = channel_edges[channel_low:channel_hi+1,-2]
    # print(channel_edges)
    return (matrix, edges, channels, channel_edges)

# (matrix, edges, channels, channel_edges) = from_response_files(ARF = '../model_data/nicer_v1.01_arf.txt',
#                                                               RMF = '../model_data/nicer_v1.01_rmf_matrix.txt',
#                                                               max_input = 500, #500
#                                                               min_input = 0,
#                                                               channel_edges = '../model_data/nicer_v1.01_rmf_energymap.txt')


(matrix, edges, channels, channel_edges) = from_response_files(ARF = '../model_data/J1808/ni2584010103mpu7_arf_aeff.txt',
                                                                            RMF = '../model_data/J1808/ni2584010103mpu7_rmf_matrix.txt',
                                                                            max_input = 2500, #1400, #500
                                                                            min_input = 0,
                                                                            channel_edges = '../model_data/J1808/ni2584010103mpu7_rmf_energymap.txt')


fig = plt.figure(figsize = (10,7))
axes = fig.subplots(1,2)
# ax = fig.add_subplot(111)
# veneer((25, 100), (10, 50), ax)



axes[0].pcolormesh(edges, channel_edges, matrix, cmap=cm.viridis)
axes[0].set_ylabel('channel_edges (keV)')
axes[0].set_xlabel('Energy (keV)')
# axes[0].set_ylim([matrix.shape[0], 0])


axes[1].imshow(matrix, cmap = cm.viridis, aspect='auto')
axes[1].set_ylabel('Channel $-\;20$')
axes[1].set_xlabel('Energy interval')

# ax.set_xlim([0,500])
