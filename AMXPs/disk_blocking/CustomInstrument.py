#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:03:59 2023

@author: bas
"""

import numpy as np
import xpsi

class CustomInstrument(xpsi.Instrument):
    """ A model of the NICER telescope response. """

    def __call__(self, signal, *args):
        """ Overwrite base just to show it is possible.

        We loaded only a submatrix of the total instrument response
        matrix into memory, so here we can simplify the method in the
        base class.

        """
        matrix = self.construct_matrix()

        self._folded_signal = np.dot(matrix, signal)

        return self._folded_signal

    @classmethod
    def from_response_files(cls, ARF, RMF, skiprows = 2, specresp_index = 2, 
                            energy_hi_index = 1, energy_low_index = 0, 
                            channel_low = 20, channel_hi = 300, max_input=1400,
                            min_input=0, channel_edges=None):
        """ Constructor which converts response files into :class:`numpy.ndarray`s.
        :param str ARF: Path to ARF which is compatible with
                                :func:`numpy.loadtxt`.
        :param str RMF: Path to RMF which is compatible with
                                :func:`numpy.loadtxt`.
        :param str channel_edges: Optional path to edges which is compatible with
                                  :func:`numpy.loadtxt`.
        """
    
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
        return cls(matrix, edges, channels, channel_edges)