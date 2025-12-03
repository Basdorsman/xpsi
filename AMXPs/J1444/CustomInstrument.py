#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 09:25:02 2025

@author: bas
"""

# load IXPE instrument
import xpsi
from xpsi.Parameter import Parameter
import numpy as np
from astropy.io import fits

import os
import sys
this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory+'/data/ixpe_products/')
# from ixpe_read_pcube3 import read_response_IXPE
from ixpe_read_pha import read_response_IXPE


class CustomInstrument_stokes(xpsi.Instrument):
    """ A model of the NICER telescope response. """

    def construct_matrix(self):
        """ Implement response matrix parameterisation. """
        matrix = self['alpha'] * self.matrix
        matrix[matrix < 0.0] = 0.0

        return matrix

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
    def from_response_files(cls, bounds, values, MRF, RMF, max_input, max_channel, min_input=0, min_channel=0,
                            channel_edges=None, **kwargs):
        """ Constructor which converts response files into :class:`numpy.ndarray`s.
        :param str MRF: Path to MRF which is compatible with
                                :...
        :param str RMF: Path to RMF which is compatible with
                                :...
        :param str channel_edges: Optional path to edges which is compatible with
                                  :func:`numpy.loadtxt`.
        """
        if min_input != 0:
            min_input = int(min_input)
        max_input = int(max_input)
        try:
            matrix, edges, channels, channel_edgesT = read_response_IXPE(MRF,RMF,min_input,max_input,min_channel,max_channel)
            if channel_edges:
                channel_edgesT = np.loadtxt(channel_edges, dtype=np.double, skiprows=3)[:,1:]
        except:
            print('A file could not be loaded.')
            raise
            
        alpha = Parameter('alpha',
                          strict_bounds = (0.1,1.9),
                          bounds = bounds.get('alpha', None),
                          doc='IXPE energy-independent scaling factor',
                          symbol = r'$\alpha_{\rm X}$',
                          value = values.get('alpha', None))
            
        return cls(matrix, edges, channels, channel_edgesT, alpha, **kwargs)


class CustomInstrument_stokes_no_alpha(xpsi.Instrument):
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
    def from_response_files(cls, MRF, RMF, max_input, max_channel, min_input=0, min_channel=0,
                            channel_edges=None):
        """ Constructor which converts response files into :class:`numpy.ndarray`s.
        :param str MRF: Path to MRF which is compatible with
                                :...
        :param str RMF: Path to RMF which is compatible with
                                :...
        :param str channel_edges: Optional path to edges which is compatible with
                                  :func:`numpy.loadtxt`.
        """
        if min_input != 0:
            min_input = int(min_input)
        max_input = int(max_input)
        try:
            matrix, edges, channels, channel_edgesT = read_response_IXPE(MRF,RMF,min_input,max_input,min_channel,max_channel)
            if channel_edges:
                channel_edgesT = np.loadtxt(channel_edges, dtype=np.double, skiprows=3)[:,1:]
        except:
            print('A file could not be loaded.')
            raise
        return cls(matrix, edges, channels, channel_edgesT)

    
class CustomInstrument_txt(xpsi.Instrument):
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
    
    



class CustomInstrument_fits(xpsi.Instrument):
    """NICER rmf and arf files"""
    
    
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
    def from_response_files(cls, 
                            RMF_file, 
                            ARF_file, 
                            max_detection_channel, 
                            max_input=False, 
                            min_detection_channel=0, 
                            min_input=0,
                            **kwargs):
        
        try:
            with fits.open(RMF_file) as hdul:
                # hdul.info()  # Print a summary of the FITS file contents
        
                # Extract data
                data1 = hdul[1].data # Adjust the index if needed
                e_min = data1['E_MIN']
                e_max = data1['E_MAX']
        
                data2 = hdul[2].data  # Adjust the index if needed
                n_grps = data2['N_GRP']    # Number of response groups per energy bin
                f_chans = data2['F_CHAN']  # First channel number for each group
                n_chans = data2['N_CHAN']  # Number of channels in each group
                data_matrix = data2['MATRIX']
                

        
        except FileNotFoundError:
            print(f"Error: The file '{RMF_file}' was not found. Please check the file path.")
        
        detection_channel_edges = np.append(e_min, e_max[-1])
        n_detection_chans_file = len(e_min)
            
        try:
            with fits.open(ARF_file) as hdul:
                # hdul.info()
                data = hdul[1].data
                
                # Extract response data
                energ_lo = data['ENERG_LO']
                energ_hi = data['ENERG_HI']
                specresp = data['SPECRESP']
        except FileNotFoundError:
            print(f"Error: The file '{ARF_file}' was not found. Please check the file path.")

        incident_channel_edges = np.append(energ_lo, energ_hi[-1])
        incident_channel_edges = incident_channel_edges.astype(np.float64)

        n_incident_chans_file = len(energ_lo)  
        
        # Initialize a zero-padded matrix to store the response data in the required format for XPSI
        parsed_data = np.zeros((n_detection_chans_file, n_incident_chans_file))
        
        # Loop through each incident channel, extracting response group information
        for i, n_grp, f_chan, n_chan, response in zip(
            range(n_incident_chans_file), n_grps, f_chans, n_chans, data_matrix
        ):
            # # Populate the matrix for the first response group
            # parsed_data[f_chan[0]:f_chan[0] + n_chan[0], i] = response[0:n_chan[0]]
            
            # # Check if there is a second response group and populate its values
            # if n_grp == 2:
            #     parsed_data[f_chan[1]:f_chan[1] + n_chan[1], i] = response[n_chan[0]:n_chan[0] + n_chan[1]]
            
            offset = 0
            for j in range(n_grp):
                start = f_chan[j]         # start index for this group
                length = n_chan[j]        # how many channels to fill
            
                # Fill parsed_data from response, using the current offset
                parsed_data[start:start+length, i] = response[offset:offset+length]
                
                offset += length          # move the offset forward

        # Multiply the response matrix by the spectral response (specresp)
        response_matrix = parsed_data * specresp

        # Define the detection channels within the specified range
        detection_channels = np.arange(min_detection_channel, max_detection_channel)

        if max_input:
            # Apply cutoffs to the response matrix and channel edge arrays
            response_matrix_cutoff = response_matrix[
                min_detection_channel:max_detection_channel, 
                min_input:max_input]
            incident_channel_edges_cutoff = incident_channel_edges[
                min_input:max_input + 1]
            # redefine with the cutoff
            response_matrix = response_matrix_cutoff
            incident_channel_edges = incident_channel_edges_cutoff
        
        detection_channel_edges_cutoff = detection_channel_edges[
            min_detection_channel:max_detection_channel + 1]
        channel_edges = detection_channel_edges_cutoff

        # Return a class instance initialized with the processed response matrix,
        # cutoff incident and detection channel edges, detection channels, and additional arguments
        
        #names
        matrix=response_matrix
        channels=detection_channels
        energy_edges=incident_channel_edges
        
        
        return cls(
            matrix,
            energy_edges,
            channels,
            channel_edges,
            **kwargs
        )
