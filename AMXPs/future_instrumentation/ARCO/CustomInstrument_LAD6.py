#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:02:05 2024

@author: bas
"""

import xpsi
import numpy as np
from astropy.io import fits

class LAD6(xpsi.Instrument):
    """MicroLAD6 instrument which was supposed to be on eXTP?"""
    
    
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
                            max_input, 
                            min_detection_channel=0, 
                            min_input=0,
                            **kwargs):
        
        try:
            with fits.open(RMF_file) as hdul:
                hdul.info()  # Print a summary of the FITS file contents
        
                # Access the first HDU (Header Data Unit) table
                data = hdul[1].data  # Adjust the index if needed
                
                # Extract matrix data
                n_grps = data['N_GRP']    # Number of response groups per energy bin
                f_chans = data['F_CHAN']  # First channel number for each group
                n_chans = data['N_CHAN']  # Number of channels in each group
                matrix = data['MATRIX']
                
                # Extract e_bounds data
                e_min = hdul[2].data['E_MIN']
                e_max = hdul[2].data['E_MAX']
        
        except FileNotFoundError:
            print(f"Error: The file '{RMF_file}' was not found. Please check the file path.")
        
        detection_channel_edges = np.append(e_min, e_max[-1])
        n_detection_chans_file = len(e_min)
            
        try:
            with fits.open(ARF_file) as hdul:
                hdul.info()
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
            range(n_incident_chans_file), n_grps, f_chans, n_chans, matrix
        ):
            # Populate the matrix for the first response group
            parsed_data[f_chan[0]:f_chan[0] + n_chan[0], i] = response[0:n_chan[0]]
            
            # Check if there is a second response group and populate its values
            if n_grp == 2:
                parsed_data[f_chan[1]:f_chan[1] + n_chan[1], i] = response[n_chan[0]:n_chan[0] + n_chan[1]]

        # Multiply the response matrix by the spectral response (specresp)
        response_matrix = parsed_data * specresp

        # Define the detection channels within the specified range
        detection_channels = np.arange(min_detection_channel, max_detection_channel)

        # Apply cutoffs to the response matrix and channel edge arrays
        response_matrix_cutoff = response_matrix[
            min_detection_channel:max_detection_channel, min_input:max_input
        ]
        incident_channel_edges_cutoff = incident_channel_edges[min_input:max_input + 1]
        detection_channel_edges_cutoff = detection_channel_edges[
            min_detection_channel:max_detection_channel + 1
        ]

        # Set the channel edges to the cutoff detection channel edges
        channel_edges = detection_channel_edges_cutoff

        # Return a class instance initialized with the processed response matrix,
        # cutoff incident and detection channel edges, detection channels, and additional arguments
        return cls(
            response_matrix_cutoff,
            incident_channel_edges_cutoff,
            detection_channels,
            channel_edges,
            **kwargs
        )


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    incident_channels = 2048
    
    instrument = LAD6.from_response_files(
        RMF_file = '../instrument_files/eXTP_LAD_260eV-oar75_v3.rmf',
        ARF_file = '../instrument_files/eXTP_LAD6_260eV-oar75_v3.arf',
        max_detection_channel = 1310,
        max_input = incident_channels)
    
    fake_signal = np.ones(incident_channels)
    
    detected = instrument(fake_signal)
    
    plt.plot(instrument.channel_edges[:-1], detected)
    plt.xlabel('channels')
    plt.ylabel('fake photons in each channel')
    plt.xscale('log')
    