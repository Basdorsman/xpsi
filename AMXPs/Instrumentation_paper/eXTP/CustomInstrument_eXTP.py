#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 13:47:44 2025

@author: bas
"""

import xpsi
import numpy as np
from astropy.io import fits

class SFA(xpsi.Instrument):
    """the Spectroscopic Focusing Array (SFA), a set of 9 X-ray telescopes 
    operating in the 0.5–10 keV energy band with a field-of-view (FoV) of 12′ 
    each with an intended spatial resolution of 1′. The SFA will have an 
    effective area of ∼0.8 m2 at 2 kev and 0.5 m2 at 6 keV. SFA will be 
    equipped with Silicon Drift Detectors offering <180 eV spectral resolution. 
    The optics are Wolter-I type nested mirrors with coated glass focal 
    elements. https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/extp.html"""
    
    
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
                # hdul.info()  # Print a summary of the FITS file contents
        
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
        
        # Determine if all response groups are single (i.e., scalar layout)
        all_single_grp = all(n == 1 for n in n_grps)
        
        # Loop through each incident channel, extracting response group information
        for i, n_grp, f_chan, n_chan, response in zip(
            range(n_incident_chans_file), n_grps, f_chans, n_chans, matrix
        ):
            if all_single_grp:
                # Simple scalar values used
                parsed_data[f_chan:f_chan + n_chan, i] = response[:n_chan]
            else:
                # Indexed list values
                parsed_data[f_chan[0]:f_chan[0] + n_chan[0], i] = response[:n_chan[0]]
        
                if n_grp == 2:
                    parsed_data[f_chan[1]:f_chan[1] + n_chan[1], i] = response[n_chan[0]:n_chan[0] + n_chan[1]]


        # Multiply the response matrix by the spectral response (specresp)
        response_matrix = parsed_data * specresp
        
        # print('matrix shape:', response_matrix.shape)
        
        # Boolean masks where True means the entire row or column is zero
        # zero_rows = np.all(response_matrix == 0, axis=1)  # shape (1180,)
        # zero_cols = np.all(response_matrix == 0, axis=0)  # shape (2980,)
        
        # Find the first index where the entire row or column is zero
        # first_zero_row = np.where(zero_rows)[0][0] if np.any(zero_rows) else None
        # first_zero_col = np.where(zero_cols)[0][0] if np.any(zero_cols) else None
        
        # print(f"First all-zero row index: {first_zero_row}")
        # print(f"First all-zero column index: {first_zero_col}")
        

        

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
    incident_channels = 2460
    
    instrument = SFA.from_response_files(
        RMF_file = '../instrument_files/eXTP_Response_Files_v20241125/eXTP_Response_Files_v20241125/eXTP_SFA_v20241125.rmf',
        ARF_file = '../instrument_files/eXTP_Response_Files_v20241125/eXTP_Response_Files_v20241125/eXTP_SFA_v20241125.arf',
        max_detection_channel = 1180,
        max_input = incident_channels)
    
    #%%
    
    from matplotlib.colors import LogNorm
    
    
   
    fig, ax = plt.subplots()
    pcm = plt.pcolormesh(instrument.energy_edges, instrument.channel_edges, instrument.matrix, norm=LogNorm(vmin=1e-2, vmax=1e2), cmap='viridis')
   
    
    fig.colorbar(pcm, label='Logarithmic Colorbar')
    # pcm = plt.pcolormesh(incident_channel_edges, detection_channel_edges, responses, cmap='viridis')
    # plt.colorbar(pcm, label='Linear Colorbar')
    ax.set_xlabel('E incident (keV)')
    ax.set_ylabel('E detected (keV)')

    # Set logarithmic scale for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    
    #%%
    
    ARF = np.sum(instrument.matrix, axis=0)
    fig, ax = plt.subplots()
    ax.plot(instrument.energy_edges[:-1], ARF)
    ax.set_xlim([1e-1,1e2])
    ax.set_ylim([1e2, 1e5])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('channel energy edges (keV)')
    ax.set_ylabel('eff. area (cm^2)')
    ax.set_title('eXTP SFA effective area')
    
    
    #%%
    
    fig, ax = plt.subplots()
    
    fake_signal = np.ones(incident_channels)
    detected = instrument(fake_signal)
    
    ax.plot(instrument.channel_edges[:-1], detected)
    ax.set_xlabel('Channel edges (keV)')
    ax.set_ylabel('fake photons in each channel')
    ax.set_xscale('log')
    