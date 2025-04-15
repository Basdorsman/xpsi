#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:45:30 2024

@author: bas
"""

from astropy.io import fits
import numpy as np

rmf_file = 'instrument_files/eXTP_LAD_260eV-oar75_v3.rmf'
arf_file = 'instrument_files/eXTP_LAD6_260eV-oar75_v3.arf'
bkg_file = 'instrument_files/eXTP_LAD6_260eV-oar75_v3.bkg'

with fits.open(rmf_file) as hdul:
    hdul.info()  # Print a summary of the FITS file contents

    # Access the first HDU (Header Data Unit) table
    data = hdul[1].data  # Adjust the index if needed
    header = hdul[1].header
    
    # matrix
    n_grps = data['N_GRP']    # Number of response groups per energy bin
    f_chans = data['F_CHAN']   # First channel number for each group
    n_chans = data['N_CHAN']   # Number of channels in each group
    matrix = data['MATRIX']
    
    # e_bounds 
    channel = hdul[2].data['CHANNEL']
    e_min = hdul[2].data['E_MIN']
    e_max = hdul[2].data['E_MAX']
    
detection_channel_edges = np.append(e_min, e_max[-1])
n_detection_chans_file = len(e_min)
    
with fits.open(arf_file) as hdul:
    hdul.info()
    data = hdul[1].data
    energ_lo = data['ENERG_LO']
    energ_hi = data['ENERG_HI']
    specresp = data['SPECRESP']
    
with fits.open(bkg_file) as hdul:
    hdul.info()
    data = hdul[1].data
    bkg_channel = data['CHANNEL']
    bkg_counts = data['COUNTS']
    print(data.columns)

incident_channel_edges = np.append(energ_lo, energ_hi[-1])
n_incident_chans_file = len(energ_lo)  
    



parsed_data = np.zeros((n_detection_chans_file, n_incident_chans_file))
for i, n_grp, f_chan, n_chan, response in zip(range(n_incident_chans_file), n_grps, f_chans, n_chans, matrix):
    parsed_data[f_chan[0]:f_chan[0] + n_chan[0], i] = response[0:n_chan[0]]
    #print('inserted in parsed_data[', f_chan[0], ':', f_chan[0] + n_chan[0],',',i,']')
    if n_grp ==2:
        # print(n_chan[0])
        # print(n_chan[1])
        # print(f_chan[1])
        # print(response)
        # print(response[n_chan[0]:n_chan[1]])
        parsed_data[f_chan[1]:f_chan[1] + n_chan[1], i] = response[n_chan[0]:n_chan[0]+n_chan[1]]
        #print('inserted in parsed_data[', f_chan[1], ':', f_chan[1] + n_chan[1],',',i,']')
    
 
response = parsed_data * specresp

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

pcm = plt.pcolormesh(incident_channel_edges, detection_channel_edges, response, norm=LogNorm(vmin=1e-1, vmax=2e3), cmap='viridis')
plt.colorbar(pcm, label='Logarithmic Colorbar')
# pcm = plt.pcolormesh(incident_channel_edges, detection_channel_edges, response, cmap='viridis')
# plt.colorbar(pcm, label='Linear Colorbar')
plt.xlabel('E incident (keV)')
plt.ylabel('E detected (keV)')

# Set logarithmic scale for both axes
plt.xscale('log')
plt.yscale('log')

#%%

ARF = np.sum(response, axis=0)
fig, ax = plt.subplots()
ax.plot(incident_channel_edges[:-1], ARF)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim([1e1, 5e4])
ax.set_xlim([1e-1, 1e2])

ax.set_ylabel('presumably Eff. Area (cm2)')
ax.set_xlabel('E incident (keV)')

