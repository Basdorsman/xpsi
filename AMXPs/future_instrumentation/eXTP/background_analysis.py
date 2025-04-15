#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 16:32:04 2025

@author: bas
"""

from astropy.io import fits
import matplotlib.pyplot as plt
# plt.close('all')

bkg_file = '../instrument_files/eXTP_Response_Files_v20241125/eXTP_Response_Files_v20241125/eXTP_SFA_v20241125.bkg'

with fits.open(bkg_file) as hdul:
     hdul.info()  # Print a summary of the FITS file contents
     data = hdul[1].data  # Adjust the index if needed
     
# Extract matrix data
bkg_counts = data['COUNTS']    # Number of response groups per energy bin in 10k seconds
bkg_channels = data['CHANNEL']  # First channel number for each group

plt.plot(bkg_channels, 10*bkg_counts, label='bkg')
plt.legend()