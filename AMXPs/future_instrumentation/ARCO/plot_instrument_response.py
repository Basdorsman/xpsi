#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:06:48 2024

@author: bas
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ebounds = np.loadtxt('../instrument_files/TACO_4mod_ebounds.txt',skiprows=3)

detection_channel_edges = []
for ebound in ebounds:
    detection_channel_edges.append(ebound[1])
detection_channel_edges.append(ebound[2])


RMF= '../instrument_files/TACO_4mod_matrix.txt'


file_size = sum(1 for line in open(RMF))        
input = open(RMF, 'r')
lines = input.readlines()
input.close()

parsed_dataset = []

# Iterate through each line
for i in range(3,file_size):
    # Split the line on whitespace
    parsed_dataset.append(lines[i].split())
    
n_incident_chans = 2048
n_detection_chans = ebounds.shape[0]

responses = np.zeros((n_detection_chans, n_incident_chans))
incident_channel_edges = []

incident_channel = 0
for i, element in zip(range(len(parsed_dataset)), parsed_dataset):
    if len(parsed_dataset[i]) == 6: #this means: for all incident channels 
        energ_lo = float(element[0])
        energ_hi = float(element[1])
        n_grp = int(element[2])
        f_chan = int(element[3])
        n_chan = int(element[4])
        for j in range(n_chan):
            responses[f_chan+j, incident_channel] = float(parsed_dataset[i+j][-1])
        if n_grp == 2:
            f_chan2 = int(parsed_dataset[i+1][0])
            n_chan2 = int(parsed_dataset[i+1][1])
            for k in range(n_chan2):
                responses[f_chan2+k, incident_channel] = float(parsed_dataset[n_chan+i+k][-1])
        incident_channel+=1
        incident_channel_edges.append(energ_lo)
incident_channel_edges.append(energ_hi)

min_detection_channel = 0
max_detection_channel = 1310
min_incident_channel = 0
max_incident_channel = 2048

channels = np.arange(min_detection_channel, max_detection_channel)
responses_cutoff = responses[min_detection_channel:max_detection_channel,min_incident_channel:max_incident_channel]
incident_channel_edges_cutoff = incident_channel_edges[min_incident_channel:max_incident_channel+1]
detection_channel_edges_cutoff = detection_channel_edges[min_detection_channel:max_detection_channel+1]

pcm = plt.pcolormesh(incident_channel_edges_cutoff, detection_channel_edges_cutoff, responses_cutoff, norm=LogNorm(vmin=1e-2, vmax=1e2), cmap='viridis')
plt.colorbar(pcm, label='Logarithmic Colorbar')
# pcm = plt.pcolormesh(incident_channel_edges, detection_channel_edges, responses, cmap='viridis')
# plt.colorbar(pcm, label='Linear Colorbar')
plt.xlabel('E incident (keV)')
plt.ylabel('E detected (keV)')

# Set logarithmic scale for both axes
plt.xscale('log')
plt.yscale('log')

#%%

ARF = np.sum(responses_cutoff, axis=0)
fig, ax = plt.subplots()
ax.plot(incident_channel_edges_cutoff[:-1], ARF)
ax.set_xscale('log')

ax.set_ylabel('presumably Eff. Area (cm2)')
ax.set_xlabel('E incident (keV)')

