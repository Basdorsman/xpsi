#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:02:05 2024

@author: bas
"""

import xpsi
import numpy as np

class TACO(xpsi.Instrument):
    """TACO instrument"""
    
    
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
                            ebounds_file, 
                            max_detection_channel, 
                            max_input, 
                            min_detection_channel=0, 
                            min_input=0, 
                            ebounds_skiprows=3, 
                            RMF_skiprows=3, 
                            n_incident_chans_file=2048, 
                            **kwargs):
        
        try:
            ebounds = np.loadtxt(ebounds_file, skiprows=ebounds_skiprows)
        except:
            print('ebounds file could not be loaded at ', ebounds_file)
            raise
            
        detection_channel_edges = []
        for ebound in ebounds:
            detection_channel_edges.append(ebound[1])
        detection_channel_edges.append(ebound[2])
        
        try:
            file_size = sum(1 for line in open(RMF_file))        
            input = open(RMF_file, 'r')
            lines = input.readlines()
            input.close()
            
            parsed_dataset = []

            # Iterate through each line
            for i in range(RMF_skiprows,file_size):
                # Split the line on whitespace
                parsed_dataset.append(lines[i].split())
        except:
            print('RMF file could not be loaded at ', RMF_file)

        n_detection_chans_file = ebounds.shape[0]
        responses = np.zeros((n_detection_chans_file, n_incident_chans_file))
        incident_channel_edges = []
        
        incident_channel = 0
        for i, element in zip(range(len(parsed_dataset)), parsed_dataset): #iterate over all lines
            if len(parsed_dataset[i]) == 6: #for+if => iterate over incident channels 
                energ_lo = float(element[0])
                energ_hi = float(element[1])
                n_grp = int(element[2])
                f_chan = int(element[3])
                n_chan = int(element[4])
                for j in range(n_chan): #first intervals with nonzero responses
                    responses[f_chan+j, incident_channel] = float(parsed_dataset[i+j][-1])
                if n_grp == 2: #sometimes we have two intervals
                    f_chan2 = int(parsed_dataset[i+1][0])
                    n_chan2 = int(parsed_dataset[i+1][1])
                    for k in range(n_chan2):
                        responses[f_chan2+k, incident_channel] = float(parsed_dataset[n_chan+i+k][-1])
                incident_channel+=1
                incident_channel_edges.append(energ_lo)
        incident_channel_edges.append(energ_hi)
        
        detection_channels = np.arange(min_detection_channel, max_detection_channel)
        responses_cutoff = responses[min_detection_channel:max_detection_channel,min_input:max_input]
        incident_channel_edges_cutoff = incident_channel_edges[min_input:max_input+1]
        detection_channel_edges_cutoff = detection_channel_edges[min_detection_channel:max_detection_channel+1]
        channel_edges = detection_channel_edges_cutoff

        return cls(responses_cutoff, incident_channel_edges_cutoff, detection_channels, channel_edges, **kwargs)
    
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    incident_channels = 2048
    
    instrument = TACO.from_response_files(
            RMF_file = 'TACO_4mod_matrix.txt',
            ebounds_file = 'TACO_4mod_ebounds.txt',
            max_detection_channel = 1310,
            max_input = incident_channels)
    
    fake_signal = np.ones(incident_channels)
    
    detected = instrument(fake_signal)
    
    plt.plot(instrument.channel_edges[:-1], detected)
    plt.xlabel('channels')
    plt.ylabel('fake photons in each channel')
    plt.xscale('log')
    