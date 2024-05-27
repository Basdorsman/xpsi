 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:19:27 2024

@author: bas
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import sys
sys.path.append('../')
from helper_functions import get_mids_from_edges
from analysis import analysis
ST = analysis('local', 'test', 'model', scenario='large_r', poisson_noise=True)
exposure_time = ST.data.exposure_time
channel_mids = get_mids_from_edges(ST.instrument.channel_edges)

# import synthesise_J1808_data as synthesised


counts_model = np.loadtxt('../data/synthetic_large_r_seed=42_realisation.dat')
counts_2019 = np.loadtxt('../data/2019_preprocessed.txt')



phase_edges = ST.phases_space
phase_mids = get_mids_from_edges(phase_edges)
channel_edges = ST.instrument.channel_edges
channel_mids = get_mids_from_edges(channel_edges)

bolometric_model = np.sum(counts_model, axis=0)/exposure_time*len(phase_mids)
bolometric_2019 = np.sum(counts_2019, axis=0)/exposure_time*len(phase_mids)

# fig, axes = plt.subplots(2,2,figsize=(4,4))
fig = plt.figure(figsize=(6,4.5))

# Define the GridSpec layout with 2 rows and 1 column
# and specify the height ratios
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])  # Adjust the height ratios as needed

axes = np.empty((2, 2), dtype=object)

# Create the subplots
axes[0,0] = fig.add_subplot(gs[0,0])
axes[0,1] = fig.add_subplot(gs[1,0], sharex=axes[0,0])

# def plot_pulse_normalised(phase, pulse, **kwargs):
#     pulse_height = np.max(pulse)
#     pulse_normalised = pulse/pulse_height
#     plt.plot(phase, pulse_normalised, **kwargs)

# plot_pulse_normalised(phase_mids, data[70], label='1 kev')
# plot_pulse_normalised(phase_mids, data[170], label='2 kev')
# plt.legend()

from helper_functions import custom_subplots, shift_phase_data

#fig, axes = custom_subplots(2,1, sharex=False, figsize=(5, 5))

phase_edges_model_shifted, bolometric_model_shifted = shift_phase_data(phase_edges, bolometric_model, np.argmin(bolometric_model))
phase_edges_2019_shifted, bolometric_2019_shifted = shift_phase_data(phase_edges, bolometric_2019, np.argmin(bolometric_2019))

axes[0,0].stairs(bolometric_model, phase_edges, baseline=None, label='model')
axes[0,0].stairs(bolometric_2019, phase_edges, baseline=None, label='data')
axes[0,0].set_ylim([np.min(bolometric_model), np.max(bolometric_model)])
axes[0,0].set_ylabel('Bolometric counts/s')
#axes[0,0].legend()

bolometric_residual = (bolometric_model-bolometric_2019)#/np.sqrt(bolometric_model_shifted)
axes[0,1].stairs(bolometric_residual, phase_edges, color='k')
axes[0,1].set_xlabel('Phase (cyles)')
axes[0,1].set_ylabel(r"$(m-d)$")
axes[0,1].grid(True)
## pulse fraction

# import scipy

# def sinfunc(phase, A, p, c):
    
#     return A * np.sin(phase*2.0*np.pi + p) + c

# def fitsin(phase, counts):

#     c0 = 6500
#     A0 = 1000
#     p0 = 0.15*2.0*np.pi

#     guess = [A0, p0,c0]
#     popt, pcov = scipy.optimize.curve_fit(sinfunc, phase, counts, p0=guess)
#     perr = np.sqrt(np.diag(pcov))
    
#     return popt, perr

# n_energies = counts.shape[0]
# fit_amplitudes = np.empty(n_energies)
# fit_phases = np.empty(n_energies)
# fit_constants = np.empty(n_energies)
# fit_errors = np.empty((n_energies, 3))

# for i in range(n_energies):
#     counts_slice = counts[i]
#     (A1, p1, c1), perr = fitsin(phase_mids, counts_slice)
#     fit_amplitudes[i] = A1
#     fit_phases[i] = p1
#     fit_constants[i] = c1
#     fit_errors[i] = perr
    
# axes[0].semilogy(channel_mids, abs(fit_amplitudes)/fit_constants)

# spectrum, decomposed

# phases = ST.signal.signal_from_star.shape[1]
# exposure_time = ST.data.exposure_time

# bkg = np.sum(ST.background.registered_background, axis=1)#*exposure_time
# star_avg = np.sum(ST.signal.signal_from_star,axis=1)/exposure_time
# star_dim = ST.signal.signal_from_star[:,16]*phases/exposure_time
# star_bright = ST.signal.signal_from_star[:,30]*phases/exposure_time



#axes[1].semilogx(channel_mids, bkg, label='disk', color='tab:orange')
#ax.loglog(channel_mids, star_dim, linestyle='-.', label='star@troth', color='tab:blue')
#ax.loglog(channel_mids, bkg+star_dim, label='bkg+star@troth', color='tab:blue')
#ax.loglog(channel_mids, star_bright, linestyle='-.', label='star@peak', color='tab:orange')
#ax.loglog(channel_mids, bkg+star_bright, label='bkg+star@peak', color='tab:orange')

#axes[1].fill_between(channel_mids, star_dim, star_bright, label='star', color='gray', alpha=0.3)
#axes[1].fill_between(channel_mids, bkg+star_dim, bkg+star_bright, color='green', alpha=0.3)

#ax.set_title('variation of the star counts in comparison to the background')
#axes.set_xlabel('energy of bin (keV)')
#axes.set_ylabel('counts/bin/s')
#axes[1].set_ylim([1e-4,1e1])

# spectrum, comparing model to data

axes[0,1] = fig.add_subplot(gs[0,1])
axes[1,1] = fig.add_subplot(gs[1,1], sharex=axes[0,1])

spectrum_model = np.sum(counts_model, axis=1)/exposure_time
spectrum_2019 = np.sum(counts_2019, axis=1)/exposure_time

axes[0,1].semilogx(channel_mids, spectrum_model, label='model')
axes[0,1].semilogx(channel_mids, spectrum_2019, label='data')


axes[0,1].set_ylabel('Counts/s/channel')

axes[0,1].legend()

spectrum_residual = (spectrum_model-spectrum_2019)/np.sqrt(spectrum_model)
axes[1,1].semilogx(channel_mids, spectrum_residual, color='k')
axes[1,1].set_xlabel('Energy (keV)')
axes[1,1].set_ylabel(r"$(m-d)/\sqrt{m}$")
axes[1,1].grid(True)
#ax2 = axes[1].twiny()
#ax2.set_xlim(axes[1].get_xlim())
#ax2.semilogx(ST.instrument.channels, bkg)
#ax2.set_xlabel('Channel number')

plt.tight_layout()


fig.savefig('figure_1.png', dpi=300)