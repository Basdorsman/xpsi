#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:48:20 2023

@author: bas
"""

import numpy as np
from xpsi.global_imports import  _keV, _k_B
k_B_over_keV = _k_B / _keV

def get_T_in_log10_Kelvin(T_keV):
  """
  Converts temperature from keV to log10(K) for a given input (scalar or tuple).

  Args:
      T_keV: The temperature in keV, can be a scalar or a tuple.

  Returns:
      A scalar or tuple containing the temperature in log10(K) for each input element.

  Raises:
      ValueError: If the input is not a scalar or a tuple.
  """

  if isinstance(T_keV, (int, float)):
    # Handle scalar case
    T_log10_Kelvin = np.log10(T_keV / k_B_over_keV)
    return T_log10_Kelvin
  elif isinstance(T_keV, tuple):
    # Handle tuple case
    T_log10_Kelvin_values = []
    for t in T_keV:
      T_log10_Kelvin_values.append(np.log10(t / k_B_over_keV))
    return tuple(T_log10_Kelvin_values)
  else:
    raise ValueError("Input must be a scalar or a tuple.")

def get_mids_from_edges(edges):
    mids_len = len(edges)-1
    mids = np.empty(mids_len)
    for i in range(mids_len):
        mids[i] = (edges[i]+edges[i+1])/2
    return mids

from matplotlib import pyplot as plt

from matplotlib.ticker import MultipleLocator, AutoLocator, AutoMinorLocator
from matplotlib import gridspec
from matplotlib import cm
from xpsi.tools import phase_interpolator

def veneer(x, y, axes, lw=1.0, length=8):
    """ Make the plots a little more aesthetically pleasing. """
    if x is not None:
        if x[1] is not None:
            axes.xaxis.set_major_locator(MultipleLocator(x[1]))
        if x[0] is not None:
            axes.xaxis.set_minor_locator(MultipleLocator(x[0]))
    else:
        axes.xaxis.set_major_locator(AutoLocator())
        axes.xaxis.set_minor_locator(AutoMinorLocator())

    if y is not None:
        if y[1] is not None:
            axes.yaxis.set_major_locator(MultipleLocator(y[1]))
        if y[0] is not None:
            axes.yaxis.set_minor_locator(MultipleLocator(y[0]))
    else:
        axes.yaxis.set_major_locator(AutoLocator())
        axes.yaxis.set_minor_locator(AutoMinorLocator())

    axes.tick_params(which='major', colors='black', length=length, width=lw)
    axes.tick_params(which='minor', colors='black', length=int(length/2), width=lw)
    plt.setp(axes.spines.values(), linewidth=lw, color='black')

def plot_2D_pulse(z, x, shift, y, ylabel,
                  num_rotations=1.0, res=1000, figsize=(5,3),
                  cm=cm.viridis, normalize=True):
    """ Helper function to plot a phase-energy pulse.

    :param array-like z:
        A pair of *ndarray[m,n]* objects representing the signal at
        *n* phases and *m* values of an energy variable.

    :param ndarray[n] x: Phases the signal is resolved at.

    :param tuple shift: Hot region phase parameters.

    :param ndarray[m] x: Energy values the signal is resolved at.

    """

    fig = plt.figure(figsize = figsize)

    gs = gridspec.GridSpec(1, 2, width_ratios=[50,1], wspace=0.025)
    ax = plt.subplot(gs[0])
    ax_cb = plt.subplot(gs[1])

    new_phase_edges = np.linspace(0.0, num_rotations, res+1)
    new_phase_mids = get_mids_from_edges(new_phase_edges)
    interpolated = phase_interpolator(new_phase_mids,
                                      x,
                                      z[0], shift[0])

    
    if len(z) == 2:
        interpolated += phase_interpolator(new_phase_mids,
                                            x,
                                            z[1], shift[1])
    if normalize:
        profile = ax.pcolormesh(new_phase_mids,
                                 y,
                                 interpolated/np.max(interpolated),
                                 cmap = cm,
                                 linewidth = 0,
                                 rasterized = True)
    elif not normalize:
        profile = ax.pcolormesh(new_phase_mids,
                                 y,
                                 interpolated,
                                 cmap = cm,
                                 linewidth = 0,
                                 rasterized = True,
                                 shading = 'flat')
    
    profile.set_edgecolor('face')

    ax.set_xlim([0.0, num_rotations])
    ax.set_yscale('log')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r'Phase')
    veneer((0.1, 0.5), (None,None), ax)
    
    if normalize:
        cb = plt.colorbar(profile, cax = ax_cb, ticks = MultipleLocator(0.2))
        cb.set_label(label=r'Signal (arbitrary units)', labelpad=25)
        cb.solids.set_edgecolor('face')
        veneer((None, None), (0.05, None), ax_cb)
        cb.outline.set_linewidth(1.0)
    elif not normalize:
        ticks = np.linspace(0, np.max(z[0]), 10)
        cb = plt.colorbar(profile, cax = ax_cb,  ticks = ticks)
    
    return fig, ax