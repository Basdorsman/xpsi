#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 09:23:54 2025

@author: bas
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
this_directory = os.path.dirname(os.path.abspath(__file__))

data = np.loadtxt(this_directory+'/../../outputs/combine_posteriors_lp1000/17640043/combine_posteriors_lp1000/run_post_equal_weights.dat')

join_samples = data[:, :7]

labels = [
    r"$M\,[M_\odot]$",
    r"$R\,[\mathrm{km}]$",
    r"$D\,[\mathrm{kpc}]$",
    r"$\cos i$",
    r"$R_{\rm in}^{\rm NICER}\,[\mathrm{km}]$",
    r"$R_{\rm in}^{\rm IXPE}\,[\mathrm{km}]$",
    r"$N_H\,[10^{21}\,\mathrm{cm}^{-2}]$",
]

import corner

fig = corner.corner(
    join_samples,
    labels=labels,
    show_titles=True,
    title_fmt=".2f",
    quantiles=[0.16, 0.5, 0.84],
)

fig.savefig('join_corner.png')

# Rin_IXPE = join_samples[:,5]
# R_eq = join_samples[:,1]

# plt.hist(Rin_IXPE - R_eq, bins=100)