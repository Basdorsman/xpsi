#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 09:23:54 2025

@author: bas
"""

import matplotlib.pyplot as plt
import numpy as np
import corner
import os
import sys
this_directory = os.path.dirname(os.path.abspath(__file__))

# # data = np.loadtxt(this_directory+'/../../outputs/combine_posteriors_lp1000/17640043/combine_posteriors_lp1000/run_post_equal_weights.dat')
data = np.loadtxt(this_directory+'/test_analysis/run_post_equal_weights.dat')

# join_samples = data[:, :7]

# labels = [
#     r"$M\,[M_\odot]$",
#     r"$R\,[\mathrm{km}]$",
#     r"$D\,[\mathrm{kpc}]$",
#     r"$\cos i$",
#     r"$R_{\rm in}^{\rm NICER}\,[\mathrm{km}]$",
#     r"$R_{\rm in}^{\rm IXPE}\,[\mathrm{km}]$",
#     r"$N_H\,[10^{21}\,\mathrm{cm}^{-2}]$",
# ]


# fig = corner.corner(
#     join_samples,
#     # labels=labels,
#     show_titles=True,
#     title_fmt=".2f",
#     quantiles=[0.16, 0.5, 0.84],
# )

# fig.savefig('join_corner.png')

from sample_kde_posteriors import *

IXPE_samples = post_IXPE_nd_eqw[:,params_IXPE_nd]
NICER_samples = post_NICER_eqw[:,params_NICER_nd]
# data = np.loadtxt(this_directory+'/combine_NICER_IXPE_nd_4k/run_post_equal_weights.dat')
#nh shared
# join_samples = data[:, [0,1,2,3,5]]
#nh not shared
join_samples = data[:, [0,1,2,3]]

labels = [
    r"$M\,[M_\odot]$",
    r"$R\,[\mathrm{km}]$",
    r"$D\,[\mathrm{kpc}]$",
    r"$\cos i$",
    r"$N_H\,[10^{21}\,\mathrm{cm}^{-2}]$",
]

fig = corner.corner(
 	IXPE_samples,
    labels=labels,
 	color="C0",
 	plot_density=True,
 	plot_datapoints=True,
 	fill_contours=True
)
True
corner.corner(
 	NICER_samples,
 	fig=fig,
 	color="C1",
 	plot_density=True,
 	plot_datapoints=True,
 	fill_contours=True
)

# fig = corner.corner(
#  	NICER_samples,
#      labels=labels,
#  	# fig=fig,
#  	color="C1",
#  	plot_density=True,
#  	plot_datapoints=True,
#  	fill_contours=True
# )


corner.corner(
 	join_samples,
 	fig=fig,
 	color="C2",
 	plot_density=True,
 	plot_datapoints=True,
 	fill_contours=True
)

import matplotlib.lines as mlines

ixpe_line  = mlines.Line2D([], [], color="C0", label="IXPE")
nicer_line = mlines.Line2D([], [], color="C1", label="NICER")
joint_line = mlines.Line2D([], [], color="C2", label="Joint")

fig.axes[0].legend(
	handles=[
        ixpe_line, 
        nicer_line, 
        joint_line],
	loc="upper left",
	bbox_to_anchor=(1.02, 1.0),
	frameon=False
)
