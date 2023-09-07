#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 17:58:24 2023

@author: bas
"""



import sys
sys.path.append('/home/bas/Documents/Projects/x-psi/xpsi-bas-fork/AMXPs/')


from custom_tools import plot_2D_pulse

num_energies=16

import dill as pickle
with open(f'energies={num_energies}_integrator=combined.pkl', 'rb') as file:
     (signal_combined, phases, energies, shifts) = pickle.load(file)
with open(f'energies={num_energies}_integrator=split.pkl', 'rb') as file:
     (signal_split, phases, energies, shifts) = pickle.load(file)

signal = abs(signal_combined-signal_split)

plot_2D_pulse((signal_split,),
              x=phases,
              shift=shifts,
              y=energies,
              ylabel=r'Energy (keV)',
              num_rotations=2.)