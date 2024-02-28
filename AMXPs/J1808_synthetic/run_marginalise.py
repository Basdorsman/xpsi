#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:45:17 2024

@author: bas
"""

from analysis import analysis


myAnalysis = analysis('local','test', bkg = 'marginalise', support_factor=0.1)
myAnalysis()

