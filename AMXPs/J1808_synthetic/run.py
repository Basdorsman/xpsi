#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:45:17 2024

@author: bas
"""

from analysis import analysis
import os


support_factor = os.environ.get('support_factor')
if os.environ.get('support_factor') == None:
    print('no support_factor passed in os, using default')
    support_factor = '9e-1'


Analysis = analysis('local','test', 'marginalise', support_factor=support_factor, scenario='literature')
Analysis()

