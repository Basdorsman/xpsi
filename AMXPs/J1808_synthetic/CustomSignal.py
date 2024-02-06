#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:10:37 2023

@author: bas
"""
import numpy as np
import os

import xpsi
from xpsi.likelihoods.default_background_marginalisation import eval_marginal_likelihood
from xpsi.likelihoods.default_background_marginalisation import precomputation
from xpsi.tools.synthesise import synthesise_exposure_no_scaling as _synthesise # no scaling!
from xpsi.likelihoods._poisson_likelihood_given_background import poisson_likelihood_given_background

class CustomSignal(xpsi.Signal):
    """ A custom calculation of the logarithm of the likelihood.

    We extend the :class:`xpsi.Signal.Signal` class to make it callable.

    We also implement data synthesis capability.

    """

    def __init__(self, workspace_intervals = 1000, epsabs = 0, epsrel = 1.0e-8,
                 epsilon = 1.0e-3, sigmas = 10.0, support = None, *args, **kwargs):
        """ Perform precomputation. """
        #print("running CustomSignal init...")
        super(CustomSignal, self).__init__(*args, **kwargs)

        try:
            self._precomp = precomputation(self._data.counts.astype(np.int32))
        except AttributeError:
            print('No data... can synthesise data but cannot evaluate a '
                  'likelihood function.')
        else:
            self._workspace_intervals = workspace_intervals
            self._epsabs = epsabs
            self._epsrel = epsrel
            self._epsilon = epsilon
            self._sigmas = sigmas

            if support is not None:
                self._support = support
            else:
                self._support = -1.0 * np.ones((self._data.counts.shape[0],2))
                self._support[:,0] = 0.0

    @property
    def support(self):
        return self._support

    @support.setter
    def support(self, obj):
        self._support = obj

    def __call__(self, *args, **kwargs):
        self.loglikelihood, self.expected_counts, self.background_signal, self.background_signal_given_support = \
                eval_marginal_likelihood(self._data.exposure_time,
                                          self._data.phases,
                                          self._data.counts,
                                          self._signals,
                                          self._phases,
                                          self._shifts,
                                          self._precomp,
                                          self._support,
                                          self._workspace_intervals,
                                          self._epsabs,
                                          self._epsrel,
                                          self._epsilon,
                                          self._sigmas,
                                          kwargs.get('llzero'))#,
                                          #slim=-1.0) # default is skipping 10^89s, so some likelihood calculations are skipped
        # print('self.background_signal', self.background_signal)
    
    def poisson_likelihood_given_background(self, background_counts):
        summed_loglike, loglike, expected_counts, star = \
            poisson_likelihood_given_background(self._data.exposure_time, 
                                                self._data.phases, 
                                                self._data.counts,
                                                self._signals,
                                                self._phases,
                                                self._shifts,
                                                background_counts,
                                                allow_negative = False)
        return summed_loglike, loglike, expected_counts, star
    
    def synthesise(self,
                   exposure_time,
                   name='no_pulse',
                   directory='./',
                   **kwargs):
        
            """ Synthesise data set.
    
            """
            
            self.star_counts, self._expected_counts, synthetic = _synthesise(exposure_time,
                                                                       self._data.phases,
                                                                       self._signals,
                                                                       self._phases,
                                                                       self._shifts,
                                                                       self._background.registered_background,
                                                                       gsl_seed=42)
            self.synthetic_data = synthetic
            
            try:
                if not os.path.isdir(directory):
                    os.mkdir(directory)
            except OSError:
                print('Cannot create write directory.')
                raise

            # POISSON NOISE
            # np.savetxt(os.path.join(directory, name+'_realisation.dat'),
            #             synthetic,
            #             fmt = '%u')
            
            #NO NOISE, FLOATS
            np.savetxt(os.path.join(directory, name+'_realisation.dat'),
                        self._expected_counts,
                        fmt = '%f')
            
            # NO NOISE
            # np.savetxt(os.path.join(directory, name+'_realisation.dat'),
            #             self._expected_counts,
            #             fmt = '%u')
    
            self._write(self.expected_counts,
                        filename = os.path.join(directory, name+'_expected_hreadable.dat'),
                        fmt = '%.8e')
    
            self._write(synthetic,
                        filename = os.path.join(directory, name+'_realisation_hreadable.dat'),
                        fmt = '%u')
    
    def _write(self, counts, filename, fmt):
            """ Write to file in human readable format. """
    
            rows = len(self._data.phases) - 1
            rows *= len(self._data.channels)
    
            phases = self._data.phases[:-1]
            array = np.zeros((rows, 3))
    
            for i in range(counts.shape[0]):
                for j in range(counts.shape[1]):
                    array[i*len(phases) + j,:] = self._data.channels[i], phases[j], counts[i,j]
    
                np.savetxt(filename, array, fmt=['%u', '%.6f'] + [fmt])