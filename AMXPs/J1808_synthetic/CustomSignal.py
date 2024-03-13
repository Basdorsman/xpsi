#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:10:37 2023

@author: bas
"""
import numpy as np
import os
import time

from xpsi.tools.energy_integrator import energy_integrator

import xpsi
from xpsi.likelihoods.default_background_marginalisation import eval_marginal_likelihood
from xpsi.likelihoods.default_background_marginalisation import precomputation
from xpsi.tools.synthesise import synthesise_exposure_no_scaling as _synthesise # no scaling!
from xpsi.likelihoods._poisson_likelihood_given_background import poisson_likelihood_given_background

this_directory = os.path.dirname(os.path.abspath(__file__))


class CustomSignal(xpsi.Signal):
    """ A custom calculation of the logarithm of the likelihood.

    We extend the :class:`xpsi.Signal.Signal` class to make it callable.

    We also implement data synthesis capability.

    """

    def __init__(self, workspace_intervals = 1000, epsabs = 0, epsrel = 1.0e-8,
                 epsilon = 1.0e-3, sigmas = 10.0, support = None, bkg = 'marginalised', allow_negative_background = False, *args, **kwargs):
        """ Perform precomputation. """
        #print("running CustomSignal init...")
        super(CustomSignal, self).__init__(*args, **kwargs)

        self.bkg = bkg
        self.allow_negative_background = allow_negative_background
        #self.background_data = np.loadtxt(this_directory+'/data/J1808_synthetic_diskbb_background.txt')

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
                if not allow_negative_background:
                    self._support[:,0] = 0.0

    @property
    def support(self):
        return self._support

    @support.setter
    def support(self, obj):
        self._support = obj

    def __call__(self, *args, **kwargs):
        #print(f'call likelihood: {self.bkg}')
        if self.bkg == 'marginalise':
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
                                              kwargs.get('llzero'),
                                              allow_negative_background = self.allow_negative_background)#,
                                              #slim=-1.0) # default is skipping 10^89s, so some likelihood calculations are skipped

        elif self.bkg == 'model':
            self.loglikelihood, self.loglikelihood_array, self.expected_counts, self.signal_from_star = \
                poisson_likelihood_given_background(self._data.exposure_time, 
                                                    self._data.phases, 
                                                    self._data.counts,
                                                    self._signals,
                                                    self._phases,
                                                    self._shifts,
                                                    self._background.registered_background,
                                                    allow_negative = False)
            #print(f'loglikelihood: {self.loglikelihood}', flush=True)
        else:
            print('error! pass bkg argument in init!')


    def register(self, signals, fast_mode=False, threads=1):
        """  Register an incident signal by operating with the response matrix.

        A :class:`numpy.ndarray` is stored as an instance attribute containing
        source signal for each *output* channel in units of counts cm^2/s
        (assuming instrument effective area units are cm^2).

        """
        if fast_mode:
            try:
                del self.fast_total_counts
            except AttributeError:
                pass

            for hotRegion in signals:
                fast_total_counts = []

                for component, phases in zip(hotRegion, self.fast_phases):
                    if component is None:
                        fast_total_counts.append(None)
                    else:
                        integrated = energy_integrator(threads,
                                                       component,
                                                       np.log10(self.fast_energies),
                                                       np.log10(self._energy_edges))

                        # move interstellar to star?
                        if self._interstellar is not None:
                            self._interstellar(self._energy_mids, integrated)

                        temp = self._instrument(integrated,
                                                self._input_interval_range,
                                                self._data.index_range)

                        fast_total_counts.append(np.sum(temp))

                self.fast_total_counts = tuple(fast_total_counts)
        else:
            try:
                del self.signals
            except AttributeError:
                pass

            if self.cache:
                try:
                    del self.incident_specific_flux_signals
                except AttributeError:
                    pass

                for hotRegion in signals: # iterate over hot regions
                    signal = None
                    for component in hotRegion: # add other components
                        try:
                            signal += component
                        except TypeError:
                            signal = component.copy()
                    # cache total hot region signal
                    self.incident_specific_flux_signals = signal

                try:
                    del self.incident_flux_signals
                except AttributeError:
                    pass

                try:
                    self.execute_custom_cache_instructions()
                except NotImplementedError:
                    pass # no custom caching targets

            for hotRegion in signals:
                integrated = None
                for component in hotRegion:
                    temp = energy_integrator(threads,
                                             component,
                                             np.log10(self._energies),
                                             np.log10(self._energy_edges))
                    try:
                        integrated += temp
                    except TypeError:
                        integrated = temp

                if self.cache:
                    self.incident_flux_signals = integrated.copy()

                if self._interstellar is not None:
                    self._interstellar(self._energy_mids, integrated)

                if self.cache:
                    self.incident_flux_attenuated = integrated.copy()

                self.signals = self._instrument(integrated,
                                                self._input_interval_range,
                                                self._data.index_range)
                


            if self._background is not None:
                try:
                    self._background(self._energy_edges,
                                     self._data.phases)
                except TypeError:
                    print('Error when evaluating the incident background.')
                    raise

                # applying interstellar over my background, which is >IN J1808< is being produced at source. - Bas 
                if self._interstellar is not None:
                    self._interstellar(self._energy_mids, self._background.incident_background)

                self._background.registered_background = \
                                self._instrument(self._background.incident_background,
                                                 self._input_interval_range,
                                                 self._data.index_range)
    
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
            np.savetxt(os.path.join(directory, name+'_realisation.dat'),
                        np.round(synthetic),
                        fmt = '%u')
            
            #NO NOISE, FLOATS
            # np.savetxt(os.path.join(directory, name+'_realisation.dat'),
            #             self._expected_counts,
            #             fmt = '%f')
            
            # NO NOISE, WHOLE COUNTS
            # np.savetxt(os.path.join(directory, name+'_realisation.dat'),
            #             np.round(self._expected_counts),
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
