#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 09:36:04 2025

@author: bas
"""

import os

import xpsi
import numpy as np
from xpsi.likelihoods._gaussian_likelihood_QnUn import gaussian_likelihood_QnUn
from xpsi.likelihoods._gaussian_likelihood_given_background_IQU import gaussian_likelihood_given_background

from scipy.interpolate import interp1d

from xpsi.tools.energy_integrator import energy_integrator

from xpsi.likelihoods.default_background_marginalisation import eval_marginal_likelihood
from xpsi.likelihoods.default_background_marginalisation import precomputation
from xpsi.tools.synthesise import synthesise_exposure as _synthesise # no scaling!
from xpsi.likelihoods._poisson_likelihood_given_background import poisson_likelihood_given_background

def get_mids_from_edges(edges):
    mids_len = len(edges)-1
    mids = np.empty(mids_len)
    for i in range(mids_len):
        mids[i] = (edges[i]+edges[i+1])/2
    return mids

def find_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class CustomSignal_gaussian(xpsi.Signal):
    """

    A custom calculation of the logarithm of the likelihood.
    We extend the :class:`~xpsi.Signal.Signal` class to make it callable.
    We overwrite the body of the __call__ method. The docstring for the
    abstract method is copied.

    """

    def __init__(self, workspace_intervals = 1000, epsabs = 0, epsrel = 1.0e-8,
                 epsilon = 1.0e-3, sigmas = 10.0, support = None, **kwargs):
        super(CustomSignal_gaussian, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        anegI = (False, False)
        anegQU = (True, True)
        background = np.zeros((np.shape(self._data.counts)))

        self.loglikelihood, self.expected_counts = \
            gaussian_likelihood_given_background(self._data.exposure_time,
                                      self._data.phases,
                                      np.ascontiguousarray(self._data.counts),
                                      np.ascontiguousarray(self._data.errors),
                                      self._signals,
                                      self._phases,
                                      self._shifts,
                                      background,                                
                                      allow_negative = True)

class CustomSignal_poisson(xpsi.Signal):
    """ A custom calculation of the logarithm of the likelihood.

    We extend the :class:`xpsi.Signal.Signal` class to make it callable.

    We also implement data synthesis capability.

    """

    def __init__(self, workspace_intervals = 1000, epsabs = 0, epsrel = 1.0e-8,
                 epsilon = 1.0e-3, sigmas = 10.0, support = None, **kwargs):
        super(CustomSignal_poisson, self).__init__(**kwargs)
        self._precomp = precomputation(np.ascontiguousarray(self._data.counts).astype(np.int32))

    def __call__(self, *args, **kwargs):
        self.expected_background_counts = 0.
        self.empty_background = np.zeros(np.asarray(self._signals[0]).shape)  

        self.loglikelihood, self.expected_counts = \
            poisson_likelihood_given_background(self._data.exposure_time, 
                                                self._data.phases, 
                                                np.ascontiguousarray(self._data.counts),
                                                self._signals,
                                                self._phases,
                                                self._shifts,
                                                self.empty_background,
                                                self._precomp,
                                                allow_negative = False)


class CustomSignal_gaussian_no_instrument_response(xpsi.Signal):
    """

    NO INSTRUMENT RESPONSE. this was used for tests of simulated data.

    """

    def __init__(self, workspace_intervals = 1000, epsabs = 0, epsrel = 1.0e-8,
                 epsilon = 1.0e-3, sigmas = 10.0, support = None, **kwargs):
        """ Perform precomputation.

        :params ndarray[m,2] support:
            Prior support bounds for background count rate variables in the
            :math:`m` instrument channels, where the lower bounds must be zero
            or positive, and the upper bounds must be positive and greater than
            the lower bound. Alternatively, setting the an upper bounds as
            negative means the prior support is unbounded and the flat prior
            density functions per channel are improper. If ``None``, the lower-
            bound of the support for each channel is zero but the prior is
            unbounded.

        """

        try:
            self.stokes = kwargs['stokes']
        except KeyError:
            raise ValueError("Missing required keyword argument: 'stokes'")

        super(CustomSignal_gaussian, self).__init__(**kwargs)


    def compute_signal_data_phase(self, photosphere, fast_mode=False, threads=1):
        # does everything you'd normally do to register the signal, except no instrument response is actually being folded.
        anegI = (False)
        anegQU = (True)

        #This is for 1 spot:
        hot = photosphere.surface
        phase_mod = hot.phases_in_cycles[0]
        def shift_phase(phi,shift):
            if shift == 0: # because then phi=1 should remain phi=1, not phi=0. otherwise interpolation outside of interval.
                return phi
            return (phi + shift) % 1 

        def extend(x_base, y_base):
            # stick a duplicate to the left and right to allow interpolation at the range 0 to 1 after a phase shift between -0.25 to 0.75 was applied
            x_extended = np.concatenate([
                x_base[:-1] - 1,   # wraparound left
                x_base,       # original
                x_base[1:] + 1    # wraparound right
                ])
            y_extended = np.concatenate([
                y_base[:-1],
                y_base,
                y_base[1:]
                ])
            return x_extended, y_extended
        
        shifts = [h['phase_shift'] for h in hot.objects] 
        self.shifts = np.array(shifts)
  
        phase1 = shift_phase(phase_mod,self._shifts[0]) # underscore actually avoids instrumental phase shift here.
        # print('phase 1 new:',phase1)
        #phase1 = shift_phase(phase_mod,signals[0][0]._shifts[0]) # this would always take the first signal, but above code takes self (for IXPE IQU signals it makes no difference).
        #print('phase 1 old (NICER):',phase1)
        
        
        StokesI = photosphere.signal[0][0]
        StokesQ = photosphere.signalQ[0][0]
        StokesU = photosphere.signalU[0][0]

        signal_energies = self.energies

        # apply interstellar on signal
        self._interstellar(signal_energies, StokesI)
        self._interstellar(signal_energies, StokesQ)
        self._interstellar(signal_energies, StokesU)

        le = find_idx(signal_energies,2.0)
        he = find_idx(signal_energies,8.0)                 

        if self.isI: # If using just stokesI, you could argue it would be better to include the instrument response here also
            Imod1 = np.zeros((len(phase1)))
            for e in range(le,he):
                # integrate over signal energies
                Imod1[:] = Imod1[:] + (StokesI[e,:]+StokesI[e+1,:])*(signal_energies[e+1]-signal_energies[e])	
            Imod1 = 1/2*Imod1

      
            # interpolate to observed phases
            extend_p, extend_I = extend(phase1, Imod1)   
            I1i = interp1d(extend_p, extend_I)     
            
            # I1i = interp1d(phase1, Imod1)
            
            phase_data = self._data.phase_IXPE_pulse
            #phase_data = get_mids_from_edges(self._data.phases)
    
            
            sign1 = I1i(phase_data)
        
        elif self.isQ:
            
            Imod1 = np.zeros((len(phase1)))
            Qmod1 = np.zeros((len(phase1)))
            for e in range(le,he):
                Imod1[:] = Imod1[:] + (StokesI[e,:]+StokesI[e+1,:])*(signal_energies[e+1]-signal_energies[e])	
                Qmod1[:] = Qmod1[:] + (StokesQ[e,:]+StokesQ[e+1,:])*(signal_energies[e+1]-signal_energies[e])
            Imod1 = 1/2*Imod1
            Qmod1 = 1/2*Qmod1

            extend_p, extend_Q = extend(phase1, Qmod1) 
            extend_p, extend_I = extend(phase1, Imod1) 
            Q1i = interp1d(extend_p, extend_Q)
            # Q1i = interp1d(phase1, Qmod1)
            
            phase_data = self._data.phase_IXPE 
            sign1 = Q1i(phase_data)
        elif self.isU:
            
            Imod1 = np.zeros((len(phase1)))
            Umod1 = np.zeros((len(phase1)))   
            for e in range(le,he):
                Imod1[:] = Imod1[:] + (StokesI[e,:]+StokesI[e+1,:])*(signal_energies[e+1]-signal_energies[e])	
                Umod1[:] = Umod1[:] + (StokesU[e,:]+StokesU[e+1,:])*(signal_energies[e+1]-signal_energies[e])
            Imod1 = 1/2*Imod1
            Umod1 = 1/2*Umod1 
            
            extend_p, extend_U = extend(phase1, Umod1)
            extend_p, extend_I = extend(phase1, Imod1) 
            U1i = interp1d(extend_p, extend_U)
            # U1i = interp1d(phase1, Umod1)

            phase_data = self._data.phase_IXPE           
            sign1 = U1i(phase_data)


        if self.isQ or self.isU:
            I1i = interp1d(extend_p, extend_I)
            # I1i = interp1d(phase1, Imod1)
            
            Isign1 = I1i(phase_data) 
            signal_dphase = np.where(Isign1==0.0, 0.0, sign1/Isign1)
        elif self.isI:
            signal_dphase = sign1/np.max(Imod1)

        self.signal_dphase = signal_dphase


    def __call__(self, *args, **kwargs):
        self.loglikelihood, self.expected_counts = \
        gaussian_likelihood_QnUn(self._data.phases,
                                  self._data.counts,
                                  self._data.errors,
                                  self.signal_dphase)





class CustomSignal(xpsi.Signal):
    """ A custom calculation of the logarithm of the likelihood.

    We extend the :class:`xpsi.Signal.Signal` class to make it callable.

    We also implement data synthesis capability.

    """

    def __init__(self, 
                 workspace_intervals = 1000, 
                 epsabs = 0, 
                 epsrel = 1.0e-8,
                 epsilon = 1.0e-3, 
                 sigmas = 10.0, 
                 support = None, 
                 bkg = 'marginalised', 
                 stokes=False,
                 combine_unpulsed=True,
                 *args, 
                 **kwargs):
        """ Perform precomputation. """
        super(CustomSignal, self).__init__(*args, **kwargs)

        self.stokes = stokes
        self.bkg = bkg
        self._combine_unpulsed=combine_unpulsed
      

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
                # if not allow_negative_background:
                self._support[:,0] = 0.0
                

    @property
    def support(self):
        return self._support

    @support.setter
    def support(self, obj):
        self._support = obj

    def __call__(self, *args, **kwargs):
        self.expected_background_counts = 0.
        self.empty_background = np.zeros(np.asarray(self._signals[0]).shape)  
        
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
                                              kwargs.get('llzero'))#,
                                              # allow_negative_background = self.allow_negative_background)#,
                                              #slim=-1.0) # default is skipping 10^89s, so some likelihood calculations are skipped

        # if disk and line are stored separately, there phases are also separate. But this breaks postprocessing, sampling, and data synthesis.
        if not self._combine_unpulsed:        
            for key in ('disk', 'line'):
                if key in self.bkg:
                    self._phases.append(np.copy(self._phases[0]))
                    self._shifts = np.append(self._shifts, self._shifts[0])

        if 'disk' in self.bkg:
            self.loglikelihood, self.expected_counts = \
                poisson_likelihood_given_background(self._data.exposure_time, 
                                                    self._data.phases, 
                                                    self._data.counts,
                                                    self._signals,
                                                    self._phases,
                                                    self._shifts,
                                                    self.empty_background,
                                                    self._precomp, # temporary fix for posterior combiner
                                                    allow_negative = False)

        elif self.bkg == 'fix':
            self.loglikelihood, self.expected_counts = \
                poisson_likelihood_given_background(self._data.exposure_time, 
                                                    self._data.phases, 
                                                    self._data.counts,
                                                    self._signals,
                                                    self._phases,
                                                    self._shifts,
                                                    self.empty_background, #self.background_data,
                                                    self._precomp,
                                                    allow_negative = False)
        else:
            print('error in CustomSignal! pass bkg argument in init!')


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

    
    def synthesise(self,
                   exposure_time,
                   seed = 42,
                   name='no_pulse',
                   directory='./',
                   **kwargs):
        
            """ Synthesise data set.
    
            """

            self.expected_background_counts = 0.
            self.empty_background = np.zeros(np.asarray(self._signals[0]).shape)


            self._expected_counts, synthetic, scale_background = _synthesise(exposure_time,
                                                                       self._data.phases,
                                                                       self._signals,
                                                                       self._phases,
                                                                       self._shifts,
                                                                       self.expected_background_counts,
                                                                       self.empty_background,
                                                                       gsl_seed=seed)
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

            # POISSON NOISE, Floats
            # np.savetxt(os.path.join(directory, name+'_realisation.dat'),
            #             synthetic,
            #             fmt = '%f')
            
            #NO NOISE, FLOATS
            # np.savetxt(os.path.join(directory, name+'_realisation.dat'),
            #             self._expected_counts,
            #             fmt = '%f')
            
            # # NO NOISE, WHOLE COUNTS
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