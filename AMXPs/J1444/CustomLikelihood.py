#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 11:07:31 2025

@author: bas
"""

import xpsi
from xpsi.global_imports import *


class CustomLikelihood(xpsi.Likelihood):

    def _driver(self, fast_mode=False, synthesise=False, force_update=False, **kwargs):
        """ Main likelihood evaluation driver routine. """

        self._star.activate_fast_mode(fast_mode)

        star_updated = False
        if self._star.needs_update or force_update: # ignore fast parameters in this version
            try:
                if fast_mode or not self._do_fast:
                    fast_total_counts = None
                else:
                    fast_total_counts = tuple(signal.fast_total_counts for\
                                                        signal in self._signals)

                self._star.update(fast_total_counts, self.threads,force_update=force_update)

            except xpsiError as e:
                if isinstance(e, HotRegion.RayError):
                    print('Warning: HotRegion.RayError raised.')
                elif isinstance(e, Elsewhere.RayError):
                    print('Warning: Elsewhere.RayError raised.')


                return self.random_near_llzero

            for photosphere, signals in zip(self._star.photospheres, self._signals):
                try:
                    if fast_mode:
                        energies = signals[0].fast_energies
                    else:
                        energies = signals[0].energies

                    photosphere.integrate(energies, self.threads)
                except xpsiError as e:
                    try:
                        prefix = ' prefix ' + photosphere.prefix
                    except AttributeError:
                        prefix = ''
                    if isinstance(e, HotRegion.PulseError):
                        print('Warning: HotRegion.PulseError raised for '
                              'photosphere%s.' % prefix)
                    elif isinstance(e, Elsewhere.IntegrationError):
                        print('Warning: Elsewhere.IntegrationError for '
                              'photosphere%s.' % prefix)
                    elif isinstance(e, HotRegion.AtmosError):
                        raise
                    elif isinstance(e, Elsewhere.AtmosError):
                        raise

                    print('Parameter vector: ', super(Likelihood,self).__call__())
                    return self.random_near_llzero

            star_updated = True

        # register the signals by operating with the instrument response
        for signals, photosphere in zip(self._signals, self._star.photospheres):
            for signal in signals:
                if star_updated or signal.needs_update:
                    if signal.stokes in ["I", "Q", "U", "Qn", "Un"]: # note that stokes="I" can be handled also in IXPE, but for now I am passing that option to "regular" signals. I think there should be a stokes=False option.
                        signal.compute_signal_data_phase(photosphere, fast_mode=fast_mode, threads=self.threads)
                    elif not signal.stokes:
                         signal.register(tuple(
                                           tuple(self._divide(component,
                                                       self._star.spacetime.d_sq)
                                                 for component in hot_region)
                                           for hot_region in photosphere.signal),
                                     fast_mode=fast_mode, threads=self.threads)
                    reregistered = True
                else:
                    reregistered=False

                if not fast_mode and reregistered:
                    if synthesise:
                        hot = photosphere.surface
                        try:
                            kws = kwargs.pop(signal.prefix)
                        except AttributeError:
                            kws = {}

                        shifts = [h['phase_shift'] for h in hot.objects]
                        signal.shifts = _np.array(shifts)
                        signal.synthesise(threads=self._threads, **kws)
                    else:
                        try:
                            hot = photosphere.surface
                            shifts = [h['phase_shift'] for h in hot.objects]
                            signal.shifts = _np.array(shifts)

                            signal(threads=self._threads, llzero=self._llzero)
                        except LikelihoodError:
                            try:
                                prefix = ' prefix ' + signal.prefix
                            except AttributeError:
                                prefix = ''
                            print('Warning: LikelihoodError raised for '
                                  'signal%s.' % prefix)
                            print('Parameter vector: ', super(Likelihood,self).__call__())
                            return self.random_near_llzero
                      
                        # hot = photosphere.surface
                        # shifts = [h['phase_shift'] for h in hot.objects]
                        # signal.shifts = _np.array(shifts)

                        # signal(threads=self._threads,  llzero=self._llzero)
                        # try:
                        #     prefix = ' prefix ' + signal.prefix
                        # except AttributeError:
                        #     prefix = ''
                        # print('Warning: LikelihoodError raised for '
                        #       'signal%s.' % prefix)
                        # print('Parameter vector: ', super(Likelihood,self).__call__())
                        # return self.random_near_llzero

        return star_updated