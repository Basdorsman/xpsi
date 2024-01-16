#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:59:51 2023

@author: bas
"""

import xpsi
from xpsi.global_imports import  _keV, _k_B, _c_cgs, _h_keV
k_B_over_keV = _k_B / _keV
import numpy as np
from scipy.integrate import quad

class CustomBackground_DiskBB(xpsi.Background):
    """ The background injected to generate synthetic data. """

    def __init__(self, bounds=None, values=None, interstellar = None):
        
        doc = """
        Temperature at inner disk radius in log10 Kelvin.
        """
        inner_temperature = xpsi.Parameter('T_in',
                                strict_bounds = (3., 10.),
                                bounds = bounds,
                                doc = doc,
                                symbol = r'$T_{in}$',
                                value = values.get('T_in', None))
        
        doc = """
        Disk normalisation cos_i*R_in^2/D^2 in (km / 10 kpc)^2.
        """
        background_normalisation = xpsi.Parameter('K_disk',
                                strict_bounds = (0., 1e8),
                                bounds = bounds,
                                doc = doc,
                                symbol = r'$K_{BB}$',
                                value = values.get('K_disk', None))
        

        super(CustomBackground_DiskBB, self).__init__(inner_temperature, background_normalisation)
        
        # Making sure the interstellar object is form Interstall class
        if interstellar is not None:
            if not isinstance(interstellar, xpsi.Interstellar):
                raise TypeError('Invalid type for an interstellar object.')
            else:
                self._interstellar = interstellar
        else:
            self._interstellar = None

    def __call__(self, energy_edges, phases):
        """ Evaluate the incident background field. """
        
        n_phase_edges = phases.shape[0]
        n_phase_bins = n_phase_edges - 1
        

        T_in = self['T_in']
        K_disk = self['K_disk']

        # KbT in keV
        T_in_keV = k_B_over_keV * pow(10.0, T_in)
        
        T_out_keV = T_in_keV*1e-1
        
        epsrel = 1e-4

        flux_integral_array = np.array([]) #photons/s/cm^2/sr/energy_bin
        for i in range(len(energy_edges)-1):
            diskbb_flux_integral,_ = quad(self.diskbb_flux, energy_edges[i], energy_edges[i+1], args=(T_in_keV, T_out_keV, self.b_E, epsrel), epsrel=epsrel)
            flux_integral_array=np.append(flux_integral_array,diskbb_flux_integral)
        
        # K_disk is cos_i*R_in^2/D^2 in (km / 10 kpc)^2.
        # (1 km / 10 kpc)^2 = 1.0502650e-35 [ cm/cm ]
        
        flux_integral_array *=K_disk*4*np.pi/3*1.0502650e-35 # photons/s/cm^2/energy_bin
        
        BB = np.zeros((energy_edges.shape[0] - 1, n_phase_bins)) # I looked at the synthesise.pyx and it is actually expecting phase_edges as an input while expecting registered background to have phase_bins.

        for i in range(n_phase_bins):   
            BB[:,i] = flux_integral_array/n_phase_bins
            
        bkg=BB
        
        # Apply Interstellar if not None
        if self._interstellar is not None:
            self._energy_mids=(energy_edges[1:]+energy_edges[:-1])/2
            self._interstellar(self._energy_mids, bkg) # bkg is overwritten here
            

        self._incident_background = bkg

    def b_E(self, E, T):
        '''
        photons/s/keV/cm^2/sr (radiance type thing) of a blackbody

        parameters:
            E in keV
            T in keV

        returns:
            b_E in photons/s/keV/cm^2/sr 
        '''

        b = 2*E**2/(_h_keV**3*_c_cgs**2)/(np.exp(E/T)-1)
        return b
        
        
    def B_E(self, E, T):
        '''
        Spectral radiance type thing of a blackbody.

        parameters:
            E in keV
            T in keV

        returns:
            B_E in keV/s/keV/cm^2/sr (you will integrate over keV)
        '''
        
        B = 2*E**3/(_h_keV**3*_c_cgs**2)/(np.exp(E/T)-1)
        return B


    def diskbb_integrand(self, T, E, T_in, spectral_radiance):
        '''
        parameters:
            T, T_in in keV
            E in keV

        returns:
            integrand in spectral radiance units/keV (you will integrate over keV)
        '''

        integrand = (T/T_in)**(-11/3)*spectral_radiance(E, T)/T_in
        return integrand
    
    def diskbb_flux(self, E, T_in, T_out, spectral_radiance, epsrel):
        '''
        parameters:
            T, T_in in keV
            E in keV

        returns:
            integrated flux spectral radiance units (you have integrated over keV)
        '''
        
        flux_integral,_= quad(self.diskbb_integrand, T_out, T_in, args=(E, T_in, spectral_radiance), epsrel=epsrel)
        return flux_integral