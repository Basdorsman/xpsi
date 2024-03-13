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

from xpsi.tools.background_integrator import disk_f
from helper_functions import get_mids_from_edges

class CustomBackground_DiskBB(xpsi.Background):
    """ The background injected to generate synthetic data. """

    def __init__(self, bounds=None, values=None, interstellar = None):
        self.interstellar = interstellar
        
        doc = """
        Temperature at inner disk radius in log10 Kelvin.
        """
        inner_temperature = xpsi.Parameter('T_in',
                                strict_bounds = (3., 10.),
                                bounds = bounds.get('T_in', None),
                                doc = doc,
                                symbol = r'$T_{in}$',
                                value = values.get('T_in', None))

        doc = """
        Disk R_in in kilometers.
        """
        inner_radius = xpsi.Parameter('R_in',
                                strict_bounds = (0., 1e3),
                                bounds = bounds.get('R_in', None),
                                doc = doc,
                                symbol = r'$R_{in}$',
                                value = values.get('R_in', None))

        
        doc = """
        Disk normalisation cos_i*R_in^2/D^2 in (km / 10 kpc)^2.
        """
        background_normalisation = xpsi.Parameter('K_disk',
                                strict_bounds = (0., 1e8),
                                bounds = bounds.get('K_disk', None),
                                doc = doc,
                                symbol = r'$K_{BB}$',
                                value = values.get('K_disk', None))
        

        super(CustomBackground_DiskBB, self).__init__(inner_temperature, inner_radius, background_normalisation)
        

    def __call__(self, energy_edges, phases):#, interstellar = None):
        """ Compute F disk and set it as incident background. 
        
        F_disk has units photons/s/cm^2 in each energy bin (defined through 
        energy_edges) in each phase bin (i.e. phases). F_disk is 
        constant, but anyways to get the total photons you add each phase
        contribution and multiply by exposure time to get total photons/cm^2.
        
        """
        
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
            diskbb_flux_integral = disk_f(energy_edges[i], energy_edges[i+1], T_in_keV, T_out_keV, epsrel) 
            # disk_f is the cythonised version of l_disk and is faster.
            flux_integral_array=np.append(flux_integral_array,diskbb_flux_integral)
        
        # K_disk is cos_i*R_in^2/D^2 in (km / 10 kpc)^2.
        # (1 km / 10 kpc)^2 = 1.0502650e-35 [ cm/cm ]
        
        flux_integral_array *=K_disk*4*np.pi/3*1.0502650e-35 # photons/s/cm^2/energy_bin
        
        BB = np.zeros((energy_edges.shape[0] - 1, n_phase_bins)) 
        # I looked at the synthesise.pyx and it is actually expecting 
        # phase_edges as an input while expecting registered background (and 
        # also incident background, see e.g. Signal.py) to have phase_bins.

        for i in range(n_phase_bins):   
            BB[:,i] = flux_integral_array/n_phase_bins
            
        bkg=BB
        self._incident_background = bkg
        
        
    def get_f_disk(self, energies, spectral_radiance, attenuate = False):#, interstellar = None):
        """ Evaluate f_disk(energies).
        
        f_disk(E) = 4/3*pi * K_disk * l_disk(b_E/B_E, E)
        Ref: Mitsuda et al. 1984, Makishima et al. 1986
        But note that Mitsuda et al. 1984 has an extra factor here because they
        don't have it in the definition for b_E/B_E.
        
        parameters
        energies[keV]
        spectral_radiance can be: b_E or B_E
        attenuate determines whether to apply interstellar medium attenuation.
        
        returns
        f_disk [photons/s/cm^2/keV] or [keV/s/cm^2/keV]
        
        """
        

        T_in = self['T_in']
        K_disk = self['K_disk']

        # KbT in keV
        T_in_keV = k_B_over_keV * pow(10.0, T_in)
        
        T_out_keV = T_in_keV*1e-1
        
        epsrel = 1e-4

        f_disk_array = np.array([]) #photons/s/cm^2/sr or keV/s/cm^2/sr 
        for energy in energies:
            f_disk_value = self.l_disk(energy, T_in_keV, T_out_keV, spectral_radiance, epsrel) 
            f_disk_array=np.append(f_disk_array,f_disk_value)
        
        # K_disk is cos_i*R_in^2/D^2 in (km / 10 kpc)^2.
        # (1 km / 10 kpc)^2 = 1.0502650e-35 [ cm/cm ]
        
        f_disk_array *=K_disk*4*np.pi/3*1.0502650e-35 # photons/s/cm^2/energy_bin
        
        
            # Apply Interstellar if not None
        if self.interstellar is not None and attenuate:
            self.interstellar(energies, f_disk_array) # bkg is overwritten here
        
        return f_disk_array
        
    def get_F_disk(self, energy_edges, spectral_radiance, attenuate = False):
        """ Evaluate F_disk(energy_bins, phase_bins).
        
        F_disk(E) = 4/3*pi * K_disk * int [l_disk(b_E/B_E, E)] dE, in this case
        we integrate to get the flux from edge to edge to get the flux in each
        energy bin. 
        
        parameters
        energy_edges[keV], the edges from the bins
        spectral_radiance can be b_E or B_E
        
        returns
        F_disk in each energy bin. [photons/s/cm^2] or 
        [keV/s/cm^2] 
        
        """

        
        T_in = self['T_in']
        K_disk = self['K_disk']

        # KbT in keV
        T_in_keV = k_B_over_keV * pow(10.0, T_in)
        
        T_out_keV = T_in_keV*1e-1
        
        epsrel = 1e-4

        F_disk_array = np.array([]) #photons/s/cm^2/sr/energy_bin
        for i in range(len(energy_edges)-1):
            # diskbb_flux_integral = disk_f(energy_edges[i], energy_edges[i+1], T_in_keV, T_out_keV, epsrel)
            F_disk_value,_ = quad(self.l_disk, energy_edges[i], energy_edges[i+1], args=(T_in_keV, T_out_keV, spectral_radiance, epsrel), epsrel=epsrel)
            F_disk_array=np.append(F_disk_array,F_disk_value)
        
        # K_disk is cos_i*R_in^2/D^2 in (km / 10 kpc)^2.
        # (1 km / 10 kpc)^2 = 1.0502650e-35 [ cm/cm ]
        
        F_disk_array *=K_disk*4*np.pi/3*1.0502650e-35 # photons/s/cm^2/energy_bin
        
        # Apply Interstellar if not None
        if self.interstellar is not None and attenuate:
            energy_mids = get_mids_from_edges(energy_edges)
            self.interstellar(energy_mids, F_disk_array) # bkg is overwritten here
        
        return F_disk_array


    def b_E(self, E, T):
        '''
        photon radiance of a blackbody

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
        Energy radiance of a blackbody.

        parameters:
            E in keV
            T in keV

        returns:
            B_E in keV/s/keV/cm^2/sr (you will integrate over keV)
        '''
        
        B = 2*E**3/(_h_keV**3*_c_cgs**2)/(np.exp(E/T)-1)
        return B


    def l_disk_integrand(self, T, E, T_in, spectral_radiance):
        '''
        parameters:
            T, T_in in keV
            E in keV

        returns:
            integrand in spectral radiance units/keV. This integrand will 
            be integrated over keV.
        '''

        # print('T: ', T)
        # print('E:', E)
        # print('T_in: ', T_in)
        # print('(T/T_in)**(-11/3)', (T/T_in)**(-11/3))
        # print('spectral_radiance(E, T)/T_in', spectral_radiance(E, T)/T_in)

        integrand = (T/T_in)**(-11/3)*spectral_radiance(E, T)/T_in
        return integrand
    
    def l_disk(self, E, T_in, T_out, spectral_radiance, epsrel):
        '''
        parameters:
            T, T_in in keV
            E in keV

        returns:
            disk luminosity [spectral radiance units]. 
        '''

        disk_luminosity,_= quad(self.l_disk_integrand, T_out, T_in, args=(E, T_in, spectral_radiance), epsrel=epsrel)
        return disk_luminosity
    
    
def get_k_disk(cos_i, r_in, distance):
  """
  This function calculates the k-disk value for a given set of input parameters.

  Args:
      cos_i: The cosine inclination angle of the disk, can be a scalar or a tuple.
      r_in: The inner radius of the disk in kilometers, can be a scalar or a tuple.
      distance: The distance to the disk in kiloparsecs, can be a scalar or a tuple.

  Returns:
      A tuple containing the k-disk values for each element in the input parameters.

  Raises:
      ValueError: If the input tuples have different lengths.
  """

  if isinstance(cos_i, tuple) and isinstance(r_in, tuple) and isinstance(distance, tuple):
    if len(cos_i) != len(r_in) or len(cos_i) != len(distance):
      raise ValueError("Input tuples must have the same length.")
    # Use a loop instead of recursion
    k_disk_values = []
    for c, r, d in zip(cos_i, r_in, distance):
      k_disk_values.append(c * (r / (d / 10))**2)
    return tuple(k_disk_values)
  else:
    return cos_i * (r_in / (distance / 10))**2

class k_disk_derive(xpsi.Derive):
    def __init__(self):
        pass

    def __call__(self, boundto, caller = None):
        # ref is a reference to another hot region object
        return get_k_disk(self.star['cos_inclination'], self.background['R_in'], self.star['distance'])