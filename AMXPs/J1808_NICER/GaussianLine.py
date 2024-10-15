#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:46:40 2024

@author: bas
"""

from xpsi.ParameterSubspace import ParameterSubspace
from xpsi.Parameter import Parameter
import numpy as np


class GaussianLine(ParameterSubspace):
    def __init__(self, bounds=None, values=None):

        doc = """
        mean value (keV)
        """
        mu = Parameter('mu',
                                strict_bounds = (1e-10,1e10),
                                bounds = bounds.get('mu', None),
                                doc = doc,
                                symbol = r'$\mu$',
                                value = values.get('mu', None))

        doc = """
        standard deviation (keV)
        """
        sigma = Parameter('sigma',
                                strict_bounds = (1e-10,1e10),
                                bounds = bounds.get('sigma', None),
                                doc = doc,
                                symbol = r'$\sigma$',
                                value = values.get('sigma', None))

        
        doc = """
        Line normalisation (photons/s/cm^2)
        """
        line_normalisation = Parameter('N',
                                strict_bounds = (1e-10,1e50),
                                bounds = bounds.get('N', None),
                                doc = doc,
                                symbol = r'N',
                                value = values.get('N', None))
        

        super(GaussianLine, self).__init__(mu, sigma, line_normalisation)
        
    def __call__(self, energies):
        self.line_flux = self.get_line_flux(energies)
        return self.line_flux
    
    def get_line_flux(self, energies):
        mu = self['mu']
        sigma = self['sigma']
        N = self['N']
    
        flux = N*np.exp(-(energies-mu*np.ones(energies.shape))**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

        
        return flux
        
    
    
if __name__ == '__main__':
    line_bounds = dict(
        mu=(0.001,1000.),
        sigma=(0.001,1000.),
        N=(0.001,1000.)
        )
    
    
    line_values = dict(
        mu=3,
        sigma=1,
        N=1)
    
    
    MyLine = GaussianLine(bounds=line_bounds, values=line_values)
    energies= np.linspace(0,6,100)
    # print(energies)
    flux = MyLine(energies)
    import matplotlib.pyplot as plt
    plt.plot(energies, flux)
    