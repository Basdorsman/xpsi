#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:26:35 2022

@author: bas
"""
import xpsi
from xpsi.global_imports import *
from xpsi.global_imports import _c, _G, _dpr, gravradius, _csq, _km, _2pi
from xpsi.Parameter import Parameter


import numpy as np
import math


class CustomInstrument(xpsi.Instrument):
    """ A model of the NICER telescope response. """

    def __call__(self, signal, *args):
        """ Overwrite base just to show it is possible.

        We loaded only a submatrix of the total instrument response
        matrix into memory, so here we can simplify the method in the
        base class.

        """
        matrix = self.construct_matrix()

        self._folded_signal = np.dot(matrix, signal)

        return self._folded_signal

    @classmethod
    def from_response_files(cls, ARF, RMF, max_input, min_input=0,
                            channel_edges=None):
        """ Constructor which converts response files into :class:`numpy.ndarray`s.
        :param str ARF: Path to ARF which is compatible with
                                :func:`numpy.loadtxt`.
        :param str RMF: Path to RMF which is compatible with
                                :func:`numpy.loadtxt`.
        :param str channel_edges: Optional path to edges which is compatible with
                                  :func:`numpy.loadtxt`.
        """

        if min_input != 0:
            min_input = int(min_input)

        max_input = int(max_input)

        try:
            ARF = np.loadtxt(ARF, dtype=np.double, skiprows=3)
            RMF = np.loadtxt(RMF, dtype=np.double)
            if channel_edges:
                channel_edges = np.loadtxt(channel_edges, dtype=np.double, skiprows=3)[:,1:]
        except:
            print('A file could not be loaded.')
            raise

        matrix = np.ascontiguousarray(RMF[min_input:max_input,20:201].T, dtype=np.double)

        edges = np.zeros(ARF[min_input:max_input,3].shape[0]+1, dtype=np.double)

        edges[0] = ARF[min_input,1]; edges[1:] = ARF[min_input:max_input,2]

        for i in range(matrix.shape[0]):
            matrix[i,:] *= ARF[min_input:max_input,3]

        channels = np.arange(20, 201)

        return cls(matrix, edges, channels, channel_edges[20:202,-2])

class CustomHotRegion(xpsi.HotRegion):
    """Custom implementation of HotRegion"""

    required_names = ['super_colatitude',
                      'super_radius',
                      'phase_shift',
                      'super_temperature (if no custom specification)']
    optional_names = ['modulator',
                      'omit_colatitude',
                      'omit_radius',
                      'omit_azimuth',
                      'cede_colatitude',
                      'cede_radius',
                      'cede_azimuth',
                      'cede_temperature']
    
    def __init__(self,
            bounds,
            values,
            symmetry = True,
            omit = False,
            cede = False,
            concentric = False,
            sqrt_num_cells = 32,
            min_sqrt_num_cells = 10,
            max_sqrt_num_cells = 80,
            num_rays = 200,
            num_leaves = 64,
            num_phases = None,
            phases = None,
            do_fast = False,
            fast_sqrt_num_cells = 16,
            fast_min_sqrt_num_cells = 4,
            fast_max_sqrt_num_cells = 16,
            fast_num_rays = 100,
            fast_num_leaves = 32,
            fast_num_phases = None,
            fast_phases = None,
            is_antiphased = False,
            custom = None,
            image_order_limit = None,
            modulated = False,
            **kwargs
            ):

        doc = """
        log10(superseding region effective temperature [K])
        """
        super_temperature = Parameter('super_temperature',
  		    strict_bounds = (3.0, 7.0), # very cold --> very hot
  		    bounds = bounds.get('super_temperature', None),
  		    doc = doc,
  		    symbol = r'$\log_{10}(T\;[\rm{K}])$',
  		    value = values.get('super_temperature', None))
      #   if cede:
      #       doc = """
    		# log10(ceding region effective temperature [K])
    		# """
      #       cede_temp = Parameter('cede_temperature',
    		# 	      strict_bounds = (3.0, 7.0), # same story
    		# 	      bounds = bounds.get('cede_temperature', None),
    		# 	      doc = doc,
    		# 	      symbol = r'$\log_{10}(\mathcal{T}\;[\rm{K}])$',
    		# 	      value = values.get('cede_temperature', None))
      #   else:
      #       cede_temp = None

        if not modulated: 
            print("not modulated")
            bounds['super_modulator'] = None
            values['super_modulator'] = 0.0    
        
        doc = """
        [Modulator] adds factor to log10(intensity), which is intensity listed in NSX file
        format.
        """
        super_modulator = Parameter('super_modulator',
                    strict_bounds = (-0.3, 0.3),
                    bounds = bounds.get('super_modulator', None),
                    doc = doc,
                    symbol = r'$modulator$',
                    value = values.get('super_modulator', None))
        
        # print("modulated")
        # print(super_modulator)
        

        custom = [super_temperature, super_modulator]
        

        super(CustomHotRegion, self).__init__(
                bounds,
                values,
                symmetry = symmetry,
                omit = omit,
                cede = cede,
                concentric = concentric,
                sqrt_num_cells = sqrt_num_cells,
                min_sqrt_num_cells = min_sqrt_num_cells,
                max_sqrt_num_cells = max_sqrt_num_cells,
                num_rays = num_rays,
                num_leaves = num_leaves,
                num_phases = num_phases,
                phases = phases,
                do_fast = do_fast,
                fast_sqrt_num_cells = fast_sqrt_num_cells,
                fast_min_sqrt_num_cells = fast_min_sqrt_num_cells,
                fast_max_sqrt_num_cells = fast_max_sqrt_num_cells,
                fast_num_rays = fast_num_rays,
                fast_num_leaves = fast_num_leaves,
                fast_num_phases = fast_num_phases,
                fast_phases = fast_phases,
                is_antiphased = is_antiphased,
                custom = custom,
                #modulated=super_modulator,
                image_order_limit = image_order_limit,
                **kwargs
                )        
        
    
    def _HotRegion__compute_cellParamVecs(self):
        self._super_radiates = _np.greater(self._super_cellArea, 0.0).astype(_np.int32)
        self._super_cellParamVecs = _np.ones((self._super_radiates.shape[0],
                                      self._super_radiates.shape[1],
                                      #2
                                      3),
                                     dtype=_np.double)
        
        #self._super_cellParamVecs[...,:-1] *= self['super_temperature']
        self._super_cellParamVecs[...,0] *= self['super_temperature']
        self._super_cellParamVecs[...,1] *= self['super_modulator']
        
        
        
        for i in range(self._super_cellParamVecs.shape[1]):
            self._super_cellParamVecs[:,i,-1] *= self._super_effGrav
        
        try:
            self._cede_radiates = _np.greater(self._cede_cellArea, 0.0).astype(_np.int32)
        except AttributeError:
            pass
        else:
            self._cede_cellParamVecs = _np.ones((self._cede_radiates.shape[0],
                                         self._cede_radiates.shape[1],
                                         2), dtype=_np.double)
        
            self._cede_cellParamVecs[...,:-1] *= self['cede_temperature']
        
            for i in range(self._cede_cellParamVecs.shape[1]):
                self._cede_cellParamVecs[:,i,-1] *= self._cede_effGrav
    

class CustomHotRegion_Accreting(xpsi.HotRegion):
    """Custom implementation of HotRegion. Accreting Atmosphere model by 
    Anna Bobrikova. The parameters are ordered I(E < mu < tau < tbb < te).
    
    E is energy.
    mu is cos of zenith angle.
    tau is the optical depth of the comptom slab.
    tbb is the black body temperature.
    te is temperature of the electron gas.
    """

    required_names = ['super_colatitude',
                      'super_radius',
                      'phase_shift',
                      'super_tbb',
                      'super_te',
                      'super_tau']
    optional_names = ['omit_colatitude',
                      'omit_radius',
                      'omit_azimuth',
                      'cede_colatitude',
                      'cede_radius',
                      'cede_azimuth',
                      'cede_temperature']
    
    def __init__(self,
            bounds,
            values,
            symmetry = True,
            omit = False,
            cede = False,
            concentric = False,
            sqrt_num_cells = 32,
            min_sqrt_num_cells = 10,
            max_sqrt_num_cells = 80,
            num_rays = 200,
            num_leaves = 64,
            num_phases = None,
            phases = None,
            do_fast = False,
            fast_sqrt_num_cells = 16,
            fast_min_sqrt_num_cells = 4,
            fast_max_sqrt_num_cells = 16,
            fast_num_rays = 100,
            fast_num_leaves = 32,
            fast_num_phases = None,
            fast_phases = None,
            is_antiphased = False,
            custom = None,
            image_order_limit = None,
            **kwargs
            ):

        doc = """
        tbb
        """
        super_tbb = Parameter('super_tbb',
  		    strict_bounds = (0.00015, 0.003), # this one is non-physical, we went for way_to_low Tbbs here, I will most probably delete results from too small Tbbs. This is Tbb(keV)/511keV, so these correspond to 0.07 - 1.5 keV, but our calculations don't work correctly for Tbb<<0.5 keV
  		    bounds = bounds.get('super_tbb', None),
  		    doc = doc,
  		    symbol = r'tbb',
  		    value = values.get('super_tbb', None))

        doc = """
        te
        """
        super_te = Parameter('super_te',
                    strict_bounds = (40., 200.), #actual range is 40-200 imaginaty units, ~20-100 keV (Te(keV)*1000/511keV is here)
                    bounds = bounds.get('super_te', None),
                    doc = doc,
                    symbol = r'te',
                    value = values.get('super_te', None))
        
        doc = """
        tau
        """
        super_tau = Parameter('super_tau',
                    strict_bounds = (0.5, 3.5),
                    bounds = bounds.get('super_tau', None),
                    doc = doc,
                    symbol = r'tau',
                    value = values.get('super_tau', None))
        

        custom = [super_tbb, super_te, super_tau]
        

        super(CustomHotRegion_Accreting, self).__init__(
                bounds,
                values,
                symmetry = symmetry,
                omit = omit,
                cede = cede,
                concentric = concentric,
                sqrt_num_cells = sqrt_num_cells,
                min_sqrt_num_cells = min_sqrt_num_cells,
                max_sqrt_num_cells = max_sqrt_num_cells,
                num_rays = num_rays,
                num_leaves = num_leaves,
                num_phases = num_phases,
                phases = phases,
                do_fast = do_fast,
                fast_sqrt_num_cells = fast_sqrt_num_cells,
                fast_min_sqrt_num_cells = fast_min_sqrt_num_cells,
                fast_max_sqrt_num_cells = fast_max_sqrt_num_cells,
                fast_num_rays = fast_num_rays,
                fast_num_leaves = fast_num_leaves,
                fast_num_phases = fast_num_phases,
                fast_phases = fast_phases,
                is_antiphased = is_antiphased,
                custom = custom,
                image_order_limit = image_order_limit,
                **kwargs
                )        
        
    
    def _HotRegion__compute_cellParamVecs(self):
        self._super_radiates = _np.greater(self._super_cellArea, 0.0).astype(_np.int32)
        self._super_cellParamVecs = _np.ones((self._super_radiates.shape[0],
                                      self._super_radiates.shape[1],
                                      #2
                                      3),
                                     dtype=_np.double)
        
        #self._super_cellParamVecs[...,:-1] *= self['super_temperature']
        self._super_cellParamVecs[...,0] *= self['super_te']
        self._super_cellParamVecs[...,1] *= self['super_tbb']
        self._super_cellParamVecs[...,2] *= self['super_tau']
        
        
        # for i in range(self._super_cellParamVecs.shape[1]):
        #     self._super_cellParamVecs[:,i,-1] *= self._super_effGrav
        
        # try:
        #     self._cede_radiates = _np.greater(self._cede_cellArea, 0.0).astype(_np.int32)
        # except AttributeError:
        #     pass
        # else:
        #     self._cede_cellParamVecs = _np.ones((self._cede_radiates.shape[0],
        #                                  self._cede_radiates.shape[1],
        #                                  2), dtype=_np.double)
        
        #     self._cede_cellParamVecs[...,:-1] *= self['cede_temperature']
        
        #     for i in range(self._cede_cellParamVecs.shape[1]):
        #         self._cede_cellParamVecs[:,i,-1] *= self._cede_effGrav

class CustomHotRegion_Accreting_te_const(xpsi.HotRegion):
    """Custom implementation of HotRegion. Accreting Atmosphere model by 
    Anna Bobrikova. The parameters are ordered I(E < mu < tau < tbb < te).
    
    E is energy.
    mu is cos of zenith angle.
    tau is the optical depth of the comptom slab.
    tbb is the black body temperature.
    te is temperature of the electron gas.
    """

    required_names = ['super_colatitude',
                      'super_radius',
                      'phase_shift',
                      'super_tbb',
                      'super_tau'] #te is gone!
    optional_names = ['omit_colatitude',
                      'omit_radius',
                      'omit_azimuth',
                      'cede_colatitude',
                      'cede_radius',
                      'cede_azimuth',
                      'cede_temperature']
    
    def __init__(self,
            bounds,
            values,
            symmetry = True,
            omit = False,
            cede = False,
            concentric = False,
            sqrt_num_cells = 32,
            min_sqrt_num_cells = 10,
            max_sqrt_num_cells = 80,
            num_rays = 200,
            num_leaves = 64,
            num_phases = None,
            phases = None,
            do_fast = False,
            fast_sqrt_num_cells = 16,
            fast_min_sqrt_num_cells = 4,
            fast_max_sqrt_num_cells = 16,
            fast_num_rays = 100,
            fast_num_leaves = 32,
            fast_num_phases = None,
            fast_phases = None,
            is_antiphased = False,
            custom = None,
            image_order_limit = None,
            **kwargs
            ):

        doc = """
        tbb
        """
        super_tbb = Parameter('super_tbb',
  		    strict_bounds = (0.00015, 0.003), # this one is non-physical, we went for way_to_low Tbbs here, I will most probably delete results from too small Tbbs. This is Tbb(keV)/511keV, so these correspond to 0.07 - 1.5 keV, but our calculations don't work correctly for Tbb<<0.5 keV
  		    bounds = bounds.get('super_tbb', None),
  		    doc = doc,
  		    symbol = r'tbb',
  		    value = values.get('super_tbb', None))

        # doc = """
        # te
        # """
        # super_te = Parameter('super_te',
        #             strict_bounds = (40., 200.), #actual range is 40-200 imaginaty units, ~20-100 keV (Te(keV)*1000/511keV is here)
        #             bounds = bounds.get('super_te', None),
        #             doc = doc,
        #             symbol = r'te',
        #             value = values.get('super_te', None))
        
        doc = """
        tau
        """
        super_tau = Parameter('super_tau',
                    strict_bounds = (0.5, 3.5),
                    bounds = bounds.get('super_tau', None),
                    doc = doc,
                    symbol = r'tau',
                    value = values.get('super_tau', None))
        

        custom = [super_tbb, super_tau] #[super_tbb, super_te, super_tau]
        

        super(CustomHotRegion_Accreting_te_const, self).__init__(
                bounds,
                values,
                symmetry = symmetry,
                omit = omit,
                cede = cede,
                concentric = concentric,
                sqrt_num_cells = sqrt_num_cells,
                min_sqrt_num_cells = min_sqrt_num_cells,
                max_sqrt_num_cells = max_sqrt_num_cells,
                num_rays = num_rays,
                num_leaves = num_leaves,
                num_phases = num_phases,
                phases = phases,
                do_fast = do_fast,
                fast_sqrt_num_cells = fast_sqrt_num_cells,
                fast_min_sqrt_num_cells = fast_min_sqrt_num_cells,
                fast_max_sqrt_num_cells = fast_max_sqrt_num_cells,
                fast_num_rays = fast_num_rays,
                fast_num_leaves = fast_num_leaves,
                fast_num_phases = fast_num_phases,
                fast_phases = fast_phases,
                is_antiphased = is_antiphased,
                custom = custom,
                image_order_limit = image_order_limit,
                **kwargs
                )        
        
    
    def _HotRegion__compute_cellParamVecs(self):
        self._super_radiates = _np.greater(self._super_cellArea, 0.0).astype(_np.int32)
        self._super_cellParamVecs = _np.ones((self._super_radiates.shape[0],
                                      self._super_radiates.shape[1],
                                      2),
                                     dtype=_np.double)
        
        #self._super_cellParamVecs[...,:-1] *= self['super_temperature']
        
        # self._super_cellParamVecs[...,0] *= self['super_te']
        # self._super_cellParamVecs[...,1] *= self['super_tbb']
        # self._super_cellParamVecs[...,2] *= self['super_tau']
        
        self._super_cellParamVecs[...,0] *= self['super_tbb']
        self._super_cellParamVecs[...,1] *= self['super_tau']
        
        
        # for i in range(self._super_cellParamVecs.shape[1]):
        #     self._super_cellParamVecs[:,i,-1] *= self._super_effGrav
        
        # try:
        #     self._cede_radiates = _np.greater(self._cede_cellArea, 0.0).astype(_np.int32)
        # except AttributeError:
        #     pass
        # else:
        #     self._cede_cellParamVecs = _np.ones((self._cede_radiates.shape[0],
        #                                  self._cede_radiates.shape[1],
        #                                  2), dtype=_np.double)
        
        #     self._cede_cellParamVecs[...,:-1] *= self['cede_temperature']
        
        #     for i in range(self._cede_cellParamVecs.shape[1]):
        #         self._cede_cellParamVecs[:,i,-1] *= self._cede_effGrav


class CustomPhotosphere_BB(xpsi.Photosphere):
    """ Implement method for imaging."""
    @property
    def global_variables(self):

        return np.array([self['p__super_colatitude'],
                          self['p__phase_shift'] * _2pi,
                          self['p__super_radius'],
                          self['p__super_temperature']#,
                          #self['s__super_colatitude'],
                          #(self['s__phase_shift'] + 0.5) * _2pi,
                          #self['s__super_radius'],
                          #self.hot.objects[1]['s__super_temperature']])
                          ])


class CustomPhotosphere_4D(xpsi.Photosphere):
    """ A photosphere extension to preload the numerical atmosphere NSX. """

    @xpsi.Photosphere.hot_atmosphere.setter
    def hot_atmosphere(self, path):
        size = (35,11,67,166)
        
        # NSX = np.loadtxt(path, dtype=np.double)
        with np.load(path) as data_dictionary:
            NSX = data_dictionary['arr_0.npy']

        _mu_opt = np.ascontiguousarray(NSX[0:size[2],1][::-1])
        logE_opt = np.ascontiguousarray([NSX[i*size[2],0] for i in range(size[3])])
        logT_opt = np.ascontiguousarray([NSX[i*size[1]*size[2]*size[3],3] for i in range(size[0])])
        logg_opt = np.ascontiguousarray([NSX[i*size[2]*size[3],4] for i in range(size[1])])

        def reorder_23(array, size):
            new_array=np.zeros(size)
            index=0
            for i in range(size[3]):
                for j in range(size[2]):
                        new_array[:,:,j,i]=array[:,:,index]
                        index+=1
            return new_array

        reorder_buf_opt=reorder_23(10**NSX[:,2].reshape(size[0],size[1],int(np.prod(size)/(size[0]*size[1]))),size)
        buf_opt=np.ravel(np.flip(reorder_buf_opt,2))
        
        self._hot_atmosphere = (logT_opt, logg_opt, _mu_opt, logE_opt, buf_opt)
        

class CustomPhotosphere_5D(xpsi.Photosphere):
    """ A photosphere extension to preload the numerical atmosphere NSX. """

    @xpsi.Photosphere.hot_atmosphere.setter
    def hot_atmosphere(self, path):
        size = (7,35,11,67,166)

        # Loading LARGE npz file which is hopefully quite fast. 
        with np.load(path) as data_dictionary:
            NSX = data_dictionary['arr_0.npy']
        
        _mu_opt = np.ascontiguousarray(NSX[0:size[3],1][::-1])
        logE_opt = np.ascontiguousarray([NSX[i*size[3],0] for i in range(size[4])])
        logT_opt = np.ascontiguousarray([NSX[i*size[2]*size[3]*size[4],3] for i in range(size[1])])
        logg_opt = np.ascontiguousarray([NSX[i*size[3]*size[4],4] for i in range(size[2])])
        modulator = np.ascontiguousarray([NSX[i*size[1]*size[2]*size[3]*size[4],5] for i in range(size[0])])

        def reorder_last_two(array, size):
            new_array=np.zeros(size)
            index=0
            for i in range(size[4]):
                for j in range(size[3]):
                        new_array[:,:,:,j,i]=array[:,:,:,index]
                        index+=1
            return new_array

        reorder_buf_opt=reorder_last_two(10**NSX[:,2].reshape(size[0],size[1],size[2],int(np.prod(size)/(size[0]*size[1]*size[2]))),size)
        buf_opt=np.ravel(np.flip(reorder_buf_opt,3))

        self._hot_atmosphere = (modulator, logT_opt, logg_opt, _mu_opt, logE_opt, buf_opt)

class CustomPhotosphere_Accreting(xpsi.Photosphere):
    """ A photosphere extension to preload the numerical atmosphere NSX. """

    @xpsi.Photosphere.hot_atmosphere.setter
    def hot_atmosphere(self, path):
        #size = (150, 9, 31, 20, 41)
        size = (150, 9, 31, 11, 41)

        with np.load(path, allow_pickle=True) as data_dictionary:
            NSX = data_dictionary['arr_0.npy']

        Energy = np.ascontiguousarray(NSX[0:size[0],0])
        cos_zenith = np.ascontiguousarray([NSX[i*size[0],1] for i in range(size[1])])
        tau = np.ascontiguousarray([NSX[i*size[0]*size[1],2] for i in range(size[2])])
        t_bb = np.ascontiguousarray([NSX[i*size[0]*size[1]*size[2],3] for i in range(size[3])])
        t_e = np.ascontiguousarray([NSX[i*size[0]*size[1]*size[2]*size[3],4] for i in range(size[4])])
        intensities = np.ascontiguousarray(NSX[:,5])

        self._hot_atmosphere = (t_e, t_bb, tau, cos_zenith, Energy, intensities)
        
class CustomPhotosphere_Accreting_te_const(xpsi.Photosphere):
    """ A photosphere extension to preload the numerical atmosphere NSX. """

    @xpsi.Photosphere.hot_atmosphere.setter
    def hot_atmosphere(self, path):
        size = (150, 9, 31, 11)#, 41) te is the 41

        with np.load(path, allow_pickle=True) as data_dictionary:
            NSX = data_dictionary['arr_0.npy']

        Energy = np.ascontiguousarray(NSX[0:size[0],0])
        cos_zenith = np.ascontiguousarray([NSX[i*size[0],1] for i in range(size[1])])
        tau = np.ascontiguousarray([NSX[i*size[0]*size[1],2] for i in range(size[2])])
        t_bb = np.ascontiguousarray([NSX[i*size[0]*size[1]*size[2],3] for i in range(size[3])])
        #t_e = np.ascontiguousarray([NSX[i*size[0]*size[1]*size[2]*size[3],4] for i in range(size[4])])
        te_step_size = size[0]*size[1]*size[2]*size[3]
        intensities = np.ascontiguousarray(NSX[0:te_step_size,5])

        self._hot_atmosphere = (t_bb, tau, cos_zenith, Energy, intensities)

from xpsi.likelihoods.default_background_marginalisation import eval_marginal_likelihood
from xpsi.likelihoods.default_background_marginalisation import precomputation

class CustomSignal(xpsi.Signal):
    """ A custom calculation of the logarithm of the likelihood.

    We extend the :class:`xpsi.Signal.Signal` class to make it callable.

    We overwrite the body of the __call__ method. The docstring for the
    abstract method is copied.

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
        #print("running CustomSignal call...")
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
                                          kwargs.get('llzero'))

from scipy.stats import truncnorm
class CustomPrior(xpsi.Prior):
    """ A custom (joint) prior distribution.

    Source: Fictitious
    Model variant: ST-U
        Two single-temperature, simply-connected circular hot regions with
        unshared parameters.

    """

    __derived_names__ = ['compactness', 'phase_separation',]

    def __init__(self):
        """ Nothing to be done.

        A direct reference to the spacetime object could be put here
        for use in __call__:

        .. code-block::

            self.spacetime = ref

        Instead we get a reference to the spacetime object through the
        a reference to a likelihood object which encapsulates a
        reference to the spacetime object.

        """
        super(CustomPrior, self).__init__() # not strictly required if no hyperparameters

    def __call__(self, p = None):
        """ Evaluate distribution at ``p``.

        :param list p: Model parameter values.

        :returns: Logarithm of the distribution evaluated at ``p``.

        """
        temp = super(CustomPrior, self).__call__(p)
        if not np.isfinite(temp):
            return temp

        # based on contemporary EOS theory
        if not self.parameters['radius'] <= 16.0:
            return -np.inf

        ref = self.parameters.star.spacetime # shortcut

        # limit polar radius to try to exclude deflections >= \pi radians
        # due to oblateness this does not quite eliminate all configurations
        # with deflections >= \pi radians
        R_p = 1.0 + ref.epsilon * (-0.788 + 1.030 * ref.zeta)
        if R_p < 1.76 / ref.R_r_s:
            return -np.inf

        # polar radius at photon sphere for ~static star (static ambient spacetime)
        #if R_p < 1.5 / ref.R_r_s:
        #    return -np.inf

        mu = math.sqrt(-1.0 / (3.0 * ref.epsilon * (-0.788 + 1.030 * ref.zeta)))

        # 2-surface cross-section have a single maximum in |z|
        # i.e., an elliptical surface; minor effect on support, if any,
        # for high spin frequenies
        if mu < 1.0:
            return -np.inf

        ref = self.parameters # redefine shortcut

        # enforce order in hot region colatitude
        if ref['p__super_colatitude'] > ref['s__super_colatitude']:
            return -np.inf

        phi = (ref['p__phase_shift'] - 0.5 - ref['s__phase_shift']) * _2pi

        ang_sep = xpsi.HotRegion.psi(ref['s__super_colatitude'],
                                     phi,
                                     ref['p__super_colatitude'])

        # hot regions cannot overlap
        if ang_sep < ref['p__super_radius'] + ref['s__super_radius']:
            return -np.inf

	#print("Calling CustomPrior with these paremeters:",p)

        return 0.0

    def inverse_sample(self, hypercube=None):
        """ Draw sample uniformly from the distribution via inverse sampling. """

        to_cache = self.parameters.vector

        if hypercube is None:
            hypercube = np.random.rand(len(self))

        # the base method is useful, so to avoid writing that code again:
        _ = super(CustomPrior, self).inverse_sample(hypercube)

        ref = self.parameters # shortcut

        idx = ref.index('distance')
        ref['distance'] = truncnorm.ppf(hypercube[idx], -2.0, 7.0, loc=0.3, scale=0.1)

        # flat priors in cosine of hot region centre colatitudes (isotropy)
        # support modified by no-overlap rejection condition
        idx = ref.index('p__super_colatitude')
        a, b = ref.get_param('p__super_colatitude').bounds
        a = math.cos(a); b = math.cos(b)
        ref['p__super_colatitude'] = math.acos(b + (a - b) * hypercube[idx])

        idx = ref.index('s__super_colatitude')
        a, b = ref.get_param('s__super_colatitude').bounds
        a = math.cos(a); b = math.cos(b)
        ref['s__super_colatitude'] = math.acos(b + (a - b) * hypercube[idx])

        # restore proper cache
        for parameter, cache in zip(ref, to_cache):
            parameter.cached = cache

        # it is important that we return the desired vector because it is
        # automatically written to disk by MultiNest and only by MultiNest
        return self.parameters.vector

    def transform(self, p, **kwargs):
        """ A transformation for post-processing. """

        p = list(p) # copy

        # used ordered names and values
        ref = dict(zip(self.parameters.names, p))

        # compactness ratio M/R_eq
        p += [gravradius(ref['mass']) / ref['radius']]

        # phase separation between hot regions
        # first some temporary variables:
        if ref['p__phase_shift'] < 0.0:
            temp_p = ref['p__phase_shift'] + 1.0
        else:
            temp_p = ref['p__phase_shift']

        temp_s = 0.5 + ref['s__phase_shift']

        if temp_s > 1.0:
            temp_s = temp_s - 1.0

        # now append:
        if temp_s >= temp_p:
            p += [temp_s - temp_p]
        else:
            p += [1.0 - temp_p + temp_s]

        return p
    
class CustomPrior_NoSecondary(xpsi.Prior):
    """ A custom (joint) prior distribution.

    Source: Fictitious
    Model variant: ST-U
        Two single-temperature, simply-connected circular hot regions with
        unshared parameters.

    """

    __derived_names__ = ['compactness', 'phase_separation',]

    # def __init__(self):
    #     """ Nothing to be done.

    #     A direct reference to the spacetime object could be put here
    #     for use in __call__:

    #     .. code-block::

    #         self.spacetime = ref

    #     Instead we get a reference to the spacetime object through the
    #     a reference to a likelihood object which encapsulates a
    #     reference to the spacetime object.

    #     """
    #     super(CustomPrior_NoSecondary, self).__init__() # not strictly required if no hyperparameters

    def __call__(self, p = None):
        """ Evaluate distribution at ``p``.

        :param list p: Model parameter values.

        :returns: Logarithm of the distribution evaluated at ``p``.

        """
        temp = super(CustomPrior_NoSecondary, self).__call__(p)
        if not np.isfinite(temp):
            return temp

        # based on contemporary EOS theory
        if not self.parameters['radius'] <= 16.0:
            return -np.inf

        ref = self.parameters.star.spacetime # shortcut

        # limit polar radius to try to exclude deflections >= \pi radians
        # due to oblateness this does not quite eliminate all configurations
        # with deflections >= \pi radians
        R_p = 1.0 + ref.epsilon * (-0.788 + 1.030 * ref.zeta)
        if R_p < 1.76 / ref.R_r_s:
            return -np.inf

        # polar radius at photon sphere for ~static star (static ambient spacetime)
        #if R_p < 1.5 / ref.R_r_s:
        #    return -np.inf

        mu = math.sqrt(-1.0 / (3.0 * ref.epsilon * (-0.788 + 1.030 * ref.zeta)))

        # 2-surface cross-section have a single maximum in |z|
        # i.e., an elliptical surface; minor effect on support, if any,
        # for high spin frequenies
        if mu < 1.0:
            return -np.inf

        ref = self.parameters # redefine shortcut

        # enforce order in hot region colatitude
        # if ref['p__super_colatitude'] > ref['s__super_colatitude']:
        #     return -np.inf

        # phi = (ref['p__phase_shift'] - 0.5 - ref['s__phase_shift']) * _2pi

        # ang_sep = xpsi.HotRegion.psi(ref['s__super_colatitude'],
        #                              phi,
        #                              ref['p__super_colatitude'])

        # # hot regions cannot overlap
        # if ang_sep < ref['p__super_radius'] + ref['s__super_radius']:
        #     return -np.inf

	#print("Calling CustomPrior with these paremeters:",p)

        return 0.0

    def inverse_sample(self, hypercube=None):
        """ Draw sample uniformly from the distribution via inverse sampling. """

        to_cache = self.parameters.vector

        if hypercube is None:
            hypercube = np.random.rand(len(self))

        # the base method is useful, so to avoid writing that code again:
        _ = super(CustomPrior_NoSecondary, self).inverse_sample(hypercube)

        ref = self.parameters # shortcut

        idx = ref.index('distance')
        ref['distance'] = truncnorm.ppf(hypercube[idx], -2.0, 7.0, loc=0.3, scale=0.1)

        # flat priors in cosine of hot region centre colatitudes (isotropy)
        # support modified by no-overlap rejection condition
        idx = ref.index('p__super_colatitude')
        a, b = ref.get_param('p__super_colatitude').bounds
        a = math.cos(a); b = math.cos(b)
        ref['p__super_colatitude'] = math.acos(b + (a - b) * hypercube[idx])

        # idx = ref.index('s__super_colatitude')
        # a, b = ref.get_param('s__super_colatitude').bounds
        # a = math.cos(a); b = math.cos(b)
        # ref['s__super_colatitude'] = math.acos(b + (a - b) * hypercube[idx])

        # restore proper cache
        for parameter, cache in zip(ref, to_cache):
            parameter.cached = cache

        # it is important that we return the desired vector because it is
        # automatically written to disk by MultiNest and only by MultiNest
        return self.parameters.vector

    def transform(self, p, **kwargs):
        """ A transformation for post-processing. """

        p = list(p) # copy

        # used ordered names and values
        ref = dict(zip(self.parameters.names, p))

        # compactness ratio M/R_eq
        p += [gravradius(ref['mass']) / ref['radius']]

        # phase separation between hot regions
        # first some temporary variables:
        if ref['p__phase_shift'] < 0.0:
            temp_p = ref['p__phase_shift'] + 1.0
        else:
            temp_p = ref['p__phase_shift']

        temp_s = 0.5 + ref['s__phase_shift']

        if temp_s > 1.0:
            temp_s = temp_s - 1.0

        # now append:
        if temp_s >= temp_p:
            p += [temp_s - temp_p]
        else:
            p += [1.0 - temp_p + temp_s]

        return p

from matplotlib import pyplot as plt

from matplotlib.ticker import MultipleLocator, AutoLocator, AutoMinorLocator
from matplotlib import gridspec
from matplotlib import cm
from xpsi.tools import phase_interpolator

def veneer(x, y, axes, lw=1.0, length=8):
    """ Make the plots a little more aesthetically pleasing. """
    if x is not None:
        if x[1] is not None:
            axes.xaxis.set_major_locator(MultipleLocator(x[1]))
        if x[0] is not None:
            axes.xaxis.set_minor_locator(MultipleLocator(x[0]))
    else:
        axes.xaxis.set_major_locator(AutoLocator())
        axes.xaxis.set_minor_locator(AutoMinorLocator())

    if y is not None:
        if y[1] is not None:
            axes.yaxis.set_major_locator(MultipleLocator(y[1]))
        if y[0] is not None:
            axes.yaxis.set_minor_locator(MultipleLocator(y[0]))
    else:
        axes.yaxis.set_major_locator(AutoLocator())
        axes.yaxis.set_minor_locator(AutoMinorLocator())

    axes.tick_params(which='major', colors='black', length=length, width=lw)
    axes.tick_params(which='minor', colors='black', length=int(length/2), width=lw)
    plt.setp(axes.spines.values(), linewidth=lw, color='black')

def plot_2D_pulse(z, x, shift, y, ylabel,
                  num_rotations=5.0, res=1000, figsize=(12,6),
                  cm=cm.viridis):
    """ Helper function to plot a phase-energy pulse.

    :param array-like z:
        A pair of *ndarray[m,n]* objects representing the signal at
        *n* phases and *m* values of an energy variable.

    :param ndarray[n] x: Phases the signal is resolved at.

    :param tuple shift: Hot region phase parameters.

    :param ndarray[m] x: Energy values the signal is resolved at.

    """

    fig = plt.figure(figsize = figsize)

    gs = gridspec.GridSpec(1, 2, width_ratios=[50,1], wspace=0.025)
    ax = plt.subplot(gs[0])
    ax_cb = plt.subplot(gs[1])

    new_phases = np.linspace(0.0, num_rotations, res)

    interpolated = phase_interpolator(new_phases,
                                      x,
                                      z[0], shift[0])
    if len(z) == 2:
        interpolated += phase_interpolator(new_phases,
                                           x,
                                           z[1], shift[1])

    profile = ax.pcolormesh(new_phases,
                             y,
                             interpolated/np.max(interpolated),
                             cmap = cm,
                             linewidth = 0,
                             rasterized = True)

    profile.set_edgecolor('face')

    ax.set_xlim([0.0, num_rotations])
    ax.set_yscale('log')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r'Phase')
    veneer((0.1, 0.5), (None,None), ax)

    cb = plt.colorbar(profile, cax = ax_cb, ticks = MultipleLocator(0.2))

    cb.set_label(label=r'Signal (arbitrary units)', labelpad=25)
    cb.solids.set_edgecolor('face')

    veneer((None, None), (0.05, None), ax_cb)
    cb.outline.set_linewidth(1.0)
    
    return ax

class CustomBackground(xpsi.Background):
    """ The background injected to generate synthetic data. """

    def __init__(self, bounds=None, value=None):

        # first the parameters that are fundemental to this class
        doc = """
        Powerlaw spectral index.
        """
        index = xpsi.Parameter('powerlaw_index',
                                strict_bounds = (-4.0, -1.01),
                                bounds = bounds,
                                doc = doc,
                                symbol = r'$\Gamma$',
                                value = value)

        super(CustomBackground, self).__init__(index)

    def __call__(self, energy_edges, phases):
        """ Evaluate the incident background field. """

        G = self['powerlaw_index']

        temp = np.zeros((energy_edges.shape[0] - 1, phases.shape[0]))

        temp[:,0] = (energy_edges[1:]**(G + 1.0) - energy_edges[:-1]**(G + 1.0)) / (G + 1.0)

        for i in range(phases.shape[0]):
            temp[:,i] = temp[:,0]

        self._incident_background= temp
        
class SynthesiseData(xpsi.Data):
    """ Custom data container to enable synthesis. """

    def __init__(self, channels, phases, first, last):

        self.channels = channels
        # print(channels)
        # print(len(channels))
        self._phases = phases

        try:
            self._first = int(first)
            self._last = int(last)
        except TypeError:
            raise TypeError('The first and last channels must be integers.')
        if self._first >= self._last:
            raise ValueError('The first channel number must be lower than the '
                             'the last channel number.')