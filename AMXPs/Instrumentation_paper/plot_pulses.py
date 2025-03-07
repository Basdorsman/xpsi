import xpsi
import numpy as np

import os
import sys
this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory+'/../')
from parameter_values import parameter_values
from CustomPrior import CustomPrior

sys.path.append(this_directory+'/../J1808_NICER/')
from CustomHotregion import CustomHotRegion_Accreting
from CustomPhotosphere import CustomPhotosphereDiskLine
from CustomInstrument_TACO import TACO
from CustomSignal import CustomSignal
from CustomInterstellar import CustomInterstellar


class plot_pulse(object):
    def __init__(self):
        self.scenario = '2022'
        self.bkg = 'disk'
        self.sqrt_num_cells = 50
        self.num_leaves = 30
        self.num_rays = 512
        self.file_atmosphere = '/home/bas/Documents/Projects/x-psi/model_datas/bobrikova/Bobrikova_compton_slab.npz'
        self.file_interstellar = "/home/bas/Documents/Projects/x-psi/xpsi-bas-fork/AMXPs/model_data/n_H/TBnew/tbnew0.14.txt"
        self.num_energies=128
        
        self.pv = parameter_values(self.scenario, self.bkg)
    
        self.set_likelihood()
        
        #self.likelihood(self.p, reinitialise=True)
        
        # print('parameters:', self.p)
        # print(self.likelihood)

        self.likelihood.check(None, [-1.8693770262e+07], 1.0e-4, physical_points=[], force_update=True)
        print(self.likelihood([]))
    
    # def __call__(self):
        



    def set_spacetime(self):
        spacetime_values = dict(frequency = 401.,
                      mass = self.pv.mass,
                      radius = self.pv.radius,
                      distance = self.pv.distance,
                      cos_inclination = self.pv.cos_i)
        

        spacetime_bounds = {}
        
        self.spacetime = xpsi.Spacetime(bounds=spacetime_bounds, values=spacetime_values)
        
    def set_hotregions(self):    
        kwargs = {'symmetry': True, #call for azimuthal invariance
                  'split': True,
                  'omit': False,
                  'cede': False,
                  'concentric': False,
                  'sqrt_num_cells': self.sqrt_num_cells,
                  'min_sqrt_num_cells': 10,
                  'max_sqrt_num_cells': 128,
                  'num_leaves': self.num_leaves,
                  'num_rays': self.num_rays,
                  'atm_ext':'Num5D'}
                  #'prefix': 'p'}
        
        hotregion_values = dict(super_colatitude = self.pv.super_colatitude,
                                super_radius = self.pv.super_radius,
                                phase_shift = self.pv.phase_shift, 
                                super_tbb = self.pv.tbb,
                                super_tau = self.pv.tau,
                                super_te = self.pv.te)
        hotregion_bounds = {}
        
        primary = CustomHotRegion_Accreting(hotregion_bounds, hotregion_values, **kwargs)


        self.hot = xpsi.HotRegions((primary,))
        
    def set_disk(self):
        from Disk import Disk, get_k_disk
  
        disk_bounds = {}
        self.k_disk = get_k_disk(self.pv.cos_i, self.pv.R_in, self.pv.distance)            
        disk_values = dict(T_in = self.pv.diskbb_T_log10_K,
                           R_in = self.pv.R_in,
                           K_disk = self.k_disk)
            
        self.disk = Disk(bounds=disk_bounds, values=disk_values)

    def set_photosphere(self):     
        self.set_spacetime()
        self.set_hotregions()
        self.set_disk()
        self.photosphere = CustomPhotosphereDiskLine(hot = self.hot, elsewhere = None, stokes=False, disk=self.disk,
                                        values=dict(mode_frequency = self.spacetime['frequency']))

        self.photosphere.hot_atmosphere = self.file_atmosphere
        
    def set_star(self):
        self.set_photosphere()
        self.star = xpsi.Star(spacetime = self.spacetime, photospheres = self.photosphere)

    def set_data(self):
        #using some pretend data
        
        settings = dict(counts = np.ones((1310, 32)),
                        channels=np.arange(0, 1310),
                        phases=np.linspace(0.0, 1.0, 33),
                        first=0, last=1309,
                        exposure_time=1e5)
        self.data = xpsi.Data(**settings)
        
    def set_instrument(self):
        self.instrument = TACO.from_response_files(
                RMF_file = 'instrument_files/TACO_4mod_matrix.txt',
                ebounds_file = 'instrument_files/TACO_4mod_ebounds.txt',
                max_detection_channel = 1310,
                max_input = 1317)

    def set_interstellar(self):
        interstellar_bounds = None
        interstellar_values = self.pv.column_density #self.pv.column_density
        self.interstellar=CustomInterstellar.from_SWG(self.file_interstellar, bounds=interstellar_bounds, value=interstellar_values)

    def set_signal(self):
        self.set_data()
        self.set_instrument()
        self.set_interstellar()

        self.signal = CustomSignal(data = self.data,
                            instrument = self.instrument,
                            background = None,
                            interstellar = self.interstellar,
                            support = None,
                            cache = False, # only true if verifying code implementation otherwise useless slowdown.
                            bkg = self.bkg,
                            epsrel = 1.0e-8,
                            epsilon = 1.0e-3,
                            sigmas = 10.0)
     
    def set_parameter_vector(self):
        self.p = self.pv.p()

    def set_prior(self):
        self.prior = CustomPrior(self.scenario, self.bkg, fix_mass=False)
  
    def set_likelihood(self):
        self.set_star()
        self.set_signal()
        self.set_parameter_vector()
        self.set_prior()
        
        self.likelihood = xpsi.Likelihood(star = self.star, signals = self.signal,
                                      num_energies=self.num_energies, #128
                                      threads=1,
                                      prior=self.prior,
                                      externally_updated=True)
        

        
if __name__ == '__main__':
    my_pulse = plot_pulse()
    # my_pulse()