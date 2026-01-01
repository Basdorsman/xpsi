import os
import sys
this_directory = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(this_directory+'/../')

import numpy as np
import time
from matplotlib import rcParams
import matplotlib.pyplot as plt

import xpsi
np.random.seed(xpsi._rank+10)
print('Rank reporting: %d' % xpsi._rank)

from CustomPrior import CustomPrior_twohotspots as CustomPrior
from CustomInstrument import CustomInstrument_fits, CustomInstrument_stokes
from CustomPhotosphere import CustomPhotosphereDiskLine
from CustomInterstellar import CustomInterstellar
from CustomSignal import CustomSignal, CustomSignal_poisson, CustomSignal_gaussian
from CustomHotregion import CustomHotRegion_Accreting

from parameter_values import parameter_values
from helper_functions import plot_2D_pulse, CustomAxes, get_mids_from_edges

from xpsi.Parameter import Derive

class analysis(object):
    def __init__(self, 
                 run_type, 
                 bkg, 
                 sampler='multi', 
                 support_factor = "None", 
                 scenario = 'None', 
                 poisson_noise=True, 
                 poisson_seed=42, 
                 fix_mass=False, 
                 eos_informed=False,
                 polarization=False,
                 NICER=True,
                 channel_min=None):

        self.scenario = os.environ.get('scenario')
        if os.environ.get('scenario') == None or os.environ.get('scenario') =='None':
            print('scenario is not in environment variables, using passed argument.')
            self.scenario=scenario
        print(f'scenario: {self.scenario}')

        self.run_type = os.environ.get('run_type')
        if os.environ.get('run_type') == None or os.environ.get('run_type') == "None":
            print('run_type is not in environment variables, using passed argument.')
            self.run_type = run_type
        print(f'run_type: {self.run_type}')
        
        self.sampler = os.environ.get('sampler')
        if os.environ.get('sampler') == None or os.environ.get('sampler') == 'None':
            print('sampler is not in environment variables, using passed argument.')
            self.sampler = sampler
        print(f'sampler: {self.sampler}')

        self.analysis_name = os.environ.get('LABEL')
        if not isinstance(self.analysis_name, str):
                print('cannot import analysis name, using test_analysis')
                self.analysis_name = 'test_analysis'
        print(f'analysis_name: {self.analysis_name}')

        try:
            self.num_energies = int(os.environ.get('num_energies'))
        except:
            print('num_energies from environment variables failed, proceeding with default.')
            self.num_energies = 64 #40 # 128
            pass
        print(f'num_energies: {self.num_energies}')
            
        try:
            self.num_leaves = int(os.environ.get('num_leaves'))
        except:
            print('num_leaves from environment variables failed, proceeding with default.')
            self.num_leaves = 32 #30 # 50 avoids interpolation error with polarisation # 30 #128
            pass
        print(f'num_leaves: {self.num_leaves}')
    
        try:
            self.sqrt_num_cells = int(os.environ.get('sqrt_num_cells'))
        except:
            print('sqrt_num_cells from environment variables failed, proceeding with default.')
            self.sqrt_num_cells = 32 #50 # 128
            pass
        print(f'sqrt_num_cells: {self.sqrt_num_cells}')
    
        try:
            self.num_rays = int(os.environ.get('num_rays'))
        except:
            print('num_rays from env. var. failed, proceeding with default.')
            self.num_rays = 100 #512
        print(f'num_rays: {self.num_rays}')
    
        try:
            self.live_points = int(os.environ.get('live_points'))
        except:
            print('live_points from environment variables failed, proceeding with default.')
            self.live_points = 64
            pass
        print(f'live_points: {self.live_points}')
        
        try:
            self.max_iter = int(os.environ.get('max_iter'))
        except:
            print('max_iter from environment variables failed, proceeding with default.')
            self.max_iter = -1
            pass
        print(f'max_iter: {self.max_iter}')

        self.bkg = os.environ.get('bkg')
        if os.environ.get('bkg') == None or os.environ.get('bkg') == "None":
            print(f'bkg environment variable is not allowed to be None, using passed argument: {bkg}.')
            self.bkg = bkg
        print(f'bkg: {self.bkg}')

        if self.bkg == 'marginalise':
                self.support_factor = os.environ.get('support_factor')
                if os.environ.get('support_factor') == None or os.environ.get('support_factor') == 'None':
                    print('No support_factor in os. Taking from passed or default argument')
                    self.support_factor = support_factor
        elif 'disk' in self.bkg or 'line' in self.bkg or self.bkg == 'fix':
            self.support_factor = 'None'
        print(f'support_factor: {self.support_factor}')   
       
        try:
            poisson_noise = os.environ.get('poisson_noise')
        except:
            print('No poisson noise decision in os. Taking default choice')
        if poisson_noise == 'True' or poisson_noise == True:
            self.poisson_seed = int(os.environ.get('poisson_seed'))
            self.poisson_noise = poisson_noise
        elif poisson_noise == None or poisson_noise == 'None':
            self.poisson_noise = poisson_noise
            self.poisson_seed = poisson_seed
        print(f'poisson_noise: {self.poisson_noise}, poisson_seed: {self.poisson_seed} (only relevant if poisson noise is True)')
       
        if os.environ.get('fix_mass') == None or os.environ.get('fix_mass') =='None':
            print('fix_mass is not in environment variables, using passed argument.')
            self.fix_mass = fix_mass
        else:
            self.fix_mass = os.environ.get('fix_mass')

        if self.fix_mass == "True" or self.fix_mass == True:
            self.fix_mass = True
        else:
            self.fix_mass = False
        print(f'fix_mass: {self.fix_mass}')
        
        if os.environ.get('eos_informed') == None or os.environ.get('eos_informed') =='None':
            print('eos_informed is not in environment variables, using passed argument.')
            self.eos_informed = eos_informed
        else:
            self.eos_informed = os.environ.get('eos_informed')

        if self.eos_informed == "True" or self.eos_informed == True:
            self.eos_informed = True
        else:
            self.eos_informed = False

        print(f'eos_informed: {self.eos_informed}')

        if os.environ.get('polarization') == None or os.environ.get('polarization') =='None':
            print('polarization is not in environment variables, using passed argument.')
            self.polarization = polarization
        else:
            self.polarization = os.environ.get('polarization')
        if self.polarization == "qu" or self.polarization == "iqu":
            self.polarization = self.polarization
        else:
            self.polarization = False
        print(f'polarization: {self.polarization}')
        
        if os.environ.get('channel_min') == None or os.environ.get('channel_min') == 'None':
            print('channel_min is not in environment variables, using passed argument.')
            self.channel_min = channel_min
        else:
            self.channel_min = int(os.environ.get('channel_min'))
        print(f'channel_min: {self.channel_min}') 

        self.NICER = NICER
        self.signal_phase_shift = False
        self.variable_params=True
        self.disk_NICER_only=False
        self.pv = parameter_values(self.scenario, self.bkg, self.fix_mass, polarization=self.polarization, signal_phase_shift=self.signal_phase_shift)
        self.file_locations()
        self.set_parameter_vector()
        self.set_bounds()
        self.set_interstellar()
        self.set_likelihood()

    def file_locations(self):
        self.this_directory = this_directory
        
        if self.scenario in ('large_r', 'small_r', 'J1444s'):
            self.file_pulse_profile = self.this_directory + f'/data/NICER_products/data/{self.scenario}_seed={self.poisson_seed}_ch{self.channel_min}_realisation.dat'
        if self.scenario in ('J1444', 'J1444_STU', 'J1444_STS'):
            self.file_pulse_profile = self.this_directory + f'/data/NICER_products/data/J1444_preprocessed_ch{self.channel_min}.txt'
       
        self.RMF_file = self.this_directory+'/data/NICER_products/srgaj1444.rmf'
        self.ARF_file = self.this_directory+'/data/NICER_products/srgaj1444.arf'
        self.file_atmosphere = self.this_directory + '/data/Bobrikova_compton_slab_I.npz'
        self.file_interstellar = self.this_directory +'/data/tbnew0.14.txt'

    def set_bounds(self):
        self.bounds = self.pv.bounds()

    def set_data_NICER(self):
        if self.scenario == '2019' or self.scenario == 'large_r' or self.scenario == 'small_r':
            self.exposure_time = 1.32366e5 #Mason's 2019 data cut
        if self.scenario == '2022':
            self.exposure_time = 7.13422e4 #Mason's 2022 data cut
        if 'J1444' in self.scenario:
            self.exposure_time = 24823.7
        
        self.phases_space = np.linspace(0.0, 1.0, 33)

      

        if self.channel_min == 100:        
            self.min_input = 700 #  700 works with channel_low = 100 (1 keV). 
            self.channel_low = 100 # 100 corresponds to 1 keV. 
        elif self.channel_min == 20:
            self.min_input = 0 #  0 is used with 0.2 keV (channel_low=20).
            self.channel_low = 20 # 20 corresponds to 0.2 keV. 
        self.max_input = 2800 # accomodates just beyond 10 keV
        self.channel_hi = 1000 # 10 keV
       


        settings = dict(counts = np.loadtxt(self.file_pulse_profile, dtype=np.double),
                        channels=np.arange(self.channel_low,self.channel_hi),
                        phases=self.phases_space,
                        first=0, 
                        last=self.channel_hi-self.channel_low-1,
                        exposure_time=self.exposure_time)

        self.NICER_data = xpsi.Data(**settings)
        
    def set_data_IXPE(self):
        from ixpe_read_pha import readData_pha

        data_path = self.this_directory + "/data/ixpe_products/phase_binned_xspec/"

        fname_ixpedata_du1 = data_path + "ixpe03250101_du1_evt2_v01_src_bary_pers_pre60376"
        fname_ixpedata_du2 = data_path + "ixpe03250101_du2_evt2_v01_src_bary_pers_pre60376"
        fname_ixpedata_du3 = data_path + "ixpe03250101_du3_evt2_v01_src_bary_pers_pre60376"

        Idat1, Qdat1, Udat1, Iderr1, Qerr1, Uerr1, channels1, phase_edges1, exposure1 = readData_pha(fname_ixpedata_du1)
        Idat2, Qdat2, Udat2, Iderr2, Qerr2, Uerr2, channels2, phase_edges2, exposure2 = readData_pha(fname_ixpedata_du2)
        Idat3, Qdat3, Udat3, Iderr3, Qerr3, Uerr3, channels3, phase_edges3, exposure3 = readData_pha(fname_ixpedata_du3)

        minchan = 50
        maxchan1 = 151

        self.IXPE_I_DU1_data = xpsi.Data(Idat1.T[minchan:maxchan1,:],
                               channels=channels1[minchan:maxchan1],
                               phases=phase_edges1,
                               first=0,
                               last=maxchan1-minchan-1,
                               exposure_time=exposure1)
        
        # print('self.IXPE_I_data.channels', self.IXPE_I_data.channels)

        self.IXPE_I_DU2_data = xpsi.Data(Idat2.T[minchan:maxchan1,:],
                               channels=channels2[minchan:maxchan1],
                               phases=phase_edges2,
                               first=0,
                               last=maxchan1-minchan-1,
                               exposure_time=exposure2)

        self.IXPE_I_DU3_data = xpsi.Data(Idat3.T[minchan:maxchan1,:],
                               channels=channels3[minchan:maxchan1],
                               phases=phase_edges3,
                               first=0,
                               last=maxchan1-minchan-1,
                               exposure_time=exposure3)

                               
        self.IXPE_Q_DU1_data = xpsi.Data(Qdat1.T[minchan:maxchan1,:],
                               channels=channels1[minchan:maxchan1],
                               phases=phase_edges1,
                               first=0,
                               last=maxchan1-minchan-1,
                               exposure_time=exposure1)

        self.IXPE_Q_DU2_data = xpsi.Data(Qdat2.T[minchan:maxchan1,:],
                               channels=channels2[minchan:maxchan1],
                               phases=phase_edges2,
                               first=0,
                               last=maxchan1-minchan-1,
                               exposure_time=exposure2)

        self.IXPE_Q_DU3_data = xpsi.Data(Qdat3.T[minchan:maxchan1,:],
                               channels=channels3[minchan:maxchan1],
                               phases=phase_edges3,
                               first=0,
                               last=maxchan1-minchan-1,
                               exposure_time=exposure3)


        self.IXPE_U_DU1_data = xpsi.Data(Udat1.T[minchan:maxchan1,:],
                               channels=channels1[minchan:maxchan1],
                               phases=phase_edges1,
                               first=0,
                               last=maxchan1-minchan-1,
                               exposure_time=exposure1)

        self.IXPE_U_DU2_data = xpsi.Data(Udat2.T[minchan:maxchan1,:],
                               channels=channels2[minchan:maxchan1],
                               phases=phase_edges2,
                               first=0,
                               last=maxchan1-minchan-1,
                               exposure_time=exposure2)

        self.IXPE_U_DU3_data = xpsi.Data(Udat3.T[minchan:maxchan1,:],
                               channels=channels3[minchan:maxchan1],
                               phases=phase_edges3,
                               first=0,
                               last=maxchan1-minchan-1,
                               exposure_time=exposure3)



        self.IXPE_I_DU1_data.errors, self.IXPE_Q_DU1_data.errors, self.IXPE_U_DU1_data.errors = Iderr1.T[minchan:maxchan1,:], Qerr1.T[minchan:maxchan1,:], Uerr1.T[minchan:maxchan1,:]
        self.IXPE_I_DU2_data.errors, self.IXPE_Q_DU2_data.errors, self.IXPE_U_DU2_data.errors = Iderr2.T[minchan:maxchan1,:], Qerr2.T[minchan:maxchan1,:], Uerr2.T[minchan:maxchan1,:]
        self.IXPE_I_DU3_data.errors, self.IXPE_Q_DU3_data.errors, self.IXPE_U_DU3_data.errors = Iderr3.T[minchan:maxchan1,:], Qerr3.T[minchan:maxchan1,:], Uerr3.T[minchan:maxchan1,:]
            
    def set_instrument_NICER(self):
        
        alpha_bounds = dict(alpha = (0.8, 1.2))  
        values = {}
        
        self.NICER = CustomInstrument_fits.from_response_files(
            bounds = alpha_bounds,
            values = values,
            RMF_file = self.RMF_file, 
            ARF_file = self.ARF_file,
            max_detection_channel=self.channel_hi, 
            min_detection_channel = self.channel_low, 
            max_input = self.max_input, #around the maximum
            min_input = self.min_input,
            prefix='NICER')

    def set_instrument_IXPE(self):
        class derive_du1(Derive):
            def __init__(self):
                pass

            def __call__(self, boundto, caller=None):
                return self.IXPE_du1_I['alpha']
                
        class derive_du2(Derive):
            def __init__(self):
                pass

            def __call__(self, boundto, caller=None):
                return self.IXPE_du2_I['alpha']
                
        class derive_du3(Derive):
            def __init__(self):
                pass

            def __call__(self, boundto, caller=None):
                return self.IXPE_du3_I['alpha']   
        
        alpha_bounds = dict(alpha = (0.95, 1.05))
        alpha_value = dict(alpha = 1.)
        
        derive_du1_inst = derive_du1()
        derive_du2_inst = derive_du2()
        derive_du3_inst = derive_du3()
        
        self.IXPE_du1_I = CustomInstrument_stokes.from_response_files(
                                                     bounds={},
                                                     values=alpha_value,
                                                     MRF = self.this_directory + '/data/ixpe_products/phase_binned_xspec/response/ixpe_d1_obssim20240101_v013.arf',
                                                     RMF = self.this_directory + '/data/ixpe_products/phase_binned_xspec/response/ixpe_d1_obssim20240101_v013.rmf',
                                                     max_input = 275,
                                                     max_channel = 150,
                                                     min_input = 0,
                                                     min_channel = 50,
                                                     channel_edges = None,
                                                     prefix="du1")
        derive_du1_inst.IXPE_du1_I = self.IXPE_du1_I

        self.IXPE_du1 = CustomInstrument_stokes.from_response_files(
                                                     bounds = {'alpha': None},
                                                     values = {'alpha': derive_du1_inst},
                                                     MRF = self.this_directory + '/data/ixpe_products/phase_binned_xspec/response/ixpe_d1_obssim20240101_v013.mrf',
                                                     RMF = self.this_directory + '/data/ixpe_products/phase_binned_xspec/response/ixpe_d1_obssim20240101_v013.rmf',
                                                     max_input = 275,
                                                     max_channel = 150,
                                                     min_input = 0,
                                                     min_channel = 50, #2 keV
                                                     channel_edges = None,
                                                     prefix="du1_pol")

        self.IXPE_du2_I = CustomInstrument_stokes.from_response_files(
                                                      bounds=alpha_bounds,
                                                      values={},
                                                      MRF = self.this_directory + '/data/ixpe_products/phase_binned_xspec/response/ixpe_d2_obssim20240101_v013.arf',
                                                      RMF = self.this_directory + '/data/ixpe_products/phase_binned_xspec/response/ixpe_d2_obssim20240101_v013.rmf',
                                                      max_input = 275,
                                                      max_channel = 150,
                                                      min_input = 0,
                                                      min_channel = 50,
                                                      channel_edges = None,
                                                      prefix="du2")
        derive_du2_inst.IXPE_du2_I = self.IXPE_du2_I
        
        self.IXPE_du2 = CustomInstrument_stokes.from_response_files(
                                                     bounds = {'alpha': None},
                                                     values = {'alpha': derive_du2_inst},
                                                     MRF = self.this_directory + '/data/ixpe_products/phase_binned_xspec/response/ixpe_d2_obssim20240101_v013.mrf',
                                                     RMF = self.this_directory + '/data/ixpe_products/phase_binned_xspec/response/ixpe_d2_obssim20240101_v013.rmf',
                                                     max_input = 275,
                                                     max_channel = 150,
                                                     min_input = 0,
                                                     min_channel = 50,
                                                     channel_edges = None,
                                                     prefix="du2_pol")
        
        self.IXPE_du3_I = CustomInstrument_stokes.from_response_files(
                                                     bounds=alpha_bounds,
                                                     values={},
                                                     MRF = self.this_directory + '/data/ixpe_products/phase_binned_xspec/response/ixpe_d3_obssim20240101_v013.arf',
                                                     RMF = self.this_directory + '/data/ixpe_products/phase_binned_xspec/response/ixpe_d3_obssim20240101_v013.rmf',
                                                     max_input = 275,
                                                     max_channel = 150,
                                                     min_input = 0,
                                                     min_channel = 50,
                                                     channel_edges = None,
                                                     prefix="du3")
        derive_du3_inst.IXPE_du3_I = self.IXPE_du3_I                                             
        
        self.IXPE_du3 = CustomInstrument_stokes.from_response_files(
                                                     bounds = {'alpha': None},
                                                     values = {'alpha': derive_du3_inst},
                                                     MRF = self.this_directory + '/data/ixpe_products/phase_binned_xspec/response/ixpe_d3_obssim20240101_v013.mrf',
                                                     RMF = self.this_directory + '/data/ixpe_products/phase_binned_xspec/response/ixpe_d3_obssim20240101_v013.rmf',
                                                     max_input = 275,
                                                     max_channel = 150,
                                                     min_input = 0,
                                                     min_channel = 50,
                                                     channel_edges = None,
                                                     prefix="du3_pol")



    def set_spacetime(self):
        fix_mass = self.fix_mass


        if fix_mass:
            spacetime_values = dict(frequency = self.pv.frequency, mass = self.pv.mass)
        if not fix_mass:
            spacetime_values = dict(frequency = self.pv.frequency)

        if fix_mass:
            spacetime_bounds = dict(distance = self.bounds["distance"],
                                    radius = self.bounds["radius"],
                                    cos_inclination = self.bounds["cos_inclination"])
        if not fix_mass:
            spacetime_bounds = dict(distance = self.bounds["distance"],                       # (Earth) distance
                                    mass = self.bounds["mass"],                          # mass
                                    radius = self.bounds["radius"],     # equatorial radius
                                    cos_inclination = self.bounds["cos_inclination"])   

        self.spacetime = xpsi.Spacetime(bounds=spacetime_bounds, values=spacetime_values)

    def set_hotregions(self):
        
        self.p_kwargs = {'symmetry': True, #call for azimuthal invariance
                  'split': True,
                  'omit': False,
                  'cede': False,
                  'concentric': False,
                  'sqrt_num_cells': self.sqrt_num_cells,
                  'min_sqrt_num_cells': 10,
                  'max_sqrt_num_cells': 128,
                  'num_leaves': self.num_leaves,  #50 avoids interp error.
                  'num_rays': self.num_rays,
                  'atm_ext':'Num5D'}
        
        self.p_bounds = dict(super_colatitude = (0.001, np.pi/2 - 0.001),
                                super_radius = self.bounds["super_radius"],
                                phase_shift = self.bounds["phase_shift"], 
                                super_tbb = self.bounds['super_tbb'],
                                super_tau = self.bounds['super_tau'],
                                super_te = self.bounds['super_te'])
        self.p_values = {}
        
        if self.variable_params:
            # while initial p_values are the same, they should not be linked, I think, and vary between NICER and IXPE during sampling.
            self.primary_NICER = CustomHotRegion_Accreting(self.p_bounds, 
                                                           self.p_values, 
                                                           prefix='NICER__p',
                                                           **self.p_kwargs)
            self.primary_IXPE = CustomHotRegion_Accreting(self.p_bounds, 
                                                          self.p_values, 
                                                          prefix='IXPE__p',
                                                          **self.p_kwargs)
        elif not self.variable_params:
            self.primary = CustomHotRegion_Accreting(self.p_bounds, self.p_values, **self.p_kwargs)

        self.s_kwargs = {'symmetry': True, #call for azimuthal invariance
                  'split': True,
                  'omit': False,
                  'cede': False,
                  'concentric': False,
                  'sqrt_num_cells': self.sqrt_num_cells,
                  'min_sqrt_num_cells': 10,
                  'max_sqrt_num_cells': 128,
                  'num_leaves': self.num_leaves,  #50 avoids interp error.
                  'num_rays': self.num_rays,
                  'atm_ext':'Num5D',
                  'is_antiphased': True}
                  # 'prefix': 's'}
        
        if self.scenario == 'J1444_STU':  
            self.s_bounds = dict(super_colatitude = self.bounds["super_colatitude"],
                                    super_radius = self.bounds["super_radius"],
                                    phase_shift = self.bounds["phase_shift"], 
                                    super_tbb = self.bounds['super_tbb'],
                                    super_tau = self.bounds['super_tau'],
                                    super_te = self.bounds['super_te'])
            self.s_values = {}
        elif self.scenario == 'J1444_STS':
            class derive_s__super_colatitude(Derive):
                def __init__(self):
                    pass

                def __call__(self, boundto, caller=None):
                    return np.pi - self.primary['super_colatitude']
                
            class derive_s__super_radius(Derive):
                def __init__(self):
                    pass

                def __call__(self, boundto, caller=None):
                    return self.primary['super_radius']    
                
            class derive_s__phase_shift(Derive):
                def __init__(self):
                    pass

                def __call__(self, boundto, caller=None):
                    return self.primary['phase_shift']
                
            class derive_s__super_tbb(Derive):
                def __init__(self):
                    pass

                def __call__(self, boundto, caller=None):
                    return self.primary['super_tbb']
            
            class derive_s__super_tau(Derive):
                def __init__(self):
                    pass

                def __call__(self, boundto, caller=None):
                    return self.primary['super_tau']
            
            class derive_s__super_te(Derive):
                def __init__(self):
                    pass

                def __call__(self, boundto, caller=None):
                    return self.primary['super_te']
                
            derive_colatitude_NICER = derive_s__super_colatitude()
            derive_radius_NICER = derive_s__super_radius()
            derive_phase_NICER = derive_s__phase_shift()
            derive_tbb_NICER = derive_s__super_tbb()
            derive_tau_NICER = derive_s__super_tau()
            derive_te_NICER = derive_s__super_te()
            
            self.s_values_NICER = {'super_colatitude': derive_colatitude_NICER,
                             'super_radius': derive_radius_NICER,
                             'phase_shift': derive_phase_NICER,
                             'super_tbb': derive_tbb_NICER,
                             'super_tau': derive_tau_NICER,
                             'super_te': derive_te_NICER}
            
            derive_colatitude_NICER.primary = self.primary_NICER
            derive_radius_NICER.primary = self.primary_NICER
            derive_phase_NICER.primary = self.primary_NICER
            derive_tbb_NICER.primary = self.primary_NICER
            derive_tau_NICER.primary = self.primary_NICER
            derive_te_NICER.primary = self.primary_NICER
            
            derive_colatitude_IXPE = derive_s__super_colatitude()
            derive_radius_IXPE = derive_s__super_radius()
            derive_phase_IXPE = derive_s__phase_shift()
            derive_tbb_IXPE = derive_s__super_tbb()
            derive_tau_IXPE = derive_s__super_tau()
            derive_te_IXPE = derive_s__super_te()
            
            self.s_values_IXPE = {'super_colatitude': derive_colatitude_IXPE,
                             'super_radius': derive_radius_IXPE,
                             'phase_shift': derive_phase_IXPE,
                             'super_tbb': derive_tbb_IXPE,
                             'super_tau': derive_tau_IXPE,
                             'super_te': derive_te_IXPE}
            
            derive_colatitude_IXPE.primary = self.primary_IXPE
            derive_radius_IXPE.primary = self.primary_IXPE
            derive_phase_IXPE.primary = self.primary_IXPE
            derive_tbb_IXPE.primary = self.primary_IXPE
            derive_tau_IXPE.primary = self.primary_IXPE
            derive_te_IXPE.primary = self.primary_IXPE
            
            self.s_bounds = dict(super_colatitude = None,
                                    super_radius = None,
                                    phase_shift = None, 
                                    super_tbb = None,
                                    super_tau = None,
                                    super_te = None)

        if self.variable_params:            
            self.secondary_NICER = CustomHotRegion_Accreting(self.s_bounds, 
                                                             self.s_values_NICER, 
                                                             prefix='NICER__s',
                                                             **self.s_kwargs)
            self.secondary_IXPE = CustomHotRegion_Accreting(self.s_bounds, 
                                                            self.s_values_IXPE,
                                                            prefix='IXPE__s',
                                                            **self.s_kwargs)
            self.hot_NICER = xpsi.HotRegions((self.primary_NICER,self.secondary_NICER))
            self.hot_IXPE = xpsi.HotRegions((self.primary_IXPE,self.secondary_IXPE))
            

        elif not self.variable_params:
            self.secondary = CustomHotRegion_Accreting(self.s_bounds, self.s_values, **self.s_kwargs)
            self.hot = xpsi.HotRegions((self.primary,self.secondary))

    def set_elsewhere(self):
        self.elsewhere = xpsi.Elsewhere(bounds=dict(elsewhere_temperature = self.bounds['elsewhere_temperature']))
        
    def set_photosphere(self):
        self.set_hotregions()
        self.set_disk()
        self.set_line()
        
        
        if self.variable_params:
            photosphere_bounds = dict(spin_axis_position_angle = (None, None))
            photosphere_values = dict(mode_frequency = self.spacetime['frequency'])
            
            self.photosphere_NICER = CustomPhotosphereDiskLine(hot = self.hot_NICER, 
                                                         elsewhere = None, 
                                                         stokes=False, 
                                                         disk=self.disk_NICER, 
                                                         values=photosphere_values, 
                                                         bounds=photosphere_bounds,
                                                         prefix='NICER')
    
            self.photosphere_NICER.hot_atmosphere = self.file_atmosphere
            # self.photosphere_NICER.hot_atmosphere_Q = this_directory+'/../model_data/Bobrikova_compton_slab_Q.npz'
    
            self.photosphere_IXPE = CustomPhotosphereDiskLine(hot = self.hot_IXPE, 
                                                         elsewhere = None, 
                                                         stokes=True, 
                                                         disk=self.disk_IXPE, 
                                                         values=photosphere_values, 
                                                         bounds=photosphere_bounds,
                                                         prefix='IXPE')
    
            self.photosphere_IXPE.hot_atmosphere = self.file_atmosphere
            self.photosphere_IXPE.hot_atmosphere_Q = this_directory+'/../model_data/Bobrikova_compton_slab_Q.npz'
            
            self.photospheres = [self.photosphere_NICER, self.photosphere_IXPE]
            
        
        elif not self.variable_params:
            photosphere_bounds = dict(spin_axis_position_angle = (None, None))
            self.photosphere = CustomPhotosphereDiskLine(hot = self.hot, 
                                                         elsewhere = None, 
                                                         stokes=True if self.polarization else False, 
                                                         disk=self.disk, 
                                                         line=self.line,
                                                         values=dict(mode_frequency = self.spacetime['frequency']), 
                                                         bounds=photosphere_bounds)
    
            self.photosphere.hot_atmosphere = self.file_atmosphere
            self.photosphere.hot_atmosphere_Q = this_directory+'/../model_data/Bobrikova_compton_slab_Q.npz'
    
    def set_star(self):
        if self.variable_params:
            photospheres=self.photospheres
        elif not self.variable_params:
            photospheres=self.photosphere
        self.star = xpsi.Star(spacetime = self.spacetime, photospheres = photospheres)
        
    def set_interstellar(self):
        # bounds = None 
        bounds = self.bounds['column_density']
        values = None #self.pv.column_density
        self.interstellar=CustomInterstellar.from_SWG(self.file_interstellar, bounds=bounds, value=values)

        
        
    def set_disk(self):
        from Disk import Disk, k_disk_derive
        bounds = dict(T_in_keV = self.bounds["T_in_keV"],
                      R_in = self.bounds["R_in"],
                      K_disk = None)
        if self.bkg == 'disk':    
            self.k_disk_NICER = k_disk_derive()
            self.disk_NICER = Disk(bounds=bounds, values={'K_disk': self.k_disk_NICER}, prefix='NICER')
            self.k_disk_NICER.disk = self.disk_NICER
            
            self.k_disk_IXPE = k_disk_derive()
            self.disk_IXPE = Disk(bounds=bounds, values={'K_disk': self.k_disk_IXPE}, prefix='IXPE')
            self.k_disk_IXPE.disk = self.disk_IXPE
            
        elif self.bkg == 'disk_NICER':              
            self.k_disk_NICER = k_disk_derive()
            self.disk_NICER = Disk(bounds=bounds, values={'K_disk': self.k_disk_NICER})
            self.k_disk_NICER.disk = self.disk_NICER
            
            self.disk_IXPE = None
            
        else:
            self.disk = None
            
    def set_line(self):
        from GaussianLine import GaussianLine
              
        if 'line' in self.bkg:
            line_values = {}
            
            line_bounds = dict(
                mu = self.bounds['mu'],
                sigma = self.bounds['sigma'],
                N = self.bounds['N'],
                )
                
            self.line = GaussianLine(bounds=line_bounds, values=line_values)
        else:
            self.line = None

    def set_signal(self):
        self.set_data_NICER()
        self.set_instrument_NICER()    
        
        if self.signal_phase_shift:
            phase_values = {}
            phase_bounds = dict(phase_shift = (-0.5, 0.5))
        else:
            phase_values = None
            phase_bounds = None
        
        self.signal_NICER = CustomSignal(data = self.NICER_data,
                            instrument = self.NICER,
                            background = None,
                            photosphere_prefix = 'NICER',
                            interstellar = self.interstellar,
                            cache = False, # only true if verifying code implementation otherwise useless slowdown.
                            bounds=phase_bounds,
                            values=phase_values,
                            bkg = self.bkg,
                            epsrel = 1.0e-8,
                            epsilon = 1.0e-3,
                            sigmas = 10.0)        
        self.signals = [[self.signal_NICER],]

        self.set_data_IXPE()
        self.set_instrument_IXPE()
        self.signal_IXPE_I_DU1 = CustomSignal_poisson(data = self.IXPE_I_DU1_data,
                                instrument = self.IXPE_du1_I,
                                photosphere_prefix = 'IXPE',
                                interstellar = self.interstellar,
                                workspace_intervals = 1000,
                                cache = False,
                                epsrel = 1.0e-8,
                                epsilon = 1.0e-3,
                                sigmas = 10.0,
                                support = None,
                                stokes="I")
        self.signals.append([self.signal_IXPE_I_DU1])
        
        self.signal_IXPE_I_DU2 = CustomSignal_poisson(data = self.IXPE_I_DU2_data,
                                instrument = self.IXPE_du2_I,
                                photosphere_prefix = 'IXPE',
                                interstellar = self.interstellar,
                                workspace_intervals = 1000,
                                cache = False,
                                epsrel = 1.0e-8,
                                epsilon = 1.0e-3,
                                sigmas = 10.0,
                                support = None,
                                stokes="I")
        self.signals[1].append(self.signal_IXPE_I_DU2)
        
        self.signal_IXPE_I_DU3 = CustomSignal_poisson(data = self.IXPE_I_DU3_data,
                                instrument = self.IXPE_du3_I,
                                photosphere_prefix = 'IXPE',
                                interstellar = self.interstellar,
                                workspace_intervals = 1000,
                                cache = False,
                                epsrel = 1.0e-8,
                                epsilon = 1.0e-3,
                                sigmas = 10.0,
                                support = None,
                                stokes="I")
        self.signals[1].append(self.signal_IXPE_I_DU3)
        
        
        
        self.signal_IXPE_Q_DU1 = CustomSignal_gaussian(data = self.IXPE_Q_DU1_data,
                                instrument = self.IXPE_du1,
                                photosphere_prefix = 'IXPE',
                                interstellar = self.interstellar,
                                workspace_intervals = 1000,
                                cache = False,
                                epsrel = 1.0e-8,
                                epsilon = 1.0e-3,
                                sigmas = 10.0,
                                support = None,
                                stokes="Q")
        self.signals[1].append(self.signal_IXPE_Q_DU1)
        
        
        self.signal_IXPE_Q_DU2 = CustomSignal_gaussian(data = self.IXPE_Q_DU2_data,
                                instrument = self.IXPE_du2,
                                photosphere_prefix = 'IXPE',
                                interstellar = self.interstellar,
                                workspace_intervals = 1000,
                                cache = False,
                                epsrel = 1.0e-8,
                                epsilon = 1.0e-3,
                                sigmas = 10.0,
                                support = None,
                                stokes="Q")
        self.signals[1].append(self.signal_IXPE_Q_DU2)
        
        
        self.signal_IXPE_Q_DU3 = CustomSignal_gaussian(data = self.IXPE_Q_DU3_data,
                                instrument = self.IXPE_du3,
                                interstellar = self.interstellar,
                                photosphere_prefix = 'IXPE',
                                workspace_intervals = 1000,
                                cache = False,
                                epsrel = 1.0e-8,
                                epsilon = 1.0e-3,
                                sigmas = 10.0,
                                support = None,
                                stokes="Q")
        self.signals[1].append(self.signal_IXPE_Q_DU3)
        
        
        self.signal_IXPE_U_DU1 = CustomSignal_gaussian(data = self.IXPE_U_DU1_data,
                                instrument = self.IXPE_du1,
                                photosphere_prefix = 'IXPE',
                                interstellar = self.interstellar,
                                workspace_intervals = 1000,
                                cache = False,
                                epsrel = 1.0e-8,
                                epsilon = 1.0e-3,
                                sigmas = 10.0,
                                support = None,
                                stokes="U")
        self.signals[1].append(self.signal_IXPE_U_DU1)
        
        
        self.signal_IXPE_U_DU2 = CustomSignal_gaussian(data = self.IXPE_U_DU2_data,
                                instrument = self.IXPE_du2,
                                photosphere_prefix = 'IXPE',
                                interstellar = self.interstellar,
                                workspace_intervals = 1000,
                                cache = False,
                                epsrel = 1.0e-8,
                                epsilon = 1.0e-3,
                                sigmas = 10.0,
                                support = None,
                                stokes="U")
        self.signals[1].append(self.signal_IXPE_U_DU2)
        
        
        self.signal_IXPE_U_DU3 = CustomSignal_gaussian(data = self.IXPE_U_DU3_data,
                                instrument = self.IXPE_du3,
                                photosphere_prefix = 'IXPE',
                                interstellar = self.interstellar,
                                workspace_intervals = 1000,
                                cache = False,
                                epsrel = 1.0e-8,
                                epsilon = 1.0e-3,
                                sigmas = 10.0,
                                support = None,
                                stokes="U")
        self.signals[1].append(self.signal_IXPE_U_DU3)

    def set_parameter_vector(self):
        self.p = self.pv.p()
        print('self.p:',self.p)

    def set_prior(self):
        self.prior = CustomPrior(self.scenario, self.bkg, fix_mass = self.fix_mass, eos_informed=self.eos_informed, variable_params=self.variable_params)
        
    def set_likelihood(self):
        self.set_spacetime() # self.spacetime is defined here
        self.set_photosphere() # self.k_disk is defined here
        self.k_disk_NICER.spacetime = self.spacetime
        if self.bkg == 'disk':
            self.k_disk_IXPE.spacetime = self.spacetime          
        self.set_star() # star is defined afterwards
        self.set_signal()
        self.set_prior()
        
        self.likelihood = xpsi.Likelihood(star = self.star, 
                                           signals = self.signals,
                                           num_energies=self.num_energies, #128
                                           threads=1,
                                           prior=self.prior,
                                           externally_updated=True)
        


        
        if self.scenario == 'J1444s':
            if self.poisson_seed == 1:
                true_logl = 1.8694056662e+05
            elif self.poisson_seed == 0:
                true_logl = 1.8788034922e+05
            elif self.poisson_seed == 42:
                if self.channel_min == 20:
                    true_logl = 1.8061835150e+05
                elif self.channel_min == 100:
                    true_logl = 1.8269371046e+05 #data start at ch 100, seed 42, low res
            # true_logl = 1.8903850924e+05 #data start at ch 100, input 700, hi res
            # true_logl = 1.8738168720e+05 #nonoise
            # true_logl = 1.8733692430e+05 #nonoise, low res data
            # true_logl = 1.8742408005e+05 #low res data
            # true_logl = 1.8751140823e+05
        if self.scenario in ('J1444', 'J1444_STS'):
            if self.channel_min == 20:
                true_logl = 1.3525318684e+07
            elif self.channel_min == 100:
                true_logl = 1.3594326983e+07
                if self.polarization == 'iqu':
                    true_logl = 1.0198724313e+07


        self.true_logl = true_logl
        
    def __call__(self):
        # start call with a likelihood check
        t_check = time.time()
        self.likelihood.check(None, [self.true_logl], 1.0e6, physical_points=[self.p], force_update=True)
        print('Likelihood check took {:.3f} seconds'.format((time.time()-t_check)))
        
        print('param values',self.likelihood.params)
        
        analysis_name = self.analysis_name
        

        folderstring = f'{analysis_name}'

        try: 
            os.makedirs(folderstring)
        except OSError:
            if not os.path.isdir(folderstring):
                raise
        
        print('plotting...')
        
        rcParams['text.usetex'] = False
        rcParams['font.size'] = 14.0
        
        # # Likelihood check and plot
        # from matplotlib import cm
        # fig, ax = plot_2D_pulse((self.photosphere.signal[0][0],),
        #               x=self.signal_NICER.phases[0],
        #               shift=self.signal_NICER.shifts,
        #               y=self.signal_NICER.energies,
        #               ylabel=r'Energy (keV)',
        #               cm=cm.jet)

        
        # plt.savefig('{}/pre_sampling_plot.png'.format(folderstring))
        # print('figure saved in {}'.format(folderstring))
        
        
        fig, axes = plt.subplots(3,1,figsize=(5,8))
       
        profile = CustomAxes.plot_2D_counts(axes[0], self.NICER_data.counts, get_mids_from_edges(self.NICER_data.phases), get_mids_from_edges(self.NICER.channel_edges))
        profile = CustomAxes.plot_2D_counts(axes[1], self.signal_NICER.expected_counts, get_mids_from_edges(self.NICER_data.phases), get_mids_from_edges(self.NICER.channel_edges))
        profile = CustomAxes.plot_2D_counts(axes[2], self.signal_NICER.expected_counts-self.NICER_data.counts, get_mids_from_edges(self.NICER_data.phases), get_mids_from_edges(self.NICER.channel_edges))
        fig.colorbar(profile, ax=axes[0])
        fig.colorbar(profile, ax=axes[1])
        fig.colorbar(profile, ax=axes[2])
        axes[2].set_title('expected-data')
        fig.tight_layout()
        
        if self.run_type == 'sample':
            if self.sampler == 'multi':
                wrapped_params = [0]*len(self.likelihood)
                if self.scenario == 'J1444_STS':
                    wrapped_params[self.likelihood.index('NICER__p__phase_shift')] = 1
                    wrapped_params[self.likelihood.index('IXPE__p__phase_shift')] = 1
                elif self.scenario == 'J1444_STU':
                    wrapped_params[self.likelihood.index('p__phase_shift')] = 1
                    wrapped_params[self.likelihood.index('s__phase_shift')] = 1
                else:
                    wrapped_params[self.likelihood.index('phase_shift')] = 1
                outputfiles_basename = f'./{folderstring}/run_ST_'
                runtime_params = {'resume': True,
                                  'importance_nested_sampling': False,
                                  'multimodal': False,
                                  'n_clustering_params': None,
                                  'outputfiles_basename': outputfiles_basename,
                                  'n_iter_before_update': 100,
                                  'n_live_points': self.live_points,
                                  'sampling_efficiency': 0.1,
                                  'const_efficiency_mode': False,
                                  'wrapped_params': wrapped_params,
                                  'evidence_tolerance': 0.5,
                                  'seed': 7,
                                  'verbose': True}
            elif self.sampler == 'ultra':
                wrapped_params = [False]*len(self.likelihood)
                wrapped_params[self.likelihood.index('phase_shift')] = True
                sampler_params = {'wrapped_params': wrapped_params, 
                                  'log_dir': folderstring}
                if self.max_iter == -1:
                    self.max_iter = None
                runtime_params={'max_iters':self.max_iter,
                                'min_num_live_points': self.live_points}


            print('runtime_params: ', runtime_params)
            
            
            
            print("sampling starts ...")
            t_start = time.time()
            
            sys.stdout.flush()
            
            
            if self.sampler == 'multi':
                xpsi.Sample.nested(self.likelihood, self.prior,**runtime_params)
            elif self.sampler == 'ultra':
                # from xpsi.UltranestSampler import UltranestCalibrator
                # sampler_instance = UltranestCalibrator(self.likelihood, self.prior, sampler_params=sampler_params, use_stepsampler=True, stepsampler_params={})      
                # for nstep, results in sampler_instance.run(**runtime_params):
                #     print(f" {nstep:d}", results, sep='\n')
                xpsi.Sample.ultranested(self.likelihood, self.prior, sampler_params=sampler_params,runtime_params=runtime_params, use_stepsampler=True, stepsampler_params={'nsteps': 200})
            print("... sampling done")
            print('Sampling took {:.3f} seconds'.format((time.time()-t_start)))
            
        elif self.run_type == 'test':
            # print('test: inverse sampling prior')

            t_start = time.time()

            n_priors = 10000
            # inverse sampling test
            test=self.prior.draw(ndraws=n_priors)[0]#[:,0:-1]
            
            print(f'time to draw {n_priors} priors:',time.time()-t_start)
            # names_dictionary = self.pv.names()
            # labels_dictionary = self.pv.labels()
            # axis_labels = [labels_dictionary[key] for key in names_dictionary]
            
            import corner
            figure=corner.corner(test)
            figure.tight_layout()
            figure.savefig(f'{folderstring}/prior_test.png',dpi=50)
            print('Test took {:.3f} seconds'.format((time.time()-t_start)))

            
            
if __name__ == '__main__':
    Analysis = analysis('test', 
                        'disk_NICER', 
                        sampler='multi', 
                        scenario='J1444_STS', 
                        support_factor=None, 
                        poisson_seed=42, 
                        eos_informed=False, 
                        polarization='iqu', 
                        channel_min=100)
    Analysis()
