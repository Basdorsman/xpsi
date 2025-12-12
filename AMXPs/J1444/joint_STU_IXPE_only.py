import os
import sys
this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory+'/../')

import numpy as np
import time
from matplotlib import rcParams
import matplotlib.pyplot as plt

import xpsi
np.random.seed(xpsi._rank+10)
print('Rank reporting: %d' % xpsi._rank)

from CustomPrior import CustomPrior_STU as CustomPrior
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
                 machine, 
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
        
        self.machine = os.environ.get('machine')
        if os.environ.get('machine') == None or os.environ.get('machine') =='None':
            print('machine variable is not in environment variables, using passed argument.')
            self.machine = machine
        print(f'machine: {self.machine}')

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

        secondary = True
        self.NICER = NICER
        self.pv = parameter_values(self.scenario, self.bkg, self.fix_mass, polarization=self.polarization, secondary=secondary)
        self.file_locations()
        self.set_parameter_vector()
        self.set_bounds()
        self.set_interstellar()
        self.set_likelihood()

    def file_locations(self):
        self.this_directory = this_directory
        
        if self.scenario in ('large_r', 'small_r', 'J1444s'):
            self.file_pulse_profile = self.this_directory + f'/data/NICER_products/data/{self.scenario}_seed={self.poisson_seed}_ch{self.channel_min}_realisation.dat'
        if self.scenario == 'J1444':
            self.file_pulse_profile = self.this_directory + f'/data/NICER_products/data/J1444_preprocessed_ch{self.channel_min}.txt'
       
        self.RMF_file = self.this_directory+'/data/NICER_products/srgaj1444.rmf'
        self.ARF_file = self.this_directory+'/data/NICER_products/srgaj1444.arf'

        if self.machine == 'local':
            self.file_atmosphere = '/home/bas/Documents/Projects/x-psi/model_datas/bobrikova/Bobrikova_compton_slab.npz'
            self.file_interstellar = "/home/bas/Documents/Projects/x-psi/xpsi-bas-fork/AMXPs/model_data/n_H/TBnew/tbnew0.14.txt"
        elif self.machine == 'snellius' or 'helios':
            self.file_atmosphere = self.this_directory + '/../model_data/Bobrikova_compton_slab.npz'
            self.file_interstellar = self.this_directory + "/../model_data/interstellar/tbnew/tbnew0.14.txt"
        if self.scenario == 'kajava' or self.scenario == 'literature' or self.scenario == '2019' or self.scenario == '2022' or self.scenario=='small_r' or self.scenario=='large_r':
            self.file_bkg = self.this_directory + '/data/disk_2019.txt'
        # self.file_bkg = self.this_directory + '/../model_data/synthetic/diskbb_background.txt'

    def set_bounds(self):
        self.bounds = self.pv.bounds()

    def set_data_NICER(self):
        if self.scenario == '2019' or self.scenario == 'large_r' or self.scenario == 'small_r':
            self.exposure_time = 1.32366e5 #Mason's 2019 data cut
        if self.scenario == '2022':
            self.exposure_time = 7.13422e4 #Mason's 2022 data cut
        if self.scenario in ('J1444','J1444s'):
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
        self.NICER = CustomInstrument_fits.from_response_files(
            self.RMF_file, 
            self.ARF_file,
            max_detection_channel=self.channel_hi, 
            min_detection_channel = self.channel_low, 
            max_input = self.max_input, #around the maximum
            min_input = self.min_input)

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
        
        alpha_bounds = dict(alpha = (0.8, 1.2))
        
        derive_du1_inst = derive_du1()
        derive_du2_inst = derive_du2()
        derive_du3_inst = derive_du3()
        
        self.IXPE_du1_I = CustomInstrument_stokes.from_response_files(
                                                     bounds=alpha_bounds,
                                                     values={},
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
                  'atm_ext':'Num5D',
                  'prefix': 'p'}
        
        self.p_bounds = dict(super_colatitude = self.bounds["super_colatitude"],
                                super_radius = self.bounds["super_radius"],
                                phase_shift = self.bounds["phase_shift"], 
                                super_tbb = self.bounds['super_tbb'],
                                super_tau = self.bounds['super_tau'],
                                super_te = self.bounds['super_te'])
        self.p_values = {}
        
        primary = CustomHotRegion_Accreting(self.p_bounds, self.p_values, **self.p_kwargs)


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
                  'is_antiphased': False,
                  'prefix': 's'}
        
        self.s_bounds = dict(super_colatitude = self.bounds["super_colatitude"],
                                super_radius = self.bounds["super_radius"],
                                phase_shift = self.bounds["phase_shift"], 
                                super_tbb = self.bounds['super_tbb'],
                                super_tau = self.bounds['super_tau'],
                                super_te = self.bounds['super_te'])
        self.s_values = {}
        
        secondary = CustomHotRegion_Accreting(self.s_bounds, self.s_values, **self.s_kwargs)

        self.hot = xpsi.HotRegions((primary,secondary))

    def set_elsewhere(self):
        self.elsewhere = xpsi.Elsewhere(bounds=dict(elsewhere_temperature = self.bounds['elsewhere_temperature']))
        
    def set_photosphere(self):
        self.set_hotregions()
        self.set_disk()
        self.set_line()
        
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
        # self.set_photosphere()
        self.star = xpsi.Star(spacetime = self.spacetime, photospheres = self.photosphere)
        
    def set_interstellar(self):
        # bounds = None 
        bounds = self.bounds['column_density']
        values = None #self.pv.column_density
        self.interstellar=CustomInterstellar.from_SWG(self.file_interstellar, bounds=bounds, value=values)
    
    def set_support(self):
        support_factor = self.support_factor
        if support_factor == "None" or support_factor == None:
            self.support = None
        else:
            data_spectrum = np.sum(self.NICER_data.counts, axis=1)/self.NICER_data.exposure_time        

            support_factor = float(support_factor)
            self.bg_spectrum = np.loadtxt(self.file_bkg)
    
            allowed_deviation_factor = support_factor  # used to be 1. + support_factor
    
            support = np.zeros((len(self.bg_spectrum), 2), dtype=np.double)
            support[:,0] = self.bg_spectrum/allowed_deviation_factor #lower limit
            support[support[:,0] < 0.0, 0] = 0.0
            support[:,1] = np.minimum(self.bg_spectrum*allowed_deviation_factor, data_spectrum) #upper limit
    
            for i in range(support.shape[0]):
                if support[i,1] == 0.0:
                    for j in range(i, support.shape[0]):
                        if support[j,1] > 0.0:
                            support[i,0] = support[j,1]
                            break
            
            self.support = support
        
        
    def set_disk(self):
        from Disk import Disk, k_disk_derive
        if 'disk' in self.bkg:    
            bounds = dict(#T_in = get_T_in_log10_Kelvin(self.bounds["T_in"]),
                          T_in_keV = self.bounds["T_in_keV"],
                          R_in = self.bounds["R_in"],
                          K_disk = None) #derived means no bounds
                
            self.k_disk = k_disk_derive()
            self.disk = Disk(bounds=bounds, values={'K_disk': self.k_disk})
            self.k_disk.disk = self.disk
            
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
        self.set_support()
        if self.NICER:
            self.set_data_NICER()
            self.set_instrument_NICER()     
            self.signal_NICER = CustomSignal(data = self.NICER_data,
                                instrument = self.NICER,
                                background = None,
                                interstellar = self.interstellar,
                                support = self.support,
                                cache = False, # only true if verifying code implementation otherwise useless slowdown.
                                bkg = self.bkg,
                                epsrel = 1.0e-8,
                                epsilon = 1.0e-3,
                                sigmas = 10.0)        
            self.signals = [[self.signal_NICER],]
        else:
            self.signals = [[],]
        
        if self.polarization:
            self.set_data_IXPE()
            self.set_instrument_IXPE()
            self.signal_IXPE_I_DU1 = CustomSignal_poisson(data = self.IXPE_I_DU1_data,
                                    instrument = self.IXPE_du1_I,
                                    interstellar = self.interstellar,
                                    workspace_intervals = 1000,
                                    cache = False,
                                    epsrel = 1.0e-8,
                                    epsilon = 1.0e-3,
                                    sigmas = 10.0,
                                    support = None,
                                    stokes="I")
            self.signals[0].append(self.signal_IXPE_I_DU1)
            
            self.signal_IXPE_I_DU2 = CustomSignal_poisson(data = self.IXPE_I_DU2_data,
                                    instrument = self.IXPE_du2_I,
                                    interstellar = self.interstellar,
                                    workspace_intervals = 1000,
                                    cache = False,
                                    epsrel = 1.0e-8,
                                    epsilon = 1.0e-3,
                                    sigmas = 10.0,
                                    support = None,
                                    stokes="I")
            self.signals[0].append(self.signal_IXPE_I_DU2)
            
            self.signal_IXPE_I_DU3 = CustomSignal_poisson(data = self.IXPE_I_DU3_data,
                                    instrument = self.IXPE_du3_I,
                                    interstellar = self.interstellar,
                                    workspace_intervals = 1000,
                                    cache = False,
                                    epsrel = 1.0e-8,
                                    epsilon = 1.0e-3,
                                    sigmas = 10.0,
                                    support = None,
                                    stokes="I")
            self.signals[0].append(self.signal_IXPE_I_DU3)
            
            
            
            self.signal_IXPE_Q_DU1 = CustomSignal_gaussian(data = self.IXPE_Q_DU1_data,
                                    instrument = self.IXPE_du1,
                                    interstellar = self.interstellar,
                                    workspace_intervals = 1000,
                                    cache = False,
                                    epsrel = 1.0e-8,
                                    epsilon = 1.0e-3,
                                    sigmas = 10.0,
                                    support = None,
                                    stokes="Q")
            self.signals[0].append(self.signal_IXPE_Q_DU1)
            
            
            self.signal_IXPE_Q_DU2 = CustomSignal_gaussian(data = self.IXPE_Q_DU2_data,
                                    instrument = self.IXPE_du2,
                                    interstellar = self.interstellar,
                                    workspace_intervals = 1000,
                                    cache = False,
                                    epsrel = 1.0e-8,
                                    epsilon = 1.0e-3,
                                    sigmas = 10.0,
                                    support = None,
                                    stokes="Q")
            self.signals[0].append(self.signal_IXPE_Q_DU2)
            
            
            self.signal_IXPE_Q_DU3 = CustomSignal_gaussian(data = self.IXPE_Q_DU3_data,
                                    instrument = self.IXPE_du3,
                                    interstellar = self.interstellar,
                                    workspace_intervals = 1000,
                                    cache = False,
                                    epsrel = 1.0e-8,
                                    epsilon = 1.0e-3,
                                    sigmas = 10.0,
                                    support = None,
                                    stokes="Q")
            self.signals[0].append(self.signal_IXPE_Q_DU3)
            
            
            self.signal_IXPE_U_DU1 = CustomSignal_gaussian(data = self.IXPE_U_DU1_data,
                                    instrument = self.IXPE_du1,
                                    interstellar = self.interstellar,
                                    workspace_intervals = 1000,
                                    cache = False,
                                    epsrel = 1.0e-8,
                                    epsilon = 1.0e-3,
                                    sigmas = 10.0,
                                    support = None,
                                    stokes="U")
            self.signals[0].append(self.signal_IXPE_U_DU1)
            
            
            self.signal_IXPE_U_DU2 = CustomSignal_gaussian(data = self.IXPE_U_DU2_data,
                                    instrument = self.IXPE_du2,
                                    interstellar = self.interstellar,
                                    workspace_intervals = 1000,
                                    cache = False,
                                    epsrel = 1.0e-8,
                                    epsilon = 1.0e-3,
                                    sigmas = 10.0,
                                    support = None,
                                    stokes="U")
            self.signals[0].append(self.signal_IXPE_U_DU2)
            
            
            self.signal_IXPE_U_DU3 = CustomSignal_gaussian(data = self.IXPE_U_DU3_data,
                                    instrument = self.IXPE_du3,
                                    interstellar = self.interstellar,
                                    workspace_intervals = 1000,
                                    cache = False,
                                    epsrel = 1.0e-8,
                                    epsilon = 1.0e-3,
                                    sigmas = 10.0,
                                    support = None,
                                    stokes="U")
            self.signals[0].append(self.signal_IXPE_U_DU3)

    def set_parameter_vector(self):
        self.p = self.pv.p()

    def set_prior(self):
        self.prior = CustomPrior(self.scenario, self.bkg, fix_mass = self.fix_mass, eos_informed=self.eos_informed)
        
    def set_likelihood(self):
        self.set_spacetime() # self.spacetime is defined here
        self.set_photosphere() # self.k_disk is defined here
        if 'disk' in self.bkg:
            self.k_disk.spacetime = self.spacetime
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
        if self.scenario == 'J1444':
            if self.channel_min == 20:
                true_logl = 1.3525318684e+07
            elif self.channel_min == 100:
                true_logl = 1.3594326983e+07
        
            
        if self.scenario == 'small_r':
            if self.polarization == 'qu':
                true_logl = 7.9265139733e+07 # with IXPE qu
            elif self.polarization == 'iqu':
                true_logl = 7.9265007576e+07 # with IXPE iqu
            elif not self.polarization:
                true_logl = 7.9265215141e+07 #without IXPE
        self.true_logl = true_logl
    
    def __call__(self):
        true_logl = -4.8856382662e+05
    
        # start call with a likelihood check
        t_check = time.time()
        self.likelihood.check(None, [true_logl], 1.0e6, physical_points=[self.p], force_update=True)
        print('Likelihood check took {:.3f} seconds'.format((time.time()-t_check)))
        
        print('param values',self.likelihood.params)
        
        def plot_signal(counts1, phases1, channels1, num_rot = 2, dpi=200, colormap='inferno'):
            """ Plot the data in a convenient way.

            :param int num_rot:
                The number of rotations to plot.

            :param int dpi:
                The resolution of the plot.

            :param str colormap:
                The colormap to use.
            """
            # Get the counts
            counts_list = [ counts1 for i in range(num_rot) ]
            phase_list = [phases1[:-1] + i for i in range(num_rot)] 
            counts = np.concatenate( (counts_list), axis=1 )
            phases = np.concatenate( (phase_list), axis=0 )

            # Do the plot
            fig,axs = plt.subplots( 2,2 , height_ratios=[1.,1.], width_ratios=[3,1.5], sharex='col',sharey='row')
            fig.subplots_adjust(wspace=0, hspace=0)
            axs[0,1].axis('off')

            # Plot the 2D data
            ax2 = axs[1,0]
            im = ax2.pcolormesh( phases, channels1 , counts, cmap=colormap)
            ax2.set_xlabel(r'Phase $\phi$ [cycles]')
            ax2.set_ylabel('Ph energy [keV]')#'PI channel')
            ax2.set_yscale('log')
            #ax2.set_ylim([0.1, 15.0])

            # Plot the pulse
            ax1 = axs[0,0]
            ax1.sharex( ax2 )
            ax1.tick_params(direction='in', which='both',labelbottom=False)

            ax1.errorbar( x=phases, y=counts.sum(axis=0), yerr=np.sqrt( counts.sum(axis=0) ), ds='steps-mid', color='black' )
            ax1.set_ylabel('Counts')
            
            # Plot the spectrum
            ax3 = axs[1,1]
            ax3.sharey( ax2 )
            ax3.set_yscale('log')
            ax3.tick_params(direction='in', which='both',labelbottom=False)

            ax3.step( counts.sum(axis=1)/2, channels1 , color='black')
            ax3.set_xlabel('Cts/chan.')

            # Add the colorbar    
            fig.colorbar( im , ax=ax3 , location='right',  label='Counts')
            fig.set_dpi(dpi)

            return fig, axs

        #print(likelihood)
        #exit()

        from matplotlib.collections import LineCollection

        def plot_QUplane():
            """ Plot Stokes in the Q-U plane """
            fig = plt.figure(figsize=(7,7))
            ax1 = fig.add_subplot(111)

            ax1.set_xlabel(r'$100 \times Q/I_{max}$')
            ax1.set_ylabel(r'$100 \times U/I_{max}$')

            ph = self.hot.phases_in_cycles[0]
            I1 = np.sum(self.photosphere.signal[0][0], axis=0)
            Q1 = np.sum(self.photosphere.signalQ[0][0], axis=0)
            U1 = np.sum(self.photosphere.signalU[0][0], axis=0)

            I2 = np.interp(ph, self.hot.phases_in_cycles[1], np.sum(self.photosphere.signal[1][0], axis=0))
            Q2 = np.interp(ph, self.hot.phases_in_cycles[1], np.sum(self.photosphere.signalQ[1][0], axis=0))
            U2 = np.interp(ph, self.hot.phases_in_cycles[1], np.sum(self.photosphere.signalU[1][0], axis=0))

            Itot = I1 + I2
            Qntot = 100*(Q1 + Q2)/np.max(Itot)
            Untot = 100*(U1 + U2)/np.max(Itot)

            ax1.axis([1.1*np.min(Qntot),1.1*np.max(Qntot),1.1*np.min(Untot),2.7*np.max(Untot)]) #set axis limits. This is [xlow, xhigh, ylow, yhigh]

            segments = [np.column_stack([Qntot[i:i+2], Untot[i:i+2]]) for i in range(len(Qntot) - 1)]
            lc = LineCollection(segments, cmap='hsv',array=ph,linewidth=4)
            line = ax1.add_collection(lc)

            l, b, h, w = .45, .65, .2, .4
            ax2 = fig.add_axes([l, b, w, h])
            segments = [np.column_stack([ph[i:i+2], Itot[i:i+2]/np.max(Itot)]) for i in range(len(ph) - 1)]
            ax2.axis([0,1,0,1.1])
            lc = LineCollection(segments, cmap='hsv',array=ph,linewidth=4)
            line = ax2.add_collection(lc)
            ax2.set_xlabel('phase')
            ax2.set_ylabel(r'$I\,/I_\mathrm{max}$')
            #veneer((0.05,0.2), (0.05,0.2), ax1)
            #veneer((0.05,0.2), (0.05,0.2), ax2)
        
        
        plot_QUplane()
        plt.savefig("qu_plane.png",bbox_inches='tight')
        
        from xpsi.utilities import PlottingLibrary as XpsiPlot
    
        
        #Plot all the data 
    
        fig, axs = plot_signal(self.IXPE_I_DU1_data.counts, self.IXPE_I_DU1_data.phases, self.IXPE_du1_I.channel_edges[1:], dpi=100 , colormap='inferno' , num_rot=2)
        plt.savefig("figs/data_I_du1_inkeV.png",bbox_inches='tight') 
        
        fig, axs = plot_signal( self.signal_IXPE_I_DU1.expected_counts, self.IXPE_I_DU1_data.phases, self.IXPE_du1_I.channel_edges[1:], dpi=100 , colormap='inferno' , num_rot=2)
        plt.savefig("figs/data_simulated_I_du1.png",bbox_inches='tight')    
        
        fig, axs = self.IXPE_I_DU2_data.plot( dpi=100 , colormap='inferno' , num_rot=2)
        plt.savefig("figs/data_I_du2.png",bbox_inches='tight')
    
        fig, axs = plot_signal( self.signal_IXPE_I_DU2.expected_counts, self.IXPE_I_DU2_data.phases, self.IXPE_du2_I.channel_edges[1:], dpi=100 , colormap='inferno' , num_rot=2)
        plt.savefig("figs/data_simulated_I_du2.png",bbox_inches='tight')    
        
        fig, axs = self.IXPE_I_DU3_data.plot( dpi=100 , colormap='inferno' , num_rot=2)
        plt.savefig("figs/data_I_du3.png",bbox_inches='tight')    
            
        fig, axs = plot_signal( self.signal_IXPE_I_DU3.expected_counts, self.IXPE_I_DU3_data.phases, self.IXPE_du3_I.channel_edges[1:], dpi=100 , colormap='inferno' , num_rot=2)
        plt.savefig("figs/data_simulated_I_du3.png",bbox_inches='tight')   
    
    
        fig, axs = self.IXPE_Q_DU1_data.plot( dpi=100 , colormap='inferno' , num_rot=2)
        plt.savefig("figs/data_Q_du1.png",bbox_inches='tight')
        
        fig, axs = self.IXPE_Q_DU2_data.plot( dpi=100 , colormap='inferno' , num_rot=2)
        plt.savefig("figs/data_Q_du2.png",bbox_inches='tight')
        
        fig, axs = self.IXPE_Q_DU3_data.plot( dpi=100 , colormap='inferno' , num_rot=2)
        plt.savefig("figs/data_Q_du3.png",bbox_inches='tight') 
    
    
        fig, axs = self.IXPE_U_DU1_data.plot( dpi=100 , colormap='inferno' , num_rot=2)
        plt.savefig("figs/data_U_du1.png",bbox_inches='tight')
        
        fig, axs = self.IXPE_U_DU2_data.plot( dpi=100 , colormap='inferno' , num_rot=2)
        plt.savefig("figs/data_U_du2.png",bbox_inches='tight')
        
        fig, axs = self.IXPE_U_DU3_data.plot( dpi=100 , colormap='inferno' , num_rot=2)
        plt.savefig("figs/data_U_du3.png",bbox_inches='tight')
        
        
        
        #Plot simulated data for the example parameters:
        
        from xpsi.tools import phase_interpolator
        
        #print("channels: ", self.IXPE_I_DU1_data.channels)
        XpsiPlot.plot_2d_pulse(pulse=self.signal_IXPE_I_DU1.expected_counts,
                           x=self.IXPE_I_DU1_data.phases,
                           y=self.IXPE_I_DU1_data.channels,
                           rotations=2,
                           ylabel='Channel',
                           cbar_label='Counts',)    
        plt.savefig("figs/model_I_du1.png")
        
        XpsiPlot.plot_2d_pulse(pulse=self.signal_IXPE_I_DU2.expected_counts, 
                           x=self.IXPE_I_DU2_data.phases,
                           y=self.IXPE_I_DU2_data.channels,
                           rotations=2,
                           ylabel='Channel',
                           cbar_label='Counts',)    
        plt.savefig("figs/model_I_du2.png")

     
        XpsiPlot.plot_2d_pulse(pulse=self.signal_IXPE_I_DU3.expected_counts,
                           x=self.IXPE_I_DU3_data.phases,
                           y=self.IXPE_I_DU3_data.channels,
                           rotations=2,
                           ylabel='Channel',
                           cbar_label='Counts',)    
        plt.savefig("figs/model_I_du3.png")

       
        XpsiPlot.plot_2d_pulse(pulse=self.signal_IXPE_Q_DU1.expected_counts,
                           x=self.IXPE_Q_DU1_data.phases,
                           y=self.IXPE_Q_DU1_data.channels,
                           rotations=2,
                           ylabel='Channel',
                           cbar_label='Counts',)    
        plt.savefig("figs/model_Q_du1.png")
        
     
        XpsiPlot.plot_2d_pulse(pulse=self.signal_IXPE_Q_DU2.expected_counts,
                           x=self.IXPE_Q_DU2_data.phases,
                           y=self.IXPE_Q_DU2_data.channels,
                           rotations=2,
                           ylabel='Channel',
                           cbar_label='Counts',)    
        plt.savefig("figs/model_Q_du2.png")

        XpsiPlot.plot_2d_pulse(pulse=self.signal_IXPE_Q_DU3.expected_counts,
                           x=self.IXPE_Q_DU3_data.phases,
                           y=self.IXPE_Q_DU3_data.channels,
                           rotations=2,
                           ylabel='Channel',
                           cbar_label='Counts',)    
        plt.savefig("figs/model_Q_du3.png")

        XpsiPlot.plot_2d_pulse(pulse=self.signal_IXPE_U_DU1.expected_counts,
                           x=self.IXPE_U_DU1_data.phases,
                           y=self.IXPE_U_DU1_data.channels,
                           rotations=2,
                           ylabel='Channel',
                           cbar_label='Counts',)    
        plt.savefig("figs/model_U_du1.png")
        
        XpsiPlot.plot_2d_pulse(pulse=self.signal_IXPE_U_DU2.expected_counts,
                           x=self.IXPE_U_DU2_data.phases,
                           y=self.IXPE_U_DU2_data.channels,
                           rotations=2,
                           ylabel='Channel',
                           cbar_label='Counts',)    
        plt.savefig("figs/model_U_du2.png")

        XpsiPlot.plot_2d_pulse(pulse=self.signal_IXPE_U_DU3.expected_counts,
                           x=self.IXPE_U_DU3_data.phases,
                           y=self.IXPE_U_DU3_data.channels,
                           rotations=2,
                           ylabel='Channel',
                           cbar_label='Counts',)    
        plt.savefig("figs/model_U_du3.png")
        
        
        analysis_name = self.analysis_name
        machine = self.machine
        
        if machine == 'local':
            folderstring = f'local_runs/{analysis_name}'
        elif machine == 'snellius' or 'helios':
            folderstring = f'{analysis_name}'

        try: 
            os.makedirs(folderstring)
        except OSError:
            if not os.path.isdir(folderstring):
                raise
        
        print('plotting...')
        
        rcParams['text.usetex'] = False
        rcParams['font.size'] = 14.0
        
        # Likelihood check and plot
        from matplotlib import cm
        fig, ax = plot_2D_pulse((self.photosphere.signal[0][0],),
                      x=self.signal_NICER.phases[0],
                      shift=self.signal_NICER.shifts,
                      y=self.signal_NICER.energies,
                      ylabel=r'Energy (keV)',
                      cm=cm.jet)

        
        plt.savefig('{}/pre_sampling_plot.png'.format(folderstring))
        print('figure saved in {}'.format(folderstring))
        
        
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
                wrapped_params[self.likelihood.index('p__phase_shift')] = 1
                wrapped_params[self.likelihood.index('s__phase_shift')] = 1
                outputfiles_basename = f'./{folderstring}/run_ST_'
                runtime_params = {'resume': False,
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

            # t_start = time.time()

            
            # # inverse sampling test
            # test=self.prior.draw(ndraws=100)[0]#[:,0:-1]
            # names_dictionary = self.pv.names()
            # labels_dictionary = self.pv.labels()
            # axis_labels = [labels_dictionary[key] for key in names_dictionary]
            
            # import corner
            # figure=corner.corner(test, labels=axis_labels[:19], label_kwargs={'fontsize': 12},)
            # figure.tight_layout()
            # figure.savefig(f'{folderstring}/prior.pdf',)
            print('Test took {:.3f} seconds'.format((time.time()-t_start)))

            
            
if __name__ == '__main__':
    Analysis = analysis('local', 
                        'test', 
                        'disk', 
                        sampler='multi', 
                        scenario='J1444', 
                        support_factor='100', 
                        poisson_seed=42, 
                        eos_informed=True, 
                        polarization='iqu', 
                        NICER=False,
                        channel_min=100)
    Analysis()