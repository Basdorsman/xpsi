import os
import sys
this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory+'/../')

import numpy as np
import math
import time
from matplotlib import rcParams
import matplotlib.pyplot as plt

import xpsi
np.random.seed(xpsi._rank+10)
print('Rank reporting: %d' % xpsi._rank)

from xpsi.global_imports import gravradius

from CustomPrior import CustomPrior
from CustomInstrument import CustomInstrument_fits, CustomInstrument_stokes
from CustomPhotosphere import CustomPhotosphereDiskLine
from CustomInterstellar import CustomInterstellar
from CustomSignal import CustomSignal, CustomSignal_gaussian
from CustomHotregion import CustomHotRegion_Accreting
from CustomLikelihood import CustomLikelihood

from parameter_values import parameter_values
from helper_functions import get_T_in_log10_Kelvin, plot_2D_pulse, CustomAxes, get_mids_from_edges

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
                 polarization=False):

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
            self.num_energies = 40 # 128
            pass
        print(f'num_energies: {self.num_energies}')
            
        try:
            self.num_leaves = int(os.environ.get('num_leaves'))
        except:
            print('num_leaves from environment variables failed, proceeding with default.')
            self.num_leaves = 30 # 50 avoids interpolation error with polarisation # 30 #128
            pass
        print(f'num_leaves: {self.num_leaves}')
    
        try:
            self.sqrt_num_cells = int(os.environ.get('sqrt_num_cells'))
        except:
            print('sqrt_num_cells from environment variables failed, proceeding with default.')
            self.sqrt_num_cells = 50 # 128
            pass
        print(f'sqrt_num_cells: {self.sqrt_num_cells}')
    
        try:
            self.num_rays = int(os.environ.get('num_rays'))
        except:
            print('num_rays from env. var. failed, proceeding with default.')
            self.num_rays = 512
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

        self.pv = parameter_values(self.scenario, self.bkg, self.fix_mass, polarization=self.polarization)
        self.file_locations()
        self.set_bounds()
        self.set_interstellar()
        self.set_likelihood()

    def file_locations(self):
        self.this_directory = this_directory
        
        if self.scenario in ('large_r', 'small_r', 'J1444s'):
            self.file_pulse_profile = self.this_directory + f'/data/NICER_products/data/{self.scenario}_seed={self.poisson_seed}_realisation.dat'
        if self.scenario == 'J1444':
            self.file_pulse_profile = self.this_directory + f'/data/NICER_products/data/J1444_preprocessed.txt'
       
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


        self.min_input = 0 # 20 is used with 0.3 keV (channel_low=30). 0 is used with 0.2 keV (channel_low=20). 900 works with channel_low = 120 (1.2 keV). 
        self.channel_low = 20 # 20 corresponds to 0.2 keV. # 30 corresponds to 0.3 keV
        self.channel_hi = 580 # 300 corresponds to 3 keV. 600 corresponds to 6 keV (98.7% of total counts retained)
        self.max_input = 1880 # 1400 works with channel-hi = 300. 2000 works with channel_hi = 600 (6 keV)



        settings = dict(counts = np.loadtxt(self.file_pulse_profile, dtype=np.double),
                        channels=np.arange(self.channel_low,self.channel_hi),
                        phases=self.phases_space,
                        first=0, 
                        last=self.channel_hi-self.channel_low-1,
                        exposure_time=self.exposure_time)

        self.NICER_data = xpsi.Data(**settings)
        
    def set_data_IXPE(self):
        from ixpe_read_pcube3 import readData_pcube_ebin

        fname_ixpedata = this_directory+"/data/ixpe_products/ixpeobssimdata_scenarioB/pcube_10bin/model_amsp_xpsi"
        fname_ixpedata_pulse = this_directory+"/data/ixpe_products/ixpeobssimdata_scenarioB/pcube_20bin/model_amsp_xpsi"
        
        
        
        phase_IXPE, Idat1, qn, un, Iderr1, qnerr, unerr, PD, PDerr, keVdat, MDP99 = readData_pcube_ebin(fname_ixpedata, NPhadat=10)
        phase_IXPE_pulse, Idat2, qn2, un2, Iderr2, qnerr2, unerr2, PD2, PDerr2, keVdat2, MDP99_2 = readData_pcube_ebin(fname_ixpedata_pulse, NPhadat=20)
        
        
        self.IXPE_I_data = xpsi.Data([Idat2[:,0]/np.max(Idat2[:,0])],
                               channels=np.arange(0, 1),
                               phases=np.linspace(0,1,len(phase_IXPE_pulse)+1),
                               first=0,
                               last=0,
                               exposure_time=1.0)
        self.IXPE_Q_data = xpsi.Data([qn[:,0]],
                               channels=np.arange(0, 1),
                               phases=np.linspace(0,1,len(phase_IXPE)+1),
                               first=0,
                               last=0,
                               exposure_time=1.0)
        self.IXPE_U_data = xpsi.Data([un[:,0]],
                               channels=np.arange(0, 1),
                               phases=np.linspace(0,1,len(phase_IXPE)+1),
                               first=0,
                               last=0,
                               exposure_time=1.0)
        
        self.IXPE_Q_data.phase_IXPE = phase_IXPE
        self.IXPE_U_data.phase_IXPE = phase_IXPE
        self.IXPE_I_data.phase_IXPE_pulse = phase_IXPE_pulse
        self.IXPE_I_data.errors, self.IXPE_Q_data.errors, self.IXPE_U_data.errors = Iderr2/np.max(Idat2[:,0]), qnerr, unerr
                
            
    def set_instrument_NICER(self):
        self.NICER = CustomInstrument_fits.from_response_files(
            self.RMF_file, 
            self.ARF_file,
            max_detection_channel=self.channel_hi, 
            min_detection_channel = self.channel_low, 
            max_input = self.max_input, #around the maximum
            min_input = self.min_input)

    def set_instrument_IXPE(self):
        self.IXPE = CustomInstrument_stokes.from_response_files(MRF = this_directory+'/data/ixpe_products/ixpe_d1_obssim_v012.mrf',
                                             RMF = this_directory+'/data/ixpe_products/ixpe_d1_obssim_v012.rmf',
                                             max_input = 275,
                                             max_channel = 200,
                                             min_input = 0,
                                             min_channel = 50,
                                             channel_edges = None)


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
        
        self.hot_kwargs = {'symmetry': True, #call for azimuthal invariance
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
                  #'prefix': 'p'}
        
        self.hotregion_bounds = dict(super_colatitude = self.bounds["super_colatitude"],
                                super_radius = self.bounds["super_radius"],
                                phase_shift = self.bounds["phase_shift"], 
                                super_tbb = self.bounds['super_tbb'],
                                super_tau = self.bounds['super_tau'],
                                super_te = self.bounds['super_te'])
        self.hot_values = {}
        
        primary = CustomHotRegion_Accreting(self.hotregion_bounds, self.hot_values, **self.hot_kwargs)


        self.hot = xpsi.HotRegions((primary,))

    def set_elsewhere(self):
        self.elsewhere = xpsi.Elsewhere(bounds=dict(elsewhere_temperature = self.bounds['elsewhere_temperature']))
        
    def set_photosphere(self):
        self.set_spacetime()
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
        self.set_photosphere()
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
            bounds = dict(T_in = get_T_in_log10_Kelvin(self.bounds["T_in"]),
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
        self.set_data_NICER()
        self.set_instrument_NICER()
        self.set_support()
        
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
    
        if self.polarization:
            if 'qu' in self.polarization:
                self.set_data_IXPE()
                self.set_instrument_IXPE()
                self.signals = [[self.signal_NICER],] # to apply disk correctly to signal, NICER must be first element.
            if 'i' in self.polarization:
                signalI = CustomSignal_gaussian(data = self.IXPE_I_data,
                                                instrument = self.IXPE,
                                                interstellar = self.interstellar,
                                                workspace_intervals = 1000,
                                                cache = False,
                                                epsrel = 1.0e-8,
                                                epsilon = 1.0e-3,
                                                sigmas = 10.0,
                                                support = None,
                                                stokes="I")
                self.signals[0].append(signalI)
            signalQ = CustomSignal_gaussian(data = self.IXPE_Q_data,
                                    instrument = self.IXPE,
                                    interstellar = self.interstellar,
                                    workspace_intervals = 1000,
                                    cache = False,
                                    epsrel = 1.0e-8,
                                    epsilon = 1.0e-3,
                                    sigmas = 10.0,
                                    support = None,
                                    stokes="Q")
            self.signals[0].append(signalQ)
            signalU = CustomSignal_gaussian(data = self.IXPE_U_data,
            	                instrument = self.IXPE,
            	                interstellar = self.interstellar,
            	                workspace_intervals = 1000,
            	                cache = False,
            	                epsrel = 1.0e-8,
            	                epsilon = 1.0e-3,
            	                sigmas = 10.0,
            	                support = None,
            	                stokes="U")
            self.signals[0].append(signalU)

    def set_parameter_vector(self):
        self.p = self.pv.p()
        print('again parameter vector', len(self.p))
        
   
    def set_prior(self):
        self.prior = CustomPrior(self.scenario, self.bkg, fix_mass = self.fix_mass, eos_informed=self.eos_informed)
        
    def set_likelihood(self):
        self.set_star()
        if 'disk' in self.bkg:
            self.k_disk.star = self.star
        self.set_signal()
        self.set_parameter_vector()
        self.set_prior()
        
        self.likelihood = CustomLikelihood(star = self.star, 
                                           signals = self.signals if self.polarization else self.signal_NICER,
                                           num_energies=self.num_energies, #128
                                           threads=1,
                                           prior=self.prior,
                                           externally_updated=True)
        
        

        
        if self.scenario == 'J1444s':
            true_logl = 1.8738168720e+05 #nonoise
            # true_logl = 1.8733692430e+05 #nonoise, low res data
            # true_logl = 1.8742408005e+05 #low res data
            # true_logl = 1.8751140823e+05
        if self.scenario == 'J1444':
            true_logl = 1.8
        
            
        if self.scenario == 'small_r':
            if self.polarization == 'qu':
                true_logl = 7.9265139733e+07 # with IXPE qu
            elif self.polarization == 'iqu':
                true_logl = 7.9265007576e+07 # with IXPE iqu
            elif not self.polarization:
                true_logl = 7.9265215141e+07 #without IXPE
        self.true_logl = true_logl
    
    def __call__(self):
        
        # start call with a likelihood check
        t_check = time.time()
        self.likelihood.check(None, [self.true_logl], 1.0e-6, physical_points=[self.p], force_update=True)
        print('Likelihood check took {:.3f} seconds'.format((time.time()-t_check)))
        
        
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
                wrapped_params[self.likelihood.index('phase_shift')] = 1
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
            print('there is no test')

            
            
if __name__ == '__main__':
    Analysis = analysis('local', 'test', 'disk', sampler='multi', scenario='J1444s', support_factor='100', fix_mass=False, eos_informed=False, polarization=False)
    Analysis()