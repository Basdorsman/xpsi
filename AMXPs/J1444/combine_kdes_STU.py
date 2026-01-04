# this file is imported by join_posteriors, which samples from NICER and IXPE 
# separate analysis. non-shared paramaters was set to true, and that modified
# the nr of param values in the parameter and pior inverse sample. Here the 
# free params are M, D, cosi, nH, NICER Rin and IXPE Rin.

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

from CustomPrior import CustomPrior_twohotspots as CustomPrior
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
                 channel_min=None,
                 combine_kdes=True,
                 nh_shared=False):

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
        
        if os.environ.get('channel_min') == None or os.environ.get('channel_min') == 'None':
            print('channel_min is not in environment variables, using passed argument.')
            self.channel_min = channel_min
        else:
            self.channel_min = int(os.environ.get('channel_min'))
        print(f'channel_min: {self.channel_min}') 


        if os.environ.get('nh_shared') == None or os.environ.get('nh_shared') =='None':
            print('nh_shared is not in environment variables, using passed argument.')
            self.nh_shared = nh_shared
        else:
            self.nh_shared = os.environ.get('nh_shared')
        if self.nh_shared == "qu" or self.nh_shared == "iqu":
            self.nh_shared = self.nh_shared
        else:
            self.nh_shared = False
        print(f'nh_shared: {self.nh_shared}')


        self.combine_kdes=combine_kdes
        self.pv = parameter_values(self.scenario, self.bkg, self.fix_mass, polarization=self.polarization, combine_kdes=self.combine_kdes, nh_shared=self.nh_shared)
        self.file_locations()
        self.set_parameter_vector()
        self.set_bounds()
        self.set_interstellar()
        self.set_likelihood()

    def file_locations(self):
        self.this_directory = this_directory
        
        if self.scenario in ('large_r', 'small_r', 'J1444s'):
            self.file_pulse_profile = self.this_directory + f'/data/NICER_products/data/{self.scenario}_seed={self.poisson_seed}_ch{self.channel_min}_realisation.dat'
        if self.scenario == 'J1444_STU':
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
        if self.scenario in ('J1444_STU','J1444s'):
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
            
    def set_instrument_NICER(self):     
        alpha_values=dict(alpha=1)
        alpha_bounds={}
        self.NICER = CustomInstrument_fits.from_response_files(
            bounds=alpha_bounds,
            values=alpha_values,
            RMF_file=self.RMF_file, 
            ARF_file=self.ARF_file,
            max_detection_channel=self.channel_hi, 
            min_detection_channel = self.channel_low, 
            max_input = self.max_input, #around the maximum
            min_input = self.min_input)

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
        
        self.p_bounds = {}
        self.p_values = dict(super_colatitude = self.pv.super_colatitude,
                             super_radius = self.pv.super_radius,
                             phase_shift = self.pv.phase_shift,
                             super_tbb = self.pv.tbb,
                             super_tau = self.pv.tau,
                             super_te = self.pv.te)
        
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
                  'is_antiphased': True,
                  'prefix': 's'}
        
        self.s_bounds = {}
        self.s_values = dict(super_colatitude = self.pv.super_colatitude_s,
                             super_radius = self.pv.super_radius_s,
                             phase_shift = self.pv.phase_shift_s,
                             super_tbb = self.pv.tbb_s,
                             super_tau = self.pv.tau_s,
                             super_te = self.pv.te_s)
        
        secondary = CustomHotRegion_Accreting(self.s_bounds, self.s_values, **self.s_kwargs)

        self.hot = xpsi.HotRegions((primary,secondary))

    def set_photosphere(self):
        self.set_hotregions()
        self.set_disk()
        
        photosphere_bounds = dict(spin_axis_position_angle = (None, None))
        self.photosphere = CustomPhotosphereDiskLine(hot = self.hot, 
                                                     elsewhere = None, 
                                                     stokes=True if self.polarization else False, 
                                                     disk=self.disk, 
                                                     line=None,
                                                     values=dict(mode_frequency = self.spacetime['frequency']), 
                                                     bounds=photosphere_bounds)

        self.photosphere.hot_atmosphere = self.file_atmosphere
    def set_star(self):
        # self.set_photosphere()
        self.star = xpsi.Star(spacetime = self.spacetime, photospheres = self.photosphere)
        
    def set_interstellar(self):
        if self.nh_shared:
            bounds = self.bounds['column_density']
            values = None #self.pv.column_density
        elif not self.nh_shared:
            bounds = None
            values = self.pv.column_density
        self.interstellar=CustomInterstellar.from_SWG(self.file_interstellar, bounds=bounds, value=values)

        
        
    def set_disk(self):
        # IXPE AND NICER NOT PROPERLY SEPARATED HERE (BUT IT DOESN'T MATTER FOR THIS PURPOSE)
        from Disk import Disk, k_disk_derive
        bounds = dict(R_in = self.bounds["R_in"],
                      K_disk = None)
        if self.bkg == 'disk':    
            self.k_disk_NICER = k_disk_derive()
            self.disk_NICER = Disk(bounds=bounds, 
                                   values=dict(T_in_keV = self.pv.T_in_keV,K_disk = self.k_disk_NICER), 
                                   prefix='NICER')
            self.k_disk_NICER.disk = self.disk_NICER
            
            self.k_disk_IXPE = k_disk_derive()
            self.disk_IXPE = Disk(bounds=bounds, 
                                   values=dict(T_in_keV = self.pv.T_in_keV,K_disk = self.k_disk_NICER), 
                                   prefix='IXPE')
            self.k_disk_IXPE.disk = self.disk_IXPE
            
            self.disk = [self.disk_NICER, self.disk_IXPE]
            
        elif self.bkg == 'disk_NICER':              
            self.k_disk_NICER = k_disk_derive()
            self.disk_NICER = Disk(bounds=bounds, 
                                   values=dict(T_in_keV = self.pv.T_in_keV,K_disk = self.k_disk_NICER), 
                                   prefix='NICER')
            self.k_disk_NICER.disk = self.disk_NICER
            
            self.disk = self.disk_NICER
            
        else:
            self.disk = None

    def set_signal(self):
        self.set_data_NICER()
        self.set_instrument_NICER()
        
        self.signal_NICER = CustomSignal(data = self.NICER_data,
                            instrument = self.NICER,
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
        self.prior = CustomPrior(self.scenario, self.bkg, fix_mass = self.fix_mass, eos_informed=self.eos_informed, combine_kdes=self.combine_kdes)
        
    def set_likelihood(self):
        self.set_spacetime() # self.spacetime is defined here
        self.set_photosphere() # self.k_disk is defined here
        self.k_disk_NICER.spacetime = self.spacetime
        if self.bkg == 'disk':
            self.k_disk_IXPE.spacetime = self.spacetime          
        self.set_star() # star is defined afterwards
        self.set_signal()
        self.set_prior()
        
        self.likelihood = CustomLikelihood(star = self.star, 
                                           signals = self.signals if self.polarization else self.signal_NICER,
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
        if self.scenario == 'J1444_STU':
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
        
        # alow failure in check (the disk is not correctly calculated)
        t_check = time.time()
        self.likelihood.check(None, [self.true_logl], 1.0e6, physical_points=[self.p], force_update=True)
        print('Likelihood check took {:.3f} seconds'.format((time.time()-t_check)))
        
        
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
            print('test: inverse sampling prior')

            t_start = time.time()

            
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
    Analysis = analysis('test', 
                        'disk', 
                        sampler='multi', 
                        scenario='J1444_STU', 
                        eos_informed=False, 
                        polarization=False, 
                        channel_min=100)
    Analysis()