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

from CustomPrior import CustomPrior_STU
from CustomInstrument import CustomInstrument
from CustomPhotosphereDisk import CustomPhotosphereDisk
from CustomInterstellar import CustomInterstellar
from CustomSignal import CustomSignal
from CustomHotregion import CustomHotRegion_Accreting

from parameter_values import parameter_values
from helper_functions import get_T_in_log10_Kelvin, plot_2D_pulse, CustomAxes, get_mids_from_edges

class analysis(object):
    def __init__(self, machine, run_type, bkg, sampler='multi', support_factor = "None", scenario = 'None', poisson_noise=True, poisson_seed=42):
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
            self.num_leaves = 30 #128
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
                if os.environ.get('support_factor') == None: #or os.environ.get('support_factor') == 'None':
                    print(f'No support_factor in os. Taken from passed or default argument: {support_factor}')
                    self.support_factor = support_factor
        elif self.bkg == 'model' or self.bkg == 'fix':
            self.support_factor = 'None'
        print(f'support_factor: {self.support_factor}')   
        
        self.poisson_noise = os.environ.get('poisson_noise')
        if self.poisson_noise == 'True':
            self.poisson_seed = int(os.environ.get('poisson_seed'))
        elif self.poisson_noise == None:
            self.poisson_noise = poisson_noise
            self.poisson_seed = poisson_seed
            print(f'No poisson noise decision in os. Taking default poisson')
        print(f'poisson_noise: {self.poisson_noise}, poisson_seed: {self.poisson_seed} (only relevant if poisson noise is True)')
       
        
        #self.integrator = 'azimuthal_invariance' #'general/azimuthal_invariance'
        # self.interpolator = 'split' #'split/combined'

        self.pv = parameter_values(self.scenario, self.bkg)
    
        self.file_locations()
        self.set_bounds()
        # self.set_values()
        self.set_interstellar()
        self.set_likelihood()
        
        t_check = time.time()
        #self.likelihood(self.p, reinitialise=True)
        self.likelihood.check(None, [self.true_logl], 1.0e-4, physical_points=[self.p], force_update=True)
        print('Likelihood check took {:.3f} seconds'.format((time.time()-t_check)))
        print(self.likelihood(self.p))


    def file_locations(self):
        self.this_directory = this_directory
        # if self.scenario == 'kajava' or self.scenario == 'literature':
        #     if self.poisson_noise:
        #         self.file_pulse_profile = self.this_directory + f'/data/synthetic_{self.scenario}_seed={self.poisson_seed}_realisation.dat' 
        #     elif not self.poisson_noise:
        #         self.file_pulse_profile = self.this_directory + f'/data/J1808_synthetic_{self.scenario}_realisation.dat'
        
        if self.scenario == 'large_r' or self.scenario == 'small_r':
                self.file_pulse_profile = self.this_directory + f'/data/synthetic_{self.scenario}_seed={self.poisson_seed}_realisation.dat'
        
        # real data
        if self.scenario == '2019' or self.scenario == '2022':
            self.file_pulse_profile = self.this_directory + f'/data/{self.scenario}_preprocessed.txt'
        
            self.file_arf = self.this_directory + f'/../model_data/instrument_data/J1808_NICER_{self.scenario}/merged_saxj1808_{self.scenario}_arf_aeff.txt'
            self.file_rmf = self.this_directory + f'/../model_data/instrument_data/J1808_NICER_{self.scenario}/merged_saxj1808_{self.scenario}_rmf_matrix.txt'
            self.file_channel_edges = self.this_directory + f'/../model_data/instrument_data/J1808_NICER_{self.scenario}/merged_saxj1808_{self.scenario}_rmf_energymap.txt'

        elif self.scenario == 'large_r' or self.scenario == 'small_r':
            self.file_arf = self.this_directory + f'/../model_data/instrument_data/J1808_NICER_2019/merged_saxj1808_2019_arf_aeff.txt'
            self.file_rmf = self.this_directory + f'/../model_data/instrument_data/J1808_NICER_2019/merged_saxj1808_2019_rmf_matrix.txt'
            self.file_channel_edges = self.this_directory + f'/../model_data/instrument_data/J1808_NICER_2019/merged_saxj1808_2019_rmf_energymap.txt'
            
            
        if self.machine == 'local':
            self.file_atmosphere = '/home/bas/Documents/Projects/x-psi/model_datas/bobrikova/Bobrikova_compton_slab.npz'
            self.file_interstellar = "/home/bas/Documents/Projects/x-psi/xpsi-bas-fork/AMXPs/model_data/n_H/TBnew/tbnew0.14.txt"
        elif self.machine == 'snellius':
            self.file_atmosphere = self.this_directory + '/../model_data/Bobrikova_compton_slab.npz'
            self.file_interstellar = "/home/dorsman/xpsi-bas-fork/AMXPs/model_data/interstellar/tbnew/tbnew0.14.txt"
        if self.scenario == 'kajava' or self.scenario == 'literature':
            self.file_bkg = self.this_directory + f'/data/background_countrate_{self.scenario}.txt'
        # self.file_bkg = self.this_directory + '/../model_data/synthetic/diskbb_background.txt'

    def set_bounds(self):
        self.bounds = self.pv.bounds()

    def set_data(self):
        if self.scenario == '2019' or self.scenario == 'large_r' or self.scenario == 'small_r':
            self.exposure_time = 1.32366e5 #Mason's 2019 data cut
        if self.scenario == '2022':
            self.exposure_time = 7.13422e4 #Mason's 2022 data cut
        
        self.phases_space = np.linspace(0.0, 1.0, 33)

        energy_range = 'large'

        if energy_range == 'small':
            self.min_input = 0 # 20 is used with 0.3 keV (channel_low=30). 0 is used with 0.2 keV (channel_low=20). 900 works with channel_low = 120 (1.2 keV). 
            self.channel_low = 20 # 20 corresponds to 0.2 keV. # 30 corresponds to 0.3 keV
            self.channel_hi = 300 # 300 corresponds to 3 keV. 600 corresponds to 6 keV (98.7% of total counts retained)
            self.max_input = 1400 # 1400 works with channel-hi = 300. 2000 works with channel_hi = 600 (6 keV)

        if energy_range == 'large':
            self.min_input = 20 # 20 is used with 0.3 keV (channel_low=30). 0 is used with 0.2 keV (channel_low=20). 900 works with channel_low = 120 (1.2 keV). 
            self.channel_low = 30 # 20 corresponds to 0.2 keV. # 30 corresponds to 0.3 keV
            self.channel_hi = 600 # 300 corresponds to 3 keV. 600 corresponds to 6 keV (98.7% of total counts retained)
            self.max_input = 2000 # 1400 works with channel-hi = 300. 2000 works with channel_hi = 600 (6 keV)



        settings = dict(counts = np.loadtxt(self.file_pulse_profile, dtype=np.double),
                        channels=np.arange(self.channel_low,self.channel_hi),
                        phases=self.phases_space,
                        first=0, 
                        last=self.channel_hi-self.channel_low-1,
                        exposure_time=self.exposure_time)

        self.data = xpsi.Data(**settings)
        
        
    def set_instrument(self):
        self.instrument = CustomInstrument.from_response_files(ARF = self.file_arf,
                RMF = self.file_rmf,
                channel_edges = self.file_channel_edges,       
                channel_low = self.channel_low,
                channel_hi = self.channel_hi,
                min_input = self.min_input,
                max_input = self.max_input)


    def set_spacetime(self):
    
        values = dict(frequency = 401.)
        # values = dict(frequency = 401.,
        #               mass = self.pv.mass,
        #               radius = self.pv.radius,
        #               distance = self.pv.distance)
        
        
    
        spacetime_bounds = dict(distance = self.bounds["distance"],                       # (Earth) distance
                                mass = self.bounds["mass"],                          # mass
                                radius = self.bounds["radius"],     # equatorial radius
                                cos_inclination = self.bounds["cos_inclination"])               # (Earth) inclination to rotation axis

        # spacetime_bounds = dict(distance = self.bounds["distance"],
        #                         cos_inclination = self.bounds["cos_inclination"])

        # spacetime_bounds = dict(cos_inclination = self.bounds["cos_inclination"])


        self.spacetime = xpsi.Spacetime(bounds=spacetime_bounds, values=values) # values=dict(frequency=self.values["frequency"]))

    def set_hotregions(self):
        # self.num_rays = 16
        
        p_kwargs = {'symmetry': True, #call for azimuthal invariance
                  'split': True,
                  'omit': False,
                  'cede': False,
                  'concentric': False,
                  'sqrt_num_cells': self.sqrt_num_cells,
                  'min_sqrt_num_cells': 10,
                  'max_sqrt_num_cells': 128,
                  'num_leaves': self.num_leaves,
                  'num_rays': self.num_rays,
                  'atm_ext':'Num5D',
                  'prefix': 'p'}
        
        hotregion_bounds = dict(super_colatitude = self.bounds["super_colatitude"],
                                super_radius = self.bounds["super_radius"],
                                phase_shift = self.bounds["phase_shift"], 
                                super_tbb = self.bounds['super_tbb'],
                                super_tau = self.bounds['super_tau'],
                                super_te = self.bounds['super_te'])
        values = {}
        
        primary = CustomHotRegion_Accreting(hotregion_bounds, values, **p_kwargs)

        s_kwargs = {'symmetry': True, #call for azimuthal invariance
                  'split': True,
                  'omit': False,
                  'cede': False,
                  'concentric': False,
                  'sqrt_num_cells': self.sqrt_num_cells,
                  'min_sqrt_num_cells': 10,
                  'max_sqrt_num_cells': 128,
                  'num_leaves': self.num_leaves,
                  'num_rays': self.num_rays,
                  'is_antiphased': True,
                  'atm_ext':'Num5D',
                  'prefix': 's'}
        
        hotregion_bounds = dict(super_colatitude = self.bounds["super_colatitude"],
                                super_radius = self.bounds["super_radius"],
                                phase_shift = self.bounds["phase_shift"], 
                                super_tbb = self.bounds['super_tbb'],
                                super_tau = self.bounds['super_tau'],
                                super_te = self.bounds['super_te'])
        values = {}
        
        secondary = CustomHotRegion_Accreting(hotregion_bounds, values, **s_kwargs)


        self.hot = xpsi.HotRegions((primary, secondary))

    def set_elsewhere(self):
        self.elsewhere = xpsi.Elsewhere(bounds=dict(elsewhere_temperature = self.bounds['elsewhere_temperature']))
        
    def set_photosphere(self):
        self.set_spacetime()
        self.set_hotregions()
        self.set_disk()
        self.photosphere = CustomPhotosphereDisk(hot = self.hot, elsewhere = None, stokes=False, custom=self.disk,
                                        values=dict(mode_frequency = self.spacetime['frequency']))

        self.photosphere.hot_atmosphere = self.file_atmosphere

    def set_star(self):
        self.set_photosphere()
        self.star = xpsi.Star(spacetime = self.spacetime, photospheres = self.photosphere)
        # print('self.star: ',self.star)
        
    def set_interstellar(self):
        # bounds = None 
        bounds = self.bounds['column_density']
        values = None #self.pv.column_density

        
        
        self.interstellar=CustomInterstellar.from_SWG(self.file_interstellar, bounds=bounds, value=values)
    
    def set_support(self):
        support_factor = self.support_factor
        if support_factor == "None":
            self.support = None
        else:
            support_factor = float(support_factor)
            bg_spectrum = np.loadtxt(self.file_bkg)
    
            allowed_deviation_factor = 1. + support_factor  # 1.00005 is Roughly 1 count difference given max count rate of 0.8/s and exp. time of 1.3e5
    
            support = np.zeros((len(bg_spectrum), 2), dtype=np.double)
            support[:,0] = bg_spectrum/allowed_deviation_factor #lower limit
            support[support[:,0] < 0.0, 0] = 0.0
            support[:,1] = bg_spectrum*allowed_deviation_factor #upper limit
    
            for i in range(support.shape[0]):
                if support[i,1] == 0.0:
                    for j in range(i, support.shape[0]):
                        if support[j,1] > 0.0:
                            support[i,0] = support[j,1]
                            break
            
            self.support = support
        
        
    def set_disk(self):
        from Disk import Disk, k_disk_derive
        if self.bkg == 'model':            
            bounds = dict(T_in = get_T_in_log10_Kelvin(self.bounds["T_in"]),
                          R_in = self.bounds["R_in"],
                          K_disk = None) #derived means no bounds
                
            self.k_disk = k_disk_derive()
            
            self.disk = Disk(bounds=bounds, values={'K_disk': self.k_disk})
            

            self.k_disk.disk = self.disk
            
        elif self.bkg == 'marginalise' or self.bkg == 'fix':
            self.background = None
        else:
            print('error! bkg must be either model or marginalised.')
            
    def set_signal(self):
        self.set_data()
        self.set_instrument()
        self.set_support()

        self.signal = CustomSignal(data = self.data,
                            instrument = self.instrument,
                            background = None, #self.background,
                            interstellar = self.interstellar,
                            support = self.support,
                            cache = False, # only true if verifying code implementation otherwise useless slowdown.
                            bkg = self.bkg,
                            epsrel = 1.0e-8,
                            epsilon = 1.0e-3,
                            sigmas = 10.0)
        
        
    def set_parameter_vector(self):
        parameters_single_hotspot = self.pv.p()
        
        # add second hotspot
        n_shift = 3
        n_hs_params = 6
        self.p = parameters_single_hotspot[:-n_shift] + parameters_single_hotspot[-(n_shift+n_hs_params):-n_shift] + parameters_single_hotspot[-n_shift:]
        # print('self.p: ',self.p)     
    
    def set_prior(self):
        self.prior = CustomPrior_STU(self.scenario, self.bkg)
        
    def set_likelihood(self):
        self.set_star()
        self.k_disk.star = self.star
        self.set_signal()
        self.set_parameter_vector()
        self.set_prior()
        
        self.likelihood = xpsi.Likelihood(star = self.star, signals = self.signal,
                  num_energies=self.num_energies, #128
                                      threads=1,
                                      prior=self.prior,
                                      externally_updated=True)



        
        if self.scenario == '2019':
            true_logl = 1.5844462356e+08 # ST-U
            # true_logl = 1.5315129891e+08 # no elsewhere
            # true_logl= -7.9418857894e+89 # 2019 data, marginalized background
        
        if self.scenario == '2022':
            true_logl = 1.5844462356e+08 # ST-U
            #true_logl = 1.0540960782e+08 # no elsewhere
            # true_logl = 1.1365193823e+08 # 2022 data
            
        if self.scenario == 'large_r':
            true_logl = 1.6881742361e+08
            
        if self.scenario == 'small_r':
            true_logl = 7.9265215639e+07
            

        self.true_logl = true_logl
    
    def __call__(self):
        
        analysis_name = self.analysis_name
        machine = self.machine
        
        if machine == 'local':
            folderstring = f'local_runs/{analysis_name}'
        elif machine == 'helios':
            folderstring = f'helios_runs/{analysis_name}'
        elif machine == 'snellius':
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
                      x=self.signal.phases[0],
                      shift=self.signal.shifts,
                      y=self.signal.energies,
                      ylabel=r'Energy (keV)',
                      cm=cm.jet)

        
        plt.savefig('{}/pre_sampling_plot.png'.format(folderstring))
        print('figure saved in {}'.format(folderstring))
        
        
        fig, ax = plt.subplots()
        #profile = CustomAxes.plot_2D_counts(ax, self.data.counts-self.signal.expected_counts, get_mids_from_edges(self.data.phases), self.data.channels)
        #profile = CustomAxes.plot_2D_counts(ax, self.data.counts, get_mids_from_edges(self.data.phases), self.data.channels)
        profile = CustomAxes.plot_2D_counts(ax, self.signal.expected_counts, get_mids_from_edges(self.data.phases), self.data.channels)
        
        
        fig.colorbar(profile, ax=ax)
        
        if self.run_type == 'sample':

            


            if self.sampler == 'multi':
                wrapped_params = [0]*len(self.likelihood)
                wrapped_params[self.likelihood.index('p__phase_shift')] = 1
                wrapped_params[self.likelihood.index('s__phase_shift')] = 1
                outputfiles_basename = f'./{folderstring}/run_ST_U_'
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
            print('test starts')
            n_repeats = 10
            t_start = time.time()
            for repeat in range(n_repeats):
                #self.star.update(force_update=True)
                #self.likelihood.check(None, [self.true_logl], 1.0e-4, physical_points=[self.p], force_update=True)
                self.likelihood(self.p, reinitialise=True)
            print('Test took {:.3f} seconds'.format((time.time()-t_start)))
            
            
if __name__ == '__main__':
    Analysis = analysis('local','sample', 'model', sampler='multi', scenario='2019')
    Analysis()
