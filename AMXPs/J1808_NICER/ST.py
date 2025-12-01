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
from CustomInstrument import CustomInstrument
from CustomPhotosphere import CustomPhotosphereDiskLine
from CustomInterstellar import CustomInterstellar
from CustomSignal import CustomSignal
from CustomHotregion import CustomHotRegion_Accreting
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
                 disk_combined=False):
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
            self.sqrt_num_cells = 50 #128
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
        
        #self.integrator = 'azimuthal_invariance' #'general/azimuthal_invariance'
        # self.interpolator = 'split' #'split/combined'
        self.pv = parameter_values(self.scenario, self.bkg, self.fix_mass)
        self.disk_combined = disk_combined
        self.file_locations()
        self.set_bounds()
        # self.set_values()
        self.set_interstellar()
        self.set_likelihood()
        
        t_check = time.time()
        #self.likelihood(self.p, reinitialise=True)
        
        
        print('parameters:', self.p)
        # print(self.likelihood)



        self.likelihood.check(None, [self.true_logl], 1e6, physical_points=[self.p], force_update=True)
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
        elif self.machine == 'snellius' or 'helios':
            self.file_atmosphere = self.this_directory + '/../model_data/Bobrikova_compton_slab.npz'
            self.file_interstellar = self.this_directory + "/../model_data/interstellar/tbnew/tbnew0.14.txt"
        if self.scenario == 'kajava' or self.scenario == 'literature' or self.scenario == '2019' or self.scenario == '2022' or self.scenario=='large_r':
            self.file_bkg = self.this_directory + f'/data/disk_2019.txt'
        elif self.scenario == 'small_r':
            self.file_bkg = self.this_directory + f'/data/disk_smallr.txt'
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
        fix_mass = self.fix_mass


        if fix_mass:
            values = dict(frequency = 401., mass = self.pv.mass)
        if not fix_mass:
            values = dict(frequency = 401.)

        if fix_mass:
            spacetime_bounds = dict(distance = self.bounds["distance"],
                                    radius = self.bounds["radius"],
                                    cos_inclination = self.bounds["cos_inclination"])
        if not fix_mass:
            spacetime_bounds = dict(distance = self.bounds["distance"],                       # (Earth) distance
                                    mass = self.bounds["mass"],                          # mass
                                    radius = self.bounds["radius"],     # equatorial radius
                                    cos_inclination = self.bounds["cos_inclination"])   

        self.spacetime = xpsi.Spacetime(bounds=spacetime_bounds, values=values)

    def set_hotregions(self):
        # self.num_rays = 16
        
        self.hot_kwargs = {'symmetry': True, #call for azimuthal invariance
                  'split': True,
                  'omit': False,
                  'cede': False,
                  'concentric': False,
                  'sqrt_num_cells': self.sqrt_num_cells,
                  #'min_sqrt_num_cells': 10,
                  #'max_sqrt_num_cells': 128,
                   'min_sqrt_num_cells': self.sqrt_num_cells,
                   'max_sqrt_num_cells': self.sqrt_num_cells,
                  'num_leaves': self.num_leaves,
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
        
        self.photosphere = CustomPhotosphereDiskLine(hot = self.hot, 
                                                     elsewhere = None, 
                                                     stokes=False, 
                                                     disk=self.disk, 
                                                     line=self.line,
                                                     disk_combined=self.disk_combined,
                                                     values=dict(mode_frequency = self.spacetime['frequency']))

        self.photosphere.hot_atmosphere = self.file_atmosphere

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
            data_spectrum = np.sum(self.data.counts, axis=1)/self.data.exposure_time        

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
            
            print('self.support', support)
            
            self.support = support
        
        
    def set_disk(self):
        from Disk import Disk, k_disk_derive
        if 'disk' in self.bkg:    
            bounds = dict(
                T_in = get_T_in_log10_Kelvin(self.bounds["T_in"]),
                # T_in_keV = self.bounds["T_in_keV"],
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
            print('no line')

    def set_signal(self):
        self.set_data()
        self.set_instrument()
        self.set_support()

        self.signal = CustomSignal(data = self.data,
                            instrument = self.instrument,
                            background = None,
                            interstellar = self.interstellar,
                            support = self.support,
                            cache = False, # only true if verifying code implementation otherwise useless slowdown.
                            bkg = self.bkg,
                            epsrel = 1.0e-8,
                            epsilon = 1.0e-3,
                            sigmas = 10.0,
                            disk_combined=self.disk_combined)
        
        
    def set_parameter_vector(self):
        self.p = self.pv.p()
        
   
    def set_prior(self):
        self.prior = CustomPrior(self.scenario, self.bkg, fix_mass = self.fix_mass, eos_informed=self.eos_informed)
        
    def set_likelihood(self):
        self.set_star()
        if 'disk' in self.bkg:
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
            if self.bkg == 'marginalise':
                true_logl = -1.1307400098e+05#-8.8549011385e+04 # marginalise
                if self.support_factor == 100 or self.support_factor == '100':
                    true_logl = -9.2194659551e+04
                if self.support_factor == 10 or self.support_factor == '10':
                    true_logl = -9.3134985012e+04
                if self.support_factor == 2 or self.support_factor == '2':
                    true_logl = -9.4081343510e+04
            elif self.bkg == 'fix':
                true_logl = 1.6789503475e+08 # empty background
            elif self.bkg == 'disk':
                true_logl = 1.6880517943e+08 # 1.5315194624e+08 #1.6880517943e+08
            elif self.bkg == 'line':
                true_logl = 1.6789503475e+08
            elif self.bkg == 'diskline':
                true_logl = 1.6880218511e+08 #1.6880517943e+08
        
        if self.scenario == '2022':
            if self.bkg == 'marginalise':
                true_logl = -1.1307400098e+05#-8.8549011385e+04 # marginalise
                if self.support_factor == 100 or self.support_factor == '100':
                    true_logl = -2.3127321809e+05
            elif self.bkg == 'fix':
                true_logl = 1.6789503475e+08 # empty background
            elif self.bkg == 'disk':
                true_logl = 1.1730546413e+08
            elif self.bkg == 'line':
                true_logl = 1.6789503475e+08
            elif self.bkg == 'diskline':
                true_logl = 1.1740674355e+08

            
        if self.scenario == 'large_r':
            if self.bkg == 'marginalise':
                true_logl = -9.0515374178e+04 #-8.7237365668e+04 # marginalise
            elif self.bkg == 'fix':
                true_logl = 1.6792913585e+08 # empty background
            elif self.bkg == 'disk':
                true_logl = 1.6880517943e+08
        
            
        if self.scenario == 'small_r':
            true_logl = 7.9265215141e+07
            if self.bkg == 'marginalise':
                if self.support_factor == 100 or self.support_factor == '100':
                    true_logl = -9.0260696431e+04
                elif self.support_factor == 'None' or self.support_factor == None:
                    true_logl = -8.7566701725e+04

        self.true_logl = true_logl
    
    def __call__(self):
        
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
        
        
        if self.run_type == 'plot':
            print('plotting...')
            
            rcParams['text.usetex'] = False
            rcParams['font.size'] = 14.0
            from matplotlib import cm
            
            # # plot photosphere signal
    
            # fig, ax = plot_2D_pulse((self.photosphere.signal[0][0],),
            #               x=self.signal.phases[0],
            #               shift=self.signal.shifts,
            #               y=self.signal.energies,
            #               ylabel=r'Energy (keV)',
            #               cm=cm.jet)
    
            
            # plt.savefig('{}/pre_sampling_plot.png'.format(folderstring))
            # print('figure saved in {}'.format(folderstring))
            
            
            # plot data, signal and residual
            fig, axes = plt.subplots(3,1,figsize=(5,8))
           
            profile = CustomAxes.plot_2D_counts(axes[0], self.data.counts, get_mids_from_edges(self.data.phases), get_mids_from_edges(self.instrument.channel_edges))
            profile = CustomAxes.plot_2D_counts(axes[1], self.signal.expected_counts, get_mids_from_edges(self.data.phases), get_mids_from_edges(self.instrument.channel_edges))
            profile = CustomAxes.plot_2D_counts(axes[2], self.signal.expected_counts-self.data.counts, get_mids_from_edges(self.data.phases), get_mids_from_edges(self.instrument.channel_edges))
            fig.colorbar(profile, ax=axes[0])
            fig.colorbar(profile, ax=axes[1])
            fig.colorbar(profile, ax=axes[2])     
            axes[2].set_title('expected-data')
            fig.tight_layout()
            
        
            #plot data
            fig, axes = plt.subplots(2,1, sharex=True)
            
            signal = self.data.counts
            phases = get_mids_from_edges(self.data.phases)
            channels = get_mids_from_edges(self.instrument.channel_edges)
            
            profile = axes[1].pcolormesh(phases,
                                       channels,
                                       signal,
                                       cmap = cm.jet,
                                       #vmin = vmin,
                                       #vmax = vmax,
                                       linewidth = 0,
                                       rasterized = True)
            axes[1].set_xlim([0.0, 1.0])
            axes[1].set_yscale('log')
            axes[1].set_ylabel(r'Energy (keV)')
            axes[1].set_xlabel(r'Phase')
            axes[1].set_yticks([0.5, 1.0, 2.0])
            axes[1].set_yticklabels([0.5, 1.0, 2.0])
            fig.colorbar(profile, ax=axes[1], label='Counts')
            
            
            signal_bol=np.sum(signal, axis=0)
            axes[0].step(self.data.phases, np.append(signal_bol,signal_bol[0]), where='post') 
            axes[0].set_ylabel('Counts')
            
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=np.min(signal), vmax=np.max(signal))
            dummy_mappable = ScalarMappable(norm=norm, cmap=cm.jet)
            dummy_mappable.set_array([])  # required for colorbar
            cbar_dummy = fig.colorbar(dummy_mappable, ax=axes[0])
            cbar_dummy.remove()  # safely remove dummy colorbar
            axes[0].set_title(f'{self.scenario} data')
            fig.tight_layout()
            
            fig.savefig(f'{folderstring}/data_plot_{self.scenario}.png')
          
        
        elif self.run_type == 'sample':

            


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
            print('test starts')
            # num_rays = [20, 512]
            # for num_ray in num_rays:
            #     self.hot_kwargs['num_rays']=num_ray
            #     print(self.hot_kwargs)
            #     primary = CustomHotRegion_Accreting(self.hotregion_bounds, self.hot_values, **self.hot_kwargs)
            #     self.hot = xpsi.HotRegions((primary,))
            #     self.hot.print_settings()

            #     self.photosphere = CustomPhotosphereDiskLine(hot = self.hot, elsewhere = None, stokes=False, disk=self.disk, line=self.line,
            #                                     values=dict(mode_frequency = self.spacetime['frequency']))
            #     self.photosphere.hot_atmosphere = self.file_atmosphere
            #     self.star = xpsi.Star(spacetime = self.spacetime, photospheres = self.photosphere)
            #     #self.star.update(force_update=True)
                
            #     self.likelihood = xpsi.Likelihood(star = self.star, signals = self.signal,
            #                                   num_energies=self.num_energies, #128
            #                                   threads=1,
            #                                   prior=self.prior,
            #                                   externally_updated=True)
            #     self.likelihood.check(None, [self.true_logl], 1.0e-4, physical_points=[self.p], force_update=True)
            


            # inverse sampling test
            # test=self.prior.draw(ndraws=10000)[0][:,0:2]
            # print(test.shape)
            # import corner
            # labels = [ "Mass (M☉)", "Radius (km)"]  # Adjust labels as needed
            # y_limits = (5, 15)  # Adjust as needed
            # x_limits = (1.0, 3.0)  # Adjust as needed
            # figure=corner.corner(test, labels=labels, quantiles=[0.16, 0.5, 0.84], 
            #            show_titles=True, title_fmt='.2f', range=[x_limits, y_limits])
            
            # 

            
            
            print('time integrator test')
            n_repeats = 1000
            i=0
            t_likelihood = 0
            timings_summed = np.zeros(4)
            
            while i < n_repeats:
                t_start = time.time()
                
                # # same sample
                # l_test = self.likelihood(self.p, reinitialise=True)
                # timings_summed = self.hot.objects[0]._integrator_timings
                # # print('phase_array: ', self.hot.objects[0]._interpolation_products[0])
                # # print('profile_array: ', self.hot.objects[0]._interpolation_products[1])
                # t_likelihood += time.time()-t_start
                # i+=1
                
                # radiating = self.hot.objects[0]._super_radiates
                
                # for i in range(radiating.shape[0]):
                #     print('radiating:', radiating[:,i])

                # random samples
                p_test = self.prior.inverse_sample()
                l_test = self.likelihood(p_test, reinitialise=True)
                if l_test > -1e89:
                    # print(l_test)
                    timings_summed += self.hot.objects[0]._integrator_timings
                    t_likelihood += time.time()-t_start
                    i+=1
   
            
            print(f'repeats={n_repeats}')
            print(f'Evaluation takes {(t_likelihood)/n_repeats:0.3f} seconds')
            
            print(f'signal eval: {timings_summed[0]/n_repeats:0.3f} seconds, {timings_summed[0]/t_likelihood*100:0.1f}% of likelihood')
            print(f'pre-atmosphere: {timings_summed[1]/n_repeats:0.3f} seconds, {timings_summed[1]/t_likelihood*100:0.1f}% of likelihood')
            print(f'intensities: {timings_summed[2]/n_repeats:0.3f} seconds, {timings_summed[2]/t_likelihood*100:0.1f}% of likelihood')
            print(f'phase interpolation: {timings_summed[3]/n_repeats:0.3f} seconds, {timings_summed[3]/t_likelihood*100:0.1f}% of likelihood')

            # profile = CustomAxes.plot_2D_counts(axes[0], self.data.counts, get_mids_from_edges(self.data.phases), get_mids_from_edges(self.instrument.channel_edges))
            # profile = CustomAxes.plot_2D_counts(axes[1], self.signal.expected_counts, get_mids_from_edges(self.data.phases), get_mids_from_edges(self.instrument.channel_edges))
            # profile = CustomAxes.plot_2D_counts(axes[2], self.signal.expected_counts-self.data.counts, get_mids_from_edges(self.data.phases), get_mids_from_edges(self.instrument.channel_edges))
            # fig.colorbar(profile, ax=axes[0])
            # fig.colorbar(profile, ax=axes[1])
            # fig.colorbar(profile, ax=axes[2])     
            # axes[2].set_title('expected-data')
            # fig.tight_layout()
    
            
if __name__ == '__main__':
    Analysis = analysis('local', 'test', 'disk', sampler='multi', scenario='2022', support_factor='None', fix_mass=False, eos_informed=False)
    Analysis()

    expected = Analysis.signal.expected_counts
    print('expected counts: ',np.sum(expected))
    
    
    phase_data_array = Analysis.hot.objects[0]._interpolation_products[0]
    intensity_data_array = Analysis.hot.objects[0]._interpolation_products[1]
    phase_query_array = Analysis.hot.objects[0]._interpolation_products[2]
    intensity_query_array = Analysis.hot.objects[0]._interpolation_products[3]
    cell_radiates = Analysis.hot.objects[0]._interpolation_products[4]
    
    np.savez_compressed("interpolation_products_reduced_correct_pulse.npz", 
                        phase_data_array=phase_data_array, 
                        intensity_data_array=intensity_data_array, 
                        phase_query_array=phase_query_array,
                        intensity_query_array=intensity_query_array,
                        cell_radiates=cell_radiates)
