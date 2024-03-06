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
from CustomPhotosphere import CustomPhotosphere
from CustomInterstellar import CustomInterstellar
from CustomSignal import CustomSignal
from CustomHotregion import CustomHotRegion_Accreting

from helper_functions import get_T_in_log10_Kelvin, plot_2D_pulse

class analysis(object):
    def __init__(self, machine, run_type, bkg, support_factor = "None"):
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
            self.num_leaves = 30 # 128
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
            self.live_points = int(os.environ.get('live_points'))
        except:
            print('live_points from environment variables failed, proceeding with default.')
            self.live_points = 20 # 128
            pass
        print(f'live_points: {self.live_points}')
        
        try:
            self.max_iter = int(os.environ.get('max_iter'))
        except:
            print('max_iter from environment variables failed, proceeding with default.')
            self.max_iter = 10
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
                    print(f'support_factor is taken from passed or default argument: {support_factor}')
                    self.support_factor = support_factor
        elif self.bkg == 'model':
            self.support_factor = 'None'
        print(f'support_factor: {self.support_factor}')        
                
        
        self.integrator = 'azimuthal_invariance' #'general/azimuthal_invariance'
        self.interpolator = 'split' #'split/combined'
    
        self.file_locations()
        self.set_bounds()
        self.set_values()
        self.set_interstellar()
        self.set_likelihood()
        
        t_check = time.time()
        #self.likelihood(self.p, reinitialise=True)
        self.likelihood.check(None, [self.true_logl], 1.0e-4, physical_points=[self.p], force_update=True)
        print('Likelihood check took {:.3f} seconds'.format((time.time()-t_check)))
        print(self.likelihood(self.p))


    def file_locations(self):
        self.this_directory = this_directory
        # self.file_pulse_profile = self.this_directory + '/data/J1808_synthetic_realisation.dat' 
        # self.file_pulse_profile = self.this_directory + '/data/2022_preprocessed.txt' 
        self.file_pulse_profile = self.this_directory + '/data/2019_preprocessed.txt' 
        
        self.file_arf = self.this_directory + '/../model_data/instrument_data/J1808_NICER_2019/merged_saxj1808_2019_arf_aeff.txt'
        self.file_rmf = self.this_directory + '/../model_data/instrument_data/J1808_NICER_2019/merged_saxj1808_2019_rmf_matrix.txt'
        self.file_channel_edges = self.this_directory + '/../model_data/instrument_data/J1808_NICER_2019/merged_saxj1808_2019_rmf_energymap.txt'
        
        if self.machine == 'local':
            self.file_atmosphere = '/home/bas/Documents/Projects/x-psi/model_datas/bobrikova/Bobrikova_compton_slab.npz'
            self.file_interstellar = "/home/bas/Documents/Projects/x-psi/xpsi-bas-fork/AMXPs/model_data/n_H/TBnew/tbnew0.14.txt"
        elif self.machine == 'snellius':
            self.file_atmosphere = self.this_directory + '/../model_data/Bobrikova_compton_slab.npz'
            self.file_interstellar = "/home/dorsman/xpsi-bas-fork/AMXPs/model_data/interstellar/tbnew/tbnew0.14.txt"
        
        self.file_bkg = self.this_directory + '/../model_data/synthetic/diskbb_background.txt'
    
    def set_bounds(self):
        bounds = {}
        bounds["distance"] = (None, None)
        bounds["cos_i"] = (0.15, 1.0) #updated lower limit due to lack of eclipses, chakrabarty & Morgan 1998
        bounds["mass"] = (1.0, 3.0)
        bounds["radius"] = (3.0 * gravradius(1.0), 16.0)     # equatorial radius
        bounds["super_colatitude"] = (None, None)
        bounds["super_radius"] = (None, None)
        bounds["phase_shift"] = (-0.25, 0.75)
        bounds['super_tbb'] = (0.001, 0.003)
        bounds['super_tau'] = (0.5, 3.5)
        bounds['super_te'] = (40., 200.)
        bounds['elsewhere_temperature'] = (5.0,7.0) #log10 K
        bounds['interstellar'] = (None, None)
        if self.bkg == 'model':
            bounds['T_in'] = (0.01, 0.6) # keV
            bounds['R_in'] = (20, 64) # km
           
        self.bounds = bounds
        
    def set_values(self):
        values = {}
        values['frequency'] = 401.0
        self.values = values

    def set_data(self):
        self.exposure_time = 1.32366e5 #Mason's 2019 data cut
        # self.exposure_time = 7.13422e4 #Mason's 2022 data cut
        self.phases_space = np.linspace(0.0, 1.0, 33)

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
    
        spacetime_bounds = dict(distance = self.bounds["distance"],                       # (Earth) distance
                                mass = self.bounds["mass"],                          # mass
                                radius = self.bounds["radius"],     # equatorial radius
                                cos_inclination = self.bounds["cos_i"])               # (Earth) inclination to rotation axis

        self.spacetime = xpsi.Spacetime(bounds=spacetime_bounds, values=dict(frequency=self.values["frequency"]))
        
    def set_hotregions(self):
        self.num_rays = 512

        kwargs = {'symmetry': self.integrator, #call general integrator instead of for azimuthal invariance
                  'interpolator': self.interpolator,
                  'omit': False,
                  'cede': False,
                  'concentric': False,
                  'sqrt_num_cells': self.sqrt_num_cells,
                  'min_sqrt_num_cells': 10,
                  'max_sqrt_num_cells': 128,
                  'num_leaves': self.num_leaves,
                  'num_rays': self.num_rays,
                  'prefix': 'p'}
        
        hotregion_bounds = dict(super_colatitude = self.bounds["super_colatitude"],
                                super_radius = self.bounds["super_radius"],
                                phase_shift = self.bounds["phase_shift"], 
                                super_tbb = self.bounds['super_tbb'],
                                super_tau = self.bounds['super_tau'],
                                super_te = self.bounds['super_te'])
        values = {}
        
        primary = CustomHotRegion_Accreting(hotregion_bounds, values, **kwargs)


        self.hot = xpsi.HotRegions((primary,))

    def set_elsewhere(self):
        self.elsewhere = xpsi.Elsewhere(bounds=dict(elsewhere_temperature = self.bounds['elsewhere_temperature']))
        
    def set_photosphere(self):
        self.set_spacetime()
        self.set_hotregions()
        self.set_elsewhere()
        self.photosphere = CustomPhotosphere(hot = self.hot, elsewhere = self.elsewhere,
                                        values=dict(mode_frequency = self.spacetime['frequency']))

        self.photosphere.hot_atmosphere = self.file_atmosphere

    def set_star(self):
        self.set_photosphere()
        self.star = xpsi.Star(spacetime = self.spacetime, photospheres = self.photosphere)
        
        
    def set_interstellar(self):
        self.interstellar=CustomInterstellar.from_SWG(self.file_interstellar, bounds=self.bounds['interstellar'], value=None)
    
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
        
        
    def set_background(self):
        from CustomBackground import CustomBackground_DiskBB, k_disk_derive
        if self.bkg == 'model':            
            bounds = dict(T_in = get_T_in_log10_Kelvin(self.bounds["T_in"]),
                          R_in = self.bounds["R_in"],
                          K_disk = None) #derived means no bounds
                
            k_disk = k_disk_derive()
            
            self.background = CustomBackground_DiskBB(bounds=bounds, values={'K_disk': k_disk}, interstellar = self.interstellar)
            
            k_disk.star = self.star
            k_disk.background = self.background
            
        elif self.bkg == 'marginalise':
            self.background = None
        else:
            print('error! bkg must be either model or marginalised.')
            
    def set_signal(self):
        self.set_data()
        self.set_instrument()
        self.set_background()
        self.set_support()

        self.signal = CustomSignal(data = self.data,
                            instrument = self.instrument,
                            background = self.background,
                            interstellar = self.interstellar,
                            support = self.support,
                            cache = False,
                            bkg = self.bkg,
                            epsrel = 1.0e-8,
                            epsilon = 1.0e-3,
                            sigmas = 10.0)
        
        
    def set_parameter_vector(self):
        # SAX J1808-like 
        mass = 1.4
        radius = 12.
        distance = 3.5
        inclination = 60
        cos_i = math.cos(inclination*math.pi/180)
        phase_shift = 0
        super_colatitude = 45*math.pi/180 # 20*math.pi/180 #
        super_radius = 15.5*math.pi/180 #  0.001 #


        # Compton slab model parameters
        tbb=0.0012 # 0.0017 # #0.001 -0.003 Tbb(data) = Tbb(keV)/511keV, 1 keV = 0.002 data
        te=100. # 50. # 40-200 corresponds to 20-100 keV (Te(data) = Te(keV)*1000/511keV), 50 keV = 100 data
        tau=1. #0.5 - 3.5 tau = ln(Fin/Fout)


        # elsewhere
        elsewhere_T_keV =  0.4 # 0.5 # keV 
        elsewhere_T_log10_K = get_T_in_log10_Kelvin(elsewhere_T_keV)

        # source background
        column_density = 1.17 #10^21 cm^-2


        p = [mass, #1.4, #grav mass
              radius,#12.5, #coordinate equatorial radius
              distance, # earth distance kpc
              cos_i, #cosine of earth inclination
              phase_shift, #phase of hotregion
              super_colatitude, #colatitude of centre of superseding region
              super_radius,  #angular radius superceding region
              tbb,
              te,
              tau,
              elsewhere_T_log10_K]

        if self.bkg == 'model':
            diskbb_T_keV = 0.25 # 0.3  #  keV #0.3 keV for Kajava+ 2011

            diskbb_T_log10_K = get_T_in_log10_Kelvin(diskbb_T_keV)
            p.append(diskbb_T_log10_K)
            
            R_in = 30 # 20 #  1 #  km #  for very small diskBB background
            p.append(R_in)
            

            # K_disk = get_k_disk(cos_i, R_in, distance)
            # K_disk = 0
            # p.append(K_disk)

        p.append(column_density)
        self.p = p
    
    def set_prior(self):
        self.prior = CustomPrior()
        
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


        ########## likelihood check
        # true_logl = -4.6402898384e+04
        # true_logl = -4.2233157248e+04 # background, support
        # true_logl = -1.1767530520e+04  # background, sf=1.00005, floated data, high res
        # true_logl = -1.0929410655e+04  # background, sf=1.001, floated data, high res
        # true_logl = -9.9746308842e+03 # # background, sf=1.1, floated data, high res
        # true_logl = -9.8546380529e+03 # # background, sf=1.5, floated data, high res
        #true_logl = -9.8076308641e+03  # background, no support, floated data, high res 
        # true_logl = -9.8013206348e+03  # background, no support, floated data, high res, allow neg. bkg. 
        # true_logl = -4.1076321631e+04 # no background, no support
        # true_logl = -1.0047370824e+04  # no background, no support, floated data, high res
        # true_logl = 1.9406875013e+08  # given background, background, support, floated data, high res,
        
        ## 2019 data
        # true_logl = 1.6202395730e+08 # 2019 data, modeled background
        true_logl= -7.9418857894e+89 # 2019 data, marginalized background
        
        
        ### 2022 data
        # true_logl = 1.1365193823e+08 # 2022 data
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

        
        plt.savefig('pre_sampling_plot.png')
        print('figure saved in {}'.format(folderstring))
        
        if self.run_type == 'sample':

            
            wrapped_params = [0]*len(self.likelihood)
            wrapped_params[self.likelihood.index('p__phase_shift')] = 1


            sampling_efficiency = 0.1

            outputfiles_basename = f'./{folderstring}/run_ST_'
            runtime_params = {'resume': False,
                              'importance_nested_sampling': False,
                              'multimodal': False,
                              'n_clustering_params': None,
                              'outputfiles_basename': outputfiles_basename,
                              'n_iter_before_update': 100,
                              'n_live_points': self.live_points,
                              'sampling_efficiency': sampling_efficiency,
                              'const_efficiency_mode': False,
                              'wrapped_params': wrapped_params,
                              'evidence_tolerance': 0.5,
                              'seed': 7,
                              'max_iter': self.max_iter, # manual termination condition for short test
                              'verbose': True}

            print('runtime_params: ', runtime_params)
            
            print("sampling starts ...")
            t_start = time.time()
            xpsi.Sample.nested(self.likelihood, self.prior,**runtime_params)
            print("... sampling done")
            print('Sampling took {:.3f} seconds'.format((time.time()-t_start)))
            
