'''
Helper functions to read IXPE data and response files.
These scripts are still to be checked and likely developed further.
'''


import numpy as np
from astropy.io import fits

def readData_pha(Filename, NPhadat=16):
# READS SIMULATED DATA FROM .fits 

	NPhase = NPhadat # number of phases in .fits
	NEnergy = 375 #275 # number of energy bins in .fits

	I = np.zeros((NPhase,NEnergy))
	Ierr = np.zeros((NPhase,NEnergy))

	Q = np.zeros((NPhase,NEnergy))
	Qerr = np.zeros((NPhase,NEnergy))

	U = np.zeros((NPhase,NEnergy))
	Uerr = np.zeros((NPhase,NEnergy))

	phase_points = np.linspace(0.0,1.0,NPhase+1)
	phase = np.zeros((NPhase))
	for i in range(0,NPhase):
		phase[i] = (phase_points[i+1]+phase_points[i])/2.0

#ixpe03250101_du1_evt2_v01_src_bary_pers_pre60376_phase0012_pha1u.fits

	for p in range(NPhase):

		# READ I
		if(p<10):
			hdulist = fits.open(str(Filename) + '_phase000' + str(p) + '_pha1.fits')
			data = hdulist[1].data
		else:
			hdulist = fits.open(str(Filename) + '_phase00' + str(p) + '_pha1.fits')
			data = hdulist[1].data		
		#print(hdulist.info())
		#print(hdulist[1].header['EXPOSURE'])
		#print(hdulist[1].data.columns)
		#exit()
		exposure = hdulist[1].header['EXPOSURE']
		#print(exposure)
		#exit()
		#print("Estimated err- stat_err: ", np.sqrt(np.abs(data.field('RATE'))*exposure)/exposure - np.abs(data.field('STAT_ERR')))

		if p==0: #Assuming same channels in all phase bins
			channels = data.field('CHANNEL')
		
		I[p,:] = data.field('RATE')*exposure
		Ierr[p,:] = data.field('STAT_ERR')*exposure

		# READ Q
		if(p<10):
			hdulist = fits.open(str(Filename) + '_phase000' + str(p) + '_pha1q.fits')
			data = hdulist[1].data
		else:
			hdulist = fits.open(str(Filename) + '_phase00' + str(p) + '_pha1q.fits')
			data = hdulist[1].data
	
		exposure2 = hdulist[1].header['EXPOSURE']
		if exposure2 != exposure:
			print("Error: We assume exposure is same for all I, Q, and U.")
			exit()								
		Q[p,:] = data.field('RATE')*exposure
		Qerr[p,:] = data.field('STAT_ERR')*exposure



		# READ U
		if(p<10):
			hdulist = fits.open(str(Filename) + '_phase000' + str(p) + '_pha1u.fits')
			data = hdulist[1].data
		else:
			hdulist = fits.open(str(Filename) + '_phase00' + str(p) + '_pha1u.fits')
			data = hdulist[1].data

		exposure = hdulist[1].header['EXPOSURE']
		exposure3 = hdulist[1].header['EXPOSURE']
		if exposure3 != exposure:
			print("Error: We assume exposure is same for all I, Q, and U.")
			exit()	
		U[p,:] = data.field('RATE')*exposure
		Uerr[p,:] = data.field('STAT_ERR')*exposure

	return I, Q, U, Ierr, Qerr, Uerr, channels, phase_points, exposure






def read_response_IXPE(MRF,RMF,min_input,max_input,min_channel,max_channel):

        hdulist_mrf = fits.open(MRF)
        #cols1 = hdulist_mrf[1].columns
        #print(cols1.info())
        specresp = hdulist_mrf[1].data["SPECRESP"]
        ene_lo = hdulist_mrf[1].data["ENERG_LO"]
        ene_hi = hdulist_mrf[1].data["ENERG_HI"]

        hdulist_rmf = fits.open(RMF)
        matrix = hdulist_rmf[1].data["MATRIX"]
        emin = hdulist_rmf[2].data["E_MIN"]
        emax = hdulist_rmf[2].data["E_MAX"]

        #print("len(ene_lo): ", len(ene_lo))
        #print("len(emin): ", len(emin))
        #print("matrix dimenions:", len(matrix[:,0]), len(matrix[0,:]))


        matrix_cut = np.ascontiguousarray(matrix[min_input:max_input+1,min_channel:max_channel+1].T, dtype=np.double)

        edges = np.zeros(specresp[min_input:max_input].shape[0]+1, dtype=np.double)
        edges[0] = ene_lo[min_input]; edges[1:] = ene_hi[min_input:max_input+1]

        for i in range(matrix_cut.shape[0]):
                matrix_cut[i,:] *= specresp[min_input:max_input+1]

        channel_edges = np.zeros(matrix[0,min_channel:max_channel+1].shape[0]+1, dtype=np.double)
        #print(matrix[0,min_channel:max_channel+1].shape[0])
        channel_edges[0] = emin[min_channel]; channel_edges[1:] = emax[min_channel:max_channel+1]

        channels = np.arange(min_channel,max_channel+1)

        #print("channels: ", channels, len(channels))
        #print("channel_edges: ", channel_edges, len(channel_edges))
        #print("edges: ", edges, len(edges))

        #print("matrix_cut=",matrix_cut, len(matrix_cut[0,:]), len(matrix_cut[:,0]))

        return matrix_cut, edges, channels, channel_edges
