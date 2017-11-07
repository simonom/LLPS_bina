# python file for plotting ETH Maxwell-Stefan results in the form of cross sections through a particle.  Able to plot bulk and schlieren.

# Copyright 2017 Simon O'Meara.  Program distributed under the terms of the 
# GNU General Public License

# This file is part of ETH23.

# ETH23 is free software: you can redistribute it and/or modify it under the 
# terms of the GNU General Public License as published by the Free Software 
# Foundation, either version 3 of the License, or (at your option) any later 
# version.

# ETH23 is distributed in the hope that it will be useful, but WITHOUT ANY 
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE.  See the GNU Public License for more details.

# You should have received a copy of the GNU Public License along with ETH23.
# If not, see <http://www.gnu.org/licenses/>.

# -----------------------------------------------------------------------------

# import stuff
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator 
from matplotlib.ticker import ScalarFormatter 

import numpy as np
import scipy.io

# define function
def MS_plotting_v_pyx():
    
	# open saved file
	res = scipy.io.loadmat("FvMS_paper_ps_test_con_es1.mat")
   	# withdraw variables contained in dictionary keys
   	
	nrec = res['nrec']
	nrecsc = res['nrecsc']
   	time2 = res['time2']
   	Vrec = res['Vrec']
	Vrecsc = res['Vrecsc']
   	#d = res['Gamma_rec'] # component activity coefficients

	# time indices at which to plot results
	it = np.array(([2, 50, 100, 999]))
	
	# number of total radii points	
	rp = np.zeros(([it.shape[0]]))
	# time point loop
	for itn in range(0, it.shape[0]):
		# number of shells with a volume
		rp[itn] = np.sum(Vrec[:, it[itn]]>0.0)+np.sum(Vrecsc[:, it[itn]]>0.0)	

	# empty radius results (for each time step in 1st dim.) and each shell (2nd dim.)
	r = np.zeros(([int(np.max(rp))+1, it.shape[0]]))
	
	
	# time step loop to get radius of 1p and 2p shells
	for itn in range(0, it.shape[0]):	
	
		# empty array for combined volumes of 1p and 2p 
		# volumes (m^3)
		Vrec2 = np.zeros((int(rp[itn])+1))
		# empty array for sv mole fractions in bulk and schlieren
		xsv = np.zeros((int(rp[itn])+1))
		# count on steps through original volume arrays
		V0c = 0
		# check on whether it's 1p or 2p to be considered in 
		# the volume valuation 	
		sch_ch = 0

		# shell loop to get radius of 1p and 2p shells
		for ir in range(0, int(rp[itn])):
			
			if Vrec[V0c, it[itn]]>0.0 and Vrecsc[V0c, it[itn]]>0.0 and sch_ch == 0:
	
				Vrec2[ir+1] = Vrec[V0c, it[itn]]
				sch_ch = 1 # check on accounting for 2p
				V0c = V0c # keep count constant
				# sv mole fraction at this radius
				xsv[ir+1] = nrec[0, V0c, it[itn]]/np.sum(nrec[:, V0c, it[itn]])
				# radius points for both 1p and 2p 
				# results at each time step
				r[ir+1, itn] = (((3.0*np.sum(Vrec2[0:ir+2]))/(4.0*np.pi))**(1.0/3.0)+((3.0*np.sum(Vrec2[0:ir+1]))/(4.0*np.pi))**(1.0/3.0))/2.0	
		
				continue
		
			if Vrec[V0c, it[itn]]>0.0 and Vrecsc[V0c, it[itn]]>0.0 and sch_ch == 1:
				Vrec2[ir+1] = Vrecsc[V0c, it[itn]]
				sch_ch = 0 # check on accounting for 2p
			
				# sv mole fraction at this radius
				xsv[ir+1] = nrecsc[0, V0c, it[itn]]/np.sum(nrecsc[:, V0c, it[itn]])
				
				# move up index for original volume array
				V0c = V0c+1 
				# radius points for both bulk and schlieren 
				# results at each time step
				r[ir+1, itn] = (((3.0*np.sum(Vrec2[0:ir+2]))/(4.0*np.pi))**(1.0/3.0)+((3.0*np.sum(Vrec2[0:ir+1]))/(4.0*np.pi))**(1.0/3.0))/2.0	
				continue		
			if Vrec[V0c, it[itn]]>0.0 and Vrecsc[V0c, it[itn]]==0.0:
				Vrec2[ir+1] = Vrec[V0c, it[itn]]
				# sv mole fraction at this radius
				xsv[ir+1] = nrec[0, V0c, it[itn]]/np.sum(nrec[:, V0c, it[itn]])
				# move up index for original volume array
				V0c = V0c+1

			if Vrec[V0c, it[itn]]==0.0 and Vrecsc[V0c, it[itn]]>0.0:

				Vrec2[ir+1] = Vrecsc[V0c, it[itn]]
				
				# sv mole fraction at this radius
				xsv[ir+1] = nrecsc[0, V0c, it[itn]]/np.sum(nrecsc[:, V0c, it[itn]])
				# move up index for original volume array
				V0c = V0c+1
		
			# radius points for both 1p and 2p 
			# results at each time step
			r[ir+1, itn] = (((3.0*np.sum(Vrec2[0:ir+2]))/(4.0*np.pi))**(1.0/3.0)+((3.0*np.sum(Vrec2[0:ir+1]))/(4.0*np.pi))**(1.0/3.0))/2.0	
		
		# set centre mole fraction same as first shell
		xsv[0] = xsv[1]
		
		if itn == 0:
			fig1, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, sharey = True)
			im = ax0.plot(r[0:int(rp[itn])+1, itn], xsv[0:int(rp[itn]+1)], '-xb')
			ax0.set_title(r't=1.84x$10^{-5}$ s')
			ax0.set_xlabel(r'$r$ (m)')
			ax0.set_ylabel(r'$x_{sv}$')

		if itn == 1:
			im = ax1.plot(r[0:int(rp[itn])+1, itn], xsv[0:int(rp[itn]+1)], '-xb')
			ax1.set_title(r't=1.89x$10^{-5}$ s')
			ax1.set_xlabel(r'$r$ (m)')
		if itn == 2:
			im = ax2.plot(r[0:int(rp[itn])+1, itn], xsv[0:int(rp[itn]+1)], '-xb')
			ax2.set_title(r't=1.94x$10^{-5}$ s')
			ax2.set_xlabel(r'$r$ (m)')
	
		if itn == 3:
			im = ax3.plot(r[0:int(rp[itn])+1, itn], xsv[0:int(rp[itn]+1)], '-xb')
			ax3.set_title(r't=1.99x$10^{-5}$ s')
			ax3.set_xlabel(r'$r$ (m)')

			plt.show()
