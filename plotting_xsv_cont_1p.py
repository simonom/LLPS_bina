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
    
	# open saved file for non-equilibrium situation
	res = scipy.io.loadmat("FvMS_paper_ps_test_loes_10shell_HX.mat")
   	# withdraw variables contained in dictionary keys
	nrec = res['nrec']
   	time2 = res['time2']
   	Vrec = res['Vrec']
	esrec = res['esrec']

	##open saved file for equilibrium case
# 	reseq = scipy.io.loadmat("FvMS_paper_fig1a_eq.mat")
# 	time2eq = reseq['time2']
# 	Vreceq = reseq['Vrec']

	d = res['Gamma_rec'] # component activity coefficients
		
	# volumes per shell (m3)
	Vshell = Vrec
	
	# sum volumes along shells (m3)
	VT = np.sum(Vshell, axis = 0)

	# convert particle volume to radius and find maximum (m)
	rmax = np.max((3.0*VT/(4.0*np.pi))**(1.0/3.0))
	# standard radius 
	rstan = np.arange(0, rmax, rmax/1000.0)
	# empty results matrix for sv mole fraction
	xsvstan = np.zeros((rstan.shape[0], 1000))
	xsvstan[:, :] = np.nan
	# empty results matrix for Gamma
	Gamstan = np.zeros((rstan.shape[0], 1000))
	Gamstan[:, :] = np.nan
	
	for it in range(0, np.sum(Vrec[0, :]>0)): # time loop
	
		# empty matrix for combined radius points
		ract = np.zeros((np.sum(Vrec[:, it]>0),1))
		# empty matrix for mole fractions
		xsvact = np.zeros((ract.shape[0], 1))
		# empty matrix for Gamma
		Gamact = np.zeros((ract.shape[0], 1))
		
		# starting index for spreading mole fractions over 
		#standard radius point array
		ish0 = 0
		# loop through volumes to get actual radius points (m) and mole
 		# fractions 
		for ir in range(0, ract.shape[0]):
			
			ract[ir, 0] = ((3.0/(np.pi*4.0))*np.sum(Vrec[0:ir+1, it]))**(1.0/3.0)
			xsvact[ir, 0] = nrec[0, ir, it]/np.sum(nrec[:, ir, it])
			Gamact[ir,0] = d[0, ir, it]
			# relevant standard radii points
			ish1 = np.sum(rstan<=np.max(ract[ir,0]))
			# spread mole fractions onto the standard radius point 
			# array		
			xsvstan[ish0:ish1, it] = xsvact[ir, 0]
			Gamstan[ish0:ish1, it] = Gamact[ir, 0]
			ish0 = ish1
			

	# contour plot of mole fractions against time and radius		
	xfmt = ScalarFormatter()
	xfmt.set_powerlimits((0, 0))

	fig1, (ax0, ax2) = plt.subplots(1, 2)
	
	z = np.ma.masked_where(np.isnan(xsvstan), xsvstan)
	z2 = np.ma.masked_where(np.isnan(Gamstan), Gamstan)
	
	levels = MaxNLocator(nbins = 40).tick_values(np.min(z[~np.isnan(z)]), np.max(z[~np.isnan(z)]))
	cmap = plt.get_cmap('winter_r')
	norm1 = BoundaryNorm(levels, ncolors = cmap.N, clip=True)
	im = ax0.pcolormesh(time2[0, 0:999], rstan, z[0:999,:], cmap=cmap, norm=norm1, label='particle phase diffusion limited')	

	levels2 = MaxNLocator(nbins = 40).tick_values(np.min(z2[~np.isnan(z)]), np.max(z2[~np.isnan(z)]))
	cmap = plt.get_cmap('jet')
	norm2 = BoundaryNorm(levels2, ncolors = cmap.N, clip=True)
	im3 = ax2.pcolormesh(time2[0, 0:999], rstan, z2[0:999,:], cmap=cmap, norm=norm2, label='particle phase diffusion limited Gamma')	


# 	#plot of equilibrium radius superimposed
# 	im2 = ax0.plot(time2[0, :], req, '--k', label='particle phase equilibrium')
	ax0.legend(loc='upper right')
	# set y-axis limits
	plt.ylim((0, 1.2e-7))	
	# standard notation on axis ticks
	ax0.xaxis.set_major_formatter(xfmt)	
	ax0.yaxis.set_major_formatter(xfmt)
	ax2.xaxis.set_major_formatter(xfmt)	
	ax2.yaxis.set_major_formatter(xfmt)
	# set labels
	ax0.set_xlabel(r'$t$ (s)', size=20)
	ax2.set_xlabel(r'$t$ (s)', size=20)
	ax0.set_ylabel(r'$r$ (m)', size=20)
	# set font size of standard notation
	ax0.xaxis.offsetText.set_fontsize(20)
	ax0.yaxis.offsetText.set_fontsize(20)
	ax2.xaxis.offsetText.set_fontsize(20)
	ax2.yaxis.offsetText.set_fontsize(20)
	# put axis ticks inside
	ax0.tick_params(axis='both', direction='in', labelsize=20)
	ax2.tick_params(axis='both', direction='in', labelsize=20)
	#ax0.set_xticks([0.0, 1.0e-1, 2.0e-1, 3.0e-1, 4.0e-1])
	

	# colorbar xsv plot
	cb = fig1.colorbar(im, ax=ax0, shrink=0.5)
	cb.set_label(r'$x_{sv}$', size=20)
	cax = cb.ax
	pos1 = cax.get_position()
	yshift = pos1.height*0.6
	pos2 = [pos1.x0, pos1.y0-yshift, pos1.width, pos1.height]
	cax.set_position(pos2)
	
	# colorbar Gamma plot
	cb2 = fig1.colorbar(im3, ax=ax2, shrink=0.5)
	cb2.set_label('$\Gamma$', size=20)
	cax2 = cb2.ax
	pos3 = cax2.get_position()
	yshift = pos3.height*0.6
	pos4 = [pos3.x0, pos3.y0-yshift, pos3.width, pos3.height]
	cax2.set_position(pos4)

	
	
	# y-axis scale on right for sv saturation ratio
	ax1 = ax0.twinx()
	ax1.plot(esrec[0, :], esrec[1, :], '--', color=(1.00,0.48,0.00), label='$e_{s,sv}$')
	ax1.set_ylabel(r'$e_{s,sv}$', size=20)
	ax1.yaxis.offsetText.set_fontsize(20)
	ax1.tick_params(axis='y', direction='in', labelsize=20)
	ax1.yaxis.label.set_color((1.00,0.48,0.00))	
	ax1.yaxis.label.set_position((1.1, 0.8))
	plt.ylim((-1.4, 0.7))
	ax1.set_yticks([0.00, 1.00, 2.00])
	#ax1.legend(loc=(0.5, 0.8))

	plt.show()
