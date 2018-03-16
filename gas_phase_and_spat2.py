# Copyright Notice: This code is in Copyright.  Any use leading to 
# publication or 
# financial gain is prohibited without the permission of the authors Simon 
# O'Meara : simon.omeara@manchester.ac.uk.  First published 2017.

# This file is part of diffusion_extend

# diffusion_extend 
# is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# diffusion_extend
# is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with diffusion_extend
# (see the LICENSE file).  If not, see 
# <http://www.gnu.org/licenses/>.


# -------------------------------------------------------
# function to call on gas phase diffusion and and new spatial dimensions of 
# shells

import numpy as np
from spat_arrays_inequal import spat_arrays_inequal
from concs2 import concs2
from phas_sep1 import phas_sep1

def gas_phase_and_spat(es, estime, shn, time, V0, tsn, Nz0, 
	Z0, M, p, rc, Dg, Cstar, ts, Diamw, idma, del0, 
	T, gam0, Db):


	# inputs:
	# T - temperature (K)    
	# gam0 - reference activity coefficients
	# Nz0 - number of moles per component (1st dim.) per
	# shell (2nd dim.) in 1st phase
	# Db - component self-diffusion coefficients (m2/s)

	# ---------------------------------------------------------------------	
	# revalue spatial variables 
	# new spatial arrays
	[delta1, rc, A] = spat_arrays_inequal(V0, shn)
    
	# new objects for new volume and number of mole matrices
	V1 = np.zeros((1, shn))
	V1[0, :] = V0[0, :]
	
	Nz1 = np.zeros((Nz0.shape[0], Nz0.shape[1]))
	Nz1[:, :] = Nz0[:, :]


	# --------------------------------------------------------------------
	# call on the gas phase diffusion equation to get new surface 
	# concentrations (mol/m3)    	

	[Z1, Nz1, V1] = concs2(es, estime, shn, time, 
		V1, tsn, Nz1, Z0, M, p, rc[0, shn-1], 
		Dg, Cstar, ts, idma, gam0, T)			
		
	# comment out to prevent phase separation
	#[Nz1,n1sc,V1,V1sc] = phas_sep1(Nz1,n1sc,shn,gam0,M,p,PS0,V1,V1sc,Db,ts)
	# new spatial arrays
	[delta1, rc, A] = spat_arrays_inequal(V1, shn)
	
	# --------------------------------------------------------------------
	# move material from surface to near-surface if it gets too big
	if delta1[-1]>(Diamw*1.0):
 		
 		# mole fractions of all components in near surface and surface shells
		xnssh = Nz1[:, -2]/np.sum(Nz1[:, -2])
		xssh = Nz1[:, -1]/np.sum(Nz1[:, -1])

		# transfer half the nv inwards
		Nz1[1::, -2] = Nz1[1::, -2]+(Nz1[1::, -1]/2.0)
		Nz1[1::, -1] = Nz1[1::, -1]-(Nz1[1::, -1]/2.0)
		# set the sv as required in both shells
		Nz1[0, -2] = Nz1[1, -2]*(xnssh[0]/xnssh[1])
		Nz1[0, -1] = Nz1[1, -1]*(xssh[0]/xssh[1])
# 		print('add')
			
		# else: # make a new shell
# 			print 'seggy'
# 			Nz1a = Nz1[:, 0:-1]
# 			new_shell = (Nz1[:, -1]/2.0)
# 			new_shell.shape = (len(M), 1) # ensure correct dimensions
# 			Nz1a = np.append(Nz1a, new_shell, 1)
# 			Nz1 = np.append(Nz1a, new_shell, 1)
# 			Nz1[0, -2] = Nz1[1, -2]*(xnssh[0]/xnssh[1])
# 			Nz1[0, -1] = Nz1[1, -1]*(xssh[0]/xssh[1])
# 			del Nz1a
# 			shn = shn+1
			
			
		# molar volume
		MV = (M[:, 0]/p[:, 0])
		MV.shape = (len(M), 1)
		# new shell volumes (m3)
		V1 = np.sum(Nz1[:, :]*(np.ones((len(M), shn))*MV), 0)
		# new spatial variables
		[delta1, rc, A] = spat_arrays_inequal(V1, shn)
		
	# split near-surface shell in two if it gets too big
	if delta1[-2]>((rc[0, -1]/shn)*2.0):
# 		print('seggy')
		Nz1a = Nz1[:, 0:-2]
		new_shell = (Nz1[:, -2]/2.0)
		new_shell.shape = (len(M), 1) # ensure correct dimensions
		Nz1a = np.append(Nz1a, new_shell, 1) # first new shell
		Nz1a = np.append(Nz1a, new_shell, 1) # second new shell
		surf_shell = Nz1[:, -1]
		surf_shell.shape = (len(M), 1) # ensure correct dimensions
		Nz1 = np.append(Nz1a, surf_shell, 1) # original end shell
		
		del Nz1a
		shn = shn+1
		
		# molar volume
		MV = (M[:, 0]/p[:, 0])
		MV.shape = (len(M), 1)
		# new shell volumes (m3)
		V1 = np.sum(Nz1[:, :]*(np.ones((len(M), shn))*MV), 0)
		
		[delta1, rc, A] = spat_arrays_inequal(V1, shn)
		
	# if surface shell too small
	if delta1[-1]<(Diamw*0.3):
# 		print('titchy')
		
		Nzfac = Nz1[0, -1]/Nz1[1, -1] # number of sv moles for every mole nv at surface
		Nzfacns = Nz1[0, -2]/Nz1[1, -2] # number of sv moles for every mole nv at near surface
		xsv = Nz1[0, -1]/np.sum(Nz1[:, -1])
		Nzadd = Nz1[1, -1]*0.5*(1.0/xsv) # number of nv moles we want to add to surface
		
		if Nz1[1, -2]>Nzadd: # if sufficient material in near-surface
		
			# move from near-surface to surface
			Nz1[1, -1] = Nz1[1, -1]+Nzadd 
			Nz1[1, -2] = Nz1[1, -2]-Nzadd 
			Nz1[0, -2] = Nz1[1, -2]*Nzfacns # correct number sv moles near-surface
		else:
		
			# move all nv from near-surface to surface
			Nz1[1, -1] = Nz1[1, -1]+Nz1[1, -2]
			# remove near-surface shell
			Nzsurf = Nz1[:, -1]
			Nzsurf.shape = (len(M), 1) # ensure correct dimensions
			Nz1 = np.append(Nz1[:, 0:-2], Nzsurf, 1)
			shn = shn-1
		# number of moles sv in surface 
		Nz1[0, -1] = Nz1[1, -1]*Nzfac
		
		# molar volume
		MV = (M[:, 0]/p[:, 0])
		MV.shape = (len(M), 1)
		# new shell volumes (m3)
		V1 = np.sum(Nz1[:, :]*(np.ones((len(M), shn))*MV), 0)
		
		[delta1, rc, A] = spat_arrays_inequal(V1, shn)
	# --------------------------------------------------------------------
	# new object for component concentrations (mol/m3)
	Z1 = np.zeros((M.size,shn))
	Z1[:,:] = Nz1/V1

	
	return A, shn, Nz1, delta1, Z1, V1, rc