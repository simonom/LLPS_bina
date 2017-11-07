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

def gas_phase_and_spat(es, estime, shn, time, V0, V0sc, tsn, Nz0, n0sc, 
	Z0, Z0sc, M, p, rc, Dg, Cstar, ts, Diamw, idma, del0, 
	T, gam0, PS0, Db):

	# inputs:
	# T - temperature (K)    
	# gam0 - reference activity coefficients
	# Nz0 - number of moles per component (1st dim.) per
	# shell (2nd dim.) in 1st phase
	# Db - component self-diffusion coefficients (m2/s)

	# ---------------------------------------------------------------------	
	# revalue spatial variables 
	# new spatial arrays
	[delta1, rc, A, Asc] = spat_arrays_inequal(V0, shn, V0sc)
    
	# new objects for new volume and number of mole matrices
	V1 = np.zeros((1, shn))
	V1[0, :] = V0[0, :]
	V1sc = np.zeros((1, shn))
	V1sc[0, :] = V0sc[0, :]
	
	Nz1 = np.zeros((Nz0.shape[0], Nz0.shape[1]))
	Nz1[:, :] = Nz0[:, :]
	n1sc = np.zeros((n0sc.shape[0], n0sc.shape[1]))
	n1sc[:, :] = n0sc[:, ]	
	# --------------------------------------------------------------------
	# call on the gas phase diffusion equation to get new surface 
	# concentrations (mol/m3)    	

	[Z1, Nz1, V1, Z1sc, n1sc, V1sc] = concs2(es, estime, shn, time, 
		V1, V1sc, tsn, Nz1, n1sc, Z0, Z0sc, M, p, rc[0, shn-1], 
		Dg, Cstar, ts, idma, gam0, T)		
	
	#if tsn==1:
		#print 'gas_phase1'
		#print Nz1[:,:]
		#print Nz1[0,:]/np.sum(Nz1[:,:],0)
		#print n1sc[:,:]
		#print n1sc[0,:]/np.sum(n1sc[:,:],0)	
		
	# comment out to prevent phase separation
	#[Nz1,n1sc,V1,V1sc] = phas_sep1(Nz1,n1sc,shn,gam0,M,p,PS0,V1,V1sc,Db,ts)
	# new spatial arrays
	[delta1, rc, A, Asc] = spat_arrays_inequal(V1,shn,V1sc)
	
	# --------------------------------------------------------------------
	# new object for component concentrations (mol/m3)
	Z1 = np.zeros((M.size,shn))
	Z1[:,:] = Nz1/V1
	Z1sc = np.zeros((M.size,shn))
	ish = (V1sc==0.0)
	Z1sc[:,ish] = 0.0
	ish = (V1sc!=0.0)
	Z1sc[:,ish] = n1sc[:,ish]/V1sc[ish]
	
	return A, Asc, shn, Nz1, delta1, Z1, V1, rc, n1sc, V1sc, Z1sc
