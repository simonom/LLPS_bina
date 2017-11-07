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
# function to quantify the initial concentrations of components in the particle phase 
# (mol/m3)

import numpy as np
from interp_x import int_x

def concs(ai0, M, shn, V0, sp, p, idma, SMILES_arr, gam0):

    	# -------------------------------------------------------------------    	
	# inputs:
	
    	# ai0 - activities of each component at start with components in 
	# 1st dim. and shells 
    	# in 2nd dim.
    	# M - molar mass of each component (g mol^{-1}) (row array)
    	# shn - number of shells
    	# V0 - shell volume (m^3)
    	# sp - number of times to save number of moles and shell volumes
    	# p - density of each component (g m^{-3}) (row array)
    	# idma - ideality marker (1 for ideal, 0 for non-ideal)
	# gam0 - activity coefficients
    	# -------------------------------------------------------------------
	# outputs:
    
    	# Z - mol m^{-3} (per component (1st dim.), per shell (2nd dim.))
    	# Nz0 - absolute number of moles per component (1st dim.), 
	# per shell (2nd dim.)
    	# nrec - record of component concentrations (1st dim.), with shell 
	# (2nd dim.) and with time (3rd dim.) (mol m^{-3})
    	# Vrec - record of shell volumes (1st dim.), with time (2nd dim.) 
	# (m^{3})
    	# gamma_rec - record of component activity coefficients (1st dim.), 
	# with shell (2nd dim.) and with time (3rd dim.) (dimensionless)
    	# SN_partrec - record of number of shells with time (1st dim.)
    	# --------------------------------------------------------------------
	# ensure shn is integer not float
	shn = int(shn)
	# concentration matrix (mol m^{-3}) for both phases
	Z0 = np.zeros((M[:, 0].shape[0], shn))	
	Z0sc = np.zeros((M[:, 0].shape[0], shn))
	# absolute number of moles per component, per bulk shell (mol)
	Nz0 = np.zeros((M[:, 0].shape[0], shn))
	# moles per component in schlieren at start (mol)
	n0sc = np.zeros((M[:, 0].shape[0], shn))
	# mole fraction required to attain the prescribed activity 
	# (based on saturation ratio)
	x0 = np.zeros((M[:, 0].shape[0], shn))	
	
	# shell loop
	for ir2 in range(0, shn):
		
		if ir2 == 0:
			if idma == 0:	
				# mole fraction at this activity
				# note, third input is starting sv mole fraction
				x0[0, ir2] = int_x(ai0[0, ir2], gam0, 0.0)
    			else:
				x0[0, ir2] = ai0[0, ir2]
			
			# resulting mole fraction of non-volatile
			x0[M[:, 0].shape[0]-1, ir2] = 1-np.sum(
					x0[0:M[:, 0].shape[0]-1, ir2])
			# ratios of component molar volumes ((M/p) m^3 mol^{-1})
			mVrat = (x0[:,ir2]*(M[:,0]/p[:,0]))/(np.sum((x0[:,ir2]*(
				M[:,0]/p[:,0]))))
			# if activities in this shell same as the previous 
			# shell then duplicate mole fractions and molar 
			# volumes from previous shell ((M/p) m^3 mol^{-1})
		if ir2>0 and np.sum((ai0[:, ir2] == 
				ai0[:, ir2-1])) == ai0.shape[0]:
			x0[:, ir2] = x0[:, ir2-1]
			mVrat = mVrat
		else:
			if idma == 0:
				# sv mole fraction at this activity
				# third input is actual mole fraction
				x0[0, ir2] = int_x(ai0[0, ir2], gam0, 0.0)
			else:
				x0[0, ir2] = ai0[0, ir2]

			# resulting mole fraction of non-volatile
			x0[M[:, 0].shape[0]-1, ir2] = 1-np.sum(
					x0[0:M[:, 0].shape[0]-1, ir2])
			# ratios of component molar volumes ((M/p) m3/mol)
			mVrat = (x0[:, ir2]*(M[:, 0]/p[:, 0]))/(np.sum((
					x0[:, ir2]*(M[:, 0]/p[:, 0]))))

        	# volumes of each component (m3)
		Vi = mVrat*V0[0, ir2]
        	# number of moles per component at start (n) (mol)
		Nz0[:, ir2] = (p[:, 0]/M[:, 0])*Vi[:]
		# concentrations (mol/m3)
		Z0[:, ir2] = Nz0[:, ir2]/V0[0, ir2] 

	del x0, mVrat, Vi
	
	# save initial number of moles (components 1st dim., 
	# number of shells 2nd dim., time steps 3rd dim.)
	nrec = np.zeros((M[:, 0].shape[0], shn*10, sp))
	nrec[:, 0:shn, 0] = Nz0

	# save number of moles in second phase (mol)
	nrecsc = np.zeros((M[:, 0].shape[0], shn*10, sp))

	# save volumes of second phase (m3)
	Vrecsc = np.zeros((shn*10, sp))
    
	# save fractions of nuclei that could form
	nuc2p = np.zeros((1,shn))
    
	# shell volumes recording (shells 1st dim., time steps 2nd dim.)
	Vrec = np.zeros((shn*10, sp))
    
    # component activity coefficients recording (dimensionless) 
	# (components 1st dim., number of shells 2nd dim., time steps 3rd dim.)
	gamma_rec = np.zeros((M[:, 0].shape[0], shn*10, sp))
    
	# record of mole fraction (and therefore saturation ratio when 
	# assuming instantaneous gas phase diffusion) of semi-volatile
	esrec = np.zeros((1, sp))
    	# number of shells recording
	shn_rec = np.zeros((sp, 1))
    
	return (Z0, Z0sc, Nz0, nrec, Vrec, gamma_rec, shn_rec, 
			esrec, n0sc, nrecsc, Vrecsc, nuc2p)
