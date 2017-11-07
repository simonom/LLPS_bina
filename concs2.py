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
# function to calculate gas phase diffusion

import numpy as np
from esnow_func import esnow_func
from gas_diff import gpp


def concs2(es, estime, shn, time, V0, V0sc, tsn, Nz0, Nz0sc, Z, Zsc, M, p, 
		Rp, Dg, Cstar, ts, idma, gam0, T):

    	#---------------------------------------------------------------------
    	# inputs:
    	# es - saturation ratios of components (1st dim) with time (2nd dim) 
    	# estime - saturation ratio times (s)
    	# shn - number of shells
    	# time - times used in solution so far (s)
    	# V0 - shell volumes (m^3)
	# V0sc - same as above for 2nd phase
    	# Nz0 - number of moles per component and shell (mol) (1st & 2nd dim.)
    	# Nz0sc - same as above for 2nd phase
	# Z - particle phase concentration of components (1st dim.) in shells 
	# (2nd dim.) (mol m^{-3})
	# Zsc - same as above for 2nd phase
    	# M - component molar masses (g mol^{-1})
    	# p - component densities (g/m^3)
    	# Rp - particle radius (m)
    	# Dg - gas phase diffusion coefficient (m^2 s^{-1})
    	# Cstar - gas phase effective saturation concentration (ug m^{-3})
    	# ts - time step (s)
    	# idma - ideality marker (1=ideal, 0=non-ideal)
	# gam0 - reference activity coefficients
	# T - temperature (K)
    	#----------------------------------------------------------------------
	# outputs:
    	# Z1 - component concentrations (1st dim.) in shells (2nd dim.)
    	# Z1sc - same as above for 2nd phase
	# Nz0 - number of moles per component and shell (mol) (1st & 2nd dim.)
    	# Nz0sc - same as above for 2nd phase
	# V1 - shell volumes (m^3)
    	#---------------------------------------------------------------------
    	# create new object for new concentration matrix (mol/m3)
	Z1 = np.zeros((Z.shape[0], Z.shape[1]))
   	Z1sc = np.zeros((Zsc.shape[0], Zsc.shape[1])) 
    	# bulk gas phase concentrations (ug/m3)
	[Cg, esnow] = esnow_func(estime, time, tsn, es, Cstar)
	# component mole fractions in surface shell following gas phase 
	# diffusion	
	
	[x2,x2sc] = gpp(Z[:,shn-1], Zsc[:,shn-1], Cstar, Cg, Rp, M, p, Dg, T, ts, idma, gam0, tsn)				

	#print 'concs2'
	#print esnow	
	# x2 = esnow # assume gas-particle equilibrium	
		
    	# new number of moles object
	Nz1 = np.zeros((Nz0.shape[0], Nz0.shape[1]))
	Nz1[:, :] = Nz0[:, :]
    	Nz1sc = np.zeros((Nz0sc.shape[0], Nz0sc.shape[1]))
	Nz1sc[:, :] = Nz0sc[:, :]

	# number of moles of semi-volatile (no. moles non-volatile constant)	
	# (from mole fraction eq.)
	Nz1[0, shn-1] = (Nz1[1, shn-1]*x2[0])/(1.0-x2[0]) 	
    	Nz1sc[0, shn-1] = (Nz1sc[1, shn-1]*x2sc[0])/(1.0-x2sc[0])
	
	# new volume surface shell (no. moles of non-volatiles constant)
    	# new volume array object (m^3)
	V1 = np.zeros((1, shn))
	V1[0, :] = V0[0, :]
	V1[0, shn-1] = np.sum(Nz1[:, shn-1]*(M[:, 0]/p[:, 0]))
	V1sc = np.zeros((1, shn))
	V1sc[0, :] = V0sc[0, :]
	V1sc[0, shn-1] = np.sum(Nz1sc[:, shn-1]*(M[:, 0]/p[:, 0]))
    	
	# concentration of each component in surface (mol m^{-3})
	Z1[:, shn-1] = Nz1[:, shn-1]/V1[0, shn-1]
	if V1sc[0,shn-1]>0.0:
		Z1sc[:, shn-1] = Nz1sc[:, shn-1]/V1sc[0, shn-1]	
	else:
		Z1sc[:, shn-1]=0.0	

	del Cg, x2
	
	#print 'concs2'
	
	return Z1, Nz1, V1, Z1sc, Nz1sc, V1sc 
