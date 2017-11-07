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
# function to interpolate thermodynamic factor as a 
# function of mole fraction
import numpy as np

def interp_Gam(xi, Gam0, shn):

	# inputs:
	# xi - sv mole fractions in shells
	# Gam0 - matrix of thermodynamic factors (2nd row)
	# as a function of sv mole fraction (3rd row)


	# -----------------------------------------------------
	# empty matrix for thermodynamic factors of each 
	# component in each shell
	Gam = np.zeros((1, 1, shn))
	# matrix of reference mole fraction spread across shells
	Gam_xref = (np.ones((shn, Gam0.shape[1]))*Gam0[2, :])
	Gam_xref = np.transpose(Gam_xref)
	
	# thermodynamic factors for this component:
	# matrix of thermodynamic factors spread across shells
	Gam_ref = np.ones((shn, Gam0.shape[1]))*Gam0[1, :]
	Gam_ref = np.transpose(Gam_ref) # transpose
		
	# get distances between sv mole fractions and reference 
	# sv mole fraction
	xi_dist = xi[0, :]-Gam_xref[:, :]
	
	
	# indices of highest reference mole fraction actual mole fraction is 
	# greater than
	xi_ind = np.sum(xi_dist>0, 0)-1
	
	# ensure code can deal with mole fraction=1
	a = (xi_ind==np.shape(Gam_xref)[0]-1)
	xi_ind[a]=xi_ind[a]-1
			
	# distance of mole fractions from lowest closest mole fraction
	dist_lo = np.abs(xi[0, :]-Gam_xref[xi_ind, 0])
	# distance of mole fractions from highest closest mole fraction
	dist_hi = np.abs(xi[0, :]-Gam_xref[xi_ind+1, 0])
		
	# thermodynamic factors found by linear interpolation
	Gam[0, 0, 0:shn] = ((dist_lo)/(dist_lo+dist_hi)*
			Gam_ref[(xi_ind+1), 0]+(dist_hi)/(dist_lo+dist_hi)*
			Gam_ref[(xi_ind), 0])
			
	
	return Gam
