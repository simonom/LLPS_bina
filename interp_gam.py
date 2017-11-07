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
# function to interpolate activity coefficients as a function 
# of mole fraction
import numpy as np
import matplotlib.pyplot as plt

def interp_gam(xi, gam0, shn):

	# empty matrix for activity coefficients of each component in each shell
	gam = np.zeros((1, 1, shn))
	# matrix of reference mole fraction spread across shells
	gam_xref = (np.ones((shn, gam0.shape[1]))*
			gam0[2, :])
	gam_xref = np.transpose(gam_xref)
	
	# activity coefficients for this component:
	# matrix of activity coefficients spread across shells
	gam_ref = np.ones((shn, gam0.shape[1]))*gam0[0, :]
	gam_ref = np.transpose(gam_ref) # transpose
	
	#print 'interp_gam'
	#plt.plot(gam0[2,:],gam0[0,:]*gam0[2,:],'-b')
	#plt.show()
	#return		
	
	# get distances between mole fractions and reference mole fraction
	xi_dist = xi[0, :]-gam_xref[:, :]
		
	# indices of highest reference mole fraction actual mole fraction is 
	# greater than
	xi_ind = np.sum(xi_dist>0, 0)-1
			
	# distance of mole fractions from lowest closest mole fraction
	dist_lo = np.abs(xi[0, :]-gam_xref[xi_ind, 0])
	# if sv mole fraction already at 1, then ensure we don't look for a 
	# higher mole fraction 
	ish = (xi_ind==(np.shape(gam_xref)[0]-1))	
	xi_ind[ish]=xi_ind[ish]-1	
	# distance of mole fractions from highest closest mole fraction
	dist_hi = np.abs(xi[0, :]-gam_xref[xi_ind+1, 0])
		
	# activity coefficients found by linear interpolation
	gam[0, 0, 0:shn] = ((dist_lo)/(dist_lo+dist_hi)*
			gam_ref[(xi_ind+1), 0]+(dist_hi)/(dist_lo+dist_hi)*
			gam_ref[(xi_ind), 0])		

	return gam
