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
# function to calculate the initial spatial arrays

import numpy as np

def spat_arrays(shn,Dp):
	
	# ---------------------------------------------------------------------
	# inputs:
	# shn - number of shells
	# Dp - particle diameter (m)
	# --------------------------------------------------------------------
	rp = Dp/2.0 # whole particle radius (m)
	if shn>1:
		Diamw = rp/(1.0e3) # surface shell width (m)
		# uncomment to get same width surface shell as bulk shells
		#Diamw = rp/shn # surface shell width (m) 
		r0 = rp-Diamw #  initial particle bulk radius (m)
		# individual radial width of particle shells (m)	
		delta = (np.append(np.ones((shn-1))*(r0/(shn-1)),Diamw))	
	elif shn==1:
		Diamw=rp
		delta=Diamw
	
	# cumulative radial widths of shells (m)
	rc = np.cumsum(delta)
	# areas of shells (m^2)
	A = (4.0*np.pi*(rc**2.0))
    	# volume of individual shells (m^3)
	V0 = np.zeros((1,shn))
	V0[0, 0] = (4.0/3.0)*np.pi*(rc[0])**3 # volume of core shell
	
	if shn>1:
		# volume of all non-core shells (m^3)
		V0[0,1:shn] = ((4.0/3.0)*np.pi)*(rc[1:shn]**3-rc[0:shn-1]**3)
	
	# volume of schlieren (m^3)
	V0sc = np.zeros((1, shn))

	return rc, V0, Diamw, V0sc
