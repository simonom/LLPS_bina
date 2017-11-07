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
# function to find new spatial dimensions of shells

import numpy as np

def spat_arrays_inequal(V0, shn, V0sc):

	# ---------------------------------------------------------------------
	# inputs:
	# V0 - shell volumes (m3)
	# shn - number of shells (dimensionless)
	# V0sc - schlieren volumes (m3)
	# outputs:
	# delta - individual shell widths (m)
	# rc - cumulative shell widths (m)
	# A - shell surface areas (m2)
	# ---------------------------------------------------------------------
	# factor schlieren volumes into diffusing distance (m3)
	V1 = V0+V0sc	
	# individual shell widths (m)
	delta = np.append((3.0*np.cumsum(V1)[0]/(4.0*np.pi))**(1.0/3.0),
			(3.0*np.cumsum(V1)[1:shn]/(4.0*np.pi))**(1.0/3.0)-
			(3.0*np.cumsum(V1)[0:shn-1]/(4.0*np.pi))**(1.0/3.0))
	
	A = np.zeros((1, shn)) # shell areas (m2)
	rc = np.zeros((1, shn)) # cumulative shell end points (m)
	rc[0, :] = (3.0*np.cumsum(V1)[0:shn]/(4.0*np.pi))**(1.0/3.0)
	A[0, :] = 4.0*np.pi*rc[0, :]**2
	
	rsc = np.zeros((1, shn)) # 2nd phase radius (m2)
	Asc = np.zeros((1, shn)) # 2nd phase areas (m2)
	# index of shells with 2nd phase
	ind = (V0sc>0)
	rsc[ind] = ((3.0/(4.0*np.pi))*V0sc[ind])**(1.0/3.0)
	Asc[ind] = 4.0*np.pi*rsc[ind]**2.0
	
	return delta, rc, A, Asc
