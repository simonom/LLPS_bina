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
# function to find whether acceptable change to number of moles per shell has 
# been exceeded

import numpy as np

def accept_change(Nz0, Nznew, shn, ut, n0sc, nscnew):

	if shn>1:
		# initial mole fraction difference between neighbouring particle 
		# shells (going from outer to inner)
		delta_act = np.abs(Nz0[:, 1:shn]/np.sum(Nz0[:, 1:shn], 0)-
						Nz0[:, 0:shn-1]/
						np.sum(Nz0[:, 0:shn-1], 0))
						
		# shells with a second phase
		ind = n0sc[0,:]>0
		
		delta_actsc = np.abs(n0sc[:, ind]/
						np.sum(n0sc[:, ind], 0)-
						Nz0[:,ind]/np.sum(Nz0[:,ind],0))			
		# define acceptable change as a function of initial difference 
		# in mole fraction between neighbouring particle shells for
		# 1st phase and as a function of initial difference in mole
		# fraction between phases per shell for 2nd phases
		utp = ut[0]+((ut[1]-ut[0])/1.0)*delta_act
		utpsc= ut[0]+((ut[1]-ut[0])/1.0)*delta_actsc
			
		# change in mole fraction of each component in 
		# shells exceeding acceptable change
		ex_i = (np.abs(Nz0[:, 0:shn-1]/np.sum(Nz0[:, 0:shn-1], 0)-
			Nznew[:, 0:shn-1]/np.sum(Nznew[:, 0:shn-1], 0))>utp)
		ex_i = ex_i[:,ind]+(np.abs(n0sc[:, ind]/np.sum(n0sc[:, ind], 0)
			-nscnew[:, ind]/np.sum(nscnew[:, ind], 0))>utpsc)

	elif shn==1 and np.sum(nscnew>0)>0: # one shell particle with two phases
		# initial mole fraction difference between phases 
		delta_act = np.abs(Nz0/np.sum(Nz0)-n0sc/np.sum(n0sc))
		# acceptable change defined as a fuction of initial difference 
		# in mole fraction between two phases
		utp = ut[0]+((ut[1]-ut[0])/1.0)*delta_act
		# whether threshold exceeded
		ex_i = (np.abs(Nz0/np.sum(Nz0))-np.abs(Nznew/np.sum(Nznew)))>utp
		ex_i = ex_i+(np.abs(n0sc/np.sum(n0sc))-
			np.abs(nscnew/np.sum(nscnew)))>utp
	else:
		ex_i=0

	return ex_i
