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
# function to find mole fractions from a given activity using interpolation

import numpy as np

def int_x(ai0, gam0, x0):
	
	# ---------------------------------------------------------------------
	# inputs:
	# ai0 - actual activity
	# gam0 - reference sv mole fractions and activity coefficients
	# x0 - actual sv mole fractions
	# ---------------------------------------------------------------------

	# reference activities (for binary system) 
	# (mole fraction*activity coefficient)
	a_sv = gam0[2, :]*gam0[0, :]

	
	# find where there's a peak in activity
	ipeak = np.where(a_sv==np.max(a_sv))	
	if ipeak==a_sv.shape[0]-1:	

		# subtract reference activity from actual activity
		del_a = ai0-a_sv
		# consider only points where actual 
		# activity>reference activity
		del_a[del_a<0]=1.0e6
		ltei = np.where(del_a==np.min(del_a))
	else:

		if x0<=gam0[2,ipeak]:		
			ltei = np.sum(a_sv[0:ipeak[0]]<=ai0)-1
		if x0>gam0[2,ipeak]:	
			ltei = ipeak+np.sum(a_sv[ipeak[0]::]>=ai0)
			

#  	print 'int_x'
	#print x0
#  	print ltei
	
	lo_diff = np.abs(a_sv[ltei]-ai0)
	hi_diff = np.abs(a_sv[ltei+1]-ai0)
	lo_fac = lo_diff/(lo_diff+hi_diff)
	hi_fac = hi_diff/(lo_diff+hi_diff)
	# interpolate to get mole fraction
	x0 = gam0[2, ltei]*hi_fac+gam0[2, ltei+1]*lo_fac		

	del a_sv, ltei, lo_diff, hi_diff, lo_fac, hi_fac

	return x0	
