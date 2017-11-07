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
# function to allocate required moles of components to/from schlieren

import numpy as np
from interp_Gamma import interp_Gam as int_Gam
from nucl_form import nucl_form
import matplotlib.pyplot as plt

def phas_sep1(Nznew, nscnew, shn, gam0, M, rho, PS0, V1, 
			V1sc, Db, ts, nuc2p):

	# --------------------------------------------------------------------
	# inputs:
	# Nznew - number of moles in 1st phase
	# nscnew - no. moles 2nd phase
	# shn - no. shells
	# gam0 - activity coefficients
	# M - molar masses (g/mol)
	# rho - densities (g/m3)
	# V1 - volumes of 1st phase shells (m3)
	# V1sc - volumes of 2nd phase shells (m3)
	# Db - component self-diffusion coefficients (m2/s)
	# ts - diffusion and nucleation time step (s)
	# nuc2p - record of fraction of nuclei to be formed
	# --------------------------------------------------------------------
	
	
	[nucn, xsv1, xsv2] = nucl_form(Nznew,PS0,shn,Db,ts)
	
	# keep a count on number of nuclei formed 
	nucc = np.zeros((1,shn))
	
	# index of where number of nuclei to form less than one
	# but greater than zero
	ind = (nucn<1.0)
	ind = ind*(nucn>0.0)
	nuc2p[0,ind] = nuc2p[0,ind]+nucn[ind]
	
	# index of where the accumulated number of fractional 
	# nuclei now equals or exceeds one 
	ind = np.squeeze(nuc2p>=1.0)
	nucn[ind] = nuc2p[0,ind] 
	
	
	while (np.sum((nucn-nucc)>=1.0))>0:
		
		# index where number of nuclei to form greater than one
		ind = np.squeeze((nucn-nucc)>=1.0)
		
		# partition current moles between phases 
		# required mole fraction in 1st phase (1st row) and 
		# 2nd phase (2nd row) of partitioning shell
		xreq = np.zeros((2,shn))
		xreq[0,ind] = xsv2[ind]
		xreq[1,ind] = xsv1[ind]
		
		a = xreq[1,ind] # mole fraction in 2nd phase
		b = 0 # original moles sv in 2nd phase	
		d = 0 # original moles nv in 2nd phase
		f = xreq[0,ind] # mole fraction in 1st phase
		
		g = Nznew[0,ind] # original moles sv in bulk (1 phase system)
		h = Nznew[1,ind] # original moles nv in bulk (1 phase system)
		
		
		ind2 = (xsv2==0.0)
		if np.sum(ind2)>0:
			# number of moles sv to transfer
			c = 0.0
			# number of moles of non-volatile to transfer
			if xsv1==1.0: # move all nv out
				e=h	
			# rearrange mole fraction sv in phase 1 eq. to find 
			# no. moles nv needed in phase 1 and subtract from 
			# existing number
			else: 
				e = h-g*((1.0-xsv1)/xsv1)
	
		else:
			# number of moles sv to transfer
			c = (g+f*(-g-h+b/a-b-d))/(1.0-f/a)
			# number of moles of non-volatile to transfer
			e = b/a+c/a-b-c-d
			
		nnvmove = e
		nsvmove = c
		
		Nznew[0,ind] = Nznew[0,ind]-nsvmove
		Nznew[1,ind] = Nznew[1,ind]-nnvmove
		
		nscnew[0,ind] = nscnew[0,ind]+nsvmove
		nscnew[1,ind] = nscnew[1,ind]+nnvmove
		# update number of new nuclei formed
		nucc = nucc+np.ones((1,shn))	
	
		[nucn, xsv1, xsv2] = nucl_form(Nznew,PS0,shn,Db,ts)
		
	# record the fraction of nuclei left over when a new
	# nucleus has formed but only the fraction of one 
	# remains
	ind = np.squeeze(ind) # lose any excess dimensions
	nuc2p[0,ind] = nuc2p[0,ind]+(nucn[ind]-nucc[0,ind])
	
	# molar volume of components (m3/mol)
	MV=M/rho
	# new shell volumes
	V1[0,:]=np.sum(Nznew[:,:]*MV, axis=0)	 
	V1sc[0,:]=np.sum(nscnew[:,:]*MV, axis=0)
	

	## the Gamma value at these mole fractions
# 	Gam1 = np.zeros((1,xsv.shape[0]))
# 	Gam2 = np.zeros((1,xsv.shape[0]))
# 	Gam1[:,:] = np.interp(xsv1,gam0[2,:],gam0[1,:])
# 	Gam2[:,:] = np.interp(xsv2,gam0[2,:],gam0[1,:])
# 	
# 	# return if diffusion does not promote separation
# 	ind1 = xsv1<xsv2
# 	ind1 = ind1*((Gam1+Gam2)/2.0>0.0)
# 	ind2 = xsv1>xsv2
# 	ind2 = ind2*((Gam1+Gam2)/2.0<0.0)
# 	if np.sum(ind1+ind2)==0:
# 		return Nznew, nscnew, V1, V1sc
	

	return Nznew, nscnew, V1, V1sc
