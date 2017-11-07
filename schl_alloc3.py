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
import matplotlib.pyplot as plt

def schl_alloc(Nznew, nscnew, shn, gam0, M, rho, PS0, V1, V1sc):

	# --------------------------------------------------------------------
	# inputs:
	# Nznew - number of moles in 1st phase
	# nscnew - no. moles 2nd phase
	# shn - no. shells
	# gam0 - activity coefficients
	# M - molar masses (g/mol)
	# rho - densities (g/m3)
	# V1 - volumes of 1st phase shells (m^3)
	# V1sc - volumes of 2nd phase shells (m^3)
	# --------------------------------------------------------------------
	# sv mole fraction in all shells
	xsv0 = Nznew[0,:]/(np.sum(Nznew[:,:],axis=0))		

	# shells without LLPS
	is0 = nscnew[0,:]==0.0 # index
	for ic in range(1,M[:,0].shape[0]): # component loop
		is1 = (nscnew[ic,:]==0.0)
		is0 = is0+is1  	

	# mole fraction of sv in shells without LLPS
	xsv = Nznew[0,is0]/(np.sum(Nznew[:,is0],axis=0))	
	
	# difference in Gibbs free energy between 1 phase and phase separated 
	# system (if positive, phase separation may be favourable)
	delG=np.interp(xsv,PS0[:,0],PS0[:,1])		
	
	# return if no phase separation needed or separation has already 
	# happened
	if np.sum(delG>0)==0 or np.sum(is0==0)>0:	
		return Nznew, nscnew, V1, V1sc
	
	# sv mole fraction in shells where phase separation thermodynamically 
	# favourable
	xsv=xsv[delG>0.0]
	
	# shell index where partitioning could occur
	is0 = (xsv0==xsv)

	# semi-volatile (water) mole fractions of 2 phases
	xsv1=np.zeros((1,xsv.shape[0]))
	xsv2 = np.zeros((1,xsv.shape[0]))
	xsv1[:,:] = np.interp(xsv,PS0[:,0],PS0[:,2]) # new phase
	xsv2[:,:] = np.interp(xsv,PS0[:,0],PS0[:,3]) # existing phase	

	# the Gamma value at these mole fractions
	Gam1 = np.zeros((1,xsv.shape[0]))
	Gam2 = np.zeros((1,xsv.shape[0]))
	Gam1[:,:] = np.interp(xsv1,gam0[2,:],gam0[1,:])
	Gam2[:,:] = np.interp(xsv2,gam0[2,:],gam0[1,:])
	
	
	# return if diffusion does not promote separation
	
	ind1 = xsv1<xsv2
	ind1 = ind1*((Gam1+Gam2)/2.0>0.0)
	ind2 = xsv1>xsv2
	ind2 = ind2*((Gam1+Gam2)/2.0<0.0)
	if np.sum(ind1+ind2)==0:
		return Nznew, nscnew, V1, V1sc	
	
	# shell index where partitioning does occur
	is0 = np.squeeze((ind1+ind2)>0)

	# otherwise do partitioning between phases:	
		
	# partition current moles between phases 
	# required mole fraction in 1st phase (1st row) and 
	# 2nd phase (2nd row) of partitioning shell
	xreq = np.append(xsv2, xsv1,axis=0)
	
	a = xreq[1,:] # mole fraction in 2nd phase
	b = 0 # original moles sv in 2nd phase	
	d = 0 # original moles nv in 2nd phase
	f = xreq[0,:] # mole fraction in 1st phase
	
	g = Nznew[0,is0] # original moles sv in bulk (1 phase system)
	h = Nznew[1,is0] # original moles nv in bulk (1 phase system)
	
	
	ind = (xsv2==0.0)
	if np.sum(ind)>0:
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
	
	Nznew[0,is0] = Nznew[0,is0]-nsvmove
	Nznew[1,is0] = Nznew[1,is0]-nnvmove
	
	nscnew[0,is0] = nscnew[0,is0]+nsvmove
	nscnew[1,is0] = nscnew[1,is0]+nnvmove
	
	# molar volume of components (m3/mol)
	MV=M/rho
	# new shell volumes
	V1[0,:]=np.sum(Nznew[:,:]*MV, axis=0)	 
	V1sc[0,:]=np.sum(nscnew[:,:]*MV, axis=0)
	
	#print 'schl'
	#print xsv	
	#print xsv1
	#print xsv2	
	#print (Gam1+Gam2)/2.0	
	#print V1
	#print V1sc	
	#print Nznew[0,:]/np.sum(Nznew[:,:],0)
	#print nscnew[0,:]/np.sum(nscnew[:,:],0)
	#return

	return Nznew, nscnew, V1, V1sc
