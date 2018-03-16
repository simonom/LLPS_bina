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
# function to calculate diffusion through the particle
# using Maxwell-Stefan approach

import numpy as np
from interp_Gamma import interp_Gam as int_Gam
from interp_gam import interp_gam as gam_interp_1D
from accept_change import accept_change
from mat_inv import mat_inv
from dotprod import dotprod
from N_flux import N_flux
from gam_interp_2D import gam_interp_2D
from Jcomp_calc import Jcomp_calc

def eqn3(A, shn, M, Nz0, delta, Z, ts, p, idma, V0, Db, gamsv, gamnv, tsn, 
	 ut, C):

	# -----------------------------------------------------
	# inputs:
	
    # A - shell surface area (m2)
    # shn - number of shells
    # M - molar mass of component (g/mol)
    # Nz0 - number of moles of each component (1st dim.) per 
	# shell (2nd dim.)
    # delta - individual shell width (m)
    # Z - concentration of components (mol/m3) (1st dim.) 
	# per shell (2nd dim.)
    # ts - time step to solve over (s)
    # p - component density (g/m3)
    # idma - marker for ideality (1==ideal, 0==non-ideal)
    # V0 - original shell volume (m3)
    # Db - self-diffusion coefficients of components 
	# (m2/s)
	# gamsv - reference semi-volatile activity coefficients
	# gamnv - reference non-volatile activity coefficients
	# tsn - time step count
	
    # ------------------------------------------------------
	# outputs:
	
    # Nznew - number of moles of each component (1st dim.) 
	# per shell (2nd dim.)
    # V - shell volumes (m3)
    # Z - concentration of components (mol/m3) 
	# (1st dim.) per shell (2nd dim.)
	# Driver - the dot product of the diffusion coefficient
	# (B) and thermodynamic terms (Gamma) in 2.2.10 of Taylor (1993)
    # -----------------------------------------------------
    	
    # prepare arrays
    
	# number of components
	ncT = Z.shape[0]
	# number of components-1
	nc = ncT-1

	# new matrix (new object) for number of moles (mol)
	Nznew = np.zeros((Nz0.shape[0], Nz0.shape[1]))
	Nznew[:, :] = Nz0[:, :]
    	
	# number of shells minus 1 and plus 1
	shnm = shn-1
	shnp = shn+1

	# empty results matrix for concentration gradient (with distance) for 
	# each component (1st dim.) between each shell (2nd dim.)
	nabC = np.zeros((nc, shnm))
	
	# empty results matrix for component (1st dim.) flux across 
	# each boundary (2nd dim.) (mol/s) (use number of boundaries+2 
	# for 2nd dim. so that the first and last columns can represent 
	# fluxes at the particle centre and surface, respectively (both=0)).
	flux = np.zeros((ncT, shnp))
	
	# component molar volumes (m3/mol)
	VM = np.zeros((M.size, 1))
	VM[:, 0] = (M[:, 0]/p[:, 0])
    	
	# mole fractions of all components 
	# (1st dim.) in all shells (2nd dim.)
	xi = np.zeros((ncT, shn)) # 1st phase
	
	# binary mole fractions
	xmat = np.zeros((ncT, ncT, shnm))

	# diffusion coefficient matrix for D at shell boundaries (m2/s)
	D = np.zeros((ncT, ncT, shnm))

	# total number of moles per shell (mol)
	xT = np.zeros((1, shn))

	
	# calculations begin --------------------------------------------------
	
	# check on whether the gas phase diffusion calculation has led to an 
	# error and therefore the code should jump straight to decreasing time 
	# step	
	if np.sum(np.sum(Nz0[:, :]<0))>0:
		return Nz0, V0, Z, 0

	# total number of moles per shell
	xT = np.sum(Nz0[:, :], 0)
	# component mole fractions per shell
	xi[:, :] = Nz0[:, :]/xT
	# average mole fractions at shell boundaries
	xibar = (xi[:, 1::]+xi[:, 0:-1])/2.0
	
	# arithmetic mean concentration of each component over bounding shells 
	# (concentration at shell boundary) (mol/m3)
	Zbar = ((Z[:, 1:shn]+Z[:, 0:shnm])/2.0)

	# binary (just 2 component) mole fraction matrix per shell 
	# to be used for D calculation:
	
	# loop through components
	for ic1 in range(0, ncT):
		# loop through components (columns of D and Gamma matrix)
		for ic2 in range(0, ncT):

				# mole fraction with respect to other component
				# prevent numerical error
				ishgt = Zbar[ic1, :]>0
				isheq = Zbar[ic1, :]==0
				xmat[ic1, ic2, ishgt] = Zbar[ic1, ishgt]/(Zbar[ic1, ishgt]+Zbar[ic2, ishgt]) 
				xmat[ic1, ic2, isheq] = 0.0 
				if ic1 == ic2 and nc==2:
					D[ic1, ic2, :] = Db[ic1]
				elif ic1 == ic2 and nc==1:
					
					# Vignes
					D[ic1, ic2, :] = ((Db[ic1]**xmat[ic1, ic2, :])*
										(Db[ic2]**(1.0-xmat[ic1, ic2, :])))
					alpha = np.exp(C*((1.0-xmat[ic1, ic2, :])**3.0))
					# Ingram 2017 eq. 1
					D[ic1, ic2, :] = ((Db[ic1]**xmat[ic1, ic2, :]*alpha)*
										(Db[ic2]**(1.0-xmat[ic1, ic2, :]*alpha)))
										
				elif ic1 != ic2:
					# Vignes
					D[ic1, ic2, :] = ((Db[ic1]**xmat[ic1, ic2, :])*
										(Db[ic2]**(1.0-xmat[ic1, ic2, :])))
	del ic1, ic2
	
	# distance between shell centres (m)
	delta2 = (np.ones((nc, delta.shape[0]-1))*
		np.transpose(0.5*(delta[0:delta.shape[0]-1]+
		delta[1:delta.shape[0]])))

	
	
	# total concentration at shell boundary (2.2.10 Taylor (1993))
	Ct = np.sum(Z, 0)
	Ctbar = ((Ct[0:-1]+Ct[1::])/2.0)


	# -----------------------------------------------------
	# calculating the B and Gamma terms in 2.1.22 of Taylor (1993):
	
	if nc==1: # binary case requires 1D interpolation
		[act_pshell_sv] = gam_interp_1D(xi, gamsv, shn)
	if nc==2: # ternary case requires 2D interpolation
		[act_pshell_sv, act_pshell_nv] = gam_interp_2D(xi, gamsv, gamnv, shn)
	
	# mole fraction gradient (/m) between shells
	# (positive means decreasing concentration inwards)
	nabC[0:nc, :] = (xi[0:nc, 1::]-xi[0:nc, 0:-1])/delta2
	# activity gradient rather than mole fraction gradient (/m), for
	# replicating Shiraiwa's activity coefficient-corrected formulaism
# 	nabC[0, :] = (xi[0, 1::]*act_pshell_sv[0, 1::]-xi[0, 0:-1]*act_pshell_sv[0, 0:-1])/delta2 
	
	# inputs for the thermodynamic factor calculation:
	numerat = np.zeros((nc, shn-1))
	denomin = np.zeros((nc, shn-1))

	
	# (2.2.5, 2.1.21 and 2.1.22 Taylor 1993)
	for ic1 in range(0, nc): # component loop
		if ic1==0:
		
			# numerator and denominator from sv
			numerat[ic1, :] = np.log(act_pshell_sv[0, 1::]/act_pshell_sv[0, 0:-1])
			denomin[ic1, :] = xi[ic1, 1::]-xi[ic1, 0:-1]
		if ic1==1:
			
			# numerator and denominator from nv
			numerat[ic1, :] = np.log(act_pshell_nv[1, 1::]/act_pshell_nv[1, 0:-1])
			denomin[ic1, :] = xi[ic1, 1::]-xi[ic1, 0:-1]
	
	# thermodynamic factors and B (2.2.5, 2.1.21 and 2.1.22 Taylor 1993)
	Gamma = np.zeros((nc, nc, shn-1))
	Beta = np.zeros((nc, nc, shn-1))
	
	
	# loop through components (rows of Gamma and B matrix)
	for ic1 in range(0, nc):
		# loop through components (columns of Gamma matrix)
		for ic2 in range(0, nc):

			# the thermodynamic factor calculation (2.2.5, 2.1.21 and 2.1.22 Taylor 1993):
			
			if ic1==ic2:
				ish = np.abs(denomin[ic1, :])>1.0e-15 # prevent numerical error
				Gamma[ic1, ic2, ish] = 1.0+xibar[ic1, ish]*(numerat[ic1, ish]/denomin[ic1, ish])
				Beta[ic1, ic2, :] = (xibar[ic1, :]/D[ic1,-1,:] + 
							(np.sum(xibar[:, :]/D[ic1, :, :], 0)-xibar[ic1,:]/D[ic1, ic1, :]))
				
			if ic1!=ic2:
				ish = np.abs(denomin[ic1, :])>1.0e-15 # prevent numerical error
				Gamma[ic1, ic2, ish] = 0.0+xibar[ic1, ish]*(numerat[ic1, ish]/denomin[ic2, ish])
				Beta[ic1, ic2, :] = -xibar[ic1, :]*(1.0/D[ic1, ic2, :]-1.0/D[ic1, -1, :])
	
	# inverse of Beta
	Betainv = mat_inv(Beta)
	
	# flux of each component (rows) per shell (columns) (2.2.10 Taylor 1993)
	# (mol/s) (+ value represents flux in direction from 
	# outer shell to inner)
	# uncomment line below for Fickian framework
	Gamma = np.ones((Gamma.shape[0], Gamma.shape[1], Gamma.shape[2])) 
	[Jcomp, Driver] = dotprod(Betainv, Gamma, nabC, nc, shn-1, Ctbar)
	
	# no flux at centre or surface so append zeros to start and end of Jcomp
	Jcomp = np.append(np.zeros((nc, 1)), Jcomp, 1)
	Jcomp = np.append(Jcomp, np.zeros((nc, 1)), 1)
	
	JcompT = Jcomp[:, 1::]*A[0, 1::]-Jcomp[:, 0:-1]*A[0, 0:-1]
	
	
	# can't have flux out of a shell if there's no component available
	ind_xi_too_low = xi[0:-1, :]<1.0e-3
	if np.sum(np.sum(ind_xi_too_low))>0 and tsn>1:
		
		ind_neg_flux = (Jcomp[:, 1::]*A[0, 1::]-Jcomp[:, 0:-1]*A[0, 0:-1])<0
		# lower and upper bound flux index
		lb_fluxi = np.append(ind_neg_flux*ind_xi_too_low, np.zeros((ind_neg_flux.shape[0], 1)), 1)
		lb_fluxi = lb_fluxi==1
		ub_fluxi = np.append(np.zeros((ind_neg_flux.shape[0], 1)), ind_neg_flux*ind_xi_too_low, 1)
		ub_fluxi = ub_fluxi==1

		for index_loop in range(1, np.sum(lb_fluxi[0, :])+1):
			index_new = (np.where(np.cumsum(lb_fluxi[0, :])==index_loop))[0][0]
			Jcomp[0, index_new] = 0.0
		for index_loop in range(1, np.sum(lb_fluxi[1, :])+1):
			index_new = (np.where(np.cumsum(lb_fluxi[1, :])==index_loop))[0][0]
			Jcomp[1, index_new] = 0.0
		for index_loop in range(1, np.sum(ub_fluxi[0, :])+1):
			index_new = (np.where(np.cumsum(ub_fluxi[0, :])==index_loop))[0][0]
			Jcomp[0, index_new] = 0.0
		for index_loop in range(1, np.sum(ub_fluxi[1, :])+1):
			index_new = (np.where(np.cumsum(ub_fluxi[1, :])==index_loop))[0][0]
			Jcomp[1, index_new] = 0.0
			
	# total flux
	JcompT = Jcomp[:, 1::]*A[0, 1::]-Jcomp[:, 0:-1]*A[0, 0:-1]

	# new number of moles of each component per shell (mol)
	Nznew[0:nc, :] = Nz0[0:nc, :]+(JcompT)*ts	
	
	# new volumes of 1st phase (m3)
	V = np.zeros((1, shn))
         
	Mv = np.zeros((M.size, 1)) # molar volumes
	Mv[:, 0] = (M[:, 0]/p[:, 0])
	V[0, :] = np.sum(Nznew*Mv, axis=0)	

	if np.sum(np.sum(Nznew<0))>0:
		return Nznew, V, 0, JcompT

	# calculate flux of component N
	if (np.sum(xi[-1,:]<1.0e-3))>0:

		[Nznew, V, JV3] = N_flux(Jcomp, ts, Mv, shn, V, Nznew, tsn, A)

	else:
		# change from original bulk shell volume (m3)
		Vdiff = (V0)-(V)
	
		# flux of component N to maintain volume (mol)
		JN = Vdiff*(1.0/Mv[nc, 0])
		#JN = 0.0
	
		# number of moles of component N per shell (mol)
		Nznew[nc, :] = Nz0[nc, :]+JN
		
		# new volume of bulk shells (m3)
		V = np.zeros((1, shn))
		V[0, :] = np.sum(Nznew*Mv, axis=0)
	
	
	Znew = np.zeros((Z.shape[0], shn))     
	# new concentrations of components (mol/m3)
	# prevent numerical error display due to zero volume
	if np.sum(np.sum(Nznew<0))>0:
		return Nznew, V, Znew, JcompT
		
	Znew[:, :] = Nznew[:, :]/(np.ones((Nznew.shape[0], shn))*V[0, :])
	
	del Mv
	return Nznew, V, Znew, JcompT
	
# 	
# 	# ----------------
# 
# 	# only consider altering flux if all new 
# 	# estimates of components above zero
# 	if np.sum(np.sum(Nznew<0))==0: 
# 		
# 		# estimate new flux
# 		Jcomp_fut = Jcomp_calc(Nznew, Znew, ncT, Db, delta, gamsv, gamnv, shn, nc, shnm, shnp, M, p)
# 		if np.sum(np.sum((Jcomp[:, 1:-1]*Jcomp_fut)<0))>0:
# 			ish1 = (Jcomp[:, 1:-1]*Jcomp_fut)<0 # index where flux changes sign
# 			Jcomp_max = np.max(np.abs(Jcomp[:, 1:-1]), 1) # largest wrt shells
# 			Jcomp_max.shape = (Jcomp[:, 1:-1].shape[0], 1) # ensure 2D
# 			Jcomp_max = Jcomp_max*np.ones((Jcomp[:, 1:-1].shape[0], Jcomp[:, 1:-1].shape[1]))
# 			ish2 = np.abs(Jcomp[:, 1:-1])==(Jcomp_max) # index where flux largest of all shells
# 			ish3 = np.abs(Jcomp[0, 1:-1])>(np.abs(Jcomp[1, 1:-1])/1.0e2)
# 			ish3.shape = ((1, len(ish3))) # ensure 2D
# 			ish4 = np.abs(Jcomp[1, 1:-1])>(np.abs(Jcomp[0, 1:-1])/1.0e2)
# 			ish4.shape = ((1, len(ish4))) # ensure 2D
# 			ish3 = np.append(ish3, ish4, 0)
# 			
# 			ish = ish1*ish2*ish3 # index of potential fluxes to reduce to realistic size
# 
# 			while np.sum(np.sum(ish))>0:
# 				
# 				Jcomp[:, 1:-1][ish] = Jcomp[:, 1:-1][ish]/2.0
# 				
# 				# total flux
# 				JcompT = Jcomp[:, 1::]*A[0, 1::]-Jcomp[:, 0:-1]*A[0, 0:-1]
# 			
# 				# new number of moles of each component per shell (mol)
# 				Nznew[0:nc, :] = Nz0[0:nc, :]+(JcompT)*ts	
# 				
# 				# new volumes of 1st phase (m3)
# 				V = np.zeros((1, shn))
# 					 
# 				Mv = np.zeros((M.size, 1)) # molar volumes
# 				Mv[:,0] = (M[:, 0]/p[:, 0])
# 				V[0, :] = np.sum(Nznew*Mv, axis=0)	
# 			
# 			
# 				# calculate flux of component N
# 				if (np.sum(xi[-1,:]<1.0e-3))>0:
# 					
# 					[Nznew, V, JV3] = N_flux(Jcomp, ts, Mv, shn, V, Nznew, tsn, A)
# 			
# 				else:
# 					# change from original bulk shell volume (m3)
# 					Vdiff = (V0)-(V)
# 					# flux of component N to maintain volume (mol)
# 					JN = Vdiff*(1.0/Mv[nc, 0])
# 					# number of moles of component N per shell (mol)
# 					Nznew[nc, :] = Nz0[nc, :]+JN
# 					# new volume of bulk shells (m3)
# 					V = np.zeros((1, shn))
# 					V[0, :] = np.sum(Nznew*Mv, axis=0)
# 			
# 			
# 				Znew = np.zeros((Z.shape[0], shn))     
# 				# new concentrations of components (mol/m3)
# 				# prevent numerical error display due to zero volume
# 				if np.sum(np.sum(Nznew<0))>1:
# 					return Nznew, V, Znew, Jcomp
# 				Znew[:, :] = Nznew[:, :]/(np.ones((Nznew.shape[0], shn))*V[0, :])
# 				
# 				if np.sum(np.sum(Nznew<0))>0:
# 					return Nznew, V, Znew, Jcomp
# 				# estimate new flux
# 				
# 				Jcomp_fut = Jcomp_calc(Nznew, Znew, ncT, Db, delta, gamsv, gamnv, shn, nc, shnm, shnp, M, p)
# 		
# 			
# 				ish1 = (Jcomp[:, 1:-1]*Jcomp_fut)<0 # index where flux changes sign
# 				Jcomp_max = np.max(np.abs(Jcomp[:, 1:-1]), 1) # largest wrt shells
# 				Jcomp_max.shape = (Jcomp[:, 1:-1].shape[0], 1) # ensure 2D
# 				Jcomp_max = Jcomp_max*np.ones((Jcomp[:, 1:-1].shape[0], Jcomp[:, 1:-1].shape[1]))
# 				ish3 = np.abs(Jcomp[0, 1:-1])>(np.abs(Jcomp[1, 1:-1])/1.0e2)
# 				ish3.shape = ((1, len(ish3))) # ensure 2D
# 				ish4 = np.abs(Jcomp[1, 1:-1])>(np.abs(Jcomp[0, 1:-1])/1.0e2)
# 				ish4.shape = ((1, len(ish4))) # ensure 2D
# 				ish3 = np.append(ish3, ish4, 0)
# 				
# 				ish = ish1*ish2*ish3 # index of potential fluxes to reduce to realistic size
# 				# ---------------