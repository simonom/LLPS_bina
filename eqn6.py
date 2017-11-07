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

import numpy as np
from interp_Gamma import interp_Gam as int_Gam
from flux_iter import flux_iter as flux_iter
from accept_change import accept_change

def eqn3(A, Asc, shn, M, Nz0, delta, Z, ts, p, idma, V0, Db, gam0, n0sc, tsn, 
	V0sc, ut):
		
	# -----------------------------------------------------
	# inputs:
    	# A - shell surface area (m^2)
    	# shn - number of shells
    	# M - molar mass of component (g/mol)
    	# Nz0 - number of moles of each component (1st dim.) per 
	# shell (2nd dim.)
    	# delta - individual shell width (m)
    	# Z - concentration of components (mol/m3) (1st dim.) 
	# per shell (2nd dim.)
    	# ts - time step to solve over (s)
    	# p - component density (g/m^3)
    	# idma - marker for ideality (1==ideal, 0==non-ideal)
    	# V0 - original shell volume (m^3)
    	# Db - self-diffusion coefficients of components 
	# (m^2 s^{-1})
	# gam0 - reference activity coefficients
	# n0sc - matrix of number of moles of each component in 
	# 2nd phase
	# tsn - time step count
	# V0sc - original shell volume of 2nd phase (m3)
    	# ------------------------------------------------------
	# outputs:
    	# Nznew - number of moles of each component (1st dim.) 
	# per shell (2nd dim.)
    	# V - shell volumes (m^3)
    	# Z - concentration of components (mol m^{-3}) 
	# (1st dim.) per shell (2nd dim.)
    	# -----------------------------------------------------
    	
	# number of components
	ncT = Z.shape[0]
	# number of components-1
	nc = ncT-1

	# new matrix (new object) for number of moles (mol)
	Nznew = np.zeros((Nz0.shape[0], Nz0.shape[1]))
	Nznew[:, :] = Nz0[:, :]
	nscnew = np.zeros((n0sc.shape[0], n0sc.shape[1]))
	nscnew[:, :] = n0sc[:, :]
    	
	# number of shells minus 1 and plus 1
	shnm=shn-1
	shnp=shn+1

	# empty results matrix for concentration gradient (with distance) for 
	# each component (1st dim.) between each shell (2nd dim.)
	nabC = np.zeros((nc, shnm)) # b-b
	nabCsc = np.zeros((nc, shnm)) # sc-sc
    	nabCsco = np.zeros((nc, shnm)) # b-sco
	nabCsci = np.zeros((nc, shnm)) # sci-b
	# empty matrix for conc. grad. between phases within shells
	nabCwsh = np.zeros((nc,shn))
	
	# empty results matrix for component (1st dim.) flux across 
	# each boundary (2nd dim.) (mol/s) (use number of boundaries+2 
	# for 2nd dim. so that the first and last columns can represent 
	# fluxes at the particle centre and surface, respectively (both=0)).
	flux = np.zeros((ncT, shnp))
    	fluxsc = np.zeros((ncT, shnp))
	fluxsco = np.zeros((ncT, shnp))
	fluxsci = np.zeros((ncT, shnp))
	fluxwsh = np.zeros((ncT, shn))
	
	# component molar volumes (m3/mol)
	VM = np.zeros((M.size, 1))
	VM[:, 0] = (M[:, 0]/p[:, 0])
    	
	# mole fractions of all components 
	# (1st dim.) in all shells (2nd dim.)
	xi = np.zeros((Z.shape[0], shn)) # 1st phase
	xisc = np.zeros((n0sc.shape[0], shn)) # 2nd phase
	
	# mole fractions of all components (1st dim.) at shell boundaries 
	# (2nd dim.)
	xbar = np.zeros((Z.shape[0], shnm)) # b-b
	xbarsc = np.zeros((Z.shape[0], shnm)) # sc-sc
	xbarsco = np.zeros((Z.shape[0], shnm)) # b-sco
	xbarsci = np.zeros((Z.shape[0], shnm)) # sci-b
	xbarwsh = np.zeros((Z.shape[0], shn)) # 1p-2p

    	# binary diffusion coefficients per component (m2/s)
	D = np.zeros((nc, nc, shnm))
    	Dsc = np.zeros((nc, nc, shnm))
	Dsco = np.zeros((nc, nc, shnm))
	Dsci = np.zeros((nc, nc, shnm))
	Dwsh = np.zeros((nc, nc, shn))

	# total number of moles per shell (mol)
	xT = np.zeros((1, shn))
    	xTsc = np.zeros((1, shn))
	# mole fractions
	# component concentrations (mol/m^3)
  	Zb = np.zeros((Z.shape[0], shn))
    	Zsc = np.zeros((Z.shape[0], shn))
	
	# calculations begin --------------------------------------------------
	
	# check on whether the gas phase diffusion calculation has led to an 
	# error and therefore the code should jump straight to decreasing time 
	# step	
	if np.sum(np.sum(Nz0[:, :]<0))>0 or np.sum(np.sum(n0sc[:, :]<0))>0:
		return Nz0, V0, Z, n0sc, V0sc
	
    	# total number of moles per shell 1st phase
	xT = np.sum(Nz0[:, :], 0)
	xTsc = np.sum(n0sc[:, :], 0) # shell 2nd phase
    	# component mole fractions per shell 1st phase
	xi[:, :] = Nz0[:, :]/xT
	ish = xTsc>0.0 # shells with 2nd phase
	xisc[:, ish] = n0sc[:, ish]/xTsc[ish] # shell 2nd phase
	
	
	# component concentrations (mol/m3)
	Zb = Nz0/np.append(V0, V0, axis=0) # b
	V0sctemp = np.append(V0sc, V0sc, axis=0)
	Zsc[:, ish] = n0sc[:, ish]/V0sctemp[:, ish] # sc
		   	
	# arithmetic mean concentration of each component over bounding shells 
	# (concentration at shell boundary) (mol/m^3)
	Zbar = ((Z[:, 1:shn]+Z[:, 0:shnm])/2.0) # b-b
	Zbarsc = ((Zsc[:, 1:shn]+Zsc[:, 0:shnm])/2.0) # sc-sc
	Zbarsco = ((Zsc[:, 1:shn]+Z[:, 0:shnm])/2.0) # b-sco	
	Zbarsci = ((Z[:, 1:shn]+Zsc[:, 0:shnm])/2.0) # sci-b
	Zbarwsh = ((Z+Zsc)/2.0) # 1p-2p
		
	if shn>1:
		# arithmetic mean mole fraction of each component over 
		# bounding shells 
		# (mole fraction at shell boundary)
		xbar = Zbar/np.sum(Zbar, 0) # b-b
		# shell bounadries where 2nd phase present
		ish = Zbarsc[0, :]>0.0 
		# sc-sc	
		xbarsc[:, ish] = Zbarsc[:, ish]/np.sum(Zbarsc[:, ish], 0) 
		# shell boundaries where 2nd phase on outside and 1st phase on 
		# inside
		ish = xi[0, 0:shnm]*xisc[0, 1::]>0.0
		# b-sco
		xbarsco[:, ish] = Zbarsco[:, ish]/np.sum(Zbarsco[:, ish], 0) 
		# shell boundaries where 2nd phase on inside and 1st phase on 
		# outside
		ish = xi[0, 1::]*xisc[0, 0:shnm]>0.0
		# sci-b
		xbarsci[:, ish] = Zbarsci[:, ish]/np.sum(Zbarsci[:, ish], 0) 	
	
	# shells with 2nd phase
	ish = xisc[0, :]>0.0
	xbarwsh[:, ish] = Zbarwsh[:, ish]/np.sum(Zbarwsh[:, ish], 0) # 1p-2p	

	# distance between shell centres (m)
	delta2 = (np.ones((Z.shape[0]-1, delta.shape[0]-1))*
		np.transpose(0.5*(delta[0:delta.shape[0]-1]+
		delta[1:delta.shape[0]])))
	# distance between phases (m)
	deltawsh = delta*0.5

	# concentration gradient (mol/m4) between shells
	nabC[0:nc, :] = (Z[0:nc, 1:shn]-
		Z[0:nc, 0:shnm])/delta2 # b
	# quick fix in case mole fraction of sv in one shell 
	# gets very low, this prevents flux of sv away from shell
	index0 = (Z[0,:]/np.sum(Z,0)<1.0e-3)
	nabC[0,index0] = 0.0
	
	
	nabCsc[0:nc, :] = (Zsc[0:nc, 1:shn]-
		Zsc[0:nc, 0:shnm])/delta2 # sc
	# set to zero if nothing in schlieren
	ish = (Zsc[0:nc, 1:shn]*Zsc[0:nc, 0:shnm]==0.0)
	nabCsc[ish] = 0.0
		
		
	nabCsco[0:nc, :] = (Zsc[0:nc, 1:shn]-
		Z[0:nc, 0:shnm])/delta2 # b-sco
	ish = (Zsc[0:nc, 1:shn]==0)
	nabCsco[ish] = 0.0 # set to zero if nothing in 2nd phase
	
	nabCsci[0:nc, :] = (Zsc[0:nc, 1:shn]-
		Z[0:nc, 0:shnm])/delta2 # sci-b
	ish = (Zsc[0:nc, 0:shnm]==0.0)
	nabCsci[ish] = 0.0 # set to zero if nothing in 2nd phase	
		
	nabCwsh[0:nc, :] = (Z[0:nc, :]-
		Zsc[0:nc, :])/deltawsh # 1p-2p
	ish = (Zsc[0:nc, :]==0.0)
	nabCwsh[ish] = 0.0 # set to zero if nothing in 2nd phase	

	
	# loop through components (rows of the D matrix)
	for ic1 in range(0, nc):
		
		# loop through components (columns of D matrix)
		for ic2 in range(0, nc):
			
			if ic1 == ic2 and Z.shape[0]>1:
				
				# Vignes
				D[ic1, ic2, :] = np.product(Db**xbar[:, :], 0)
				Dsc[ic1, ic2, :] = np.product(Db**
							xbarsc[:, :], 0)
				Dsco[ic1, ic2, :] = np.product(Db**
							xbarsco[:, :], 0)
				Dsci[ic1, ic2, :] = np.product(Db**
							xbarsci[:, :], 0)
				Dwsh[ic1, ic2, :] = np.product(Db**
							xbarwsh[:, :], 0)
				# correct D for thermodynamic factor
				if idma == 0:
  
					# thermodynamic factor in 1st phase
					Gam = int_Gam(xi, gam0, shn)
					# thermodynamic factor in 2nd phase
					Gam2 = int_Gam(xisc, gam0, shn)
									
					# average of factors between 
					# neighbouring 1st phases	
					Gam11 = ((Gam[0,0,1::]+Gam[0,0,0:shnm])
						/2.0)
					# correct D
					D[ic1, ic2, :] = D[ic1, ic2, :]*Gam11
					
					# average of factors between 
					# neighbouring 2nd phases	
					Gam22 = ((Gam2[0,0,1::]+Gam2[0,0,0:shnm])
						/2.0)
					# correct D
					Dsc[ic1, ic2, :] = Dsc[ic1, ic2, :]*Gam22
					
					# average of factors between 
					# neighbouring 2 ph outside 1 ph inside
					Gamsco = ((Gam2[0,0,1::]+
						Gam[0,0,0:shnm])/2.0)
					# correct D
					Dsco[ic1, ic2, :] = (Dsco[ic1, ic2, :]*
							Gamsco)
	

					# average of factors between 
					# neighbouring 1 ph outside 2 ph inside
					Gamsci = ((Gam[0,0,1::]+
						Gam2[0,0,0:shnm])/2.0)
					# correct D
					Dsci[ic1, ic2, :] = (Dsci[ic1, ic2, :]*
							Gamsci)

					# average of factors between 
					# neighbouring phases within shells
					Gamwsh = ((Gam[0,0,:]+
						Gam2[0,0,:])/2.0)
					# index of where Gamma very low and 
					# therefore shells are effectively at 
					# thermodynamic equilibrium
					# ind = np.abs(Gamwsh)<1.0e-5
# 					if np.sum(ind)>0:
# 						Gamwsh[ind] = 0.0
					# correct D
					Dwsh[ic1, ic2, :] = (Dwsh[ic1, ic2, :]*
							Gamwsh)

			else:
				# Vignes
				D[ic1, ic2, :] = np.product(Db**xbar[:, :], 0)
				Dsc[ic1, ic2, :] = np.product(Db**
							xbarsc[:, :], 0)
				Dsco[ic1, ic2, :] = np.product(Db**
							xbarsco[:, :], 0)
				Dsci[ic1, ic2, :] = np.product(Db**
							xbarsci[:, :], 0)
				# Darken relation
				#D[ic1, ic2, :] = np.sum(Db*xbar[:, :], 0)
				#Dsc[ic1, ic2, :] = np.sum(Db*xbarsc[:, :], 0)
				#Dsco[ic1, ic2, :] = np.sum(Db*xbarsco[:, :], 0)
				#Dsci[ic1, ic2, :] = np.sum(Db*xbarsci[:, :], 0)
		
	del ic1, ic2	
	
	for in1 in range(0, nc): # component loop (i)
			
		# flux of each component across shell boundary 
		# (mol/s) (+ value represents flux in direction from 
		# outer shell to inner)	
		d = (np.sum(D[in1, :, :]*nabC[in1, :], 0))
		#d.shape = (d.size//d.shape[0], d.shape[0]) # transpose
		
		dsc = (np.sum(Dsc[in1, :, :]*nabCsc[in1, :], 0))
		#dsc.shape = (dsc.size//dsc.shape[0], dsc.shape[0]) # transpose
		
		dsco = (np.sum(Dsco[in1, :, :]*nabCsco[in1, :], 0))
		# transpose
		#dsco.shape = (dsco.size//dsco.shape[0], dsco.shape[0]) 
		
		dsci = (np.sum(Dsci[in1, :, :]*nabCsci[in1, :], 0))
		# transpose
		#dsci.shape = (dsci.size//dsci.shape[0], dsci.shape[0])
		
		dwsh = (np.sum(Dwsh[in1, :, :]*nabCwsh[in1, :], 0))
	
		flux[in1, 1:flux.shape[1]-1] = A[0, 0:shnm]*d
		fluxsc[in1, 1:fluxsc.shape[1]-1] = 0.0# Asc[0, 0:shnm]*dsc
		fluxsco[in1, 1:fluxsco.shape[1]-1] = 0.0#Asc[0, 1::]*dsco
		fluxsci[in1, 1:fluxsci.shape[1]-1] = 0.0#Asc[0, 0:shnm]*dsci
		fluxwsh[in1, :] = Asc[0, :]*dwsh

		
	# new number of moles of each component per shell (mol)
	# b-b
	Nznew[:, :] = Nz0[:, :]+(flux[:, 1:flux.shape[1]]-
			flux[:, 0:flux.shape[1]-1])*ts	

	# sc-sc
	nscnew[:, :] = n0sc[:, :]+(fluxsc[:, 1:fluxsc.shape[1]]-
			fluxsc[:, 0:fluxsc.shape[1]-1])*ts
			

	# b-sco		
	Nznew[:, :] = Nznew[:, 0:shn]+(fluxsco[:, 1:fluxsco.shape[1]])*ts
	nscnew[:, 1::] = nscnew[:, 1::]-(fluxsco[:, 1:fluxsco.shape[1]-1])*ts

	
	# sci-b	
	Nznew[:, 1::] = Nznew[:, 1::]+(fluxsci[:, 1:fluxsci.shape[1]-1])*ts
	nscnew[:, :] = nscnew[:, 0:shn]-(fluxsci[:, 1:fluxsci.shape[1]])*ts
	
	
	# 1p-2p		
	Nznew[:, :] = Nznew[:, :]-(fluxwsh)*ts
	nscnew[:, :] = nscnew[:, :]+(fluxwsh)*ts
	
	
	# new volumes of 1st phase (m3)
	V = np.zeros((1, shn))
	# new volume of 2nd phase (m3)
	Vscnew = np.zeros((1, shn))
         
	Mv = np.zeros((M.size, 1)) # molar volumes
	Mv[:,0] = (M[:, 0]/p[:, 0])
	V[0, :] = np.sum(Nznew*Mv, axis=0)
	Vscnew[0, :] = np.sum(nscnew*Mv, axis=0) 
	# change from original bulk shell volume (m3)
	Vdiff = (V0)-(V)
	Vdiffsc = (V0sc)-(Vscnew)
	# number of moles of component N per shell (mol)
	Nznew[Nznew.shape[0]-1,:] = Nz0[Nz0.shape[0]-1,:]+Vdiff*(1.0/
					Mv[Mv.shape[0]-1,0])
	nscnew[Nznew.shape[0]-1,:] = n0sc[n0sc.shape[0]-1,:]+Vdiffsc*(1.0/
		Mv[Mv.shape[0]-1,0])	


	# -----------------------------------------------------
	# flux between phases in one shell can switch 
	# from uphill to downhill over a time step rather than 
	# steadily approach equilibrium as would happen 
	# realistically, therefore reduce flux if it causes the 
	# sign of Gamma (diffusion direction) to change
	
	# Gamma in each shell and phase before diffusion
	Gam10 = int_Gam(Nz0/np.sum(Nz0,0), gam0, shn)
	Gam20 = np.zeros((Gam10.shape))
	# index where 2nd phase exists
	ind = n0sc[0,:]>0
	if np.sum(ind)>0:
		Gam20[0,0,ind] = (int_Gam(n0sc[:,ind]/np.sum(
			n0sc[:,ind],0), gam0, np.sum(ind)))[0,0,:]
			
	# Gamma in each shell and phase after diffusion
# 	Gam11 = int_Gam(Nznew/np.sum(Nznew,0), gam0, shn)
# 	
# 	Gam21 = np.zeros((Gam11.shape))
# 	ind = nscnew[0,:]>0
# 	if np.sum(ind)>0:
# 		Gam21[0,0,ind] = (int_Gam(nscnew[:,ind]/np.sum(
# 						nscnew[:,ind],0), gam0, np.sum(ind)))[0,0,:]
# 		
# 	# thermodynamic factor per shell before flux
# 	Gambf = (Gam10+Gam20)/2.0
# 	# thermodynamic factor per shell after flux
# 	Gamaf = (Gam11+Gam21)/2.0
# 	
# 	#just to make sure we don't waste time changing 
# 	# the flux when it will change anyway due to time 
# 	# step decrease due to negative presence of a 
# 	# component or excess change
# 	if (np.sum(nscnew[:,ind]/np.sum(nscnew[:,ind],0)<0)+
# 		np.sum(Nznew[:,ind]/np.sum(Nznew[:,ind],0)<0)>0):
# 			Gambf=Gamaf 
# 	# check whether diffusion threshold passed
# 	ex_i = accept_change(Nz0, Nznew, shn, ut, n0sc, nscnew)
# 	if np.sum(ex_i)>0:
# 		Gambf=Gamaf 
# 	
# 	
# 	#if necessary alter flux between phases
# 	if np.sum((Gambf*Gamaf)<0.0)>0:
# 		# index of shells where switch occurs
# 		ind_swit = (Gambf*Gamaf)<0.0
# 		[Nznew, nscnew] = flux_iter(Nz0, n0sc, Nznew, nscnew, gam0, shn, fluxwsh,
# 	 						flux, fluxsc, fluxsco, fluxsci, ts, M, p, V0, V0sc, ind_swit,tsn)
		
	
	# new volume of bulk shells (m3)
	V = np.zeros((1, shn))
	V[0, :] = np.sum(Nznew*Mv, axis=0)
	
	Znew = np.zeros((Z.shape[0], shn))     
	# new concentrations of components (mol/m3)
	for ic in range(0, Nznew.shape[0]):
		Znew[ic, :] = Nznew[ic, :]/V[0, :]
	del ic
	
	# new volume of 2nd phase (m3)
	Vscnew = np.zeros((1, shn))
	Vscnew[0, :] = np.sum(nscnew*Mv, axis=0)
	del Mv
	
	return Nznew, V, Znew, nscnew, Vscnew, Gam10
