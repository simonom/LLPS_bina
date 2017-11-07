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
# main function for the ETH model with Maxwell-Stefan solution 
# and liquid-liquid phase separation

# import required functions
import numpy as np

from spat_arrays import spat_arrays
from concs import concs
from gas_phase_and_spat2 import gas_phase_and_spat
from eqn6 import eqn3
from accept_change import accept_change
from interp_Gamma import interp_Gam as int_Gam
from spat_arrays_inequal import spat_arrays_inequal
from concs2 import concs2
from phas_sep1 import phas_sep1

def ETH23(ts, shn, Dp, Db, ut, es, estime, ai0, M, rho, Dg, Cstar, idma, 
	SMILES_arr, T, gam0, PS0):

	# -----------------------------------------------------------------
	# inputs:

	# ts - time step (s)
    # shn - number of shells
    # Dp - initial particle diameter (m)
    # Dmethod - diffusion coefficient dependence on mole fraction
    # Db - self-diffusion coefficients of components (row vector) 
	# (m2/s) 
    # ut - acceptable percentage change in number of moles in a single 
	# shell per time step
    # es - gas phase saturation ratio (fraction) per component (1st dim.), 
	# with time (2nd 
    # dim.)
    # estime - times at which saturation ratios in es occur
    # ai0 - activity per component (1st dim.) per shell (2nd dim.) at start
    # M - array of component molar masses (g mol^{-1})
    # rho - array of component densities (g m^{-3})
    # Dg - gas phase diffusion coefficients (m^2 s^{-1})
    # Cstar - effective gas phase saturation concentration (ug m^{-3})
    # idma - ideality marker (1 for ideal, 0 for non-ideal)
    # SMILES_arr - component SMILES strings 
    # T - temperature (K)
	# gam0 - reference activity coefficients
	# -----------------------------------------------------------------
	# outputs:
    
	# nrec - record of component concentrations (1st dim.), with shell 
	# (2nd dim.) and with time (3rd dim.) (mol m^{-3})
	# Vrec - record of shell volumes (1st dim.), with time (2nd dim.) (m^3)
	# time - all time steps used in the ETH calculation (s)
	# time2 - times that contents of nrec and Vrec correspond to
	# ---------------------------------------------------------------------
	
	# array preparation
	# spatial arrays
	[rc, V0, Diamw, V0sc] = spat_arrays(shn, Dp)
	# remember starting width of inner shells (m)
	del0 = rc[0]	
	# number of time points we want to save results at 
	# (i.e. size of nrec and Vrec arrays)
	sp = int(1e3)
	# initial concentration and number of moles array (mol/m3 and mol),
	# also initial diffusion coefficient array
	[Z0, Z0sc, Nz0, nrec, Vrec, Gamma_rec, SN_partrec, esrec, n0sc, 
	nrecsc, Vrecsc, nuc2p] = concs(ai0, M, shn, V0, sp, rho, idma, 
			SMILES_arr, gam0)

	# ---------------------------------------------------------------------
	# time loop preparation
    
	time = np.zeros(1) # time array (s) (grows with time steps)
	tsn = 0 # count on time steps
	ts0 = ts # original time interval (s)
    
	# time array to relate to nrec and Vrec matrices (s)
	time2 = np.zeros((1, sp))
	zreci = 0 # count on result storing

	# update time count and array (s)
	tsn = tsn+1
	time = np.append(time, time[tsn-1]+ts)	
	
	# ---------------------------------------------------------------------
	# time loop
	while estime[estime.shape[0]-1]-time[tsn]>0:
   			
		# rudimental approach to increasing time step
		if ts<ts0:
			ts=ts*1.5
		
        # find new surface concentrations of components and surface 
		# shell size, and call on LLPS partitioning	
		[A1, Asc, shn1, Nz1, delta1, Z1, V1, rc, n1sc, 
		V1sc, Z1sc] = gas_phase_and_spat(es, estime, shn, time, 
		V0, V0sc, tsn, Nz0, n0sc, Z0, Z0sc, M, rho, rc, Dg, Cstar, ts, 
		Diamw, idma, del0, T, gam0, PS0, Db)	
		
# 		if tsn==1:
# 			print 'ETH23_0'
# 			print Nz0
# 			print Nz1
# 			#print n0sc[0]/np.sum(n0sc,0)
# 			print Nz0[0]/np.sum(Nz0,0)	
# 			
# 			Gam1 = int_Gam(Nz1/np.sum(Nz1,0),gam0,shn)
# 			#Gam2 = int_Gam(n1sc/np.sum(n1sc,0),gam0,shn)
#  			print Nz1[0,:]/np.sum(Nz1,0)
# 			print Gam1
#  			#print n1sc[0]/np.sum(n1sc,0)
# 			#print Gam2
# 			return
					
		# diffusion between shells and phases	
		[Nznew,Vnew,Znew,nscnew,Vscnew,Gamman] = eqn3(A1,Asc,shn1,M,Nz1,delta1,
			Z1,ts,rho,idma,V1,Db,gam0,n1sc,tsn,V1sc,ut)		
		
		if tsn==1.0e6:
			print 'ETH23_0'
			print Nz1[0,:]/np.sum(Nz1,0)	
# 			
			Gam1 = int_Gam(Nz1/np.sum(Nz1,0),gam0,shn)
 			print Nznew[0,:]/np.sum(Nznew,0)
# 			print Gam1
			print tsn

		
		# check whether diffusion threshold passed
		ex_i = accept_change(Nz1, Nznew, shn1, ut, n1sc, nscnew)	
			
		# if acceptable change exceeded or any component has negative 
		# prescence, decrease time step
		while(np.sum(np.sum(ex_i))>0 or np.sum(np.sum(Nz1<0))>0 or 
			np.sum(np.sum(Nznew<0))>0 or 
			np.sum(np.sum(n1sc<0))>0 or 
			np.sum(np.sum(nscnew<0))>0):
			
			# decrease time step
			ts=ts/1.5
			time[tsn]=time[tsn-1]+ts
				
			# find new surface concentrations of components and 
			# surface shell size, and call on LLPS partitioning	
			[A1, Asc, shn1, Nz1, delta1, Z1, V1, rc, n1sc, 
			V1sc, Z1sc] = gas_phase_and_spat(es, estime, shn, time, 
			V0, V0sc, tsn, Nz0, n0sc, Z0, Z0sc, M, rho, rc, Dg, 
			Cstar, ts, Diamw, idma, del0, T, gam0, PS0, Db)
	
			# diffusion between shells and phases	
			[Nznew,Vnew,Znew,nscnew,Vscnew,Gamman] = eqn3(A1,Asc,shn1,M,Nz1,
			delta1,Z1,ts,rho,idma,V1,Db,gam0,n1sc,tsn,V1sc,ut)
			
			# check whether diffusion threshold passed
			ex_i=accept_change(Nz1, Nznew, shn1, ut, n1sc, nscnew)
			
		
		# comment out to prevent phase separation
		#[Nznew,nscnew,V1,V1sc] = phas_sep1(Nznew,nscnew,
		#			shn1,gam0,M,rho,PS0,Vnew,Vscnew,Db,ts,nuc2p)

		# new spatial arrays
		[delta1, rc, A, Asc] = spat_arrays_inequal(V1,shn1,V1sc)
		
		# record important information
		if (time[tsn])>(estime[(estime.size)-1]/sp)*zreci:
			
			nrec[:, 0:shn, zreci] = Nz0	
			Vrec[0:shn, zreci] = V0	
			SN_partrec[zreci, 0] =  shn
			time2[0, zreci] = time[tsn]
			nrecsc[:, 0:shn, zreci] = n0sc
			Vrecsc[0:shn, zreci] = V0sc
			Gamma_rec[0, 0:shn, zreci] = Gamman
			zreci = zreci+1
			if (zreci== 1):	
				estime.shape = (1, estime.size)
				esrec = np.append(estime, es, axis=0)	
				estime = estime.reshape(estime.size)
				
# 			print 'zreci = ', zreci
						

		
		
		# revalue number of moles (mol), shell volumes (m3) 
		# and concentrations (mol/m3)
		Nz0 = Nznew
		V0 = Vnew
		Z0 = Znew
		Z0sc = Z1sc
		ish = Vscnew[0,:]>0
		Z0sc[0,ish] = nscnew[0,ish]/Vscnew[0,ish]
		Z0sc[1,ish] = nscnew[1,ish]/Vscnew[0,ish]
		shn = shn1
		n0sc = nscnew
		V0sc = Vscnew
			
		# update time count and array (s)
		tsn = tsn+1
		time = np.append(time, time[tsn-1]+ts)
		
		
	
	return nrec, Vrec, time, time2, Gamma_rec, esrec, nrecsc, Vrecsc