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
from eqn7 import eqn3
from accept_change import accept_change
from interp_Gamma import interp_Gam as int_Gam
from spat_arrays_inequal import spat_arrays_inequal
from concs2 import concs2
from phas_sep1 import phas_sep1
import pdb #Command line debugger - see points of interest in the code below.
           #When stopped at such points, type 'c' to continue

def ETH23(ts, shn, Dp, Db, ut, es, estime, ai0, M, rho, Dg, Cstar, idma, 
	SMILES_arr, T, gam0, C):

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
	[rc, V0, Diamw] = spat_arrays(shn, Dp)
	# remember starting width of inner shells (m)
	del0 = rc[0]	
	# number of time points we want to save results at 
	# (i.e. size of nrec and Vrec arrays)
	sp = int(1e3)
	# initial concentration and number of moles array (mol/m3 and mol),
	# also initial diffusion coefficient array
	[Z0, Nz0, nrec, Vrec, Gamma_rec, SN_partrec, esrec] = concs(ai0, M, shn, V0, sp, rho, idma, 
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
	ts_count = 0
	
	# ---------------------------------------------------------------------
	# time loop
	while estime[estime.shape[0]-1]-time[tsn]>0:
   			
		# rudimental approach to increasing time step
		if ts<ts0 and ts_count<=0:
			ts=ts*1.5
		
        # find new surface concentrations of components and surface 
		# shell size, and call on LLPS partitioning	
		[A1, shn1, Nz1, delta1, Z1, V1, rc] = gas_phase_and_spat(es, estime, shn, time, 
		V0, tsn, Nz0, Z0, M, rho, rc, Dg, Cstar, ts, 
		Diamw, idma, del0, T, gam0, Db)	
					
# 		if tsn>=7460:
# 			print tsn
# 			print Nz0[:,-3::]/np.sum(Nz0[:,-3::], 0)
# 			print Nz1[:,-3::]/np.sum(Nz1[:,-3::], 0)
# 			if tsn==7305:
# 				return			
					
		# diffusion between shells and phases	
		[Nznew, Vnew, Znew, Gamman] = eqn3(A1, shn1, M, Nz1, delta1,
			Z1, ts, rho, idma, V1, Db, gam0, 0.0, tsn, ut, C)		
		# if zreci==125:
# 			print(Nz1/np.sum(Nz1,0))
# 			print(Nznew/np.sum(Nznew,0))
# 			pdb.set_trace()
			
		# check whether diffusion threshold passed
		ex_i = accept_change(Nz1, Nznew, shn1, ut)	
			
		# if acceptable change exceeded or any component has negative 
		# prescence, decrease time step
		
		while(np.sum(np.sum(ex_i))>0 or np.sum(np.sum(Nz1<0))>0 or 
			np.sum(np.sum(Nznew<0)))>0:
			ts_count = 200
			# decrease time step
			ts = ts/1.5
			time[tsn]=time[tsn-1]+ts
			
			# if ts<1.0e-3 and tsn>2200:
# 				print 'uhoh'
# 				print tsn
# 				print ex_i
# 				print Z1/np.sum(Z1,0)
# 				print Znew/np.sum(Znew,0)
# 				print Gamman
# 				return
			# find new surface concentrations of components and 
			# surface shell size, and call on LLPS partitioning	
			[A1, shn1, Nz1, delta1, Z1, V1, rc] = gas_phase_and_spat(es, estime, shn, time, 
			V0, tsn, Nz0, Z0, M, rho, rc, Dg, 
			Cstar, ts, Diamw, idma, del0, T, gam0, Db)
	
			# diffusion between shells and phases	
			[Nznew, Vnew, Znew, Gamman] = eqn3(A1, shn1, M, Nz1,
			delta1, Z1, ts, rho, idma, V1, Db, gam0, 0.0, tsn, ut, C)
			
			# check whether diffusion threshold passed
			ex_i = accept_change(Nz1, Nznew, shn1, ut)
			
		ts_count = ts_count-1
		# comment out to prevent phase separation
		#[Nznew,nscnew,V1,V1sc] = phas_sep1(Nznew,nscnew,
		#			shn1,gam0,M,rho,PS0,Vnew,Vscnew,Db,ts,nuc2p)
		
		
		
		# new spatial arrays
		[delta1, rc, A] = spat_arrays_inequal(V1, shn1)
		
		# record important information
		if (time[tsn])>(estime[(estime.size)-1]/sp)*zreci:
			
			nrec[:, 0:shn, zreci] = Nz0	
			Vrec[0:shn, zreci] = V0	
			SN_partrec[zreci, 0] =  shn
			time2[0, zreci] = time[tsn]
			Gamma_rec[0, 0:shn1-1, zreci] = Gamman[0, 1::]
			zreci = zreci+1
			if zreci== 1:	
				estime.shape = (1, estime.size)
				esrec = np.append(estime, es, axis=0)	
				estime = estime.reshape(estime.size)
				
# 			print('zreci = ', zreci)
# 			print(Nznew/np.sum(Nznew,0))

		
		
		# revalue number of moles (mol), shell volumes (m3) 
		# and concentrations (mol/m3)
		Nz0 = Nznew
		V0 = Vnew
		Z0 = Znew
		shn = shn1

			
		# update time count and array (s)
		tsn = tsn+1
		time = np.append(time, time[tsn-1]+ts)
		
		
	
	return nrec, Vrec, time, time2, Gamma_rec, esrec