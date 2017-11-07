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
# function to estimate gas phase diffusion 

import numpy as np
import matplotlib.pyplot as plt
from k1calc import k1calc
from interp_gam import interp_gam as int_gam
from interp_x import int_x

def gpp(Cp, Cpsc, Cstar, Cg, Rp, M, p, Dg, T, ts, idma, gam0, tsn):
	
	# ---------------------------------------------------------------------
    	# inputs: 
	# Cp - particle phase concentration (mol/m3 (particle))
	# Cpsc - concentration in 2nd phase (mol/m3 (particle))
	# Cstar - effective saturation concentration (mol m^{-3} (air))
	# Cg - bulk gas phase concentration (mol/m3 (air))i
	# Rp - particle radius (m)
	# M - molar mass (g/mol)
	# p - density (g/m3)
	# Dg - gas phase diffusion coefficient (m^{2} s^{-1})
	# T - temperature (K)
	# ts - time step (s)
	# idma - ideality marker
	# gam0 - reference activity coefficients
	# ---------------------------------------------------------------------
	# outputs:
    	# x2 - new mole fractions
	# x2sc - new mole fractions 2nd phase
    	# ---------------------------------------------------------------------	
	
	# current mole fraction sv in particle surface	
	xsv = np.zeros((1,1))
	xsv[0,0] = Cp[0]/np.sum(Cp)
	xsvsc = np.zeros((1,1))
	 
	if (Cpsc[0]!=0.0):
		xsvsc[0,0] = Cpsc[0]/np.sum(Cpsc)
	xsvsc0=xsvsc
	
	if idma == 0:
		# activity coefficient at this mole fraction
		gam = int_gam(xsv,gam0,1)
		gamsc = int_gam(xsvsc,gam0,1)	
	elif idma == 1:
		gam = 1.0	
	
	# convert particle sv concentration from mol/m3 (particle) to 
	# mol/m3 (air) using modified Raoult's law (eq. 1 thesis3.pdf)
	Cpa = xsv*gam*Cstar[0]
	Cpasc = xsvsc*gamsc*Cstar[0]
		
    	# gas side mass transfer coefficient (m/s (air))
	k2 = k1calc(Rp, Dg, M, p, T)	
	
	# check whether condensation or evaporation happening
	if Cg[0]-Cpa>=0:
		evap = 0 # condensation
	else:	
		evap = 1 # evaporation
	if Cg[0]-Cpasc>=0:
		evapsc=0 # condensation
	else:
		evapsc=1 # evaporation	
		

	# new particle phase concentration of sv (mol/m3 (air)) (16.69)		
	Cpa = Cpa+ts*k2[0]*(Cg[0]-Cpa)		
	Cpasc = Cpasc+ts*k2[0]*(Cg[0]-Cpasc)

	# possible activities in the particle phase
	act = gam0[2,:]*gam0[0,:]
	# maximum activity in the particle phase
	act_max = np.max(act)
	
		
	if Cpa>Cg[0] and evap==0: # exceed equilibrium check
		if Cg[0]/Cstar[0]<=act_max: 
			Cpa = Cg[0]
		# unable to equilibriate due to supersaturation, so just set very high
		else: 
			Cpa = 0.99*Cstar[0]
			
	if Cpa<Cg[0] and evap==1:
		Cpa = Cg[0]	

	if Cpasc>Cg[0] and evapsc==0: # exceed equilibrium check
		Cpasc = Cg[0]
	if Cpasc<Cg[0] and evapsc==1:
		Cpasc = Cg[0]

	
	# saturation ratio of sv at surface now (activity in particle)
	es = Cpa/Cstar[0]
	# es = 1.11 # fix saturation ratio
	essc = Cpasc/Cstar[0]
	
	# activity coefficient at which this activity occurs (in particle)	
	# index of where activity is maximal
	
	ix = np.where(act==np.max(act))[0] # index of maximum activity
	
	
	
	# interpolation to find new activity coefficient 
	# based on new activity (es)
	# multiplication in brackets gives activity across mole fraction space

	# if es<np.max(act) and xsv<gam0[2, ix]: 
# 		gam = np.interp(es,act[0:ix+1],gam0[0,0:ix+1])
# 		# mole fraction at which this activity occurs
# 		xsv = Cpa/(gam*Cstar[0])
# 	else:
		# mole fraction at which this activity occurs

	if es<=1.0:
		xsv = int_x(es, gam0, 0.0)
	else:	
		xsv = int_x(es, gam0, 1.0)
	if xsvsc!=0:
		if essc<np.max(act):
			gamsc = np.interp(essc,act[0:ix+1],gam0[0,0:ix+1])
		else:
			gamsc = np.interp(essc,act[ix+1::],gam0[0,ix+1::])	
		# mole fraction at which this activity occurs
		xsvsc = Cpasc/(gamsc*Cstar[0])

	#if tsn==100:
		#print 'gas_diff'
		#print es
		#print essc	
		#return
		#print np.max(act)
		#print Cg[0]/Cstar[0]
		#print Cpa/Cstar[0]
		#print xsv
		#return

    	# convert sv from mol/m3 (air) to mol/m3 (particle) 
	# substituting 1-Cpa/Cstar=x_nv into x=Cpnv/(Cpsv+Cpnv)
	Cp_new = (Cp[1]/(1.0-xsv))-Cp[1]	    	
	Cp_newsc=(Cpsc[1]/(1.0-xsvsc))-Cpsc[1]
		
	# new mole fractions (dimensionless)
	if Cp_new!=0.0:
		x2 = np.array((Cp_new/((Cp_new+Cp[1])))) # mole fraction sv
	else:
		x2 = np.array(0.0)
	x2.shape = [1, 1] # ensure x2 is 2D
	x2 = np.append(x2, 1.0-x2, axis=0)
	if (Cp_newsc!=0.0):
		x2sc = np.array((Cp_newsc/((Cp_newsc+Cpsc[1])))) # mole fraction sv
	else:
		x2sc=np.array(0.0)
	x2sc.shape = [1, 1] # ensure x2sc is 2D
	x2sc = np.append(x2sc, 1.0-x2sc, axis=0)
	# in case there was no nv in the 2nd phase and all sv has evaporated
	if x2sc[0]==0.0 and Cpsc[1]==0.0:
		x2sc[1]=0.0
	
	x2sc=xsvsc0 # simulate embedded morphology of second phase 
	return x2, x2sc
