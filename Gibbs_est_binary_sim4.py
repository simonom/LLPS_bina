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


# ---------------------------------------------------------
# code outline:
# 1 activities and Gibbs free energy of single phase
# 2 open matrix of thermodynamic factors for single phase
# 3 thermodynamic factors of two phase system
# 4 Gibbs free energy of two phase system
# 5 minimum Gibbs free energy (Gfe) of two phases 
# 6 compare Gfe in two phases to one phase, if less record

# ---------------------------------------------------------
# code to estimate the difference in Gibb's free energy 
# between one phase and the two
# phase system with lowest Gibb's free energy

import numpy as np
from SMILES_pars import SMILES_pars
from umansysprop import activity_coefficient_models as gam_f
import matplotlib.pyplot as plt 


# ---------------------------------------------------------
# preparatory stuff

# mole fractions in one phase (sv 1st row, nv 2nd row)
# (should be same as used in thermodynamic factor matrix)
x1 = np.zeros((2,1001))

# molar masses (g/mol)
M=np.array([[18.015], [200.318]])#sucrose:342.296, dodecanoic acid: 200.318
rho=np.array([[1.0e6], [0.8679e6]])#dodecanoic:0.8679e6, 
# sucrose: 1.581e6 #densities (g/m3)
MV=M/rho # molar volumes (m3/mol)

# estimate activity coefficients for components
# component SMILES
S_arr=np.array((['O','CCCCCCCCCCCC(O)=O']))
# dodecanoic acid: CCCCCCCCCCCC(O)=O 
# sucrose: 'C(O)C1C(O)C(O)C(O1)OC1OC(CO)C(O)C(O)C(O)C1(O)'
# parse SMILES strings
S_arr=SMILES_pars(S_arr)

# mole fractions of water (sv) at which to estimate activity
xsvref = np.arange(0.0,1.0+1.0/(x1.shape[1]-1.0),1.0/(x1.shape[1]-1.0))
# empty matrix for activities 
gam = np.zeros((xsvref.shape[0],x1.shape[0]+1))
# empty matrix for number of moles in phase 1
n1 = np.zeros((x1.shape[0],xsvref.shape[0]))
# empty matrix for Gibb's free energy of one phase system and minimum Gibb's 
# free energy of two phase system
Gres1 = np.zeros((xsvref.shape[0],4))
Gres0 = np.zeros((xsvref.shape[0],4))
# water (semi-volatile) mole fractions in 1 phase of 1 phase system
Gres1[:,0] = xsvref


# ---------------------------------------------------------
# 1 activities and Gibbs free energy of single phase
for xi in range(0, xsvref.shape[0]):
	
	x1[0,xi] = xsvref[xi] # sv mole fraction	
	x1[1,xi] = 1.0-x1[0,xi] # nv mole fraction
	
	# number of moles of each phase
	if x1[0,xi]==0.0:
		n1[1,xi]=1.0/MV[1]
	elif x1[0,xi]==1.0:
		n1[0,xi]=1.0/MV[0]
	else:
		# sv moles in single phase
		n1[0,xi]=(1.0/((x1[1,xi]/x1[0,xi])*MV[1]+MV[0]))
		n1[1,xi]=n1[0,xi]*(x1[1,xi]/x1[0,xi])# nv moles	
		
	# associate SMILES and mole fractions
	SMILES_arr = dict(zip(S_arr, x1[:,xi]))
	
	# get ln(gamma) from activity coefficient estimation
	gam_UNI = (gam_f.aiomfac_sr(SMILES_arr,{},298.15))
	
	# reorder into known order
	count=0 # component count
	for s in SMILES_arr.keys():
		# activity coefficients at this mole fraction
		gam[xi,count] = (np.exp(gam_UNI[s]))		
		count=count+1 # component count
	
	# activity coefficient multiplied by mole fraction for 
	# activity
	gam_temp1 = gam[xi,0]*x1[0,xi]
	gam_temp2 = gam[xi,1]*x1[1,xi]
	gam[xi,0] = gam_temp1
	gam[xi,1] = gam_temp2

	# Gibb's free energy in single phase	
	gam[xi,2] = (gam[xi,0]*n1[0,xi])+(gam[xi,1]*n1[1,xi])	
	if xi>0 and xi<xsvref.shape[0]-1:
		# Gibb's free energy in single phase
		Gres1[xi,1] = gam[xi,2]
 
# empty array for single phase mole fractions
x1 = np.zeros((M.shape[0]))
# empty array for single phase number of moles
n1 = np.zeros((M.shape[0]))
error=0 # break loop condition
# record minimum Gibbs free energy in potential 2 phase systems
GFEmin = np.zeros((xsvref.shape[0]))

# ---------------------------------------------------------
# 2 import thermodynamic factors as a function of mole 
# fraction for single phase
gam0 = np.load("WDD_Gamma.npy")


# ---------------------------------------------------------
# 3 thermodynamic factors of two phase system

# loop through mole fractions of first phase
for xi in range(1, xsvref.shape[0]-1):
	print xi
	# empty results matrix for Gibbs free energy of second and third phases
	Gres2 = np.zeros((1,xsvref.shape[0]))
	xp3 = np.zeros((1,xsvref.shape[0])) # remember sv mole fraction phase 3	
	
	# mole fractions in single phase	
	x1[0] = xsvref[xi] # sv mole fraction
	x1[1] = 1.0-x1[0] # nv mole fraction
		
	# number sv moles in single phase
	n1[0] = (1.0/((x1[1]/x1[0])*MV[1]+MV[0]))
	n1[1] = n1[0]*(x1[1]/x1[0])# nv moles

	# the sv mole fraction in the second phase to loop 
	# through needs to be higher/lower than the single 
	# phase if the thermodynamic factor in the single
	# phase is positive/negative 
	if gam0[1,xi]>0: 
		loop_start = xi+1
		loop_end = xsvref.shape[0]
	else:
		loop_start = 1
		loop_end = xi
	
	# loop through potential mole fractions of second 
	# phase, which represents the nucleating phase to 
	# estimate Gibbs Free energy of two phase system
	for xi2 in range(loop_start, loop_end):
		
		# mole fractions in phase 2
		xsv2 = xsvref[xi2]
		xnv2 = 1.0-xsv2

		# no. moles phase 2
		# allocate relatively very small amount to 2nd 
		# (new) phase
		nT2 = np.sum(n1)/1.0e4
		nsv2 = xsv2/(2.0-xsv2)*nT2
		nnv2 = nsv2*(xnv2/xsv2)
		# no. moles phase 3
		nsv3 = n1[0]-nsv2
		nnv3 = n1[1]-nnv2
		# mole fractions phase 3
		xsv3 = nsv3/(nsv3+nnv3)
		xnv3 = 1.0-xsv3
		xp3[0,xi2] = xsv3
		
		# distance between xsv3 and reference mole fractions
		delx = xsv3-xsvref
	
		ihi = delx==np.min(delx[delx>=0])
		disthi = np.min(delx[delx>=0])
		ilo = delx==np.max(delx[delx<0])
		distlo = np.max(delx[delx<0])
		# interpolate to find activities in phase 3
		gamsv3 = (gam[ihi,0]*(distlo/(disthi+distlo))+
					gam[ilo,0]*(disthi/(disthi+distlo)))
		gamnv3 = (gam[ihi,1]*(distlo/(disthi+distlo))+
					gam[ilo,1]*(disthi/(disthi+distlo)))	
		
		# thermodynamic factor in 2nd and 3rd phase
		Gam2 = np.interp(xsv2,gam0[2,:],gam0[1,:])
		Gam3 = np.interp(xsv3,gam0[2,:],gam0[1,:])
		# average thermodynamic factor:
		Gamav = (Gam2+Gam3)/2.0
		
		
		# -------------------------------------------------
		# 4 Gibbs free energy of second and third phases
		Gres2[0,xi2]=(gam[xi2,0]*nsv2+gam[xi2,1]*nnv2+
			gamsv3*nsv3+gamnv3*nnv3)

		# next bit rules out phase separation if
		# the thermodynamic factor and therefore
		# diffusion acts against separation
		if xsv2>xsv3 and Gamav>0:
			Gres2[0,xi2] = Gres1[xi,1]*2.0 
		if xsv2<xsv3 and Gamav<0:
			Gres2[0,xi2] = Gres1[xi,1]*2.0	

	
	# difference between Gibbs energy in 1 phase and minimum Gibbs free 
	# energy of 2 phase system at this mole fraction of 1 phase (if this is
	# positive, phase separation may be thermodynamically favourable)
	Gres0[xi,0]=Gres1[xi,1]	
	
	
	# -----------------------------------------------------
	# 5 minimum Gibbs free energy of two phase system
	Gres0[xi,1] = np.min(Gres2[Gres2!=0])
			
			
	# -----------------------------------------------------
	# 6 Difference in Gibbs free energy of two phase system
	# and single phase, if less record
	Gres1[xi,1] = Gres1[xi,1]-np.min(Gres2[Gres2!=0])
	
	# index where this minimum occurs
	imin=np.where(Gres2==np.min(Gres2[Gres2!=0])) 
	GFEmin[xi] = np.min(Gres2[Gres2!=0])
	
		
	# only record phase mole fractions if LLPS favoured
	if Gres1[xi,1]>=0: 
		# mole fraction of sv in 1st phase of 2 phase system 
		Gres1[xi,2] = xsvref[imin[1]]
		# mole fraction of sv in 2nd phase of 2 phase system
		Gres1[xi,3] = xp3[0,imin[1]]
	else:
		continue	


# plot difference in Gibb's free energy between single phase and phase separated
#plt.plot(Gres1[:,0],Gres1[:,1])
# plot mole fraction in second phase against mole fraction in single phase
plt.plot(Gres1[:,0],Gres1[:,2],'--xb')
plt.plot(Gres1[:,0],Gres1[:,3],'--xr')
plt.show()

# plot the Gibbs free energy of one phase system and the
# minimum in two phase systems against sv mole fraction in
# one phase
#plt.plot((Gres1[:,0],Gres1[:,2],'--b'))
#plt.plot((Gres1[:,0],GFEmin,'--b'))
#plt.show()

# save Gibb's free energies and mole fractions in phase separated phases
# 1st column is mole fraction of one phase, 2nd col. is difference in Gibb's 
# free energy of 1 phase and in 2 phase, 3rd col. is mole
# fraction of 1st phase in 2 phase system and 4th col. is mole fraction in 2nd 
# phase of 2 phase system     
np.save('WDD_Gibb', Gres1)