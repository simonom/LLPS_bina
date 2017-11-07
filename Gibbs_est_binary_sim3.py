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
# code to estimate the difference in Gibb's free energy between one phase and 
# the two
# phase system with lowest Gibb's free energy

import numpy as np
from SMILES_pars import SMILES_pars
from umansysprop import activity_coefficient_models as gam_f
import matplotlib.pyplot as plt 

# mole fractions in one phase (sv 1st row, nv 2nd row)
x1 = np.zeros((2,1001))

# molar masses (g/mol)
M=np.array([[18.015],[200.318]])#sucrose:342.296, dodecanoic acid: 200.318
rho=np.array([[1.0e6],[0.8679e6]])#dodecanoic:0.8679e6, sucrose: 1.581e6 #densities (g/m^3)
MV=M/rho # molar volumes (m^3/mol)

# estimate activity coefficients for components
# component SMILES
S_arr=np.array((['O','CCCCCCCCCCCC(O)=O']))
#CCCCCCCCCCCC(O)=O #'C(O)C1C(O)C(O)C(O1)OC1OC(CO)C(O)C(O)C(O)C1(O)'
# parse SMILES strings
S_arr=SMILES_pars(S_arr)

# mole fractions of water (sv) at which to estimate activity
xsvref=np.arange(0.0,1.0+1.0/(x1.shape[1]-1.0),1.0/(x1.shape[1]-1.0))
# empty matrix for activities 
gam=np.zeros((xsvref.shape[0],x1.shape[0]+1))
# empty matrix for number of moles in phase 1
n1=np.zeros((x1.shape[0],xsvref.shape[0]))
# empty matrix for Gibb's free energy of one phase system and minimum Gibb's 
# free energy of two phase system
Gres1=np.zeros((xsvref.shape[0],4))
Gres0=np.zeros((xsvref.shape[0],4))
# water (semi-volatile) mole fractions in 1 phase of 1 phase system
Gres1[:,0]=xsvref

# loop through reference mole fractions
for xi in range(0, xsvref.shape[0]):
	
	x1[0,xi]=xsvref[xi] # sv mole fraction	
	x1[1,xi]=1.0-x1[0,xi] # nv mole fraction
	

	if x1[0,xi]==0.0:
		n1[1,xi]=1.0/MV[1]
	elif x1[0,xi]==1.0:
		n1[0,xi]=1.0/MV[0]
	else:
		# sv moles in single phase
		n1[0,xi]=(1.0/((x1[1,xi]/x1[0,xi])*MV[1]+MV[0]))
		n1[1,xi]=n1[0,xi]*(x1[1,xi]/x1[0,xi])# nv moles	
		
	# associate SMILES and mole fractions
	SMILES_arr=dict(zip(S_arr,x1[:,xi]))
	
	# get ln(gamma) from activity coefficient estimation
	gam_UNI=(gam_f.aiomfac_sr(SMILES_arr,{},298.15))
	
	# reorder into known order
	count=0 # component count
	for s in SMILES_arr.keys():
		# record gamma*xnow for activity at this mole fraction
		gam[xi,count] = (np.exp(gam_UNI[s]))*x1[count,xi]		
		count=count+1 # component count
	
	# Gibb's free energy in single phase	
	gam[xi,2] = (gam[xi,0]*n1[0,xi])+(gam[xi,1]*n1[1,xi])	
	if xi>0 and xi<xsvref.shape[0]-1:
		# Gibb's free energy in single phase
		Gres1[xi,1]=gam[xi,2]
	
# plot activities against mole fraction		
#plt.plot(xsvref,gam[:,0],'--b')
#plt.plot(xsvref,gam[:,1],'--r')
# plot Gibb's free energy in single phase
#plt.plot(xsvref,gam[:,2],'--r')
#plt.show()

# empty array for single phase mole fractions
x1=np.zeros((M.shape[0]))
# empty array for single phase number of moles
n1=np.zeros((M.shape[0]))
error=0 # break loop condition

# import thermodynamic factors as a function of mole fraction
gam0 = np.load("WDD_Gamma.npy")

# loop through mole fractions of first phase
for xi in range(1,xsvref.shape[0]-1):

	# empty results matrix for Gibb's free energy of second and third phases
	Gres2=np.zeros((1,xsvref.shape[0]))
	xp3=np.zeros((1,xsvref.shape[0])) # remember sv mole fraction phase 3	
	# mole fractions in single phase	
	x1[0]=xsvref[xi] # sv mole fraction
	x1[1]=1.0-x1[0] # nv mole fraction
		
	# sv moles in single phase
	n1[0]=(1.0/((x1[1]/x1[0])*MV[1]+MV[0]))
	n1[1]=n1[0]*(x1[1]/x1[0])# nv moles

	# loop through potential mole fractions of second phase
	for xi2 in range(xi+1,xsvref.shape[0]):
		
		# mole fractions in phase 2
		xsv2 = xsvref[xi2]
		xnv2 = 1.0-xsv2

		# no. moles phase 2
		# allocate relatively very small amount to sv in 2nd phase
		nsv2 = n1[0]/1000.0 
		nnv2 = nsv2*(xnv2/xsv2)
		# no. moles phase 3
		nsv3 = n1[0]-nsv2
		nnv3 = n1[1]-nnv2
		# mole fractions phase 3
		xsv3 = nsv3/(nsv3+nnv3)
		xnv3 = 1.0-xsv3
		xp3[0,xi2]=xsv3
		
		# distance between xsv3 and reference mole fractions
		delx = xsv3-xsvref
		ihi = delx==np.min(delx[delx>=0])
		disthi = np.min(delx[delx>=0])
		ilo = delx==np.max(delx[delx<0])
		distlo = np.max(delx[delx<0])
		# interpolate to find activities in phase 3
		gamsv3 = gam[ihi,0]*(distlo/(disthi+distlo))+gam[ilo,0]*(disthi/(disthi+distlo)) 
		gamnv3 = gam[ihi,1]*(distlo/(disthi+distlo))+gam[ilo,1]*(disthi/(disthi+distlo))	
		
		# thermodynamic factor in 2nd and 3rd phase
		Gam2 = np.interp(xsv2,gam0[2,:],gam0[1,:])
		Gam3 = np.interp(xsv3,gam0[2,:],gam0[1,:])
		# average thermodynamic factor:
		Gamav = (Gam2+Gam3)/2.0
		
		# Gibb's free energy of second and third phases
		Gres2[0,xi2]=(gam[xi2,0]*nsv2+gam[xi2,1]*nnv2+
			gamsv3*nsv3+gamnv3*nnv3)
		# invalidate phase separation for these possible phases if 
		# the Gamma is not below zero, i.e., downhill diffusion 
		# would occur
		if Gamav>0.0:
			Gres2[0,xi2]=Gres1[xi,1]*2.0	
		
		if x1[0]>2.0:
			print x1[0]
			print xsv2
			print xsv3
			print nsv2
			print nsv3
			# print Gibb's free energy in single phase
			print Gres1[xi,1]
			# print individual contributions to Gibb's free 
			# energy in 2 phases
			print gam[xi,0]*n1[0]
			print gam[xi,1]*n1[1]
			print gam[xi2,0]*nsv2
			print gamsv3*nsv3  
			print gam[xi2,1]*nnv2
			print gamnv3*nnv3
			break	
	# difference between Gibb's energy in 1 phase and minimum Gibb's free 
	# energy of 2 phase system at this mole fraction of 1 phase (if this is
	# positive, phase separation may be thermodynamically favourable)
	if x1[0]>2.0:
		break		
	Gres0[xi,0]=Gres1[xi,1]	
		
	Gres0[xi,1]=np.min(Gres2[Gres2!=0])
	
	Gres1[xi,1]=Gres1[xi,1]-np.min(Gres2[Gres2!=0])
	
	# index where this minimum occurs
	imin=np.where(Gres2==np.min(Gres2[Gres2!=0])) 
	
		
	# only record phase mole fractions if LLPS favoured
	if Gres1[xi,1]>=0: 
		# mole fraction of sv in 1st phase of 2 phase system 
		Gres1[xi,2] = xsvref[imin[1]]
		# mole fraction of sv in 2nd phase of 2 phase system
		Gres1[xi,3] = xp3[0,imin[1]]
	else:
		continue	
	
# preferred two phase composition when xsv=1 in single phase
Gres1[xsvref.shape[0]-1,2]=Gres1[xsvref.shape[0]-2,2]
Gres1[xsvref.shape[0]-1,3]=1.0

# plot difference in Gibb's free energy between single phase and phase separated
#plt.plot(Gres1[:,0],Gres1[:,1])
# plot mole fractions in second phase against mole fraction in single phase
plt.plot(Gres1[:,0],Gres1[:,2],'--b')
plt.plot(Gres1[:,0],Gres1[:,3],'--r')
plt.show()

#plt.plot(Gres1[:,0],Gres1[:,2])
#plt.plot(Gres1[:,0],Gres1[:,3],'--r')
#plt.show()	
# save Gibb's free energies and mole fractions in phase separated phases
# 1st column is mole fraction of one phase, 2nd col. is difference in Gibb's 
# free energy of 1 phase and in 2 phase, 3rd col. is mole
# fraction of 1st phase in 2 phase system and 4th col. is mole fraction in 2nd 
# phase of 2 phase system     
#snp.save('WDD_Gibb', Gres1)
