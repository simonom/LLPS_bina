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
x1 = np.zeros((2,101))

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
	
	gam[xi,2] = (gam[xi,0]*n1[0,xi])+(gam[xi,1]*n1[1,xi])
	if x1[0,xi]==0.87:
		print 'whoop0'
		print gam[xi,0]
		print n1[0,xi]
		print gam[xi,1]
		print n1[1,xi]	
	# Gibbs free energy in one phase
	Gres1[xi,1]=gam[xi,2]
		
plt.plot(xsvref,gam[:,0],'--b')
plt.plot(xsvref,gam[:,1],'--r')
plt.show()
 
# empty array for single phase mole fractions
x1=np.zeros((M.shape[0]))
# empty array for single phase number of moles
n1=np.zeros((M.shape[0]))
error=0 # break loop condition
# loop through mole fractions of first phase
for xi in range(1,xsvref.shape[0]-1):

	# empty results matrix for Gibb's free energy of second and third phases
	Gres2=np.zeros((xsvref.shape[0],xsvref.shape[0]))
	
	# mole fractions in single phase	
	x1[0]=xsvref[xi] # sv mole fraction
	x1[1]=1.0-x1[0] # nv mole fraction
		
	# sv moles in single phase
	n1[0]=(1.0/((x1[1]/x1[0])*MV[1]+MV[0]))
	n1[1]=n1[0]*(x1[1]/x1[0])# nv moles
	
	# loop through potential mole fractions of second phase
	for xi2 in range(0,xi):

		# mole fractions in phase 2
		xsv2 = xsvref[xi2]
		xnv2 = 1.0-xsv2
	
		# loop through potential mole fractions of third phase
		for xi3 in range(xi+1,xsvref.shape[0]):

			# mole fractions in phase 3
			xsv3 = xsvref[xi3]
			xnv3 = 1.0-xsv3		
			
			# number of moles second phase
			if xsv2==0.0:
				nsv2=0.0
				if xnv3>0:
					nnv2=(n1[0]-n1[1]*(xsv3/xnv3))/(xsv2/xnv2-xsv3/xnv3)
				else:
					nnv2=n1[1]
			elif xsv3==0.0:
				nsv2=n1[0]
				nnv2=nsv2*(xnv2/xsv2)
			else:
				nsv2=((n1[1]-n1[0]*(xnv3/xsv3))/((xnv2/xsv2)-(xnv3/xsv3)))
				nnv2=nsv2*(xnv2/xsv2)
				
			# number of moles third phase
			nsv3=n1[0]-nsv2
			nnv3=n1[1]-nnv2	

			# Gibb's free energy of second and third phases
			Gres2[xi2,xi3]=(gam[xi2,0]*nsv2+gam[xi2,1]*nnv2+
				gam[xi3,0]*nsv3+gam[xi3,1]*nnv3)
			if x1[0]==0.87 and xsv2==0.50 and xsv3==0.875:
				print 'whoop1'
				print gam[xi2,0]
				print nsv2
				print gam[xi2,1]
				print nnv2
				print gam[xi3,0]
				print nsv3
				print gam[xi3,1]
				print nnv3	
		if error==1:
			break
	if error==1:
		break
	# difference between Gibb's energy in 1 phase and minimum Gibb's free 
	# energy of 2 phase system at this mole fraction of 1 phase (if this is
	# positive, phase separation may be thermodynamically favourable)		
	Gres0[xi,0]=Gres1[xi,1]	
	Gres0[xi,1]=np.min(Gres2[Gres2>0])
	Gres1[xi,1]=Gres1[xi,1]-np.min(Gres2[Gres2>0])
			
	# index where this minimum occurs
	imin=np.where(Gres2==np.min(Gres2[Gres2>0])) 
	
	# mole fraction of organic (non-volatile) in 1st phase of 2 phase system
	Gres1[xi,2] = xsvref[imin[0]]
	# mole fraction of organic (non-volatile) in 2nd phase of 2 phase system
	Gres1[xi,3] = xsvref[imin[1]]
	
	# preferred two phase composition when xsv=1 in single phase
	Gres1[xsvref.shape[0]-1,2]=Gres1[xsvref.shape[0]-2,2]
	Gres1[xsvref.shape[0]-1,3]=1.0

#plt.plot(Gres1[:,0],Gres1[:,2])
#plt.plot(Gres1[:,0],Gres1[:,3],'--r')
#plt.show()	
# save Gibb's free energies and mole fractions in phase separated phases
# 1st column is mole fraction of one phase, 2nd col. is difference in Gibb's 
# free energy of 1 phase and in 2 phase, 3rd col. is mole
# fraction of 1st phase in 2 phase system and 4th col. is mole fraction in 2nd 
# phase of 2 phase system     
#np.save('WDD_Gibb', Gres1)
