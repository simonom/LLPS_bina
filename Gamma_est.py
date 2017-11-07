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
# function to estimate activity coefficients and thermodynamic factors
import numpy as np
import sys
from collections import OrderedDict
from SMILES_pars import SMILES_pars
from umansysprop import activity_coefficient_models as gamma_func
import matplotlib.pyplot as plt


# SMILES array
S_arr = np.array((['O', 'CCCCCC(=O)O'])) # 'C(O)C1C(O)C(O)C(O1)OC1OC(CO)C(O)C(O)C(O)C1(O)'
# CCCCCCCCCCCC(=O)O, C(=O)(O)C=CC(=O)(O)

# parsed version of SMILES
S_arr = SMILES_pars(S_arr)
# mole fractions to estimate activity coefficient & thermodynamic factors at
x0 = np.arange(0, 1.0+1.0/100.0, 1.0/100.0)

# empty results matrix first row for sv gamma (activity coefficients), second 
# row for sv Gamma (thermodynamic factors), 
# third row for sv mole fraction, fourth row for first (if any) higher mole fraction with 
# 0 intercept between the lower mole fraction and it
gam0 = np.zeros((4, np.shape(x0)[0]))
gam0[2, :] = x0

for i in range(0, x0.shape[0]): # mole fraction loop

	# associate components with their mole fractions
	SMILES_arr = dict(zip(S_arr, [x0[i], 1.0-x0[i]]))	
	# call on activity coefficient estimation method (results are ln(gamma))
	gamma_UNIFAC = (gamma_func.aiomfac_sr(SMILES_arr, {}, 298.15))
	# reorder into known order:
	gamma = np.zeros(2)
	count = 0 # component count
	for s in SMILES_arr.keys():
		gamma[count] = gamma_UNIFAC[s]
		count = count+1 # component count

	gam0[0, i] = (gamma[0]) # sv activity coefficient
	
	# estimate sv thermodynamic factor
	if i == 0 or i == x0.shape[0]-1 :
		gam0[1, i] = 1.0
	else:
		gam0[1, i] = 1.0+x0[i]*((np.log(gam0[0,i])-np.log(gam0[0,i-1]))/(x0[i]-x0[i-1]))		

for i in range(0, x0.shape[0]): # mole fraction loop
	Gami=gam0[1,i]
	if Gami>=0:
		ichange=gam0[1,i+1::]<0
		# if no zero intercept move onto next mole fraction
		if np.sum(ichange)==0.0: 
			continue
		else: # index of first element with zero intercept
			i2 = np.where(ichange==1)[0][0]
			# first higher mole fraction with intercept
			gam0[3,i]=gam0[2,i2+i+1]
	if Gami<=0:
		ichange=gam0[1,i+1::]>0
		# if no zero intercept move onto next mole fraction
		if np.sum(ichange)==0.0: 
			continue
		else: # index of first element with zero intercept
			i2 = np.where(ichange==1)[0][0]
			# first higher mole fraction with intercept
			gam0[3,i]=gam0[2,i2+i+1]

# plot of sv mole fraction vs. activity (a.k.a. saturation ratio)
#plt.plot(gam0[0,:]*gam0[2,:], gam0[2,:])
# plot of thermodynamic factor vs. sv mole fraction
plt.plot(gam0[2,:], gam0[0,:])
plt.show()
#np.save('WHX_Gamma_test', gam0)
