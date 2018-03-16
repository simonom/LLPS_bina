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
# Front end script of function to use a Euler forward step method to solve 
# diffusion through a particle, based on the ETH model, first proposed by 
# Zobrist et al. 2011.  Includes non-ideality and diffusion solved using the 
# Fick approach.

# import required functions
import numpy as np
import scipy.io as io
import time
from Dg_calc import Dg_calc
from ETH23 import ETH23
from SMILES_pars import SMILES_pars
import matplotlib.pyplot as plt
import pdb


shn = 36 # initial number of particle phase shells
Dp = 2.0e-5 # particle diameter (m)

# smiles array
SMILES_arr = np.array((['O', 'C(O)C1C(O)C(O)C(O1)OC1OC(CO)C(O)C(O)C(O)C1(O)'])) #butanoic acid: 
# 'CCCCCC(=O)(O)'  water: 'O', sucrose: 'C(O)C1C(O)C(O)C(O1)OC1OC(CO)C(O)C(O)C(O)C1(O)',
# suberic: 'OC(=O)CCCCCC(=O)(O)'
# parsed version of SMILES array
SMILES_arr = SMILES_pars(SMILES_arr)
# component self-diffusion coefficients (row vector, 
# same order as SMILES above) (m2/s)
# Db = np.array([[1.0e-13], [1.0e-25]]) # for sucrose
# Db = np.array([[1.0e-11], [1.0e-23]]) # for suberic

# possible range of self-diffusion coefficients for Monte Carlo
Dbr = (np.array([[1.0e-15, 1.0e-11], [1.0e-27, 1.0e-23]]))

# possible range of C for mixing rule function
Cr = (np.array([2.0, -4.0]))
delt_VRF_best=1.0e6 # fit value to beat

attempts = np.zeros((1000,3))

for MCi in range(0, 1000):	
	
	# acceptable lower and upper bounds for change of mole fraction per shell 
	# per time step, see accept_change.py for dependence of acceptable change on 
	# difference in mole fractions of bounding shells
	ut = np.array(([1.0e-3, 1.0e-2])) 
	
	ts = 1.0e0 # initial time step for solution (s)
	estime = np.array([0.0, 1.8e3, 7.2e3]) # saturation ratio times (s)
	# saturation ratios of components (rows) vs time in estime (columns)
	es = np.array([[0.50, 0.90, 0.90]])
	# initial activities in particle shells
	ai0 = np.transpose(np.ones((shn, Dbr.shape[0]))*es[:, 0])
	
	# component molar masses (g/mol), water, maleic acid and sucrose 
	# from CRC online handbook
	M = np.array([[18.015], [342.296]]) # water: 18.015, malonic acid: 104.062, 
	# octanol: [130.228], sucrose: 342.296, hexanoic: 116.158, suberic: 174.195 
	# heptanoic acid: 130.185, dodecanoic acid: 200.318
	# component densities (g/m3) from CRC online handbook
	rho = np.array([[1.0e6], [1.581e6]]) # water: 1.0e6, malonic acid: 1.619e6,  
	# octanol: 0.8262e6, sucrose: 1.581e6, hexanoic: 0.9274e6, heptanoic: 0.9181, 
	# dodecanoic 0.8679, no suberic acid density in CRC, wikipedia says 1.272e6 g/m3
	#Mvol = (M/rho)/6.022e23 # molecular volume (m3/molecule)
	# molecular diameter (m)
	#sigma = (((3*Mvol)/(4*np.pi))**(1.0/3.0))*2
	#del Mvol
	T = 190.0 # temperature (K)
	# column array of gas phase diffusion coefficient (m2/s), 
	# (typical value is suggested on pp. 5159 of Zaveri et al. (2014)) (m2/s)
	#Dg = Dg_calc(M, T, sigma)
	Dg = 1.0e-6
	# component vapour pressures (atm (at 298.15 K)) (1st value for water from: 
	# http://www.kayelaby.npl.co.uk/chemistry/3_4/3_4_2.html (originally 0.0313 atm), 2nd 
	# value for maleic acid from: umansysprop using Nannoolal methods for Vp and Tb (original
	# value is 2.2248e-07 atm), 3rd value for the assumed non-volatile sucrose), 
	# malonic acid: 
	vp = np.array([[3.13e-2], [0.0]])
	# component gas phase effective saturation vapour concentrations (ug m^{3}) 
	# (eq.1 O'Meara et al. 2014)
	Cstar = ((1.0e6*M[:, 0]*vp[:, 0])/(8.2057e-5*T))
	
	# ideality marker (1 for ideal, 0 for non-ideal)
	idma = 0
	if idma==0:
		# activity coefficient array
		gam0 = np.load('WSUC_loT_Gamma_test.npy')
		
	elif idma==1:
		gam0 = 1.0
		
	# load the reference Maxwell-Stefan results
	# open saved file for non-equilibrium situation
	resref = io.loadmat("FvMS_paper_suc_MS_RH0p50_0p90_loT_test.mat")
	# withdraw variables contained in dictionary keys
	time2ref = resref['time2']
	Vrecref = np.sum(resref['Vrec'],0) # sum volumes of size bins
	# denominator for radial response function (eq. 5 Ingram et al. 2017)
	den = np.abs(Vrecref[0]-Vrecref[-1])
	# volume response function for reference
	VRFref = (np.abs(Vrecref-Vrecref[-1]))/den
	
	
	# empty matrix for randomly chosen self-diffusion coefficients (m2/s)
	# loop through Monte Carlo simulations
	Db = np.zeros((2,1))

	
	# self-diffusion coefficients
	Db[0,0] = np.random.uniform(Dbr[0,0], Dbr[0,1])
	Db[1,0] = np.random.uniform(Dbr[1,0], Dbr[1,1])
	
	# C parameter for mixing rule
	C = np.random.uniform(Cr[0], Cr[1])

	attempts[MCi,0] = Db[0]
	attempts[MCi,1] = Db[1]
	attempts[MCi,2] = C

	print('Monte Carlo attempt no.')
	print(MCi)
	# call on the solution to diffusion
	[nrec, Vrec, time, time2, Gamma_rec, esrec] = ETH23(ts, 
		shn, Dp, Db, ut, es, estime, ai0, M, rho, Dg, Cstar, idma, SMILES_arr, 
		T, gam0, C)

	# calculate normalised sum of residuals:
	# first do linear interpolation of output Vrec from this fitting run to match
	# times reference Vrec collected at
	Vrec = np.sum(Vrec,0) # sum volumes of size bins

	Vrecint = np.interp(time2ref[0,:],time2[0,:],Vrec)
	# volume response function for run
	VRFrun = (np.abs(Vrec-Vrecref[-1]))/den
	# difference in response functions (eq. 4 Ingram)
	delt_VRF = (np.sum(np.abs(VRFrun-VRFref)))/time2ref.shape[1]
	print(delt_VRF)
	print(delt_VRF_best)
	if delt_VRF<delt_VRF_best:
		print('best results=')
		print(Db)
		print(C)
		# revalue value to beat
		delt_VRF_best = delt_VRF

		# collect results into one dictionary
		res = dict()
		res['nrec'] = nrec
		res['Vrec'] = Vrec
		res['time2'] = time2
		res['Gamma_rec'] = Gamma_rec
		res['esrec'] = esrec
		# str1 = 'FvMS_paper_suc_FK_RH0p50_0p99_loT_bestfit'
# 		str2 = num2str(MCi)
# save results in .mat form		       
io.savemat('FvMS_paper_suc_FK_RH0p50_0p99_loT_bestfit', res)
