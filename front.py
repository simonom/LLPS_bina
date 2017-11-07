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

now = time.time()
print "- local time:", time.localtime(now)

shn = 10 # initial number of particle phase shells
Dp = 2.0e-7 # particle diameter (m)

# smiles array
SMILES_arr = np.array((['O', 'CCCCCC(=O)(O)'])) #hexanoic acid: 
# 'CCCCCC(=O)(O)'  water: 'O', sucrose: 
# 'C(O)C1C(O)C(O)C(O1)OC1OC(CO)C(O)C(O)C(O)C1(O)'
# parsed version of SMILES array
SMILES_arr = SMILES_pars(SMILES_arr)
# component self-diffusion coefficients (row vector, 
# same order as SMILES above) (m2/s)
Db = np.array([[1.0e-9], [1.0e-15]])
# acceptable lower and upper bounds for change of mole fraction per shell 
# per time step, see accept_change.py for dependence of acceptable change on 
# difference in mole fractions of bounding shells
ut = np.array(([3.0e-5, 1.5e-4])) 

ts = 5.0e-6 # initial time step for solution (s)
estime = np.array([0.0, 1.0e-3, 5.0e-3, 1.0e-2, 2.0e-2]) # saturation ratio times (s)
# saturation ratios of components (rows) vs time in estime (columns)
es = np.array([[0.6, 1.15, 1.15, 1.05, 1.05], [1.00, 0.05, 0.05, 1.00, 1.00]])
# initial activities in particle shells
ai0 = np.transpose(np.ones((shn, Db.shape[0]))*es[:, 0])

# component molar masses (g/mol), water, maleic acid and sucrose 
# from CRC online handbook
M = np.array([[18.015], [200.318]]) # water: 18.015, malonic acid: 104.062, 
# octanol: [130.228], sucrose: 342.296, hexanoic: 116.158, 
# heptanoic acid: 130.185, dodecanoic acid: 200.318
# component densities (g/m3) from CRC online handbook
rho = np.array([[1.0e6], [0.8679e6]]) # water: 1.0e6, malonic acid: 1.619e6,  
# octanol: 0.8262e6, sucrose: 1.581e6, hexanoic: 0.9274e6, heptanoic: 0.9181, 
# dodecanoic 0.8679
#Mvol = (M/rho)/6.022e23 # molecular volume (m3/molecule)
# molecular diameter (m)
#sigma = (((3*Mvol)/(4*np.pi))**(1.0/3.0))*2
#del Mvol
T = 298.15 # temperature (K)
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
	gam0 = np.load('WHX_Gamma_test.npy')
	#print 'front'
	# activity vs xsv plot
	#plt.plot(gam0[2,:],gam0[0,:]*gam0[2,:],'--b')
	# thermodynamic factor vs xsv plot
	#plt.plot(gam0[2,:],gam0[1,:],'--r')
	#plt.show()
	
	# Gibb's free energy array
	PS0 = np.load('WDD_Gibb.npy')
	
elif idma==1:
	gam0 = 0.0

# call on the solution to diffusion
[nrec, Vrec, time, time2, Gamma_rec, esrec, nrecsc, Vrecsc] = ETH23(ts, 
	shn, Dp, Db, ut, es, estime, ai0, M, rho, Dg, Cstar, idma, SMILES_arr, 
	T, gam0,PS0)

# collect results into one dictionary
res = dict()
res['nrec'] = nrec
res['Vrec'] = Vrec
res['time2'] = time2
res['Gamma_rec'] = Gamma_rec
res['nrecsc'] = nrecsc
res['Vrecsc'] = Vrecsc
res['esrec'] = esrec
for key in res:
	print key

import time
now = time.time()
print "- local time:", time.localtime(now)

# save results in .mat form		       
io.savemat('FvMS_paper_ps_test_loes_10shell_HX', res)
