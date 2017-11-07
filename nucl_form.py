# function to estimate the number of nuclei formed per 
# second in phase separation
import numpy as np


def nucl_form(Nznew,PS0,shn,Db,ts):
	
	# sv mole fraction in 1p of all shells
	xsv0 = Nznew[0,:]/(np.sum(Nznew[:,:], axis=0))	
	
	# difference in Gibbs free energy between 1 phase and 
	# phase separated system (if positive, phase 
	# separation may be favourable)
	delG = np.interp(xsv0, PS0[:,0], PS0[:,1])		
	
	# return if no phase separation possible
	if np.sum(delG>0)==0:	
		nucn = np.zeros((1,shn))
		return nucn, 0.0, 0.0
	
	# sv mole fractions in phases
	xsv1 = np.interp(xsv0, PS0[:,0], PS0[:,2]) # new phase
	xsv2 = np.interp(xsv0, PS0[:,0], PS0[:,3]) # existing phase
	
	# difference in mole fractions between phases
	delx = np.abs(xsv1-xsv2)
	
	# diffusion coefficient in initial phase  (m2/s)
	D = np.zeros((2,shn))
	D[0, :] = (Db[0]**xsv0) # sv contribution
	D[1, :] = (Db[1]**(1.0-xsv0)) # nv contribution
	D[0,:] = D[0, :]*D[1, :] # total diffusivity
	
	# nucleation rate (#nuclei/s)
	nucr = 0.0+(D[0,:]*1.0e21)*(delG/1.0e2)*(1.0e-2/delx) 
	# number of nuclei to form
	nucn = nucr*ts

	
	return nucn, xsv1, xsv2