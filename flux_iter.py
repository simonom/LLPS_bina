# function to find the flux required between phases to 
# reach approximately equal Gamma and therefore prevent further flux

import numpy as np
from interp_Gamma import interp_Gam as int_Gam

def flux_iter(Nz0, n0sc, Nznew, nscnew, gam0, shn, fluxwsh,
	 flux, fluxsc, fluxsco, fluxsci, ts, M, p, V0, V0sc, ind_swit,tsn):

	# Gamma in each shell and phase before diffusion
	Gam10 = int_Gam(Nz0/np.sum(Nz0,0), gam0, shn)
	Gam20 = np.zeros((Gam10.shape))
	# index where 2nd phase exists
	ind = n0sc[0,:]>0
	ind = nscnew[0,:]>0
	Gam20[0,0,ind] = (int_Gam(n0sc[:,ind]/np.sum(
			n0sc[:,ind],0), gam0, np.sum(ind)))[0,0,:]
	Gambf = (Gam10+Gam20)/2.0
	# Gamma in each shell and phase after diffusion
	Gam11 = int_Gam(Nznew/np.sum(Nznew,0), gam0, shn)
	Gam21 = np.zeros((Gam11.shape))
	Gam21[0,0,ind] = (int_Gam(nscnew[:,ind]/np.sum(
				nscnew[:,ind],0), gam0, np.sum(ind)))[0,0,:]
	Gamaf = (Gam11+Gam21)/2.0
	
	# difference in Gamma between phases after diffusion
	diff = np.abs(Gam11-Gam21)
	# empty array for factors to change flux by
	fac = np.ones((1,1,shn)) 
	
	# new fluxes affecting 2nd phase 
	fluxscn = np.zeros((fluxsc.shape))
	fluxscn[:,:] = fluxsc[:,:]
	fluxscin = np.zeros((fluxsci.shape))
	fluxscin[:,:] = fluxsci[:,:]
	fluxscon = np.zeros((fluxsco.shape))
	fluxscon[:,:] = fluxsco[:,:]
	fluxwshn = np.zeros((fluxwsh.shape))
	fluxwshn[:,:] = fluxwsh[:,:]
	count = 0

	
	while np.sum(diff[ind_swit]>1.0e-3)>0 and (np.sum((Gamaf[ind_swit]*Gambf[ind_swit])<0))>0 and np.sum(fluxwshn[0,np.squeeze(ind_swit)]*ts>1.0e-30)>0:
	#while np.sum(Gambf*Gamaf<0)>0:
		count = count+1
		
		if count>100:
			print tsn
			return
			break
		
		# -------------------------------------------------
		# index of where flux switches
		#ind10 =  np.abs(Gam20)>np.abs(Gam10)
		#ind11 = diff>0
		#ind = ind10*ind11
		#ind20 = np.abs(Gam10)>np.abs(Gam20)
		#ind21 = diff<0
		#ind2 = ind20*ind21
		#index = ind+ind2
		index =  ind_swit
		# -------------------------------------------------
		# factor to alter flux by if flux excessive
		fac[ind_swit] = (1.0/(1.0+diff[ind_swit]))*1.0e-2 # decrease flux
		if count>50:
			print 'whoop3'
			print n0sc[0,:]/np.sum(n0sc,0)
			print nscnew[0,:]/np.sum(nscnew,0)
			print fac
			print fluxwshn
			print fluxwshn[np.squeeze(ind_swit)]*ts
		# -------------------------------------------------
		# and if flux insufficient
		#ind =  np.abs(Gam20)>np.abs(Gam10) and diff<0
		#ind2 = np.abs(Gam10)>np.abs(Gam20) and diff>0
		#index = ind+ind2
		#fac[index] = 1.0*(1.0+diff[index])
		
		# alter flux
		ind_swit_ext1 = np.append(ind_swit,0)
		ind_swit_ext1 = ind_swit_ext1==1
		ind_swit_ext2 = np.append(0,ind_swit)
		ind_swit_ext2 = ind_swit_ext2==1
		ind_swit = ind_swit==1
		#fluxscn[0,np.squeeze(ind_swit_ext2)] = fluxscn[0,np.squeeze(ind_swit_ext2)]/2.0
		#fluxscin[0,np.squeeze(ind_swit_ext2)] = fluxscin[0,np.squeeze(ind_swit_ext2)]/2.0
		#fluxscon[0,np.squeeze(ind_swit_ext1)] = fluxscon[0,np.squeeze(ind_swit_ext1)]/2.0
# 		fluxwshn[0,np.squeeze(ind_swit)] = fluxwshn[0,np.squeeze(ind_swit)]/3.0
		fluxwshn[0,np.squeeze(ind_swit)] = fluxwshn[0,np.squeeze(ind_swit)]*(fac[ind_swit])
		# repeat flux calculations as above for all but 
		# between phases fluxes
		Nznew[:, :] = Nz0[:, :]+(flux[:, 1:flux.shape[1]]-
			flux[:, 0:flux.shape[1]-1])*ts
			
		# flux between 2nd phases
		nscnew[:, :] = n0sc[:, :]+(fluxscn[:, 1:fluxscn.shape[1]]-
			fluxsc[:, 0:fluxsc.shape[1]-1])*ts
			
		# inner bulk to outer 2nd phase
		Nznew[:, :] = Nznew[:, 0:shn]+(fluxscon[:, 1:fluxsco.shape[1]])*ts
		nscnew[:, 1::] = nscnew[:, 1::]-(fluxscon[:, 1:fluxsco.shape[1]-1])*ts
	
		# print 'whoop20'
# 		print nscnew
		# outer bulk to inner 2nd phase
		Nznew[:, 1::] = Nznew[:, 1::]+(fluxscin[:, 1:fluxsci.shape[1]-1])*ts
		nscnew[:, :] = nscnew[:, 0:shn]-(fluxscin[:, 1:fluxsci.shape[1]])*ts
		
		
		Nznew[:, :] = Nznew[:, :]-(fluxwshn)*ts
		nscnew[:, :] = nscnew[:, :]+(fluxwshn)*ts	
		
		V = np.zeros((1, shn))
		Vscnew = np.zeros((1, shn))
		Mv = np.zeros((M.size, 1)) # molar volumes
		Mv[:,0] = (M[:, 0]/p[:, 0])
		V[0, :] = np.sum(Nznew*Mv, axis=0)
		Vscnew[0, :] = np.sum(nscnew*Mv, axis=0) 	
		Vdiff = (V0)-(V)
		Vdiffsc = (V0sc)-(Vscnew)	
		Nznew[Nznew.shape[0]-1,:] = Nz0[Nz0.shape[0]-1,:]+Vdiff*(1.0/
					Mv[Mv.shape[0]-1,0])
		nscnew[Nznew.shape[0]-1,:] = n0sc[n0sc.shape[0]-1,:]+Vdiffsc*(1.0/
		Mv[Mv.shape[0]-1,0])
		
		
		# Gamma in each shell and phase after diffusion
		Gam11 = int_Gam(Nznew/np.sum(Nznew,0), gam0, shn)
		Gam21 = np.zeros((Gam11.shape))
		ind = nscnew[0,:]>0
		if np.sum(ind)>0:
			Gam21[0,0,ind] = (int_Gam(nscnew[:,ind]/np.sum(
				nscnew[:,ind],0), gam0, np.sum(ind)))[0,0,:]
		Gamaf = (Gam11+Gam21)/2.0
		if Gamaf[ind_swit<1.0e-4]:
			break
		
		# difference in Gamma between phases after diffusion
		diff = np.abs(Gam11-Gam21)

	
	return Nznew, nscnew