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
# nested function containing constants for input to the partitioning equation

import numpy as np

def k1calc(Rp, Dg, M, p, T):

	# ---------------------------------------------------------------------
	# inputs:
    	# Rp - most recent radius of particle (m)
    	# Dg - gas phase diffusion coefficient (m2/s)
    	# M - component molar masses (g/mol)
    	# p - component densities (g/m3)
    	# T - temperature (K)

	# outputs:
    	# k2 - gas side mass transfer coefficient (m/s (air))
	# --------------------------------------------------------------------
    
	# accommodation coefficient (dimensionless)
    	alpha = 1.0
    	# number density of air (molecules/m3) (ideal gas equation, with P in 
	# units of kg/(m s2), R in units kg m2/(s2 mol K)
    	n = (101325.0/(8.31*T))*6.022e23
    	Mvol = ((M[:, 0]/p[:, 0])/6.022e23) # molecular volume (m^3/molecule)
    	# molecular diameter (m)
    	sigma = (((3.0*Mvol[:])/(4.0*np.pi))**(1.0/3.0))*2.0
    	# gas phase mean free path (5.972e-8 m is what MOSAIC gets when iv==12 
	# in line 9147 of mosaic_box.25.f90) (m)
    	# mean free path of molecules (m) 
	# (see gas_phase_diffusion_equation_testing.pdf)
    	lamb = 0.707/(np.pi*n*(sigma**2.0)) 
    	Kn = lamb/Rp # Knudsen number (-)
    	# transition regime correction factor (eq. 14 Zaveri et al. 2014)
    	f = (0.75*alpha*(1.0+Kn))/(Kn*(1.0+Kn)+0.283*alpha*Kn+0.75*alpha);
    	k1 = (Dg/Rp)*f # gas side mass transfer coefficient (m/s)
    	k2 = ((3.0/Rp)*k1)

    	return k2
