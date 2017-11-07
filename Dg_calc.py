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
# function to calculate the gas phase diffusion coefficients (m2/s),
# for explanation of calculations see pp. 44 of thesis2.pdf.
import numpy as np
def Dg_calc(M,T,sigma):
    
    # ------------------------------------------------------------------------------------
    # inputs:
    # M - molar mass of components (g/mol)
    # T - temperature (K)
    # sigma - molecular diameter (m)
    # ------------------------------------------------------------------------------------
    
    # molecular mass (kg molecule^{-1}) (convert from g mol^{-1})
    Mmolec = (M/6.022e23)*1e-3
    # mean molecular speed in gas phase (m s^{-1})
    v = ((3*T*1.381e-23)/Mmolec)**(0.5);
    # air pressure (kg m^{-1} s^{-2}
    Pair = 101325.0
    # ideal gas constant (kg m^2s^{-2}mol^{-1}K^{-1}
    R = 8.31
    # number density (molecules m^(-3))
    n = ((Pair)/(R*T))*6.022e23   
    # mean free path of molecules (m)
    lam = 0.707/(np.pi*n*(sigma**2))
    # gas phase diffusion coefficient (m2/s) (p.44 thesis2.pdf)
    Dg = (v*lam)/3
    return Dg
