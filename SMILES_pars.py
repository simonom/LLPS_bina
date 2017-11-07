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
# function to parse SMILES strings

from collections import OrderedDict
import pybel

def SMILES_pars(SMILES_arr):
	# ordered dictionary for component SMILES and concentrations (concentrations dealt 
	# with later in code)
	SMILES_arr2 = OrderedDict() 
	for s in SMILES_arr: # component (SMILES) loop
		# SMILES parsing
		SMILES = pybel.readstring(b'smi',s)
		SMILES_arr2[SMILES] = 0.0 # add passed SMILES to dictionary
	SMILES_arr = SMILES_arr2
	del SMILES_arr2
	return SMILES_arr
