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
# function to work out saturation ratios and gas phase concentrations now (ug m^{-3})

import numpy as np

def esnow_func(estime, time, tsn, es, Cstar):

    	# saturation ratio times bounding current time (s)
    	estimeindex = np.sum(estime[:]<time[tsn])
    	# time difference between saturation ratio times and actual time (s)
    	tdifflow = time[tsn]-estime[estimeindex-1]
    	tdiffhig = estime[estimeindex]-time[tsn]
    	# linear interpolation factors
    	tfraclow = tdiffhig/(estime[estimeindex]-estime[estimeindex-1])
    	tfrachig = tdifflow/(estime[estimeindex]-estime[estimeindex-1])
    	# linear interpolation to get current saturation ratios (dimensionless)
    	esnow = es[:,estimeindex-1]*(tfraclow)+es[:,estimeindex]*[tfrachig]
    	# gas phase concentration of components (ug m^{-3} (air))
    	Cg = Cstar[:]*esnow
    
    	del estimeindex, tdifflow, tdiffhig, tfraclow, tfrachig
    	return Cg, esnow
