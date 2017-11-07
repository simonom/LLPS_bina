import numpy as np
import matplotlib.pyplot as plt

# nucleation rate (#nuclei/s)
delG = 3000
delx = 1.0e-1
D = np.logspace(-18.0,-9.0,base=10.0)

nucr = 0.0+(D*1.0e9)*(delG/3000)*(1.0e-3/delx) 

plt.plot(D,nucr,'xr')
plt.xscale('log')
plt.yscale('log')
plt.show()
