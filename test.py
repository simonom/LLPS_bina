import numpy as np
import matplotlib.pyplot as plt

PS0=np.load('WDD_Gibb.npy')
plt.plot(PS0[:,0],PS0[:,3])
plt.show()
