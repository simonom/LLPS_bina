import numpy as np
from scipy.integrate import odeint 


# mole fractions in shells
x = np.array([[0.2],[0.3]])
# distance between shells (m)
delt = 2.0e-6
# self-diffusion coefficients (m2/s)
Dpur = np.array([[1.0e-12],[1.0e-9]])


# area to diffuse over (m)
Area = np.pi*((delt*2.0)**2.0)
D = np.product(Dpur**x)
# time step to diffuse over (s)
t = np.linspace(0,2.0e10,2)
# initial concentrations (mol/m3) in two shells
C0=[2.0,20.0]		

# define the equation
def deriv(C, t, delt, D, Area):
	dy = np.array([0, 3.6])
	for i in range(0,2):
		C1,C2 = C
		if i==0:
			dy[i] = [-(D*((C2-C1)/delt))*Area][0]
		if i==1:
			dy[i] = [(D*((C2-C1)/delt))*Area][0]
	return dy
	
sol = odeint(deriv, C0, t, args=(delt, D, Area))
print sol
del sol, C0