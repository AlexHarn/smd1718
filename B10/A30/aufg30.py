import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat


psi, y = np.loadtxt("data.txt", unpack=True)
psi = np.deg2rad(psi)
# a)
A = np.vstack((np.cos(psi), np.sin(psi))).T

# b)
a = np.linalg.inv(A.T@A)@A.T@y
print("Lösungsvektor a =", a)

# c)
X_v2 = np.average((y - A@a)**2/0.011**2)/2
print("Chi_v^2 =", X_v2)
a_err = np.sqrt(1/6*X_v2)
print("sqrt(1/6*Chi_v^2) =", a_err)
a1 = ufloat(a[0], a_err)
a2 = ufloat(a[1], a_err)

# d)
delta = unp.arctan(-a2/a1)
A0 = a1/unp.cos(delta)
print("A0 =", A0)
print("delta =", delta*180/np.pi)

delta = np.arctan(-a[1]/a[0])
delta_err = np.sqrt((-a[1]/(a[0]**2 + a[1]**2)*a_err)**2
                    + (a[0]/(a[0]**2 + a[1]**2)*a_err)**2)
A0 = a[0]/np.cos(delta)
A0_err = np.sqrt((a_err/np.cos(delta))**2
                    + (a[0]*np.tan(delta)/np.cos(delta)
                    *delta_err)**2)
print(np.rad2deg(delta_err))
print(A0_err)
