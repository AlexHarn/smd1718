import numpy as np


e = 0.4
g1 = 200
g2 = 169

g = np.array([g1, g2])

A = np.matrix(([1 - e, e], [e, 1 - e]))
B = np.linalg.inv(A)
f = B@g.T

covg = np.diag(g)
covf = B@covg@B.T
df = np.sqrt(np.diag(covf))
corr = covf[0, 1]/(df[0]*df[1])

print('f =', f)
print('V[f] =', covf)
print('sigmas =', df)
print('rho =', corr)
