import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k
from scipy.optimize import newton


def f(v, m, T):
    return (m/(2*np.pi*k*T))**(1.5)*4*np.pi*v**2*np.exp(-m*v**2/(2*k*T))


def v_m(m, T):
    return np.sqrt(2*k*T/m)


# v = np.linspace(0, 10, 1e5)
# plt.plot(v, f(v, 1, 1/k))
# plt.plot(v_m(1, 1/k), f(v_m(1, 1/k), 1, 1/k), "o")
# plt.show()

# c)
# h = 1e-7
# s = 0
# i = 1
# while s < 0.5:
    # s += h*4/np.sqrt(np.pi)*(i*h)**2*np.exp(-(i*h)**2)
    # i += 1
# print("v_0.5 = v_m *", i*h)
# print("Int_0^v_0.5 f(v)dv =", s)

x1 = newton(lambda x: x**2*np.exp(-x**2) - 1/(2*np.e), 0.8, lambda x:
            2*x*np.exp(-x**2)*(1 - x**2))
x2 = newton(lambda x: x**2*np.exp(-x**2) - 1/(2*np.e), 1.2, lambda x:
            2*x*np.exp(-x**2)*(1 - x**2))
print("x_1 =", x1, "x_2 =", x2)
print("v_FWHM = v_m *", x2 - x1)
