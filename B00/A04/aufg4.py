import matplotlib.pyplot as plt
from scipy.constants import alpha
import numpy as np
from pylab import rcParams
# from scipy.misc import derivative as diff

rcParams['figure.figsize'] = 10, 5.8
rcParams['legend.numpoints'] = 1

# -------------------------------- Konstanten ---------------------------------
E_e = 50
s = (2*E_e)**2
gamma = E_e/511e-6
beta = np.sqrt(1 - gamma**(-2))


# -------------------------------- Funktionen ---------------------------------
def dwq(theta):
    """Differentieller Wirkungsquerschnitt in ursprünglicher Form"""
    return alpha**2/s*(2 + np.sin(theta)**2)/(1 - beta**2*np.cos(theta)**2)


def dwq2(theta):
    """Differentieller Wirkungsquerschnitt in numerisch stabiler Form"""
    return gamma**2*alpha**2/s*(2 + np.sin(theta)**2)\
        / (gamma**2*np.sin(theta)**2 + np.cos(theta)**2)


# def k(x):
    """Numerisch berechnete Konditionszahl"""
    # return abs(x*diff(dwq2, x, dx=1e-8)/dwq2(x))


def K(x):
    """Analytisch berechnete Konditionszahl"""
    return abs(x*2*np.sin(x)*np.cos(x)*(1 + beta**2)
               / ((1 - beta**2*np.cos(x)**2)*(2 + np.sin(x)**2)))


def aufg4():
    # theta = np.arange(0.5, 361, 1)
    theta = np.linspace(179.999995, 180.000005, 1e4)
    plt.plot(theta, dwq(np.deg2rad(theta)), ".", ms=0.5, label="Ursprünglich")
    plt.plot(theta, dwq2(np.deg2rad(theta)), "x", ms=0.5, label="Umgeformt")
    plt.xlim(179.999995, 180.000005)
    plt.xlabel(r"$\theta\,/\,°$")
    plt.ylabel(r"$\frac{\rm{d}\sigma}{\rm{d}\Omega}$")
    plt.legend()
    # plt.show()
    plt.savefig("A4_stab.pdf")
    plt.clf()

    theta = np.linspace(0, 360, 1e4)
    plt.plot(theta, K(theta/180*np.pi))
    # plt.plot(theta, k(theta/180*np.pi))
    plt.xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    plt.xlim(0, 360)
    plt.yscale("log")
    plt.axhline(1, ls="--", color="k")
    plt.xlabel(r"$\theta\,/\,°$")
    plt.ylabel(r"$K$")
    # plt.show()
    plt.savefig("A4_kond.pdf")


# ----------------------------------- Main ------------------------------------
if __name__ == '__main__':
    aufg4()
