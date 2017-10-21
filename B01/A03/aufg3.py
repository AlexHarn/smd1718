import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5.8
rcParams['legend.numpoints'] = 1


# -------------------------------- Funktionen ---------------------------------
def f(x):
    return (x**3 + 1/3) - (x**3 - 1/3)


def g(x):
    return ((3 + x**3/3) - (3 - x**3/3))/x**3


def aufg3():
    x = np.logspace(-20, 20, num=410)
    plt.plot(x, f(x), "-", label=r"$f(x)$")
    plt.xscale('log')
    # plt.axhline(2/3, color='g', label="Analytisches Ergebnis")

    plt.plot(x, g(x), "-", label=r"$g(x)$")
    plt.yticks([0, 2/3, 1], ["0", r"$\frac{2}{3}$", "1"])
    plt.xscale('log')
    plt.ylim(0, 1)
    plt.legend()
    plt.ylim(-0.01, 1.2)
    plt.xlim(1e-20, 1e20)
    plt.xlabel(r"$x$")
    plt.ylabel("Funktionswert")
    # plt.show()
    plt.savefig("A3.pdf")


# ----------------------------------- Main ------------------------------------
if __name__ == '__main__':
    aufg3()
