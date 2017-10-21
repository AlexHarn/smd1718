import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5.8
rcParams['legend.numpoints'] = 1


# -------------------------------- Funktionen ---------------------------------
def aufg2():
    x = np.logspace(-1, -20, num=20)
    y = (np.sqrt(9 - x) - 3)/x

    plt.ylim(-0.5, 0.05)
    plt.axhline(-1/6, color='g', label="Analytischer Grenzwert")
    plt.plot(x, y, ".", label="Numerische Ergebnisse")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$(\sqrt{9 - x } - 3)/x$")
    plt.xscale('log')
    plt.legend()
    plt.xticks([1e-20, 1e-15, 1e-10, 1e-5, 0.1])
    plt.savefig('A2.pdf')
    plt.clf()


# ----------------------------------- Main ------------------------------------
if __name__ == '__main__':
    aufg2()
