import h5py
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
from uncertainties import ufloat

rcParams['figure.figsize'] = 10, 5.8
rcParams['legend.numpoints'] = 1


# ------------------------------------ b) -------------------------------------
def aufg12b(plot=True):
    p0 = np.random.multivariate_normal([0, 3], [[3.5**2, 0.9*3.5*2.5],
                                                [0.9*3.5*2.5, 2.5**2]],
                                       size=10000).T
    p1 = np.empty((2, 10000))
    p1[0] = np.random.normal(6, 3.5, size=10000)
    p1[1] = np.random.normal(-0.5 + 0.6*p1[0], 1)

    if plot:
        plt.scatter(p0[0], p0[1], s=1, alpha=0.25, label="$P_0$")
        plt.scatter(p1[0], p1[1], s=1, alpha=0.25, label="$P_1$")
        plt.legend()
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        # plt.show()
        plt.savefig("A12b.pdf")
    return (p0, p1)


def aufg12c(p0, p1):
    # Vereinigung der beiden Polpulationen bilden
    # und alles in eine Liste packen
    ps = [np.array([np.concatenate((p0[0], p1[0])),
                    np.concatenate((p0[1], p1[1]))]),
          p0, p1]

    for i, p in enumerate(ps):
        # Kovarianz
        cov = np.cov(p)

        # Mittelwerte und Standardabweichungen
        m = [ufloat(np.average(p[0]), np.sqrt(cov[0][0])),
             ufloat(np.average(p[1]), np.sqrt(cov[1][1]))]

        # Korrelationskoeffizient
        r = cov[0][1]/(m[0].s*m[1].s)
        if i == 0:
            print("Gesamtpolulation:\n-----------------")
        else:
            print("Population %1i\n------------" % (i - 1))
        print("cov =", cov)
        print("m =", m)
        print("r =", r)
        print("\n")


def aufg12d(p0, p1):
    p0_1000 = np.random.multivariate_normal([0, 3], [[3.5**2, 0.9*3.5*2.5],
                                                     [0.9*3.5*2.5, 2.5**2]],
                                            size=1000).T

    fh = h5py.File("pops.hdf5", 'w')
    fh.create_dataset("p0", data=p0)
    fh.create_dataset("p_0_1000", data=p0_1000)
    fh.create_dataset("p1", data=p1)
    fh.close()


# ----------------------------------- Main ------------------------------------
if __name__ == '__main__':
    np.random.seed(1234)
    p0, p1 = aufg12b(True)
    aufg12c(p0, p1)
    aufg12d(p0, p1)
