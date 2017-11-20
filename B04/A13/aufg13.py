import numpy as np
import h5py
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5.8
rcParams['legend.numpoints'] = 1


# ------------------------------------ a) -------------------------------------
def aufg13ab():
    fh = h5py.File("../A12/pops.hdf5", "r")
    p0 = np.copy(fh["p0"])
    p1 = np.copy(fh["p1"])
    fh.close()

    x = np.linspace(-10, 20, 1000)
    plt.scatter(p0[0], p0[1], marker=".", s=2.5, alpha=0.5, label=r"$p_0$")
    plt.scatter(p1[0], p1[1], marker=".", s=2.5, alpha=0.5, label=r"$p_1$")
    plt.plot(x, 0*x, label=r"$g_1$")
    plt.plot(x, -0.75*x, label=r"$g_2$")
    plt.plot(x, -1.25*x, label=r"$g_3$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.xlim(-10, 20)
    plt.ylim(-7.5, 12.5)
    plt.legend()
    # plt.show()
    plt.savefig("A13a_scatter.pdf")
    plt.clf()

    proj = np.zeros((3, 2, len(p0[0])))
    g = [[-1, 0], [-0.8, 0.6], [-4/np.sqrt(41), 5/np.sqrt(41)]]
    for i in range(3):
        proj[i][0] = np.add(np.multiply(p0[0], g[i][0]),
                            np.multiply(p0[1], g[i][1]))
        proj[i][1] = np.add(np.multiply(p1[0], g[i][0]),
                            np.multiply(p1[1], g[i][1]))
        bins = np.linspace(-20, 10, 100)
        plt.hist(proj[i][0], bins=bins, histtype="barstacked", label="$P_0$")
        plt.hist(proj[i][1], bins=bins, histtype="barstacked", label="$P_1$",
                 alpha=0.6)
        plt.xlabel("$x_{g_%1i}$" % (i + 1))
        plt.ylabel("$n$")
        plt.legend()
        # plt.show()
        plt.savefig("A13b_%1i.pdf" % (i + 1))
        plt.clf()
    return proj


# ------------------------------------ a) -------------------------------------
def aufg13c(proj, nl=10000):
    for i, p in enumerate(proj):
        p[0][::-1].sort()
        p[1][::-1].sort()
        ls = np.linspace(max(p[0][0], p[1][0]),
                         min(p[0][-1], p[1][-1]), nl)
        tp = np.zeros(nl)
        fp = np.zeros(nl)
        fn = np.zeros(nl)
        tn = np.zeros(nl)
        j = 0
        k = 0
        for n, l in enumerate(ls):
            while (j < len(p[0])) and (p[0][j] >= l):
                j += 1
            while k < len(p[1]) and p[1][k] >= l:
                k += 1
            tp[n] = j
            fn[n] = len(p[0]) - j
            fp[n] = k
            tn[n] = len(p[0]) - k
        # plt.plot(ls, tp)
        # plt.plot(ls, fp)
        # plt.plot(ls, fn)
        plt.plot(ls, tp/(tp + fp), label="Reinheit")
        plt.plot(ls, tp/(tp + fn), label="Effizienz")
        plt.plot(ls, (tp + tn)/(tp + tn + fp + fn), label="Genauigkeit")
        plt.xlim(ls[-1], ls[0])
        plt.legend()
        plt.xlabel("$x_{g_%1i}$" % (i + 1))
        # plt.show()
        plt.savefig("A13c_%1i.pdf" % i)
        plt.clf()


# ----------------------------------- Main ------------------------------------
if __name__ == '__main__':
    proj = aufg13ab()
    aufg13c(proj)
