import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5.8
rcParams['legend.numpoints'] = 1


# -------------------------------- Funktionen ---------------------------------
def poly(x, a0, a1, a2, a3, a4, a5, a6):
    return a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4 + a5*x**5 + a6*x**6


def aufg31a(x, y, A):
    print("a)")
    a = np.linalg.inv(A.T@A)@A.T@y
    print("Koeffizienten:", a)
    xr = np.linspace(min(x) - 0.5, max(x) + 0.5, 1000)
    plt.xlim(min(x) - 0.5, max(x) + 0.5)
    plt.plot(xr, poly(xr, *a), label="Unregularisiert")
    plt.plot(x, y, "ro", label="Daten")
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    # plt.show()
    plt.savefig("A31a.pdf")
    plt.clf()
    return a


def aufg31b(x, y, A, a):
    print("b)")
    C = np.asarray(diags([np.ones(A.shape[0])*(-2), np.ones(A.shape[0] - 1),
                          np.ones(A.shape[0] - 1)], [0, -1, 1]).todense())
    C[0][0] = -1
    C[-1][-1] = -1

    ls = [0.1, 0.3, 0.7, 3, 10]
    a_reg = []
    for l in ls:
        a_reg.append(np.linalg.inv(A.T@A + l*(C@A).T@(C@A))@A.T@y)

    xr = np.linspace(min(x) - 0.5, max(x) + 0.5, 1000)
    plt.xlim(min(x) - 0.5, max(x) + 0.5)
    plt.plot(xr, poly(xr, *a), label="Unregularisiert")
    for a, l in zip(a_reg, ls):
        print("Koeffizienten mit λ = {}: {}".format(l, a))
        plt.plot(xr, poly(xr, *a), label="$\lambda = {}$".format(l))
    plt.plot(x, y, "ro", label="Daten")
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    # plt.show()
    plt.savefig("A31b.pdf")
    plt.clf()


def aufg31c():
    X = np.loadtxt("aufg_c.csv", delimiter=", ", skiprows=1)
    x = X[:, 0]
    y = np.mean(X[:, 1:], axis=1)
    A = np.vstack((x**i for i in range(0, 7))).T
    y_err = np.std(X[:, 1:], axis=1)
    W = np.diag(1/y_err)

    a = np.linalg.inv(A.T@A)@A.T@y
    a_weighted = np.linalg.inv(A.T@W@A)@A.T@W@y

    xr = np.linspace(min(x) - 0.5, max(x) + 0.5, 1000)
    plt.xlim(min(x) - 0.5, max(x) + 0.5)
    plt.plot(xr, poly(xr, *a_weighted), label="Gewichtet")
    plt.plot(xr, poly(xr, *a), "--", label="Ungewichtet")
    plt.errorbar(x, y, yerr=y_err, fmt="ro", label="Daten")
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    # plt.show()
    plt.savefig("A31c.pdf")
    plt.clf()


# ----------------------------------- Main ------------------------------------
if __name__ == "__main__":
    x, y = np.loadtxt("aufg_a.csv", unpack=True, delimiter=", ", skiprows=1)
    A = np.vstack((x**i for i in range(0, 7))).T
    a = aufg31a(x, y, A)
    aufg31b(x, y, A, a)
    aufg31c()
