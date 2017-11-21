from mcmc import MCMC
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5.8
rcParams['legend.numpoints'] = 1


# ------------------------------------ c) -------------------------------------
def aufg15c():
    normal = MCMC(2, -3, 2)
    sample = normal.sample(15, 10000)
    bins = plt.hist(sample, bins=50,
                    density=True, label="Gezogen")[1]
    x = np.linspace(bins[0], bins[-1], 1000)
    plt.plot(x, normal.pdf(x), label="PDF")
    plt.xlabel("$x$")
    plt.ylabel("Wahrscheinlichkeitsdichte")
    plt.legend()
    # plt.show()
    plt.savefig("A15c.pdf")
    plt.clf()
    return sample


# ------------------------------------ d) -------------------------------------
def aufg15d(sample):
    plt.plot(np.arange(len(sample)), sample, ".", ms=3)
    plt.xlim(-100, len(sample))
    plt.xlabel("$t\,/\,$a.u.")
    plt.ylabel("$x\,/\,$a.u.")
    plt.savefig("A15d.pdf")
    plt.xlim(-5, 200)
    plt.savefig("A15d_zoom.pdf")


# ----------------------------------- Main ------------------------------------
if __name__ == '__main__':
    np.random.seed(1234)
    sample = aufg15c()
    aufg15d(sample)
