import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import factorial
from scipy.optimize import newton
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5.8
rcParams['legend.numpoints'] = 1


def F(l):
    return -30*np.log(l) + 3*l + \
        np.log(factorial(13)*factorial(8)*factorial(9))


def taylorF(l):
    return F(10) + 0.15*(l - 10)**2


def aufg29a():
    x = np.linspace(.001, 50, 1000)
    plt.xlim(.001, 50)
    plt.ylim(0, 100)
    plt.xlabel("$\lambda$")
    plt.ylabel("$F(\lambda)$")
    plt.plot(x, F(x))
    # plt.show()
    plt.savefig("A29a.pdf")


def aufg29c():
    print(newton(lambda x: F(x) - F(10) - 0.5, 9), "< λ <",
          newton(lambda x: F(x) - F(10) - 0.5, 11))

    print(newton(lambda x: F(x) - F(10) - 2, 9), "< λ <",
          newton(lambda x: F(x) - F(10) - 2, 11))

    print(newton(lambda x: F(x) - F(10) - 4.5, 7), "< λ <",
          newton(lambda x: F(x) - F(10) - 4.5, 12))


def aufg29d():
    print(newton(lambda x: taylorF(x) - F(10) - 0.5, 9), "< λ <",
          newton(lambda x: taylorF(x) - F(10) - 0.5, 11))

    print(newton(lambda x: taylorF(x) - F(10) - 2, 9), "< λ <",
          newton(lambda x: taylorF(x) - F(10) - 2, 11))

    print(newton(lambda x: taylorF(x) - F(10) - 4.5, 7), "< λ <",
          newton(lambda x: taylorF(x) - F(10) - 4.5, 12))

    x = np.linspace(.001, 20, 1000)
    plt.xlim(.001, 20)
    plt.ylim(0, 30)
    plt.xlabel("$\lambda$")
    plt.plot(x, F(x), label="$F(\lambda)$")
    plt.plot(x, taylorF(x), label=r"$\rm{T}[F](\lambda)$")
    plt.legend()
    # plt.show()
    plt.savefig("A29d.pdf")


if __name__ == "__main__":
    # aufg29a()
    aufg29c()
    aufg29d()
