import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from tqdm import tqdm


# -------------------------------- Funktionen ---------------------------------
def line(x, n):
    a0 = np.random.normal(size=n)
    a1 = np.random.normal(size=n)
    # einfach damit es lesbar ist alles hier ausrechnen
    a0 = np.sqrt(1 - 0.8**2)*0.2*a0 - 0.8*0.2*a1 + 1
    a1 = 0.2*a1 + 1

    return a0 + a1*x


# ----------------------------------- Main ------------------------------------
if __name__ == "__main__":
    np.random.seed(1234)
    x = np.linspace(-10, 10, 1000)

    # wir nehmen jetzt einfach für alle x die selben a0 und a1
    a0 = np.random.normal(size=1000000)
    a1 = np.random.normal(size=1000000)
    a0 = np.sqrt(1 - 0.8**2)*0.2*a0 - 0.8*0.2*a1 + 1
    a1 = 0.2*a1 + 1

    y_err = np.empty(len(x))
    y_m = np.empty(len(x))
    for idx, xi in tqdm(enumerate(x), total=len(x)):
        y = a0 + a1*xi
        y_m[idx] = np.mean(y)
        y_err[idx] = np.std(y)

    # Plotten
    plt.scatter(a0, a1, marker=".")
    plt.xlabel("$a_0$")
    plt.ylabel("$a_1$")
    # plt.show()
    plt.savefig("A23_scatter.png", dpi=600)
    plt.clf()

    plt.plot(x, y_err, label="Numerisch")
    plt.plot(x, 0.2*np.sqrt(1 + x**2 - 1.6*x), label="Analytisch")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    # plt.show()
    plt.savefig("A23_resultat.pdf")

    print("y(-3) =", ufloat(y_m[350], y_err[350]))
    print("y(0) =", ufloat(y_m[500], y_err[500]))
    print("y(3) =", ufloat(y_m[650], y_err[650]))
