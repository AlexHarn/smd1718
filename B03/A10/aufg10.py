import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5.8
rcParams['legend.numpoints'] = 1


# ------------------------------------ a) -------------------------------------


# ------------------------------------ b) -------------------------------------
def dist_b(tau, N=1e6):
    """Liefert N nach f(x)=exp(-x/tau)/tau verteilte Zufallszahlen"""
    return -tau*np.log((1 - np.random.uniform(
        low=1, high=1 - 1/tau, size=int(N)))*tau)


def aufg10b(tau, N=1e6):
    """Validiert das Ergebnis für Aufgabe 10b graphisch"""
    t = dist_b(tau, N)

    x = np.linspace(0, 6, 10000)
    plt.hist(t, bins=1000, density=True, label="Generierte Zahlen")
    plt.plot(x, 1/tau*np.exp(-x/tau), label=r"$f(t)$")
    plt.xlim(0, 6)
    plt.xlabel(r"$t$")
    plt.ylabel("Wahrscheinlichkeitsdichte")
    plt.legend()
    # plt.show()
    plt.savefig("A10b.pdf")
    plt.clf()


# ------------------------------------ c) -------------------------------------
def dist_c(n, x_min, x_max, N=1e6):
    """
    Liefert N nach (1-n)/(x_max^(1-n) - x_min^(1-n))/x^n verteilte
    Zufallszahlen
    """
    if n < 2 or not isinstance(n, int):
        raise ValueError("n muss ein Integer größer oder gleich 2 sein")
    z = np.random.uniform(size=int(N))
    return (z*x_max**(1 - n) - (z - 1)*x_min**(1 - n))**(1/(1 - n))


def aufg10c(n, x_min, x_max, N=1e6):
    """Validiert das Ergebnis für Aufgabe 10c graphisch"""
    x = dist_c(n, x_min, x_max, N)

    x1 = np.linspace(x_min, x_max, 10000)
    plt.hist(x, bins=1000, density=True, label="Generierte Zahlen")
    plt.plot(x1, (1 - n)/(x_max**(1 - n) - x_min**(1 - n))/x1**n,
             label=r"$f(x)$")
    plt.xlim(x_min, x_max)
    plt.xlabel(r"$x$")
    plt.ylabel("Wahrscheinlichkeitsdichte")
    plt.legend()
    # plt.show()
    plt.savefig("A10c.pdf")
    plt.clf()


# ------------------------------------ d) -------------------------------------
def dist_d(N=1e6):
    """Liefert N cauchy-verteilte Zufallszahlen"""
    return -1/np.tan(np.pi*np.random.uniform(size=int(N)))


def aufg10d(N=1e6):
    """Validiert das Ergebnis für Aufgabe 10d graphisch"""
    x = dist_d(N)

    x1 = np.linspace(-5, 5, 10000)
    # Histogramm ist hier sehr problematisch, weil auch sehr große und sehr
    # kleine Werte noch relativ wahrscheinlich sind darum einschränken,
    # dadurch ist die Normierung aber nicht mehr ganz exakt
    # folgendes funktioniert warum auch immer nicht korrekt:
    # plt.hist(x, bins=np.concatenate(([min(x)], np.linspace(-10, 10, 1000),
    #                                 [max(x)])),
    #          density=True, label="Generierte Zahlen")
    # daher gute Näherung nehmen
    plt.hist(x, bins=np.linspace(-100, 100, 10000), density=True,
             label="Generierte Zahlen")
    plt.plot(x1, 1/(np.pi*(1 + x1**2)), label=r"$f(x)$")
    # plt.plot(x1, np.arctan(x1)/np.pi + 0.5)
    plt.xlim(-5, 5)
    plt.xlabel(r"$x$")
    plt.ylabel("Wahrscheinlichkeitsdichte")
    plt.legend()
    # plt.show()
    plt.savefig("A10d.pdf")
    plt.clf()


# ------------------------------------ e) -------------------------------------
def dist_e(N=1e6):
    """
    Liefert N nach der empirischen Verteilung verteilte Zufallszahlen unter
    Verwendung einer interpolierten Verteilungsfunktion
    """
    data = np.load("empirisches_histogramm.npy")
    # 1/50 = 0.02 ist die Binbreite
    cumsum = np.cumsum(data["hist"]/(np.sum(data["hist"])*0.02))*0.02
    # Inverse der Verteilungsfunktion interpolieren
    # Würde man natürlich normalerweise speichern und nicht jedes mal neu
    # interpolieren
    inverse = interp1d(cumsum, data["bin_mid"])
    return inverse(np.random.uniform(cumsum[0], cumsum[-1], size=int(N)))


def aufg10e():
    """Validiert das Ergebnis für Aufgabe 10e graphisch"""
    data = np.load("empirisches_histogramm.npy")
    plt.hist(data['bin_mid'], bins=np.linspace(0, 1, 51),
             weights=data['hist'], density=True, label="Original")
    plt.hist(dist_e(1e7), bins=np.linspace(0, 1, 51), density=True, alpha=0.5,
             label="Generiert")
    plt.legend()
    plt.xlabel(r"$x$")
    plt.ylabel("Wahrscheinlichkeitsdichte")
    plt.xlim(0, 1)
    # plt.show()
    plt.savefig("A10e.pdf")


# ----------------------------------- Main ------------------------------------
if __name__ == '__main__':
    np.random.seed(1234)
    aufg10b(1)
    aufg10c(2, 1, 5)
    aufg10d()
    aufg10e()
