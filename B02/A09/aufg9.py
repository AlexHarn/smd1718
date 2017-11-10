import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5.8
rcParams['legend.numpoints'] = 1


# ------------------------------------ a) -------------------------------------
def findPeriod(a, x0=1):
    """
    Findet die Periodenlänge des Linear-Kongruenten-Generators aus Aufgabe
    9a) in Abhängigkeit von a (und x0)
    """

    numbers = [x0]
    while not np.isin([numbers[-1]], numbers[:-1]):
        numbers.append((a*numbers[-1] + 3) % 1024)

    numbers = np.asarray(numbers)
    first = np.where(numbers[:-1] == numbers[-1])
    return len(numbers) - first[0][0] - 1


def aufg9a():
    """
    Findet für alle ganzen a von 0 bis 99 die Periodenlänge und erstellt
    daraus einen Plot
    """
    fig, ax = plt.subplots()
    a = np.arange(0, 1024)
    ax.plot(a, np.vectorize(findPeriod)(a), ".")
    ax.set_xlabel(r"$a$")
    ax.set_ylabel("Periodenlänge")
    ax.set_yscale("log", basey=2)
    ax.set_yticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    ax.set_xticks(range(0, 1025, 64))
    ax.grid(axis="y")
    # plt.show()
    fig.savefig("A9a_full.pdf")
    ax.set_xticks(range(0, 33))
    ax.set_xlim(-1, 33)
    fig.savefig("A9a_zoom.pdf")
    plt.clf()


# ------------------------------------ b) -------------------------------------
def uniform_b(x0, n):
    """
    Liefert ein Array mit den ersten n Zufallszahlen des Generators aus
    Aufgabenteil b)
    """
    numbers = [x0]
    while len(numbers) < n + 1:
        numbers.append((1601*numbers[-1] + 3456) % 1e4)
    return np.asarray(numbers[1:])/1e4


# ------------------------------------ c) -------------------------------------
def aufg9c():
    """Erstellt das in Aufgabe 9c geforderte Histogramm"""
    # x0 = [1, 10, 223, 5412, 2812, 899, 1e4 - 39]
    # for x in x0:
        # plt.hist(uniform_b(x, 1e6), bins=100, normed=True)
        # plt.show()
        # plt.clf()
    # plt.hist(uniform_b(42, 1e6), bins=100, density=True)
    plt.hist(uniform_b(42, 10000), bins=100, density=True)
    plt.xlabel(r"$x$")
    plt.ylabel("Wahrscheinlichkeitsdichte")
    # plt.savefig("A9c.pdf")
    plt.savefig("A9c.png", dpi=300)
    # plt.show()
    plt.clf()


# ------------------------------------ d) -------------------------------------
def aufg9d():
    """Erstellt die in Aufgabe 9d geforderten Streudiagramme"""
    numbers = uniform_b(1, 1e4).reshape(5000, 2).T
    plt.scatter(numbers[0], numbers[1], marker=".")
    plt.xlabel(r"$x_n$")
    plt.ylabel(r"$x_{n+1}$")
    # plt.show()
    plt.savefig("A9d_2D.pdf")
    plt.clf()

    numbers = uniform_b(1, 9999).reshape(3333, 3).T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(numbers[0], numbers[1], numbers[2], marker=".")
    ax.view_init(elev=10, azim=20)
    ax.set_xlabel(r"$x_n$")
    ax.set_ylabel(r"$x_{n+1}$")
    ax.set_zlabel(r"$x_{n+2}$")
    # plt.show()
    plt.savefig("A9d_3D.pdf", tight_layout=True)
    # plt.savefig("A9d.png", dpi=300, tight_layout=True)
    plt.clf()


# ------------------------------------ e) -------------------------------------
def aufg9e():
    """Erstellt die in Aufgabe 9e geforderten Streudiagramme"""
    numbers = np.random.uniform(size=10000)

    numbers_tuples = numbers.reshape(5000, 2).T
    plt.scatter(numbers_tuples[0], numbers_tuples[1], marker=".")
    plt.xlabel(r"$x_n$")
    plt.ylabel(r"$x_{n+1}$")
    # plt.show()
    plt.savefig("A9e_2D.pdf")
    plt.clf()

    numbers_triplets = numbers[:-1].reshape(3333, 3).T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(numbers_triplets[0], numbers_triplets[1], numbers_triplets[2],
               marker=".", alpha=0.75)
    ax.view_init(elev=10, azim=20)
    ax.set_xlabel(r"$x_n$")
    ax.set_ylabel(r"$x_{n+1}$")
    ax.set_zlabel(r"$x_{n+2}$")
    # plt.show()
    plt.savefig("A9e_3D.pdf")
    plt.clf()


# ------------------------------------ f) -------------------------------------
def aufg9f(N=1e5):
    """Testet, für welche Startwerte unofirm_b wie oft den Wert 0.5 liefert"""
    count = []
    for x in range(0, 1025):
        numbers = uniform_b(x, N)
        count.append(np.count_nonzero(numbers == 0.5)/N)
    count = np.asarray(count)
    # print(count)
    idx = np.where(count > 0)[0]
    print(idx/8)
    print(count[idx])


# ----------------------------------- Main ------------------------------------
if __name__ == '__main__':
    np.random.seed(1234)
    aufg9a()
    aufg9c()
    aufg9d()
    aufg9e()
    aufg9f()
