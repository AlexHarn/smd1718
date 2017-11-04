import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ------------------------------------ a) -------------------------------------
def findPeriod(a, x0=1):
    """Findet die Periodenlänge des Linear-Kongruenten-Generators aus Aufgabe
    9a) in Abhängigkeit von a (und x0)
    """

    numbers = [x0]
    while not np.isin([numbers[-1]], numbers[:-1]):
        numbers.append((a*numbers[-1] + 3) % 1024)

    numbers = np.asarray(numbers)
    first = np.where(numbers[:-1] == numbers[-1])
    return len(numbers) - first[0][0] - 1


def aufg9a():
    a = np.arange(0, 100)
    plt.plot(a, np.vectorize(findPeriod)(a), ".")
    plt.show()
    plt.clf()


# ------------------------------------ b) -------------------------------------
def uniform_b(x0, n):
    """Liefert ein Array mit den ersten n Zufallszahlen des Generators aus
    Aufgabenteil b)
    """
    numbers = [x0]
    while len(numbers) < n + 1:
        numbers.append((1601*numbers[-1] + 3456) % 1e4)
    return np.asarray(numbers[1:])/1e4


# ------------------------------------ c) -------------------------------------
def aufg9c():
    x0 = [1, 10, 223, 5412, 2812, 899, 1e4 - 39]
    for x in x0:
        plt.hist(uniform_b(x, 1e4), bins=100, normed=True)
        plt.show()
        plt.clf()


# ------------------------------------ d) -------------------------------------
def aufg9d():
    # numbers = uniform_b(1, 1e4).reshape(5000, 2).T
    # plt.scatter(numbers[0], numbers[1], marker=".")
    # plt.show()
    # plt.clf()

    numbers = uniform_b(1, 9999).reshape(3333, 3).T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(numbers[0], numbers[1], numbers[2], marker=".")
    ax.view_init(elev=10, azim=20)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    # plt.show()
    plt.savefig("A9d.pdf")


# ------------------------------------ e) -------------------------------------
def aufg9e():
    np.random.seed(1)
    numbers = np.random.uniform(size=10000)

    # numbers_tuples = numbers.reshape(5000, 2).T
    # plt.scatter(numbers_tuples[0], numbers_tuples[1], marker=".")
    # plt.show()
    # plt.clf()

    numbers_triplets = numbers[:-1].reshape(3333, 3).T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(numbers_triplets[0], numbers_triplets[1], numbers_triplets[2],
               marker=".")
    ax.view_init(elev=10, azim=20)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    # plt.show()
    plt.savefig("A9e.pdf")


# ------------------------------------ f) -------------------------------------
def aufg9f():
    count = []
    for x in range(0, 200):
        numbers = uniform_b(x, 1e4)
        count.append(np.count_nonzero(numbers == 0.5))
    count = np.asarray(count)
    print(np.where(count > 0)[0]/8)


# ----------------------------------- Main ------------------------------------
if __name__ == '__main__':
    # aufg9a()
    # aufg9c()
    # aufg9d()
    # aufg9e()
    # Scheinbar liefern alle Startwerte, die ungerade Vielfache von 8 sind
    # 16 mal den Wert 0.5
    aufg9f()
