import matplotlib.pyplot as plt
import numpy as np


# -------------------------------- Funktionen ---------------------------------
def horner(x, a):
    """Hornerschema für Polynom mit Koeffizientenvektor a"""
    f = a[0]
    # Lässt sich diese Schleife irgendwie schlau durch numpy Befehle ersetzen?
    for i in range(1, len(a)):
        f = f*x+a[i]
    return f


def f1(x):
    """Das Polynom in faktorisierter Form"""
    return (1-x)**6


def f2(x):
    """Das Polynom in ausmulitplizierter Form"""
    return x**6 - 6*x**5 + 15*x**4 - 20*x**3 + 15*x**2 - 6*x + 1


def aufg1():
    x16 = np.linspace(0.999, 1.001, 1000, dtype='float16')
    x32 = np.linspace(0.999, 1.001, 1000, dtype='float32')
    x64 = np.linspace(0.999, 1.001, 1000, dtype='float64')

    # Koeffizientenvektor für Horner
    a = ([1, -6, 15, -20, 15, -6, 1])

    f, axarr = plt.subplots(3, 3)  # , sharex='col', sharey='row'
    axarr[0][0].plot(x16, f1(x16))
    axarr[0][1].plot(x32, f1(x32))
    axarr[0][2].plot(x64, f1(x64))
    axarr[1][0].plot(x16, f2(x16))
    axarr[1][1].plot(x32, f2(x32))
    axarr[1][2].plot(x64, f2(x64))
    axarr[2][0].plot(x16, horner(x16, a))
    axarr[2][1].plot(x32, horner(x32, a))
    axarr[2][2].plot(x64, horner(x64, a))

    f.tight_layout()
    f.set_figwidth(25)
    f.set_figheight(15)
    plt.setp(axarr, xlim=[0.999, 1.001], xticks=[0.999, 1.000, 1.001])
    # , ylim=[-0.5e-18,1.5e-18]
    f.subplots_adjust(wspace=0.3)
    f.subplots_adjust(hspace=0.2)
    axarr[1][0].set_ylim([-0.004, 0.004])
    cols = ['16-Bit', '32-Bit', '64-Bit']
    rows = ['a) direkt', ' b) ausmultipliziert', 'c) Horner']

    for ax, col in zip(axarr[0], cols):
        ax.set_title(col)

    for ax, row in zip(axarr[:, 0], rows):
        ax.set_ylabel(row, rotation=90, size='large')
    plt.savefig('A1.pdf')
    plt.clf()


# ----------------------------------- Main ------------------------------------
if __name__ == '__main__':
    aufg1()
