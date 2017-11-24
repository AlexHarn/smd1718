import numpy as np
import matplotlib.pyplot as plt


class Fisher(object):
    def __init__(self, signal=None, background=None):
        """
        Konstruktor
        """
        self.p = [signal, background]
        if p0 is not None and p1 is not None:
            self.calc()

    def calc(self):
        """
        Berechnet alles
        """
        # Die Variablennamen halten sich ans Skript
        # Streuung
        self.sw = np.matrix(np.add(np.cov(self.p[0])*len(self.p[0] - 1),
                                   np.cov(self.p[1])*len(self.p[1] - 1)))

        # Mittelwerte
        self.m = np.matrix([[np.average(self.p[0][0]),
                            np.average(self.p[0][1])],
                           [np.average(self.p[1][0]),
                            np.average(self.p[1][1])]])

        # Fisher Schnitt berechnen
        self.lam = np.array((self.sw.I*(self.m[0] - self.m[1]).T).T)[0]
        print(self.lam)

    def show(self, save=False):
        """
        Zeigt an!
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.p[0][0], self.p[0][1], marker=".", s=2.5,
                   alpha=0.5, label=r"$p_0$")
        ax.scatter(self.p[1][0], self.p[1][1], marker=".", s=2.5,
                   alpha=0.5, label=r"$p_1$")
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 1000)
        ax.plot(x, self.lam[1]/self.lam[0]*x)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.legend()
        if save:
            fig.savefig(save)
        else:
            plt.show()
