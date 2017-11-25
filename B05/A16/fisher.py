import numpy as np
import matplotlib.pyplot as plt


class Fisher(object):
    def __init__(self, signal=None, background=None):
        """
        Konstruktor
        """
        self.p = [signal, background]
        self._ls = None
        self._proj = None
        if signal is not None and background is not None:
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

        # Projektion durchführen
        self._proj = []
        self._proj.append(np.add(np.multiply(self.p[0][0], self.lam[0]),
                                 np.multiply(self.p[0][1], self.lam[1])))
        self._proj.append(np.add(np.multiply(self.p[1][0], self.lam[0]),
                                 np.multiply(self.p[1][1], self.lam[1])))
        if np.average(self._proj[0]) < np.average(self._proj[1]):
            self._proj = -self._proj
            self.lam = -self.lam

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

    def showHist(self, bins=25, save=False):
        """
        Zeigt das Histogramm der Projektion an
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.hist(self._proj[0], bins=bins, label="$p_0$")
        ax.hist(self._proj[1], bins=bins, label="$p_1$",
                alpha=0.6)
        ax.set_xlabel("$x_\lambda$")
        ax.set_ylabel("$n$")
        ax.legend()
        if save:
            fig.savefig(save)
        else:
            plt.show()

    def cut(self, nl=1000):
        p = [np.sort(self._proj[0])[::-1], np.sort(self._proj[1])[::-1]]
        self._ls = np.linspace(max(p[0][0], p[1][0]),
                               min(p[0][-1], p[1][-1]), nl)
        self.tp = np.zeros(nl)
        self.fp = np.zeros(nl)
        self.fn = np.zeros(nl)
        self.tn = np.zeros(nl)
        j = 0
        k = 0
        for n, l in enumerate(self._ls):
            while (j < len(p[0])) and (p[0][j] >= l):
                j += 1
            while k < len(p[1]) and p[1][k] >= l:
                k += 1
            self.tp[n] = j
            self.fn[n] = len(p[0]) - j
            self.fp[n] = k
            self.tn[n] = len(p[1]) - k

    def showCuts(self, save=False):
        if self._ls is None:
            raise ValueError("Die Schnitte müssen zuerst durchgeführt werden!")

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(self._ls, self.tp/(self.tp + self.fp), label="Reinheit")
        ax.plot(self._ls, self.tp/(self.tp + self.fn), label="Effizienz")
        ax.plot(self._ls, (self.tp + self.tn) /
                (self.tp + self.tn + self.fp + self.fn), label="Genauigkeit")
        ax.set_xlim(self._ls[-1], self._ls[0])
        ax.legend()
        ax.set_xlabel("$\lambda_\mathrm{cut}$")
        if save:
            fig.savefig(save)
        else:
            plt.show()

    def showSignificance(self, save=False):
        if self._ls is None:
            raise ValueError("Die Schnitte müssen zuerst durchgeführt werden!")

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(self._ls, (self.tp + self.fp) /
                np.sqrt(len(self._proj[0]) + len(self._proj[1])))
        ax.set_xlim(self._ls[-1], self._ls[0])
        ax.legend()
        ax.set_xlabel("$\lambda_\mathrm{cut}$")
        ax.set_ylabel(r"$\frac{S}{\sqrt{S + B}}$")
        if save:
            fig.savefig(save)
        else:
            plt.show()

    def showSignalToBackground(self, save=False):
        if self._ls is None:
            raise ValueError("Die Schnitte müssen zuerst durchgeführt werden!")

        fig = plt.figure()
        ax = fig.add_subplot(111)

        with np.errstate(divide='ignore'):
            ax.plot(self._ls, (self.tp + self.fp)/(self.tn + self.fn))

        ax.set_xlim(self._ls[-1], self._ls[0])
        ax.legend()
        ax.set_xlabel("$\lambda_\mathrm{cut}$")
        ax.set_ylabel(r"$\frac{S}{B}$")
        if save:
            fig.savefig(save)
        else:
            plt.show()
