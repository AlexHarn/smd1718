from fisher import Fisher
import numpy as np
import h5py
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5.8
rcParams['legend.numpoints'] = 1


# ----------------------------------- Main ------------------------------------
if __name__ == '__main__':
    # Daten laden (Aufgabe 12)
    fh = h5py.File("pops.hdf5", "r")
    p0 = np.copy(fh["p0"])
    p0_1000 = np.copy(fh["p_0_1000"])
    p1 = np.copy(fh["p1"])
    fh.close()

    # Instanz unserer Fisher Klasse erstellen
    # Führt Berechnungen für a) bis c) durch
    fisher = Fisher(p0, p1)
    # fisher.show()
    print("Mittelwerte für p_0_10000:", fisher.m)
    print("kombinierte Kovarianzmatrix für p_0_10000:", fisher.sw)

    # c)
    print("Geradengleichung der Projektion für P_0_10000: g(x) = %1.4f*x"
          % (fisher.lam[1]/fisher.lam[0]))

    # d)
    fisher.showHist(save="A16d_10000.pdf")

    # e)
    # Schnitte berechnen
    fisher.cut()
    # und anzeigen bzw. speichern
    fisher.showCuts(save="A16e_10000.pdf")

    # f)
    fisher.showSignalToBackground(save="A16f_10000.pdf")
    # g)
    fisher.showSignificance(save="A16g_10000.pdf")

    # h)
    fisher.p[0] = p0_1000
    fisher.calc()
    print("Mittelwerte für p_0_1000:", fisher.m)
    print("kombinierte Kovarianzmatrix für p_0_1000:", fisher.sw)
    print("Geradengleichung der Projektion für P_0_1000: g(x) = %1.4f*x"
          % (fisher.lam[1]/fisher.lam[0]))
    fisher.showHist(save="A16d_1000.pdf")
    fisher.cut()
    fisher.showCuts(save="A16e_1000.pdf")
    fisher.showSignalToBackground(save="A16f_1000.pdf")
    fisher.showSignificance(save="A16g_1000.pdf")
