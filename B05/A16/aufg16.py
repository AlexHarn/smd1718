from fisher import Fisher
import numpy as np
import h5py
import matplotlib.pyplot as plt
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
    fisher = Fisher(p0, p1)
    fisher.show()
