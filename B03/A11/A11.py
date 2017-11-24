import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5.8
rcParams['legend.numpoints'] = 1

size = 10**5
backgroundSize = 10**7
# Phi_0 = 1


def aufg11():
    # Signal erstellen
    print("Signaldaten erstellen")
    # 1e5 gleichverteilte Zufahlszahlen zwischen u(Emin = 0) und u(Emax = inf)
    # "-10*epsilon", damit auf keinen Fall bei der Berechnung von
    # E der Fall 1/0 auftritt
    u = np.random.uniform(0, 1 / 1.7 - 10 * np.finfo(np.float64).eps,
                          size=size)
    signal = pd.DataFrame({"Energy": (1 - 1.7*u)**(-1/1.7)})

    # Akzeptanz
    print("Akzeptanz berechnen")
    # Gleichverteilte Zahlen zwischen 0 und 1
    r = np.random.uniform(0, 1, size=size)
    # Detektionswahrscheinlichkeit
    P = (1 - np.exp(-1 * signal["Energy"]/2))**3
    # Test ob r < P(E) ist
    signal = signal.assign(AcceptanceMask=np.less(r, P))
    print("Anteil der akzeptierten Werte:",
          len(signal["AcceptanceMask"][signal["AcceptanceMask"]])/size)

    print("Signaldaten und Akzeptanz plotten")
    plt.yscale('log', nonposy='clip')
    plt.xscale("log")
    plt.hist(signal["Energy"],
             bins=np.logspace(0, np.log10(np.amax(signal["Energy"])), 50),
             label="Erzeugte Signale")
    # signal["Energy"][signal["AcceptanceMask"]] sind die signal["Energy"]
    # Werte, bei denen signal["AcceptanceMask"] == True ist
    plt.hist(signal["Energy"][signal["AcceptanceMask"]],
             bins=np.logspace(0, np.log10(np.amax(signal["Energy"])), 50),
             label="Akzeptierte Signale")
    plt.legend(loc="best")
    plt.xlabel("Energie / TeV")
    plt.ylabel("Anzahl")
    plt.savefig("Energie.pdf")

    # Energiemessung
    def Normalverteilung(sigma, mu, low=-np.inf, high=np.inf):
        while(True):
            v1 = np.random.uniform(-1, 1)
            v2 = np.random.uniform(-1, 1)
            s = v1**2 + v2**2
            if(s >= 1):
                continue
            x1 = sigma*v1*np.sqrt(-2*np.log(s)/s) + mu
            if((x1 < low) | (x1 > high)):
                x2 = sigma*v2*np.sqrt(-2*np.log(s)/s) + mu
                if((x2 < low) | (x2 > high)):
                    continue
                return x2
            return x1

    print("Anzahl der Hits berechnen")
    numberOfHits = np.zeros(size)
    i = 0
    while(i < size):
        if(~signal["AcceptanceMask"][i]):
            i += 1
            continue
        numberOfHits[i] = np.round(Normalverteilung(2*signal["Energy"][i],
                                                    10*signal["Energy"][i],
                                                    low=0))
        i += 1

    signal = signal.assign(numberOfHits=numberOfHits)

    print("Hits plotten")
    plt.clf()
    plt.yscale('log', nonposy='clip')
    plt.xscale("log")
    plt.hist(signal["numberOfHits"][signal["AcceptanceMask"]],
             bins=np.logspace(0, np.log10(np.amax(signal["numberOfHits"])),
                              50))
    plt.xlabel("Hits")
    plt.ylabel("Anzahl")
    plt.savefig("Hits.pdf")

    # Ortsmessung
    print("Orte berechnen")
    x = np.zeros(size)
    y = np.zeros(size)
    i = 0
    while(i < size):
        if(~signal["AcceptanceMask"][i]):
            i += 1
            continue
        sigma = 1/(np.log10(signal["numberOfHits"][i] + 1))
        x[i] = Normalverteilung(sigma, 7, low=0, high=10)
        y[i] = Normalverteilung(sigma, 3, low=0, high=10)
        i += 1

    signal = signal.assign(x=x, y=y)

    print("Orte plotten")
    plt.clf()
    heatmap, xedges, yedges = \
        np.histogram2d(signal["x"][signal["AcceptanceMask"]],
                       signal["y"][signal["AcceptanceMask"]],
                       bins=50, range=[[0, 10], [0, 10]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap="Greens")
    plt.colorbar(label="Häufigkeit")
    plt.xlabel(r"$x$ / Längeneinheiten")
    plt.ylabel(r"$y$ / Längeneinheiten")
    plt.plot(7, 3, "rx", ms=10)
    plt.savefig("Orte.pdf", bbox_inches='tight')

    signal.to_hdf("NeutrinoMC.hdf5", key="Signal")

    # Background
    print("Background Daten erstellen...")
    os.system('setterm -cursor off')
    # numberOfHits = np.zeros(backgroundSize)
    # x = np.zeros(backgroundSize)
    # y = np.zeros(backgroundSize)
    """for i in range(backgroundSize):
        numberOfHits[i] = 10**Normalverteilung(1, 2)
        while(True):
            x[i] = Normalverteilung(1, 0)
            y[i] = Normalverteilung(1, 0)
            x[i] = np.sqrt(1-0.25)*3*x[i] + 0.5*3*y[i] + 5
            y[i] = 3*y[i] + 5
            if((x[i] < 10) & (x[i] > 0) & (y[i] < 10) & (y[i] > 0)):
                break
        if(i/backgroundSize*100 % 5 == 0):
            print(i/backgroundSize*100, "%   ", end="\r")
    """
    numberOfHits = np.fromiter((10**Normalverteilung(1, 2)
                                for i in range(0, backgroundSize)),
                               numberOfHits.dtype, count=backgroundSize)

    x = np.fromiter((Normalverteilung(1, 0) for i in range(0, backgroundSize)),
                    x.dtype, count=backgroundSize)
    y = np.fromiter((Normalverteilung(1, 0) for i in range(0, backgroundSize)),
                    y.dtype, count=backgroundSize)
    x = np.sqrt(1 - 0.25)*3*x + 0.5*3*y + 5
    y = 3*y + 5
    idx = np.concatenate((np.where(x < 0)[0], np.where(x > 10)[0],
                         np.where(y < 0)[0], np.where(y > 10)[0]))

    while(len(idx) > 0):
        print(len(idx), "Background Koordinaten sind noch ungültig...     ",
              end="\r")
        x[idx] = np.fromiter((Normalverteilung(1, 0)
                              for i in range(0, backgroundSize)),
                             x.dtype, count=len(idx))
        y[idx] = np.fromiter((Normalverteilung(1, 0)
                              for i in range(0, backgroundSize)),
                             x.dtype, count=len(idx))
        x[idx] = np.sqrt(1 - 0.25)*3*x[idx] + 0.5*3*y[idx] + 5
        y[idx] = 3*y[idx] + 5
        idx = np.concatenate((np.where(x < 0)[0], np.where(x > 10)[0],
                             np.where(y < 0)[0], np.where(y > 10)[0]))

    print("0 Background Koordinaten sind noch ungültig     ")
    os.system('setterm -cursor on')
    background = pd.DataFrame({
        "numberOfHits": numberOfHits,
        "x": x,
        "y": y,
    })

    # print("100 %  \nBackground Orte plotten")
    plt.clf()
    heatmap, xedges, yedges = np.histogram2d(background["x"], background["y"],
                                             bins=50,
                                             range=[[0, 10], [0, 10]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap="Greens")
    plt.colorbar(label="Häufigkeit")
    plt.xlabel(r"$x$ / Längeneinheiten")
    plt.ylabel(r"$y$ / Längeneinheiten")
    plt.savefig("OrteBackground.pdf", bbox_inches='tight')

    print("Background Hits plotten")
    plt.clf()
    # plt.yscale("log")
    # plt.xscale("log")
    plt.hist(np.log10(background["numberOfHits"]), bins=50)
    plt.xlabel("Hits")
    plt.ylabel("Anzahl")
    plt.savefig("HitsBackground.pdf")

    background.to_hdf("NeutrinoMC.hdf5", key="Background")


# ----------------------------------- Main ------------------------------------
if __name__ == '__main__':
    np.random.seed(1234)
    aufg11()
