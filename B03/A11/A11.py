import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Size = 10**4
BackgroundSize = 10**5
# Phi_0 = 1


# Signal erstellen

print("Signaldaten erstellen")
# 1e5 gleichverteilte Zufahlszahlen zwischen u(Emin = 0) und u(Emax = inf)
# "-10*epsilon", damit auf keinen Fall bei der Berechnung von E der Fall 1/0 auftritt
u = np.random.uniform(0, 1 / 1.7 - 10 * np.finfo(np.float64).eps, size = Size)
signal = pd.DataFrame({
    "Energy": (1 - 1.7 * u)**(-1 / 1.7)
})

# Akzeptanz

print("Akzeptanz berechnen")
# Gleichverteilte Zahlen zwischen 0 und 1
r = np.random.uniform(0, 1, size = Size)
# Detektionswahrscheinlichkeit
P = (1 - np.exp(-1 * signal["Energy"]/2))**3
# Test ob r < P(E) ist
signal = signal.assign(AcceptanceMask=np.less(r,P))
print("Anteil der akzeptierten Werte:", len(signal["AcceptanceMask"][signal["AcceptanceMask"]])/Size)

print("Signaldaten und Akzeptanz plotten")
plt.yscale("log")
plt.xscale("log")
plt.hist(signal["Energy"], bins=np.logspace(0, np.log10(np.amax(signal["Energy"])), 30), label="Erzeugte Signale")
# signal["Energy"][signal["AcceptanceMask"]] sind die signal["Energy"] Werte, bei denen signal["AcceptanceMask"] == True ist
plt.hist(signal["Energy"][signal["AcceptanceMask"]], bins=np.logspace(0, np.log10(np.amax(signal["Energy"])), 30), label="Akzeptierte Signale")
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
NumberOfHits = np.zeros(Size)
i = 0
while(i < Size):
    if(~signal["AcceptanceMask"][i]):
        i += 1
        continue
    NumberOfHits[i] = np.round(Normalverteilung(2*signal["Energy"][i], 10*signal["Energy"][i], low=0))
    i += 1

"""   Alte Methode, wird nicht mehr verwendet.
NumberOfHits = np.zeros(Size)
i = 0
while(i < Size):
    if(~signal["AcceptanceMask"][i]):
        i += 1
        continue
    v1 = np.random.uniform(-1, 1)
    v2 = np.random.uniform(-1, 1)
    s = v1**2 + v2**2
    if(s >= 1):
        continue
    x1 = np.round(2*signal["Energy"][i]*v1*np.sqrt(-2*np.log(s)/s) + 10*signal["Energy"][i])
    # x1 = np.round(2*signal["Energy"][i]*np.sqrt(s)*np.cos(np.arctan(v2/v1))*np.sqrt(-2*np.log(s)/s) + 10*signal["Energy"][i])
    if(x1 < 0):
        x2 = np.round(2*signal["Energy"][i]*v2*np.sqrt(-2*np.log(s)/s) + 10*signal["Energy"][i])
        # x2 = np.round(2*signal["Energy"][i]*np.sqrt(s)*np.sin(np.arctan(v2/v1))*np.sqrt(-2*np.log(s)/s) + 10*signal["Energy"][i])
        if(x2 < 0):
            continue
        NumberOfHits[i] = x2
        i += 1
        continue
    NumberOfHits[i] = x1
    i += 1
"""

signal = signal.assign(NumberOfHits=NumberOfHits)

print("Hits plotten")
plt.clf()
plt.yscale("log")
plt.xscale("log")
plt.hist(signal["NumberOfHits"][signal["AcceptanceMask"]], bins=np.logspace(0, np.log10(np.amax(signal["NumberOfHits"])), 30))
plt.xlabel("Hits")
plt.ylabel("Anzahl")
plt.savefig("Hits.pdf")

# Ortsmessung

print("Orte berechnen")
x = np.zeros(Size)
y = np.zeros(Size)
i = 0
while(i < Size):
    if(~signal["AcceptanceMask"][i]):
        i += 1
        continue
    sigma = 1/(np.log10(signal["NumberOfHits"][i] + 1))
    x[i] = Normalverteilung(sigma, 7, low=0, high=10)
    y[i] = Normalverteilung(sigma, 3, low=0, high=10)
    i += 1

signal = signal.assign(x=x, y=y)

print("Orte plotten")
plt.clf()
heatmap, xedges, yedges = np.histogram2d(signal["x"][signal["AcceptanceMask"]], signal["y"][signal["AcceptanceMask"]], bins=50, range = [[0, 10], [0, 10]])
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.imshow(heatmap.T, extent = extent, origin='lower', cmap="Greens")
plt.colorbar(label="Häufigkeit")
plt.xlabel(r"$x$ / Längeneinheiten")
plt.ylabel(r"$y$ / Längeneinheiten")
plt.plot(7, 3, "rx", ms=10)
plt.savefig("Orte.pdf")

signal.to_hdf("NeutrinoMC.hdf5", key="Signal")

# Background

print("Background Daten erstellen")
NumberOfHits = np.zeros(BackgroundSize)
x = np.zeros(BackgroundSize)
y = np.zeros(BackgroundSize)
for i in range(BackgroundSize):
    NumberOfHits[i] = 10**Normalverteilung(1, 2)
    while(True):
        x[i] = Normalverteilung(1, 0)
        y[i] = Normalverteilung(1, 0)
        x[i] = np.sqrt(1-0.25)*3*x[i] + 0.5*3*y[i] + 5
        y[i] = 3*y[i] + 5
        if((x[i] < 10) & (x[i] > 0) & (y[i] < 10) & (y[i] > 0)):
            break
    if(i/BackgroundSize*100 % 5 == 0):
        print(i/BackgroundSize*100, "%")
background = pd.DataFrame({
    "NumberOfHits": NumberOfHits,
    "x": x,
    "y": y,
})

print("Background Orte plotten")
plt.clf()
heatmap, xedges, yedges = np.histogram2d(background["x"], background["y"], bins=50, range = [[0, 10], [0, 10]])
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.imshow(heatmap.T, extent = extent, origin='lower', cmap="Greens")
plt.colorbar(label="Häufigkeit")
plt.xlabel(r"$x$ / Längeneinheiten")
plt.ylabel(r"$y$ / Längeneinheiten")
plt.savefig("OrteBackground.pdf")

print("Background Hits plotten")
plt.clf()
# plt.yscale("log")
# plt.xscale("log")
plt.hist(np.log10(background["NumberOfHits"]), bins=50)
plt.xlabel("Hits")
plt.ylabel("Anzahl")
plt.savefig("HitsBackground.pdf")

background.to_hdf("NeutrinoMC.hdf5", key="Background")