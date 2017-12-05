import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5.8
rcParams['legend.numpoints'] = 1


def infgain(data, target, cut):
    cut = np.atleast_1d(cut)
    p_ytrue = np.count_nonzero(target == True) / len(target)
    p_yfalse = 1 - p_ytrue
    H_y = -p_ytrue * np.log2(p_ytrue) - p_yfalse * np.log2(p_yfalse)

    p_xhigh = np.zeros(len(cut))
    p_xlow = np.zeros(len(cut))
    p_xhigh_ytrue = np.zeros(len(cut))
    p_xhigh_yfalse = np.zeros(len(cut))
    p_xlow_ytrue = np.zeros(len(cut))
    p_xlow_yfalse = np.zeros(len(cut))
    H_y_x = np.zeros(len(cut))
    for i in range(len(cut)):
        p_xhigh[i] = np.count_nonzero(data > cut[i]) / len(data)
        p_xlow[i] = 1 - p_xhigh[i]
        p_xhigh_ytrue[i] = np.count_nonzero((data >= cut[i]) & (target == True)) / len(data)
        p_xhigh_yfalse[i] = np.count_nonzero((data >= cut[i]) & (target == False)) / len(data)
        p_xlow_ytrue[i] = np.count_nonzero((data < cut[i]) & (target == True)) / len(data)
        p_xlow_yfalse[i] = np.count_nonzero((data < cut[i]) & (target == False)) / len(data)

        H_y_x[i] = -p_xhigh[i]*((0 if p_xhigh_ytrue[i] == 0 else p_xhigh_ytrue[i]*np.log2(p_xhigh_ytrue[i])) +
                          (0 if p_xhigh_yfalse[i] == 0 else p_xhigh_yfalse[i]*np.log2(p_xhigh_yfalse[i]))) \
                -p_xlow[i]*((0 if p_xlow_ytrue[i] == 0 else p_xlow_ytrue[i]*np.log2(p_xlow_ytrue[i])) +
                         (0 if p_xlow_yfalse[i] == 0 else p_xlow_yfalse[i]*np.log2(p_xlow_yfalse[i])))

    return H_y - H_y_x

if __name__ == "__main__":
    data = pd.read_csv("data.csv")

    # Temperatur
    T = np.linspace(18, 30, 24)
    plt.plot(T, infgain(data["Temperatur"], data["Fußball"], T), 'rx')
    plt.xlabel(r"Cut$_{\rm{Temperatur}}\,$/ °C")
    plt.ylabel("Informationsgewinn")
    plt.xlim(T[0], T[-1])
    plt.savefig("Temperatur.pdf")

    # Wettervorhersage
    forecast = np.linspace(-0.5, 3, 4)
    plt.plot(forecast, infgain(data["Wettervorhersage"], data["Fußball"], forecast), 'rx')
    plt.xlabel(r"Cut$_{\rm{Wettervorhersage}}$")
    plt.ylabel("Informationsgewinn")
    plt.xlim(forecast[0], forecast[-1])
    plt.savefig("Wettervorhersage.pdf")

    # Luftfeuchtigkeit
    air = np.linspace(62.5, 100, 8)
    plt.plot(air, infgain(data["Luftfeuchtigkeit"], data["Fußball"], air), 'rx')
    plt.xlabel(r"Cut$_{\rm{Luftfeuchtigkeit}}$")
    plt.ylabel("infgain")                                    
    plt.xlim(air[0], air[-1])
    plt.savefig("Luftfeuchtigkeit.pdf")