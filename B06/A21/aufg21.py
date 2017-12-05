import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def infgain(data, target, cut):
    p_ytrue = np.count_nonzero(target == True) / len(target)
    p_yfalse = 1 - p_ytrue
    H_y = -p_ytrue * np.log2(p_ytrue) - p_yfalse * np.log2(p_yfalse)

    p_xhigh = np.count_nonzero(data > cut) / len(data)
    p_xlow = 1 - p_xhigh
    p_xhigh_ytrue = np.count_nonzero((data >= cut) & (target == True)) / len(data)
    p_xhigh_yfalse = np.count_nonzero((data >= cut) & (target == False)) / len(data)
    p_xlow_ytrue = np.count_nonzero((data < cut) & (target == True)) / len(data)
    p_xlow_yfalse = np.count_nonzero((data < cut) & (target == False)) / len(data)

    # a*log(a) -> 0 für a -> 0, numpy liefert sonst NaN
    if p_xhigh_ytrue == 0:
        p1 = 0                                                 f
    else:
        p1 = p_xhigh_ytrue * np.log2(p_xhigh_ytrue)
    if p_xhigh_yfalse == 0:
        p2 = 0
    else:
        p2 = p_xhigh_yfalse * np.log2(p_xhigh_yfalse)
    if p_xlow_ytrue == 0:
        p3 = 0
    else:
        p3 = p_xlow_ytrue * np.log2(p_xlow_ytrue)
    if p_xlow_yfalse == 0:
        p4 = 0
    else:
        p4 = p_xlow_yfalse * np.log2(p_xlow_yfalse)

    H_y_x = -p_xhigh * (p1 + p2) - p_xlow * (p3 + p4)
    #H_y_x = -p_xhigh*(p_xhigh_ytrue*np.log2(p_xhigh_ytrue) +
    #                  p_xhigh_yfalse*np.log2(p_xhigh_yfalse)) \
    #        -p_xlow*(p_xlow_ytrue*np.log2(p_xlow_ytrue) +
    #                 p_xlow_yfalse*np.log2(p_xlow_yfalse))

    return H_y - H_y_x

if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    #print(data)
    # print(infgain(data["Temperatur"], data["Fußball"], 20))

    # Temperatur
    T = np.linspace(18, 30, 24)
    infT = np.zeros(len(T))
    for i in range(len(infT)):
        infT[i] = infgain(data["Temperatur"], data["Fußball"], T[i])
    plt.plot(T, infT, 'rx')
    plt.xlabel("Cut_{Temperatur}$\,$/ °C")
    plt.ylabel("infgain")
    plt.xlim(T[0], T[-1])
    plt.savefig("Temperatur.pdf")

    # Wettervorhersage
    forecast = np.linspace(-0.5, 3, 4)
    infForecast = np.zeros(len(forecast))
    for i in range(len(infForecast)):
        infForecast[i] = infgain(data["Wettervorhersage"], data["Fußball"], forecast[i])
    plt.plot(forecast, infForecast, 'rx')
    plt.xlabel("Cut_{Wettervorhersage}")
    plt.ylabel("infgain")
    plt.xlim(forecast[0], forecast[-1])
    plt.savefig("Wettervorhersage.pdf")

    # Luftfeuchtigkeit
    air = np.linspace(62.5, 100, 8)
    infAir = np.zeros(len(air))
    for i in range(len(infAir)):
        infAir[i] = infgain(data["Luftfeuchtigkeit"], data["Fußball"], air[i])
    plt.plot(air, infAir, 'rx')
    plt.xlabel("Cut_{Luftfeuchtigkeit}")
    plt.ylabel("infgain")                                    
    plt.xlim(air[0], air[-1])
    plt.savefig("Luftfeuchtigkeit.pdf")