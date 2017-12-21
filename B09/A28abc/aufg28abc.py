import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5.8
rcParams['legend.numpoints'] = 1


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
cols = df_train.corr().nlargest(4, "SalePrice")["SalePrice"].index[1:]
print("Attribute mit der höchsten Korrelation in absteigender Reihenfolge:")
print(cols.values)

for feature in cols:
    a, b, _, _, _ = linregress(df_train[feature], df_train["SalePrice"])
    x = np.array([np.min(df_train[feature])-1, np.max(df_train[feature]) + 1])
    plt.scatter(df_train[feature], df_train["SalePrice"], marker=".",
                label="Datenpunkte")
    plt.plot(x, a*x + b, "r-", label="Lineare Regression")
    plt.xlabel(feature)
    plt.xlim(x)
    plt.ylabel("SalePrice")
    plt.legend()
    # plt.show()
    plt.savefig("Scatter_{}.pdf".format((feature)))
    plt.clf()

    y_pred = a*df_train[feature] + b
    plt.hist(y_pred - df_train["SalePrice"], bins=25)
    plt.xlabel("Vorhersage - Verkaufspreis")
    plt.ylabel("Häufigkeit")
    # plt.show()
    plt.savefig("Hist_{}.pdf".format((feature)))
    plt.clf()
