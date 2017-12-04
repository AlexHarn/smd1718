import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5.8
rcParams['legend.numpoints'] = 1


def doEverything(df, fname):
    train, test = train_test_split(df, test_size=0.1)

    if os.path.isfile("{}.pkl".format(fname)):
        rf = joblib.load("{}.pkl".format(fname))
    else:
        rf = RandomForestRegressor(n_estimators=200)
        rf.fit(train.drop(columns=["x3"]), train["x3"])
        joblib.dump(rf, "{}.pkl".format(fname))
    x3_predict = rf.predict(test.drop(columns=["x3"]))
    print("MSE =", mean_squared_error(test["x3"], x3_predict))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(test["x1"], test["x2"], test["x3"], marker=".", s=0.5,
               label="Echte Daten")
    ax.scatter(test["x1"], test["x2"], x3_predict, marker=".", s=0.5,
               label="Regression")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    plt.legend()
    fig.savefig("A20_3D_{}.pdf".format(fname))
    # plt.show()
    plt.clf()

    plt.scatter(test["x1"], test["x3"], marker=".", s=0.5, label="Echte Daten")
    plt.scatter(test["x1"], x3_predict, marker=".", s=0.5, label="Regression")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_3$")
    plt.legend()
    # plt.show()
    plt.savefig("A20_x1_x3_{}.pdf".format(fname))
    plt.clf()

    plt.scatter(test["x2"], test["x3"], marker=".", s=0.5, label="Echte Daten")
    plt.scatter(test["x2"], x3_predict, marker=".", s=0.5, label="Regression")
    plt.xlabel("$x_2$")
    plt.ylabel("$x_3$")
    plt.legend()
    # plt.show()
    plt.savefig("A20_x2_x3_{}.pdf".format(fname))
    plt.clf()

    plt.scatter(test["x3"], x3_predict, marker=".", s=0.5)
    plt.xlabel("$x_3$")
    plt.ylabel("$\hat{x}_3$")
    # plt.show()
    plt.savefig("A20_x3_x3_{}.pdf".format(fname))
    plt.clf()


# ----------------------------------- Main ------------------------------------
if __name__ == '__main__':
    np.random.seed(1234)

    print("Teilaufgabe e)")
    df = pd.DataFrame({"x1": np.random.uniform(size=100000),
                       "x2": np.random.uniform(size=100000)})
    df["x3"] = 15*np.sin(4*np.pi*df["x1"]) + 50*(df["x2"] - 0.5)**2 + \
        np.random.normal(size=100000)
    doEverything(df, "e")

    print("Teilaufgabe f)")
    df = pd.DataFrame({"x1": np.random.uniform(1, 2, size=100000),
                       "x2": np.random.uniform(1, 2, size=100000)})
    df["x3"] = 15*np.sin(4*np.pi*df["x1"]) + 50*(df["x2"] - 0.5)**2 + \
        np.random.normal(size=100000)
    doEverything(df, "f")
