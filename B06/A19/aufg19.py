import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import cdist
# -------------------------------- Funktionen ---------------------------------


def kNN(x_train, y_train, x_test, k):
    """
    Implementierung des kNN Lerners.

    Parameters
    ----------
    x_train: Trainingsdatensample
    y_train: Labels der Trainingsdaten
    x_test: zu klassifizierende Daten

    Returns
    -------
    Die Labels der zu klassifizierenden Daten
    """
    if k > len(x_train):
        raise ValueError("k kann nicht so groß sein")

    idx = np.argsort(cdist(x_test, x_train), axis=1)
    y_test = np.empty(len(x_test))

    for i in range(len(x_test)):
        (values, counts) = np.unique(y_train[idx[i]][:k], return_counts=True)
        y_test[i] = values[np.argmax(counts)]
    return y_test


def evaluate(x_train, y_train, x_test, y_test, k, fsave=None):
    if fsave is not None:
        if not os.path.isfile(fsave):
            y_test_knn = kNN(x_train, y_train, x_test, k)
            np.save(fsave, y_test_knn)
        else:
            y_test_knn = np.load(fsave)
    else:
        y_test_knn = kNN(x_train, y_train, x_test, k)

    tp = np.count_nonzero(y_test_knn[:len(signal.index) - 5000])
    fn = len(signal.index) - 5000 - tp
    fp = np.count_nonzero(y_test_knn[len(signal.index) - 5000:])
    tn = len(background.index) - 5000 - fp
    print("Reinheit =", tp/(tp + fp))
    print("Effizienz =", tp/(tp + fn))
    print("Genauigkeit =", (tp + tn)/len(y_test))
    print("Signifikanz =", (tp + fp)/np.sqrt(len(y_test)))
    return y_test_knn


# ----------------------------------- Main ------------------------------------
if __name__ == '__main__':
    signal = pd.read_hdf("NeutrinoMC.hdf5", "Signal")
    background = pd.read_hdf("NeutrinoMC.hdf5", "Background")

    signal.drop(columns=["Energy"], inplace=True)
    x_train = signal.iloc[:5000].append(background.iloc[:5000])
    y_train = np.concatenate((np.ones(5000), np.zeros(5000)))
    x_test = signal.iloc[5000:].append(background.iloc[5000:])
    y_test = np.concatenate((np.ones(len(signal.index) - 5000),
                             np.zeros(len(background.index) - 5000)))
    print("Teilaufgabe d)")
    evaluate(x_train, y_train, x_test, y_test, 10, "A19d.npy")

    print("Teilaufgabe e)")
    with np.errstate(divide='ignore'):
        x_train["NumberOfHits"] = np.log10(x_train["NumberOfHits"])
        x_test["NumberOfHits"] = np.log10(x_test["NumberOfHits"])

    evaluate(x_train, y_train, x_test, y_test, 10, "A19e.npy")

    print("Teilaufgabe f)")
    evaluate(x_train, y_train, x_test, y_test, 20, "A19f.npy")
