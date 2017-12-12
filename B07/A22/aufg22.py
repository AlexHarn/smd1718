from sklearn.datasets import make_blobs
from sklearn.preprocessing import label_binarize
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5.8
rcParams['legend.numpoints'] = 1


# -------------------------------- Functions ----------------------------------
#              Aus der Vorlesung kopiert, aber besser kommentiert
# -----------------------------------------------------------------------------

def softmax(f):
    """
    Verschobene Softmax für bessere numerische Stabilität
    """
    f_shifted = f - f.max()
    p = np.exp(f_shifted).T/np.sum(np.exp(f_shifted), axis=1)
    return p.T


def loss_cross_ent(X, y, W):
    """
    Berechnet die Cross-Entropy des gesamten Datensatzes mit dem gegebenen W
    """
    # Falls die erste Spalte von X Einsen sind, ist die erste Zeile von W
    # der Biasvektor
    f = X@W
    q = softmax(f)
    # Mittelwert über alle Datenpunkte, wie nach (1), definitiv ein Skalar...
    return -np.sum(y*np.log2(q), axis=1).mean()


def gradient(W, X, y):
    """
    Berechnet den Gradienten der Cross-Entropy für das aktuelle W
    """
    # Falls die erste Spalte von X Einsen sind, ist die erste Zeile von W
    # der Biasvektor
    # würde reichen f und softmax nur einmal zu berechnen für loss_cross_end
    # und gradient, ist aber hier von der Performance kein Problem und so
    # besser lesbar
    f = X@W
    p = softmax(f)
    # Herleitung war aufgabe b). Hier sieht man auch ganz klar, dass es
    # definitiv ein Vektor ist was in (4) falsch ist...
    dh = (p - y)  # Eigentlich noch /m aber das kommt in der nächsten Zeile
    dW = X.T@dh/dh.shape[0]
    return dW


def gradient_descent(X, y, max_iter=10000, step_size=0.01):
    """
    Führt den gradient descent auf der Cross-Entropy durch.
    Es macht absolut keinen Sinn hier, wie in der VL, die Lossfunktion
    als Parameter zu übergeben, weil die Gradienten Funktion bereits analytisch
    aus dieser hergeleitet ist. Wenn dann muss man die auch übergeben
    oder alles numerisch ableiten.
    """
    K = y.shape[1]
    p = X.shape[1]

    # leere liste um loss historie zu speichern
    losses = []
    # mit zufällig initiallisierten Gewichten (und Biasen) anfangen
    W = np.random.normal(size=(p, K))*step_size
    for i in tqdm(range(max_iter)):
        W = W - gradient(W, X, y)*step_size
        loss = loss_cross_ent(X, y, W)
        losses.append(loss)

    return losses, W


if __name__ == "__main__":
    p0 = np.load("P0.npy").T
    p1 = np.load("P1.npy").T

    X = np.concatenate((p0, p1))
    y = np.concatenate((np.ones(len(p0)), np.zeros(len(p1))))

    """
    X, y = make_blobs(n_samples=1500, n_features=2, center_box=(-5, 5),
                      centers=2, cluster_std=0.3, random_state=1)
    """

    # Einserspalte an stelle 0 einfügen, sodass die erste Zeile
    # der W Matrix zum Biasvektor wird
    ones = np.ones(shape=(len(X), 1))
    X_b = np.hstack([ones, X])
    y_b = label_binarize(y, range(0, 3))[:, :-1]

    losses, W = gradient_descent(X_b, y_b, max_iter=150, step_size=0.5)

    """
    prediction = np.argmax(softmax(X_b@W), axis=1)
    discrete_cmap = LinearSegmentedColormap.from_list('discrete',
                                                      colors=[(0.8, 0.2, 0.3),
                                                              (0, 0.4, 0.8)],
                                                      N=2)
    plt.scatter(X[:, 0], X[:, 1], c=discrete_cmap(y), marker=".", s=0.5,
                alpha=1)
    plt.scatter(X[:, 0], X[:, 1], facecolor='', marker=".", s=5, alpha=0.25,
                edgecolors=discrete_cmap(prediction))
    """

    plt.scatter(p0[:, 0], p0[:, 1], marker=".", s=0.5, alpha=1, label="P0")
    plt.scatter(p1[:, 0], p1[:, 1], marker=".", s=0.5, alpha=1, label="P1")

    x = np.array([min(X[:, 0]), max(X[:, 0])])
    plt.plot(x, (x*(W[1][0] - W[1][1]) + W[0][0] - W[0][1])/(W[2][1] -
                                                             W[2][0]), "g-",
             label="Trennungsgerade")

    plt.legend()
    plt.xlim(x)
    plt.ylim((min(X[:, 1]), max(X[:, 1])))
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    # plt.show()
    plt.savefig("A22.png", dpi=300)
