from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5.8
rcParams['legend.numpoints'] = 1


# -------------------------------- Aufgabe 18 ---------------------------------
def aufg18():
    samples, mask = make_blobs(n_samples=1000, centers=2,
                               n_features=4, random_state=0)
    plt.scatter(samples.T[0][mask == 0],
                samples.T[1][mask == 0], marker=".", label="$p_0$")
    plt.scatter(samples.T[0][mask == 1],
                samples.T[1][mask == 1], marker=".", label="$p_1$")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.savefig("A18_scatter.pdf")
    plt.clf()
    pca = PCA()
    transformed = pca.fit_transform(samples)
    print("Eigenwerte der Kovarianzmatrix:", pca.explained_variance_)
    for i, xp in enumerate(transformed.T):
        plt.hist(xp[mask == 0], bins=25, label="$p_0$")
        plt.hist(xp[mask == 1], bins=25, label="$p_1$", alpha=0.5)
        plt.legend()
        plt.xlabel("$x'_%1i$" % (i + 1))
        plt.ylabel("$n$")
        # plt.show()
        plt.savefig("A18_hist_%1i.pdf" % (i + 1))
        plt.clf()
    plt.scatter(transformed.T[0][mask == 0],
                transformed.T[1][mask == 0], marker=".", label="$p_0$")
    plt.scatter(transformed.T[0][mask == 1],
                transformed.T[1][mask == 1], marker=".", label="$p_1$")
    plt.xlabel("$x'_1$")
    plt.ylabel("$x'_2$")
    plt.legend()
    plt.savefig("A18_transformed_scatter.pdf")


# ----------------------------------- Main ------------------------------------
if __name__ == '__main__':
    aufg18()
