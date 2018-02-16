import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.optimize import minimize
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5.8
rcParams['legend.numpoints'] = 1


# -------------------------------- Funktionen ---------------------------------
def aufg38a(train):
    # print(train.head())
    for atr in train.columns.values[1:]:
        print(atr)
        plt.hist2d(train[atr], train['energy_true'], bins=100)
        plt.xlabel(atr)
        plt.ylabel('energy_true')
        plt.colorbar()
        # plt.show()
        plt.savefig('A37a_{}.pdf'.format(atr))
        plt.clf()


def aufg38b(train, plot=False):
    A, _, _ = np.histogram2d(train['size'], train['energy_true'], [24, 16],
                             [[120, 500], [15, 200]])

    # f, _ = np.histogram(train['energy_true'], 16, [15, 200])
    # g, _ = np.histogram(train['size'], 24, [120, 500])
    f = np.sum(A, axis=0)
    g = np.sum(A, axis=1)

    # das geht mit Sicherheit irgendwie eleganter
    for i in range(A.shape[0]):
        A[i] = np.divide(A[i], f)
    # print(g, A@f) # stimmt
    if plot:
        plt.matshow(A)
        plt.colorbar()
        # plt.show()
        # plt.savefig('A37b.pdf')
        plt.clf()
    return (g, A, f)


def F(f, g, A, t, C):
    """
    negative log likelihood ohne konstante Verschiebung
    """
    return np.sum([-g[i]*np.log(A[i]@f) + A[i]@f for i in range(len(g))]) + \
        t/2*np.linalg.norm(C@f)**2


def aufg38f(test, A):
    f, _ = np.histogram(test['energy_true'], 16, [15, 200])
    g, _ = np.histogram(test['size'], 24, [120, 500])

    # Regularisierungsmatrix
    C = np.asarray(diags([np.ones(len(f))*(-2), np.ones(len(f) - 1),
                          np.ones(len(f) - 1)], [0, -1, 1]).todense())
    C[0][0] = -1
    C[-1][-1] = -1
    # Regularisierungsstärken
    ts = [0, 1e-6, 1e-3, 1e-1]

    x = np.arange(len(f))

    for t in ts:
        result = minimize(F, x0=np.ones(len(f))*100, args=(g, A, t, C),
                          bounds=[(0, 500) for i in range(len(f))], tol=1e-6)
        f_min = np.round(result.x)
        print(t, f, f_min)

        plt.step(x, f_min, label=r'$\tau = {}$'.format(t), where='post')

    plt.step(x, f, 'k', label='Wahr', where='post')
    plt.xlim(0, 15)
    plt.legend()
    plt.xlabel('$i$')
    plt.ylabel('Ereignisse $f_i$')
    plt.savefig('A38f.pdf')
    # plt.show()


# ----------------------------------- Main ------------------------------------
if __name__ == '__main__':
    train = pd.read_hdf('unfolding_data.hdf5', key='train')
    test = pd.read_hdf('unfolding_data.hdf5', key='test')
    aufg38a(train)
    # also size als Entfaltungs-Observable nehmen
    g, A, f = aufg38b(train)
    aufg38f(test, A)
