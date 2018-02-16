import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5.8
rcParams['legend.numpoints'] = 1


# -------------------------------- Funktionen ---------------------------------
def getA(n, eps):
    A = np.asarray(diags([np.ones(n - 1)*eps, np.ones(n)*(1 - 2*eps),
                          np.ones(n - 1)*eps], [-1, 0, 1]).todense())
    A[0, 0] = 1 - eps
    A[n - 1, n - 1] = A[0, 0]
    return A


# ----------------------------------- Main ------------------------------------
if __name__ == '__main__':
    np.random.seed(1234)
    f = np.array([193, 485, 664, 763, 804, 805, 779, 736, 684, 626, 566, 508,
                  452, 400, 351, 308, 268, 233, 202, 173])

    # im Folgenden bezeichnen alle Variablen mit _m die gezogenen "Messwerte"
    A = getA(len(f), 0.23)
    g = A@f
    g_m = np.random.poisson(g)

    w, U = np.linalg.eig(A)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    U = U[:, idx]

    b = U.T@f
    c = np.diag(w)@b
    c_m = U.T@g_m

    # Entfaltung
    invD = np.diag(1/w)
    b_m = invD@c_m
    covb = invD@U.T@np.diag(g)@U@invD
    covb_m = invD@U.T@np.diag(g_m)@U@invD
    # Regularisierung
    b_m_reg = np.copy(b_m)
    covb_m_reg = np.copy(covb_m)
    covb_m_reg[10:, 10:] = 0
    b_m_reg[10:] = 0
    # Rücktrafo
    f_m = U@b_m
    f_m_reg = U@b_m_reg
    covf_m = U@covb_m@U.T
    covf_m_reg = U@covb_m_reg@U.T
    # Normierung
    b = np.divide(b, np.sqrt(np.diag(covb)))
    b_m = np.divide(b_m, np.sqrt(np.diag(covb_m)))
    b_m_reg = np.divide(b_m_reg, np.sqrt(np.diag(covb_m)))

    ax = plt.gca()
    x = np.arange(len(b))
    plt.step(x, abs(b), 'g', label='Wahr', where='post')
    plt.step(x, abs(b_m), color='darksalmon', label='Entfaltung', where='post')
    plt.step(x, abs(b_m_reg), 'y', label='Entfaltung regularisiert',
             where='post')
    plt.xticks(np.arange(len(b)))
    plt.xlim(0.01, len(b) - 1)
    plt.hlines(1, 0, len(b) - 1)
    ax.set_yscale('log')
    plt.legend()
    plt.ylabel('Koeffizient $b_j$')
    plt.xlabel('Index $j$')
    plt.savefig('Eigenbasis.pdf')
    # plt.show()
    plt.clf()

    x += 1
    plt.plot(np.concatenate(([0], x)), np.concatenate(([0], f)), 'k-',
             label='pdf')
    plt.errorbar(x, f_m_reg, yerr=np.sqrt(np.diag(covf_m_reg)), xerr=0.5,
                 fmt='.', color='g', capsize=5, markersize=0, elinewidth=2,
                 label='regularisiert')
    plt.errorbar(x, f_m, yerr=np.sqrt(np.diag(covf_m)), xerr=0.5, fmt='.',
                 color='darksalmon', capsize=5, markersize=0, elinewidth=2,
                 label='nicht reg.')
    # plt.plot(x, f_m_reg)
    plt.xlim(0, x[-1])
    plt.ylim(0, 1200)
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('Ereignisse')
    plt.savefig('Realbasis.pdf')
    # plt.show()
