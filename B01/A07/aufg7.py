import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5.8


# ------------------------------------ A7 -------------------------------------
def aufg7():
    # Drehwinkel zur Diagonalisierung
    a = -0.5*np.arctan(2*4.2/(3.5**2 - 1.5**2))
    cosa = np.cos(a)
    sina = np.sin(a)
    sxp = np.sqrt(3.5**2*cosa**2 + 1.5**2*sina**2 - 2*4.2*cosa*sina)
    syp = np.sqrt(3.5**2*sina**2 + 1.5**2*cosa**2 + 2*4.2*cosa*sina)

    # Sehr große Auflösung, auf langsamen Rechnern eventuell verringern
    xlin = np.linspace(0, 8, 1e3)
    ylin = np.linspace(-0.5, 4.5, 1e3)

    x, y = np.meshgrid(xlin, ylin)

    ux = (x - 4)/3.5
    uy = (y - 2)/1.5

    E = (ux**2 + uy**2 - 2*ux*uy*0.8)/(1 - 0.8**2)
    Ex = (uy - ux*0.8)**2/(1 - 0.8**2)      # f(y|x)
    Ey = (ux - uy*0.8)**2/(1 - 0.8**2)      # f(x|y)
    f = 1/(2*np.pi*np.sqrt(3.5**2*1.5**2 - 4.2**2))*np.exp(-0.5*E)

    # Alles ein bisschen getrickst mit contour und den Labels, geht besser
    # aber leider keine Zeit mehr gerade
    plt.plot(-100, -100, "k-", label=r"$1\sigma$-Ellipse")  # Nur Label
    plt.errorbar(4, 2, xerr=3.5, yerr=1.5, capsize=3, color="k",
                 label=r"$(\mu_x \pm \sigma_x, \mu_y \pm \sigma_y)$")

    sina = -sina
    plt.plot([cosa*sxp + 4, -cosa*sxp + 4],
             [sina*sxp + 2, -sina*sxp + 2], "k--", label="Hauptachsen")
    plt.plot([-sina*syp + 4, sina*syp + 4],
             [cosa*syp + 2, -cosa*syp + 2], "k--")

    plt.contour(x, y, E, [1], colors="k")
    plt.contour(x, y, Ex, [1], colors="b", linestyles="--")
    plt.plot(xlin, (lambda x: 2 + 1.5*0.8*(x - 4)/3.5)(xlin), "b-",
             label=r"$E(y|x)$")
    plt.plot((lambda y: 4 + 3.5*0.8*(y - 2)/1.5)(ylin), ylin, "g-",
             label=r"$E(x|y)$")
    plt.contour(x, y, Ey, [1], colors="g", linestyles="--")
    plt.pcolor(x, y, f, cmap="hot")

    plt.axis("scaled")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.xlim(0, 8)
    plt.ylim(-0.5, 4.5)
    plt.legend()
    plt.grid(color='w', linestyle=':', linewidth=0.5, alpha=0.5)
    plt.colorbar(label=r"$f(x, y)$")
    # plt.savefig("A7.pdf", bbox_inches='tight')  # Als Vektorgraphik sehr groß
    plt.savefig("A7.png", dpi=300, bbox_inches='tight')
    # plt.show()


# ----------------------------------- Main ------------------------------------
if __name__ == '__main__':
    aufg7()
