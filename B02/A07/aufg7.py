import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.patches import Patch
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5.8

# Drehwinkel zur Diagonalisierung
a = -0.5*np.arctan(2*4.2/(3.5**2 - 1.5**2))
cosa = np.cos(a)
sina = np.sin(a)
sxp = np.sqrt(3.5**2*cosa**2 + 1.5**2*sina**2 - 2*4.2*cosa*sina)
syp = np.sqrt(3.5**2*sina**2 + 1.5**2*cosa**2 + 2*4.2*cosa*sina)

xlin = np.linspace(0, 8, 1e2)
# ylin = np.linspace(-2, 6, 1e2)
ylin = np.linspace(-0.5, 4.5, 1e2)

x, y = np.meshgrid(xlin, ylin)

ux = (x - 4)/3.5
uy = (y - 2)/1.5

E = (ux**2 + uy**2 - 2*ux*uy*0.8)/(1 - 0.8**2)
Ex = (uy - ux*0.8)**2/(1 - 0.8**2)      # f(y|x)
ErwX = lambda x: 2 + 1.5*0.8*(x - 4)/3.5
Ey = (ux - uy*0.8)**2/(1 - 0.8**2)      # f(x|y)
ErwY = lambda y: 4 + 3.5*0.8*(y - 2)/1.5
f = 1/(2*np.pi*np.sqrt(3.5**2*1.5**2 - 4.2**2))*np.exp(-0.5*E)

plt.contour(x, y, E, [1], colors="k")
plt.contour(x, y, Ex, [1], colors="b")
plt.plot(xlin, ErwX(xlin), "b--")
plt.plot(ErwY(ylin) , ylin, "g--")
plt.contour(x, y, Ey, [1], colors="g", ls="--")
plt.pcolor(x, y, f, cmap="hot")
plt.errorbar(4, 2, xerr=3.5, yerr=1.5, capsize=3, color="k",
             label=r"$(\mu_x \pm \sigma_x, \mu_y \pm \sigma_y)$")

sina = -sina
plt.plot([cosa*sxp + 4, -cosa*sxp + 4], [sina*sxp + 2, -sina*sxp + 2], "k--",
         label="Hauptachsen")
plt.plot([-sina*syp + 4, sina*syp + 4], [cosa*syp + 2, -cosa*syp + 2], "k--")

plt.axis("scaled")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.xlim(0, 8)
plt.ylim(-0.5, 4.5)
# red_patch = Patch(color="k", label=r"$1\sigma-$Ellipse")
# plt.legend(handles=[red_patch])
plt.legend()
plt.grid(color='w', linestyle=':', linewidth=0.5, alpha=0.5)
plt.colorbar(label=r"$f(x, y)$")
plt.savefig("A7.pdf")
# plt.show()
