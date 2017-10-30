import numpy as np
import numexpr as ne
import time
from scipy.optimize import newton


# ------------------------------------ c) -------------------------------------
def aufg5cIterativ(h=1e-6):
    """Rein iterative Version, viel zu langsam

    h: Genauigkeit (Schrittweite der Integration)
    """
    start = time.time()
    s = 0
    i = 1
    while s < 0.5/(h**3*4/np.sqrt(np.pi)):
        # s += h*4/np.sqrt(np.pi)*(i*h)**2*np.exp(-(i*h)**2)
        s += i**2*np.exp(-(i*h)**2)
        i += 1
    # so ist es die korrekte Trapezregel, macht aber praktisch
    # keinen Unterschied:
    i -= 1
    s = s - i**2*np.exp(-(i*h)**2)*(1 - h/2)
    stop = time.time()
    print(i*h)
    print("Gesamtzeit:", round(stop - start, 2))


def aufg5c(h=1e-10, N=1e8):
    """Variante mit numpy und numexpr.
    ACHTUNG: Benötigt sehr viel Hauptspeicher, weil alle Summandne
    gespeichert werden. Ist dafür sehr viel schneller als Schleifenmethode
    (siehe oben). In C/C++ würde das deutlich besser gehen und noch besser
    natürlich in OpenCL oder CUDA

    N: Anzahl der Funktionsauswertungen, die gleichzeitig im RAM stehen dürfen.
    Anpassen je nach Hauptspeichergröße, umso höhere Auslastung umso besser.
    h: Genauigkeit (Schrittweite der Integration)
    """
    f = np.vectorize(lambda x: x**2*np.exp(-x**2))
    # Zuerst einen groben Wert finden, die obere Grenze muss hier vorher
    # noch gröber abgeschätzt werden. 2*v_m sollte sicher drüber liegen.
    h1 = 1e-6
    a = h1*(np.cumsum(f(np.arange(0, 2, h1)))
            * h1*4/np.sqrt(np.pi)).searchsorted(0.5)
    print("Grobe Näherung für Median:", a)
    # print("1.087652")

    # Jetzt genauer
    start = time.time()
    s = 0
    S = 0
    i = 0
    N_total = a/h + N
    print("Starte Berechnung mit Genauigkeit", h)
    while S + s < 0.5/(h*4)*np.sqrt(np.pi):
        loopS = time.time()
        S += s
        i += 1
        x = np.arange((i - 1)*N*h, i*N*h, h) # doch wird benutzt
        eva = ne.evaluate("x**2*exp(-x**2)")
        s = ne.evaluate("sum(eva)")
        loopT = time.time() - loopS
        print("Status:", round(N*i/N_total*100, 2), "% ca.",
              int(round(loopT*(N_total/N - i))), "s verbleibend          ",
              end="\r", flush=True)

    median = h*((np.cumsum(eva)).searchsorted(0.5/(h*4)*np.sqrt(np.pi) - S)
                + (i - 1)*N)
    print("Status: 100 %                                       ")
    stop = time.time()
    print("Gesamtzeit:", round(stop - start, 2))
    print("v_0.5 = v_m *", median)


# ------------------------------------ d) -------------------------------------
def aufg5d():
    x1 = newton(lambda x: x**2*np.exp(-x**2) - 1/(2*np.e), 0.8,
                lambda x: 2*x*np.exp(-x**2)*(1 - x**2))
    x2 = newton(lambda x: x**2*np.exp(-x**2) - 1/(2*np.e), 1.2,
                lambda x: 2*x*np.exp(-x**2)*(1 - x**2))

    print("x_1 =", x1, "x_2 =", x2)
    print("v_FWHM = v_m *", x2 - x1)


# ----------------------------------- Main ------------------------------------
if __name__ == '__main__':
    # aufg5cIterativ()
    aufg5c(N=0.75e9) # ca. 17 GB RAM
    # aufg5d()
