import numpy as np
from scipy import sparse


def omega_0_validation_sinx(nx, ny, dx, nxny):
    omega_start = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            x = dx * j
            omega_start[i, j] = np.sin(np.pi * x)
    # plot_2dcont(X_mesh, Y_mesh, omega_start, "Startverteilung omega")
    return sparse.csc_matrix(omega_start.reshape(nxny, 1))


def omega_0_validation_siny(nx, ny, dy, nxny):
    omega_start = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            y = dy * i
            omega_start[i, j] = np.sin(np.pi * y)
    # plot_2dcont(X_mesh, Y_mesh, omega_start, "Startverteilung omega")
    return sparse.csc_matrix(omega_start.reshape(nxny, 1))


def omega_0_validation_cosx(nx, ny, dx, nxny):
    omega_start = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            x = dx * j
            omega_start[i, j] = np.cos(np.pi * x)
    # plot_2dcont(X_mesh, Y_mesh, omega_start, "Startverteilung omega")
    return sparse.csc_matrix(omega_start.reshape(nxny, 1))


def omega_0_validation_sinx_siny(nx, ny, dx, dy, nxny):
    omega_start = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            x = dx * j
            y = dy * i
            omega_start[i, j] = np.sin(np.pi * x) + np.sin(np.pi * y)

    # plot_2dcont(X_mesh, Y_mesh, omega_start, "Startverteilung omega")
    return sparse.csc_matrix(omega_start.reshape(nxny, 1))


def omega_0_validation_cosx_cosy(nx, ny, dx, dy, nxny):
    omega_start = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            x = dx * j
            y = dy * i
            omega_start[i, j] = np.cos(np.pi * x) + np.cos(np.pi * y)
    return sparse.csc_matrix(omega_start.reshape(nxny, 1))

def omega_0_validation_vortex(nx, ny, dx, dy, nxny):
    """
    Erzeugt einen Quelle innerhalb der Berechnungsgebiets, bestehend aus einem Sin-Anteil mit Maximum=1 bei x,y=0,5
    und einem EXP-Anteil mit Maximum=1 bei x,y=0,3.

    Zusammen mit dem Faktor 6 ergibt sich ein Max.=3,927 bei x,y=0,3. Die umliegenden Werte sind nicht symmetrisch.

    Die durch die Lösung der Poissongleichung entstehende Wirbelstruktur dreht sich gegen den Uhrzeigersinn. Der
    Mittelpunkt liegt dabei in x,y=0,35
    """
    omega_start = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            x = dx * j
            y = dy * i
            """
            omega_start[i, j] = (1 * np.sin(np.pi * x) *
                                 np.sin(np.pi * y) * 6 *
                                 np.exp(-50 * ((x - 0.30) ** 2 + (y - 0.30) ** 2)))"""
            omega_start[i, j] = 6 * np.exp(-50 * ((x - 0.30) ** 2 + (y - 0.30) ** 2))
    # plot_2dcont(X_mesh, Y_mesh, omega_start, "Startverteilung omega")
    return sparse.csc_matrix(omega_start.reshape(nxny, 1))