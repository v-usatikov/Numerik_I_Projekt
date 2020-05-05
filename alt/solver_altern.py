import numpy as np
import pandas as pd
import numpy.linalg as linalg
from math import pi, sin, exp
import matplotlib.pyplot as plt


def border_vector(nx: int, ny: int) -> np.ndarray:
    matrix = np.zeros((ny, nx))
    matrix[0, :] = 1
    matrix[-1, :] = 1
    matrix[:, 0] = 1
    matrix[:, -1] = 1
    matrix = matrix.flatten()
    return matrix


def negativ(vector: np.ndarray) -> np.ndarray:
    return np.array(list(map(lambda x: int(not bool(x)), vector)))


def dirichlet_BC(nx: int, ny: int):
    matrix = np.zeros((ny, nx))
    matrix.flatten()
    matrix = matrix.flatten()
    return matrix


def omega_null(nx: int, ny: int, h: float) -> np.ndarray:
    omega = np.zeros((ny, nx))
    for i in range(nx):
        for j in range(ny):
            x = h * i
            y = h * j
            omega[i, j] = sin(pi * x) * sin(pi * y) * 6 * exp(-50 * ((x - 0.30) ** 2 + (y - 0.30) ** 2))
    omega = omega.flatten()
    return omega


def __l_border_der(f: float, fp1: float, fp2: float, h: float) -> float:
    return 1 / (2 * h) * (-3 * f + 4 * fp1 - fp2)


def __r_border_der(fm2: float, fm1: float, f: float, h: float) -> float:
    return 1 / (2 * h) * (fm2 - 4 * fm1 + 3 * f)


def __in_der(fm1: float, fp1: float, h: float) -> float:
    return 1 / (2 * h) * (fp1 - fm1)


def part_der_y(i: int, j: int, array: np.ndarray) -> float:
    if j == 0:
        return __l_border_der(array[i, j], array[i, j + 1], array[i, j + 2])
    elif j == len(array):
        return __r_border_der(array[i, j - 2], array[i, j - 1], array[i, j])
    else:
        return __in_der(array[i, j - 1], array[i, j + 1])


def part_der_x(i: int, j: int, array: np.ndarray) -> float:
    if i == 0:
        return __l_border_der(array[i, j], array[i + 1, j], array[i + 2, j])
    elif i == len(array):
        return __r_border_der(array[i - 2, j], array[i - 1, j], array[i, j])
    else:
        return __in_der(array[i - 1, j], array[i + 1, j])


def matrix_Dx(nx: int, ny: int, h: float) -> np.ndarray:
    matrix = np.zeros((nx * ny, nx * ny))
    for i in range(ny):
        row = nx * i
        matrix[row, row:row + 3] = -3, 4, -1
        for _ in range(nx - 2):
            row += 1
            matrix[row, row - 1:row + 2] = -1, 0, 1
        row += 1
        matrix[row, row - 2:row + 1] = 1, -4, 3
    return matrix / (2 * h)


def matrix_Dy(nx: int, ny: int, h: float) -> np.ndarray:
    matrix = np.zeros((nx * ny, nx * ny))
    for i in range(nx):
        matrix[ny * i, i] = -3
        matrix[ny * i, nx + i] = 4
        matrix[ny * i, 2 * nx + i] = -1

        for j in range(1, ny - 1):
            matrix[ny * i + j, nx * (j - 1) + i] = -1
            matrix[ny * i + j, nx * (j + 1) + i] = 1

        matrix[ny * (i + 1) - 1, nx * (ny - 1 - 2) + i] = 1
        matrix[ny * (i + 1) - 1, nx * (ny - 1 - 1) + i] = -4
        matrix[ny * (i + 1) - 1, nx * (ny - 1) + i] = 3
    return matrix / (2 * h)


def nabla_matrix(nx: int, ny: int, h: float) -> np.ndarray:
    return matrix_Dx(nx=nx, ny=ny, h=h) + matrix_Dy(nx=nx, ny=ny, h=h)


def laplas_matrix(nx: int, ny: int, h: float) -> np.ndarray:
    nabla = nabla_matrix(nx=nx, ny=ny, h=h)
    return np.dot(nabla, nabla)


def laplas_matrix2(nx: int, ny: int, h: float):
    dx=h
    dy=h
    # Erstellen der Formel-Matrix aus der Umformung (d2p/dx2 + d2p/dy2) = omega_n
    diag_block = np.eye(ny) * (- 2 / dx**2 - 2 / dy**2)
    diag_block = diag_block + np.eye(ny, k=1) * (1 / dy ** 2)
    diag_block = diag_block + np.eye(ny, k=-1) * (1 / dy ** 2)
    Mat = np.kron(np.eye(nx), diag_block)
    Mat = Mat + np.eye(ny*nx, k=ny) * (1 / dx ** 2)
    Mat = Mat + np.eye(ny*nx, k=-ny) * (1 / dx ** 2)
    return Mat


class Solver:
    def __init__(self,
                 d: float = 1,
                 l: float = 1,
                 dt: float = 0.01,  # Zeitschritt
                 ny: int = 19 + 1,  # Große der Diskretisieren-Gitter für y-Achse
                 V0: float = 0.1  # Eingang-Strom-Geschwindigkeit
                 ):

        self.d = d
        self.L = l
        self.dt = dt
        self.ny = ny
        self.h = d / (ny - 1)  # Schritt für x y
        self.nx = int(round(l / self.h)) + 1  # Große der Diskretisieren-Gitter für x-Achse
        print(self.ny, self.nx)
        laplas = laplas_matrix(self.nx, self.ny, self.h)
        print(linalg.det(laplas))
        R = border_vector(self.nx, self.ny)
        self.nR = negativ(R)
        Rd = np.diag(R)
        nRd = np.diag(self.nR)

        self.dx_matrix = matrix_Dx(self.nx, self.ny, self.h)
        self.dy_matrix = matrix_Dy(self.nx, self.ny, self.h)

        self.laplas_BC = laplas + Rd
        self.laplas_BC_test = np.dot(nRd, laplas)*self.h**2 + Rd

        self.dirichlet_BC = dirichlet_BC(self.nx, self.ny)
        self.omega = np.zeros((self.ny, self.nx)).flatten()
        self.psi = np.zeros((self.ny, self.nx))
        self.vx = np.zeros((self.ny, self.nx))
        self.vy = np.zeros((self.ny, self.nx))

    def set_omega0(self):
        omega = np.zeros((self.ny, self.nx))
        for i in range(self.nx):
            for j in range(self.ny):
                x = self.h * i
                y = self.h * j
                omega[j, i] = sin(pi * x) * sin(pi * y) * 6 * exp(-50 * ((x - 0.30) ** 2 + (y - 0.30) ** 2))
        self.omega = omega.flatten()

    def get_psi_from_omega(self):
        mat = self.laplas_BC
        b = (-1)*self.omega + self.dirichlet_BC
        print(linalg.det(mat))
        self.psi = linalg.solve(mat, b)  # Lösen mat*p = b

    def get_speed_feld_from_psi(self):
        self.vx = np.dot(self.dy_matrix, self.psi)
        self.vy = -1*np.dot(self.dx_matrix, self.psi)

    def plot_speed_feld(self):

        vx = self.vx.reshape((self.ny, self.nx))
        vy = self.vy.reshape((self.ny, self.nx))

        # vx = -0.1*np.ones((self.ny, self.nx))
        # vy = 0.06 * np.ones((self.ny, self.nx))

        X = np.arange(0, 1+self.h, self.h)
        Y = np.arange(0, 1+self.h, self.h)

        fig, ax = plt.subplots()
        q = ax.quiver(X, Y, vx, vy)
        ax.quiverkey(q, X=0.3, Y=1.1, U=10,
                     label='Quiver key, length = 10', labelpos='E')

        plt.show()


if __name__ == "__main__":
    # print(pd.DataFrame(matrix_Dx()))
    # A = matrix_Dx()
    # print(pd.DataFrame(A))
    #
    # B = matrix_Dy()
    # print(pd.DataFrame(B))
    #
    # print(omega_null())
    #
    # array = np.ones((3, 5))
    # print(len(array), array)
    # array.flatten()
    # print(type(array))

    solver = Solver(ny=4)
    # solver = Solver(ny=3, l=4, d=3)
    solver.set_omega0()
    solver.get_psi_from_omega()
    solver.get_speed_feld_from_psi()
    solver.plot_speed_feld()
    # print(pd.DataFrame(solver.psi))