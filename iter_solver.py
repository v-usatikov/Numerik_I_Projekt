import time
from copy import deepcopy

import numpy as np
import pandas as pd
import numpy.linalg as linalg
from math import pi, sin, exp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from interfaces import TimeSolver, Solver


def omega_null(nx: int, ny: int, h: float) -> np.ndarray:
    omega = np.zeros((ny, nx))
    for i in range(nx):
        for j in range(ny):
            x = h * i
            y = h * j
            omega[i, j] = sin(pi * x) * sin(pi * y) * 6 * exp(-50 * ((x - 0.30) ** 2 + (y - 0.30) ** 2))
    omega = omega.flatten()
    return omega


def ruku_schritt(t, y, f, h):
    """RK Schritt"""
    k1 = f(t, y)
    k2 = f(t + h/2, y + h * k1 / 2)
    k3 = f(t + h/2, y + h * k2 / 2)
    k4 = f(t + h, y + h * k3)

    return y + h*(k1/6 + k2/3 + k3/3 + k4/6)


class RukuTimeSolver(TimeSolver):

    def time_step(self, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
        self.vx = vx
        self.vy = vy
        self.omega = ruku_schritt(self.t, self.omega, self._f, self.dt)
        self.t += self.dt
        return self.omega

    def _f(self, t: float, omega: np.ndarray):
        f = np.zeros(omega.shape)

        f[1:-1, 1:-1] = -(self.vx*(omega[1:-1, 2:] - omega[1:-1, :-2]) / (2 * self.h) +
                          self.vy*(omega[2:, 1:-1] - omega[:-2, 1:-1]) / (2 * self.h))
        return f


class IterSolver(Solver):
    def __init__(self,
                 time_solver: TimeSolver,
                 d: float = 1,
                 L: float = 1,
                 dt: float = 0.01,  # Zeitschritt
                 ny: int = 19 + 1,  # Große der Diskretisieren-Gitter für y-Achse
                 V0: float = 0.1,  # Eingang-Strom-Geschwindigkeit
                 end_time = 50
                 ):
        super().__init__(time_solver)

        self.d = d
        self.L = L
        self.dt = dt
        self.ny = ny
        self.h = d / (ny - 1)  # Schritt für x y
        self.nx = int(round(L / self.h)) + 1  # Große der Diskretisieren-Gitter für x-Achse
        self.tol = 0.00001

        self.omega = np.zeros((self.ny, self.nx))
        self.psi = np.zeros((self.ny, self.nx))
        self.vx = np.zeros((self.ny, self.nx))
        self.vy = np.zeros((self.ny, self.nx))

        self.t = 0
        self.end_time = end_time

    def set_omega0(self):
        omega = np.zeros((self.ny, self.nx))
        for i in range(self.nx):
            for j in range(self.ny):
                x = self.h * i
                y = self.h * j
                omega[j, i] = 1*sin(pi * x) * sin(pi * y) * 6 * exp(-50 * ((x - 0.30) ** 2 + (y - 0.30) ** 2))
        self.omega = omega
        self.time_solver.init(self.omega, self.dt, self.h)

    def solve_poisson_iter(self):
        psi = np.ones((self.ny, self.nx))
        i = 1
        psi_m10 = deepcopy(psi)
        while True:
            psi[1:-1, 1:-1] = (((psi[2:, 1:-1] + psi[:-2, 1:-1]) * self.h ** 2 +
                                (psi[1:-1, 2:] + psi[1:-1, :-2]) * self.h ** 2 +
                                self.omega[1:-1, 1:-1] * (self.h ** 2 * self.h ** 2))/(2 * (self.h ** 2 + self.h ** 2)))

            # Wall boundary conditions, pressure
            psi[:, 0] = 0  # dp/dy = 0 at y = 2
            psi[0, :] = 0  # dp/dy = 0 at y = 0
            psi[-1, :] = 0
            psi[:, -1] = 0

            if i % 10 == 0:
                if np.allclose(psi, psi_m10, atol=self.tol):
                    # print(i)
                    break
                psi_m10 = deepcopy(psi)
            i += 1
        self.psi = psi

    def get_speed_feld_from_psi(self):
        # self.vx = np.zeros((self.ny, self.nx))
        # self.vy = np.zeros((self.ny, self.nx))

        self.vx = (self.psi[2:, 1:-1] - self.psi[:-2, 1:-1]) / (2 * self.h)
        self.vy = - (self.psi[1:-1, 2:] - self.psi[1:-1, :-2]) / (2 * self.h)

    def plot_speed_feld(self):
        x = np.arange(self.h, self.L, self.h)
        y = np.arange(self.h, self.d, self.h)

        X, Y = np.meshgrid(x, y)

        fig, ax = plt.subplots()
        q = ax.quiver(X, Y, self.vx, self.vy)
        # ax.quiverkey(q, X=0.3, Y=1.1, U=10,
        #              label='Quiver key, length = 10', labelpos='E')
        plt.show()

    def plot_psi_feld(self):
        dx, dy = 0.00001, 0.00001

        X = np.arange(0, self.L+self.h, self.h)
        Y = np.arange(0, self.d+self.h, self.h)

        levels = MaxNLocator(nbins=300).tick_values(self.psi.min(), self.psi.max())

        plt.contourf(X + dx/2., Y + dy/2., self.psi, levels = levels)
        plt.colorbar()

        # fig, ax = plt.subplots()
        # q = ax.contourf()
        # ax.quiverkey(q, X=0.3, Y=1.1, U=10,
        #              label='Quiver key, length = 10', labelpos='E')

        plt.show()

    def time_step(self):
        self.omega = self.time_solver.time_step(self.vx, self.vy)

    def __iter__(self) -> (float, np.array, np.array, np.array):
        self.set_omega0()
        self.solve_poisson_iter()
        self.get_speed_feld_from_psi()
        yield self.time_solver.t, self.psi, self.vx, self.vy

        while self.t < self.end_time:
            self.time_step()
            self.solve_poisson_iter()
            self.get_speed_feld_from_psi()
            yield self.time_solver.t, self.psi, self.vx, self.vy

if __name__ == "__main__":

    # solver = Solver(ny=501)
    # solver = Solver(ny=21, l=1, d=1)
    # solver.set_omega0()
    # solver.solve_poisson_iter()
    # solver.get_speed_feld_from_psi()
    # solver.plot_speed_feld()
    # solver.plot_psi_feld()

    start = time.time()
    solver = IterSolver(RukuTimeSolver(), ny=21, L=1, d=1)
    for t, psi, vx, vy in solver:

        if np.isclose(t, 1):
            print(time.time() - start)
            solver.plot_speed_feld()

        if np.isclose(t, 10):
            print(time.time() - start)
            solver.plot_speed_feld()

        if np.isclose(t, 20):
            print(time.time() - start)
            solver.plot_speed_feld()

        if np.isclose(t, 30):
            print(time.time() - start)
            solver.plot_speed_feld()
            break

        # if abs(round(t)-t) < 0.0001:
        #     solver.plot_speed_feld()
