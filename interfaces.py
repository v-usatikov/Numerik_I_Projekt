import numpy as np


class TimeSolver:
    omega: np.ndarray
    t: float
    dt: float
    h: float

    def time_step(self, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
        """Gibt das neue Omega-Feld für den nächsten Zeitschritt zurück."""
        raise NotImplementedError

    def init(self, omega: np.ndarray, dt: float, h: float):
        self.omega = omega
        self.dt = dt
        self.h = h
        self.t = 0


class Solver:
    h: float  # Gitter Schritt
    L: float  # Lange
    d: float  # Breite
    omega: np.ndarray

    def __init__(self, time_solver: TimeSolver):
        self.time_solver = time_solver

    def __iter__(self) -> (float, np.ndarray, np.ndarray, np.ndarray):
        """Gibt zurück: self.t, self.psi, self.vx, self.vy"""
        raise NotImplementedError
