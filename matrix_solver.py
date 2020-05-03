"""
Programm zur numerischen Berechnung von Wirbelströmungen

Literatur:
[1] - Michael Knorrenschild, Numerische Mathematik, Carl Hanser Verlag,2010
"""
import numpy as np
import numpy.linalg as lg
import sympy as sp
from scipy.sparse import spdiags
from scipy import sparse
from scipy.sparse import linalg
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from sympy import symbols

import function_generation as fg  # Import für Funktionen von omega
import matrix_creation as mc
from interfaces import TimeSolver, Solver





class EulerTimeSolver(TimeSolver):
    """ explizit, erste Ordnung genau"""
    def time_step(self, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
        self.vx = vx
        self.vy = vy
        # TODO: dO/dt = nue*laplace(O) - u*dO/dx - v*dO/dy
        #       -> dO/dt = RHS -> O_n1 = O_n + RHS*dt

class LeapfrogTimeSolver(TimeSolver):
    """ explizit, zweite Ordnung genau"""
    def time_step(self, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
        self.vx = vx
        self.vy = vy
        # TODO: dO/dt = nue*laplace(O) - u*dO/dx - v*dO/dy
        #       -> dO/dt = RHS -> O_n1 = O_n-1 + RHS*2dt

class AdamBashfordTimeSolver(TimeSolver):
    """ explizit, zweite Ordnung genau"""
    def time_step(self, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
        self.vx = vx
        self.vy = vy
        # TODO: dO/dt = nue*laplace(O) - u*dO/dx - v*dO/dy
        #       -> dO/dt = RHS -> O_n1 = O_n-1 + RHS*2dt


class MatrixSolver(Solver):
    """Superclass CFD-Solver
    Zur allgemeinen Erläuterung der Variablennamen:
    - Vektoren / 1D-Arrays werden nach Möglichkeit mit kleinen Buchstaben bezeichnet
    - Matrizen / 2D-Arrays werden nach Möglichkeit mit großen Buchstaben bezeichnet
    """

    def __init__(self,
                 time_solver: TimeSolver,
                 x: float,  # Höhe in m
                 y: float,  # Breite in m
                 aufloesung: float,  # Auflösung des Gitters
                 nx: int,  # Anzahl Knoten in x-Richtung
                 ny: int,  # Anzahl Knoten in z-Richtung
                 t_max: float,  # max. simulierte Zeitspanne
                 dt: float,  # gewünschte zeitliche Auflösung (ggf. durch CFL überschrieben)
                 conv_n1_target: float,  # gewünschte Genauigkeit der iterariven Löser
                 iter_outer: int,  # max. Anzahl innerer Iterationen
                 iter_inner: int):  # max. Anzahl innerer Iterationen
        super().__init__(time_solver)

        self.x = float(x)  # Höhe in m
        self.y = float(y)  # Breite in m
        self.nx = int(nx)  # Anzahl Knoten in x-Richtung
        self.ny = int(ny)  # Anzahl Knoten in z-Richtung
        self.dx = y / (nx - 1)  # Abstand zwischen Knoten x-Richtung
        self.dy = x / (ny - 1)  # Abstand zwischen Knoten y-Richtung
        self.nxny = self.nx * self.ny
        #self.relax_faktor = 1
        #self.nt = t_max  # max. simulierte Zeitspanne
        #self.dt = dt  # gewünschte zeitliche Auflösung (ggf. durch CFL überschrieben)
        #self.nue = 1  # kinematische Viskosität
        #self.conv_n1_target = conv_n1_target
        #self.iter_outer = iter_outer  # max. Anzahl innerer Iterationen
        #self.iter_inner = iter_inner  # max. Anzahl innerer Iterationen

        np.set_printoptions(precision=3, suppress=True)

        # Verfahrensmatrizen
        self.M = sparse.csr_matrix(0)
        self.M_x = sparse.csr_matrix(0)
        self.M_y = sparse.csr_matrix(0)
        self.r = sparse.csr_matrix(0)
        self.R_sp = sparse.csr_matrix(0)
        self.R_sp_inv = sparse.csr_matrix(0)

        # Lösungsvariablen
        self.omega = np.zeros((self.ny, self.nx))
        self.psi = np.zeros((self.ny, self.nx))
        self.vx = np.zeros((self.ny, self.nx))
        self.vy = np.zeros((self.ny, self.nx))

        # Vorbereitung des Gitters
        # TODO: ggf. in Funktion überführen (def create_mesh(self):)

        self.omega = np.zeros((self.nx, self.ny))

    def set_omega0(self):
        omega = fg.omega_0_validation_vortex(self.nx, self.ny, self.dx, self.dy)
        omega = sparse.csc_matrix(omega.reshape(self.nxny, 1))
        self.omega = omega

    def matrix_init(self):
        " Initialisierung der notwendigen Matrizen"

        # Randbedingungen
        self.r, self.R_sp, self.R_sp_inv = mc.create_boundary(self, False)
        d = mc.create_dirichlet(self, self.r, 0)

        # Verfahrensmatrizen
        M_x = mc.create_coeff_gradient_dx(self)
        M_y = mc.create_coeff_gradient_dy(self)
        M_x, M_y = mc.assign_polynom_to_gradient(self, M_x, M_y)
        M = mc.create_coeff_laplace_polynom(self)

        self.M_x = mc.assign_r_to_M(self, self.R_sp, self.R_sp_inv, M_x)
        self.M_y = mc.assign_r_to_M(self, self.R_sp, self.R_sp_inv, M_y)
        self.M = mc.assign_r_to_M(self, self.R_sp, self.R_sp_inv, M)


    def gradient_field(self,
                       M,  # Koeffizientenmatrix
                       x):
        """ Gradient in Abhängigkeit der eingegeben Matrix, siehe self.create_coeff_gradient.

            Parameters:
                M   : Koeffizientenmatrix, z.B: M_x oder M_y übergeben
                v   : Abzuleitende Funktion
        """
        return M.dot(x)


    def solve_laplace(self):
        self.b = self.M.dot(self.x)

    def solve_poisson_iter(self):