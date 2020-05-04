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
from interfaces import TimeSolver, Solver
from main_evaluator import Evaluator
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from sympy import symbols

import function_generation as fg  # Import für Funktionen von omega
import matrix_creation as mc
from interfaces import TimeSolver, Solver

def ruku_schritt(t, y, f, h):
    """RK Schritt"""
    k1 = f(t, y)
    k2 = f(t + h/2, y + h * k1 / 2)
    k3 = f(t + h/2, y + h * k2 / 2)
    k4 = f(t + h, y + h * k3)

    return y + h*(k1/6 + k2/3 + k3/3 + k4/6)


class RukuTimeSolver(TimeSolver):

    def time_step(self, vx: np.ndarray, vy: np.ndarray, rhs) -> np.ndarray:
        self.rhs = rhs
        self.vx = vx
        self.vy = vy
        self.omega = ruku_schritt(self.t, self.omega, self._f, self.dt)
        CFL = np.linalg.norm(vx + vy) * self.dt / self.h  # Spektral/Frobeniusnorm
        if CFL >= 1:
            self.dt /= CFL
        self.t += self.dt
        self.t += self.dt
        return self.omega

    def _f(self, t: float, omega: np.ndarray):

        f = self.rhs

        return f

class EulerTimeSolver(TimeSolver):
    """ explizit, erste Ordnung genau

        dO/dt = nue*laplace(O) - u*dO/dx - v*dO/dy
        -> dO/dt = rhs -> O_n1 = O_n + rhs*dt
    """
    def time_step(self, vx: np.ndarray, vy: np.ndarray, rhs) -> np.ndarray:
        self.vx = vx
        self.vy = vy
        self.omega = self.omega + np.multiply(rhs, self.dt)
        CFL = np.linalg.norm(vx+vy) * self.dt / self.h  # Spektral/Frobeniusnorm
        if CFL >= 1:
            self.dt /= CFL
        self.t += self.dt
        return self.omega


class LeapfrogTimeSolver(TimeSolver):
    """ explizit, zweite Ordnung genau

        dO/dt = nue*laplace(O) - u*dO/dx - v*dO/dy
        -> dO/dt = rhs -> O_n1 = O_n-1 + rhs*2dt
    """
    def time_step(self, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
        self.vx = vx
        self.vy = vy
        # TODO: dO/dt = nue*laplace(O) - u*dO/dx - v*dO/dy
        #       -> dO/dt = rhs -> O_n1 = O_n-1 + rhs*2dt
        #   2 OMEGA-Werte benötigt!


class CrankNicholsonTimeSolver(TimeSolver):
    """ explizit, zweite Ordnung genau"""
    def time_step(self, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
        self.vx = vx
        self.vy = vy
        # TODO: dO/dt = nue*laplace(O) - u*dO/dx - v*dO/dy
        #       -> dO/dt = rhs -> O_n1 = O_n-1 + rhs*2dt
        #   2 OMEGA-Werte benötigt!


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
                 ny: int,  # Anzahl Knoten in z-Richtung
                 t_max: float,  # max. simulierte Zeitspanne
                 dt: float,  # gewünschte zeitliche Auflösung (ggf. durch CFL überschrieben)
                 tol = 1.e-05):  # gewünschte Genauigkeit der iterariven Löser
        super().__init__(time_solver)

        self.x = float(x)  # Höhe in m
        self.y = float(y)  # Breite in m
        self.ny = int(ny)  # Anzahl Knoten in y-Richtung
        self.dy = x / (self.ny - 1)  # Abstand zwischen Knoten y-Richtung
        self.nx = int(round(x / self.dy)) + 1  # Anzahl Knoten in x-Richtung
        self.dx = x / (self.nx - 1)  # Abstand zwischen Knoten x-Richtung
        assert self.dx == self.dy
        self.nxny = self.nx * self.ny  # Anzahl aller Knoten
        self.nue = 0  # kinematische Viskosität
        #self.nue = 10e-06
        self.tol = tol
        self.dt = dt  # gewünschte zeitliche Auflösung (ggf. durch CFL überschrieben)
        self.t = 0
        self.t_max = t_max
        self.iteration = 1

        self.h = self.dy
        self.d = self.y
        self.L = self.x

        iter_outer: int  # max. Anzahl innerer Iterationen
        iter_inner: int  # max. Anzahl innerer Iterationen

        np.set_printoptions(precision=5, suppress=True)

        # Verfahrensmatrizen
        self.M = sparse.csr_matrix(0)
        self.M_x = sparse.csr_matrix(0)
        self.M_y = sparse.csr_matrix(0)
        self.r = sparse.csr_matrix(0)
        self.R_sp = sparse.csr_matrix(0)
        self.R_sp_inv = sparse.csr_matrix(0)
        self.rhs = sparse.csr_matrix(0)

        # Lösungsvariablen
        self.omega = np.zeros((self.ny, self.nx))
        self.psi = np.zeros((self.ny, self.nx))
        self.vx = np.zeros((self.ny, self.nx))
        self.vy = np.zeros((self.ny, self.nx))

        # Vorbereitung des Gitters
        # TODO: ggf. in Funktion überführen (def create_mesh(self):)

        self.omega = np.zeros((self.ny, self.nx))

    def set_omega0(self):
        omega = fg.omega_0_validation_vortex(self.nx, self.ny, self.dx, self.dy)
        #omega = -fg.omega_0_validation_pi2sinx_pi2siny(self.nx, self.ny, self.dx, self.dy)
        #omega = fg.omega_0_validation_siny(self.nx, self.ny, self.dx)*np.pi**2
        omega = omega.ravel()
        self.omega = omega
        xxx_array_omega = omega.reshape(self.ny, self.nx)
        self.time_solver.init(self.omega, self.dt, h=self.dx)


    def set_matrices(self):
        """ Initialisierung der notwendigen Matrizen"""

        # Randbedingungen
        self.r, self.R_sp, self.R_sp_inv = mc.create_boundary(self, False)
        self.D = mc.create_dirichlet(self, self.r, 0)

        # Verfahrensmatrizen
        M_x = mc.create_coeff_gradient_dx(self)
        M_y = mc.create_coeff_gradient_dy(self)
        M_x, M_y = mc.assign_polynom_to_gradient(self, M_x, M_y)
        M = mc.create_coeff_laplace_polynom(self)
        #mx_array = M_x.toarray()
        #my_array = M_y.toarray()
        #m_array = M.toarray()
        #self.x_target = fg.omega_0_validation_sinx_siny(self.nx, self.ny, self.dx, self.dy)
        #self.x_target = fg.omega_0_validation_siny(self.nx, self.ny, self.dx)
        #self.x_target_sparse = sparse.csc_matrix(self.x_target.reshape(self.nxny, 1))
        #self.omega = mc.assign_d_to_b(self, self.r, self.R_sp_inv, -self.x_target_sparse, self.omega)

        self.M_x = mc.assign_r_to_M(self, self.R_sp, self.R_sp_inv, M_x)
        self.M_y = mc.assign_r_to_M(self, self.R_sp, self.R_sp_inv, M_y)
        self.M = mc.assign_r_to_M(self, self.R_sp, self.R_sp_inv, M)
        self.omega = mc.assign_d_to_b(self, self.r, self.R_sp_inv, self.D, self.omega)

        xx_mx_array = self.M_x.toarray().reshape(self.nxny, self.nxny)
        xx_my_array = self.M_y.toarray().reshape(self.nxny, self.nxny)
        xx_m_array = self.M.toarray().reshape(self.nxny, self.nxny)
        xx_omega_array = self.omega.reshape(self.ny, self.nx)

        # Matrizen für das Gauss-Seidel-Verfahren
        self.M_L = sparse.tril(self.M, k=-1)  # untere/linke Dreiecksmatrix
        self.M_U = sparse.triu(self.M, k=1)  # obere/rechte Dreiecksmatrix
        self.M_D = sparse.diags(M.diagonal())
        # Diagonale Matrix (Werte auf der Hauptdiagonalen)
        self.M_C = sparse.linalg.inv(self.M_D + self.M_L)
        xx_M_L_array = self.M_L.toarray()
        xx_M_U_array = self.M_U.toarray()
        xx_M_D_array = self.M_D.toarray()
        xx_M_C_ega_array = self.M_C.toarray()
        # TODO: Testzuweisungen entfernen



    def gradient_field(self,
                       M,  # Koeffizientenmatrix
                       x):
        """ Gradient in Abhängigkeit der eingegeben Matrix, siehe self.create_coeff_gradient.

            Parameters:
                M   : Koeffizientenmatrix, z.B: M_x oder M_y übergeben
                x   : Abzuleitende Funktion
        """
        return M.dot(x)

    def solve_velocity_field(self):
        """ Berechnung der Gradientenfelder für d/dx und d/dy"""
        self.vx = self.gradient_field(self.M_y, self.psi)
        self.vy = -self.gradient_field(self.M_x, self.psi)
        psi_array = self.psi.reshape(self.ny, self.ny)
        vx_array = self.vx.reshape(self.ny, self.nx)
        vy_array = self.vy.reshape(self.ny, self.nx)
        # TODO: Testzuweisungen entfernen

    def solve_laplace(self, M, x):
        """ Berechnung von d²/dx²+d²/dy² (Laplace-Matrix)"""
        return M.dot(x)

    def solve_poisson_spsolve(self):
        # TODO: bereinigen

        omega = self.omega.copy()
        psi = sparse.linalg.spsolve(self.M, -omega)
        omega_probe = -self.M.dot(psi)
        #norm = np.linalg.norm(self.x_target_sparse.toarray() - psi)
        #print("Norm der Poissonlösung", norm)
        #psi_array = psi.reshape(self.ny, self.nx)
        #psi_diff = self.x_target-psi.reshape(self.ny, self.nx)
        diff = omega_probe - omega
        diff_array = diff.reshape(self.ny, self.nx)
        try:
            assert np.allclose(omega_probe, omega, atol=self.tol)  # Prüfung ob, Toleranz erreicht
        except:
            raise Warning("Die Vorgegebene Toleranz konnte nicht erreicht werden.")
        # self.psi = self.x_target_sparse.toarray()
        self.psi = psi

    def solve_rhs(self):
        """ Lösung der rechte-Hand-Seite (right hand side)

            -u*gradO(x) -v*gradO(y) + nue*laplaceO(xy)"""
        rhs_1 = (self.nue * self.solve_laplace(self.M, self.omega))
        rhs_2 = np.multiply(self.vx, (self.gradient_field(self.M_x, self.omega)))
        rhs_3 = np.multiply(self.vy, (self.gradient_field(self.M_y, self.omega)))
        return rhs_1 - rhs_2 - rhs_3

    def time_step(self):
        rhs = self.solve_rhs()  # righthandside -u*gradO(x) -v*gradO(y) + nue*laplaceO(xy)
        if self.time_solver is EulerTimeSolver:
            self.omega = self.time_solver.time_step(self.vx, self.vy, rhs)
        else:
            self.omega = self.time_solver.time_step(self.vx, self.vy, rhs)

    def __iter__(self) -> (float, np.array, np.array, np.array):
        self.set_omega0()
        self.set_matrices()
        self.solve_poisson_spsolve()
        self.solve_velocity_field()
        yield self.time_solver.t, self.psi.reshape(self.nx, self.ny), \
              self.vx.reshape(self.ny, self.nx)[2:, 1:-1], \
              self.vy.reshape(self.ny, self.nx)[1:-1, 2:]

        while self.t < self.t_max:
            self.time_step()
            self.omega = mc.assign_d_to_b(self, self.r, self.R_sp_inv, self.D, self.omega)
            self.solve_poisson_spsolve()
            self.solve_velocity_field()
            yield self.time_solver.t, self.psi.reshape(self.nx, self.ny), \
                  self.vx.reshape(self.ny, self.nx)[2:, 1:-1].transpose(), \
                  self.vy.reshape(self.ny, self.nx)[1:-1, 2:].transpose()

        """

        i = 1
        relax_faktor = 1
        while True:
        xxx_array_psi = psi.reshape((self.ny, self.nx))

            if np.allclose(psi, psi_n, atol=self.tol):
                print(i)
                break
        # psi_n = sparse.csc_matrix(sparse.linalg.spsolve_triangular(self.M_L+self.M_D, omega.toarray(), lower=True))
        # psi = self.M.dot(psi-omega)
        # psi = (-np.dot(np.dot(self.M_C, self.M_U), psi_n) + np.dot(self.M_C, omega))

        if np.allclose(psi, psi_n, atol=self.tol):
            print(np.linalg.norm(self.x_target_sparse.toarray() - psi))
            print(i)
        psi = psi_n + relax_faktor * (psi - psi_n)
        if i % 10 == 0:
            #print("Norm:", sparse.linalg.norm(psi - psi_n, np.inf))
            print("Norm:", np.linalg.norm(psi - psi_n, np.inf))
            if np.allclose(psi, psi_n, atol=self.tol):
                print(np.linalg.norm(self.x_target_sparse.toarray()-psi))
                print(i)
            break
        psi_n = psi.copy()
        norm_n = norm.copy()
        i += 1"""

        """
        def solver_poisson_iter(self):
            psi = -fg.omega_0_validation_sinx_siny(self.nx, self.ny, self.dx, self.dy)+np.sin(self.iteration)
            psi = sparse.csc_matrix(psi.reshape(self.nxny, 1))
            self.psi = psi
            self.iteration += 1"""

        """def solve_rhs(self):
            Lösung der rechte-Hand-Seite (right hand side)

                -u*gradO(x) -v*gradO(y) + nue*laplaceO(xy)
            rhs_1 = self.nue * self.solve_laplace(self.M, self.omega)
            rhs_2 = sparse.csc_matrix(self.vx).multiply(self.gradient_field(self.M_x, self.omega))
            rhs_3 = sparse.csc_matrix(self.vy).multiply(self.gradient_field(self.M_y, self.omega))
            return sparse.csr_matrix(rhs_1 - rhs_2 - rhs_3)"""

if __name__ == "__main__":

    # solver = Solver(ny=501)
    # solver = Solver(ny=21, l=1, d=1)
    # solver.set_omega0()
    # solver.solve_poisson_iter()
    # solver.get_speed_feld_from_psi()
    # solver.plot_speed_feld()
    # solver.plot_psi_feld()

    # start = time.time()
    # solver = IterSolver(RukuTimeSolver(), ny=21, L=1, d=1)
    # for t, psi, vx, vy in solver:
    #
    #     if np.isclose(t, 1):
    #         print(time.time() - start)
    #         solver.plot_speed_feld()
    #
    #     if np.isclose(t, 10):
    #         print(time.time() - start)
    #         solver.plot_speed_feld()
    #
    #     if np.isclose(t, 20):
    #         print(time.time() - start)
    #         solver.plot_speed_feld()
    #
    #     if np.isclose(t, 30):
    #         print(time.time() - start)
    #         solver.plot_speed_feld()
    #         break

    solver = MatrixSolver(RukuTimeSolver(), ny=21, x=1, y=1, t_max=50, dt=0.1)
    evaluator = Evaluator(solver)
    evaluator.v_feld_animation()
    #evaluator.psi_feld_animation()