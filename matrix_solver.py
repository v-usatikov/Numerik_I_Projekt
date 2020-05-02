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
import types

import function_generation as fg  # Import für Funktionen von omega


# todo implementation Sympy
# TODO: Implementierung von sparse.linalg.linearOperator,
#  siehe https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html
# TODO: banachscher Fixpunktsatz
# TODO: Interpolation spline (Stromlinien)
# TODO: Abwägung inwiefern die Performance wichtiger als Lesbarkeit ist ( Reduzierung der Unterfunktionen und
#  Variablen-Zuweisungen/ Operationen)
# TODO: Geometrie und Omega-Start-Generation auslagern
# TODO: auf Äquidistante Gitter prüfen
# todo func: geometry

# todo subclass Slicing
# todo report statistics
# todo RungeKutta

class CfdSolver:
    """Superclass CFD-Solver
    Zur allgemeinen Erläuterung der Variablennamen:
    - Vektoren / 1D-Arrays werden nach Möglichkeit mit kleinen Buchstaben bezeichnet
    - Matrizen / 2D-Arrays werden nach Möglichkeit mit kleinen Buchstaben bezeichnet
    """

    def __init__(self: object,
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
        self.x = float(x)  # Höhe in m
        self.y = float(y)  # Breite in m
        self.aufloesung = aufloesung  # aktuell nicht notwendig
        self.nx = int(nx)  # Anzahl Knoten in x-Richtung
        self.ny = int(ny)  # Anzahl Knoten in z-Richtung
        self.dx = y / (nx - 1)  # Abstand zwischen Knoten x-Richtung
        self.dy = x / (ny - 1)  # Abstand zwischen Knoten y-Richtung
        self.nxny = self.nx * self.ny
        self.relax_faktor = 1
        self.nt = t_max  # max. simulierte Zeitspanne
        self.dt = dt  # gewünschte zeitliche Auflösung (ggf. durch CFL überschrieben)
        self.nue = 1  # kinematische Viskosität
        self.conv_n1_target = conv_n1_target
        self.iter_outer = iter_outer  # max. Anzahl innerer Iterationen
        self.iter_inner = iter_inner  # max. Anzahl innerer Iterationen
        np.set_printoptions(precision=3, suppress=True)

        # Lösungsvariablen
        # self.b = np.zeros(self.nx, self.ny)

        self.solution_psi = []
        self.solution_omega = []
        self.solution_u = []
        self.solution_v = []

        # Vorbereitung des Gitters
        # TODO: ggf. in Funktion überführen (def create_mesh(self):)
        self.sx = np.linspace(0, self.x, self.nx)  # Abstand/Strecke Nullpunkt bis x
        self.sy = np.linspace(0, self.y, self.ny)  # Abstand/Strecke Nullpunkt bis y
        self.X_mesh, self.Y_mesh = np.meshgrid(self.sx, self.sy)
        self.c = np.zeros((self.nx, self.ny))  # Colormapping Initialwerte für Scatterplot

    def figure_init(self):
        # Initialisierung Artist-Objekte;
        # ax1 = Mesh-Darstellung (Scatter)
        # ax2 = Vektor-Darstellung (Vektor)
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        self.fig.suptitle("Gitterdarstellung")
        plt.rcParams["figure.figsize"] = (4.8, 4.8)  # ca. 12,2"
        plt.rcParams["figure.dpi"] = 100

        # Initialisierung für mesh
        self.scat_mesh = self.ax1.scatter(x=np.hstack(self.X_mesh),  # Vektor
                                          y=np.hstack(self.Y_mesh),  # Vektor
                                          c=np.hstack(self.c),  # Vektor
                                          s=5,  # Skalar oder Vektor
                                          cmap='gist_rainbow',
                                          vmin=0,  # Helligkeitsnormalisierung min
                                          vmax=11)  # Helligkeitsnormalisierung max
        self.fig.colorbar(self.scat_mesh, ax=self.ax1)

        # Initialisierung Plot Vektor
        self.quiver_sol = self.ax2.quiver(self.X_mesh, self.Y_mesh, self.X_mesh, self.Y_mesh)
        self.fig.colorbar(self.quiver_sol, ax=self.ax2)

        return self.fig, self.ax1, self.ax2, self.scat_mesh, self.quiver_sol

    def animation_init(self):
        self.ax1.set_xlabel("x")
        self.ax2.set_xlabel("X")
        self.ax1.set_ylabel("y")
        self.ax2.set_ylabel("Y")
        geom_x, geom_y = self.create_geometry()
        self.scat_geom = self.ax1.scatter(x=np.hstack(geom_x), y=np.hstack(geom_y), c='k', s=10)
        self.quiver_sol.set_UVC(self.solution_u[0].toarray().reshape(self.ny,self.nx),
                                self.solution_v[0].toarray().reshape(self.ny,self.nx))
        return self.quiver_sol,

    def animation_time(self, frame):
        self.ax1.set_xlabel(frame)
        self.ax2.set_xlabel(frame + 1)
        #self.plot_scatter(self.X_mesh, self.Y_mesh, self.solution_omega[frame], "Psi")
        self.quiver_sol.set_UVC(self.solution_u[frame].toarray().reshape(self.ny,self.nx),
                                self.solution_v[frame].toarray().reshape(self.ny,self.nx))
        return self.quiver_sol,

    def animation_local(self, frame):
        self.ax1.set_xlabel(frame)
        self.ax2.set_xlabel(frame + 1)
        # Scatter-Plot
        self.plot_scatter(self.X_mesh, self.Y_mesh, self.solution_psi[frame], "Psi")
        self.quiver_sol.set_UVC(self.solution_u[frame], self.solution_v[frame])
        return self.scat_geom,

    def plot_scatter(self, x, y, c, titel: str):
        fig, ax = plt.subplots()
        scat = ax.scatter(x, y, c=c)
        fig.colorbar(scat)
        fig.suptitle(titel)

    def plot_quiver(self, x, y, u, v, titel: str):
        fig, ax = plt.subplots()
        quiver = ax.quiver(x, y, u, v)
        fig.colorbar(quiver)
        fig.suptitle(titel)

    def plot_2dcont(self, x, y, c, titel: str):
        fig, ax = plt.subplots()
        contour = ax.contourf(x, y, c)
        fig.colorbar(contour)
        fig.suptitle(titel)

    def create_geometry(self):
        # TODO: Anpassung der Geometrie an das Gitter, falls Punkte nicht darauf liegen!
        # TODO: ggf. Auslagerung aus der Klasse
        rechteck = np.array([0.12, 0.12, .4, .4])
        r_len_x = rechteck[2] - rechteck[0]
        r_len_y = rechteck[3] - rechteck[1]

        # Anpassung an Gitter (Offset)
        for i in range(0, 4):
            r_off_x = rechteck[i] % self.dx
            rechteck[i] -= r_off_x
        for i in range(0, 4):
            r_off_y = rechteck[i] % self.dy
            rechteck[i] -= r_off_y

        # Überführung in Mesh-Struktur
        x_r = np.linspace(rechteck[0], rechteck[2], int(round(r_len_x / self.dx, 0) + 1))
        y_r = np.linspace(rechteck[1], rechteck[3], int(round(r_len_y / self.dy, 0) + 1))
        X_geom, Y_geom = np.meshgrid(x_r, y_r)
        return X_geom, Y_geom

    def omega_0_validation_3by3_matrix(self):
        """Voraussetzung sollte R == 0 und D == 1 sein, mit R[0,0,0,0,1,0,0,0,0] als Test für die Randbedingungen
        """

        """
        Alternatives Array
        self.A = np.array([[[4,-1,1,-1,1,0,1,2,-2], [-2,5,1,-2,1,-2,1,-2,1], [1,-2,3,-1,-1,2,-1,-2,1]],
                           [[-2,5,1,-2,1,-1,1,-2,1], [1,-2,1,-1,1,1,-1,1,2], [1,-2,5,-1,1,-2,2,-1,2,-1]],
                           [[1,-2,3,-1,-1,2,-1,-2,1], [4,-1,1,-1,1,0,1,2,-2], [1,-2,3,-1,-1,2,-1,-2,1]]])
        """
        x_target = np.arange(1, self.nx * self.ny + 1)
        diag_main = np.ones(self.nxny).ravel() * (-4)
        diag_rechts = np.tile(np.array([0, 1, 1]), self.nx)
        diag_links = np.tile(np.array([1, 1, 0]), self.nx)
        data_diag = np.array([diag_main, diag_rechts, diag_links])
        diags = np.array([0, 1, -1])
        A = spdiags(data_diag, diags, 9, 9)
        b = A.dot(x_target)  # Erzeugt np.array([ -2.  -4. -10. -11. -10. -19. -20. -16. -28.])
        return A, b

    def create_boundary(self, override_zeros: bool):
        """ Erzeugt aus einer Randmatrix (zur Eingabe der gewünschte Werte)
            eine dünnbesetzte Matrix, deren Invertierte (R==0) sowie einen Randvektor.

            Um die Randmatrix = 0 zu setzen, mit "override zeros" als "ja" ausführen"""
        # TODO: Fehlerfixing, Probleme, wenn R nicht überschrieben -> debugging
        if override_zeros is True:
            # keine Randzellen
            r = np.zeros(self.nxny)
            R = r.reshape(self.ny, self.nx)
        else:
            # R = sparse.dok_matrix((self.nx, self.ny))
            R = np.zeros((self.nx, self.ny))
            R[0, :] = 1  # erste Zeile, Werte u_1..u_nx
            R[1:, 0] = 1  # erste Spalte, Werte u_2,u_nx+1..u_ny
            R[1:, -1] = 1  # letzte Spalte, Werte zwischen u_nx..u_ny*nx
            R[-1, 1:-1] = 1  # letzte Zeile, Werte u_ny..u_ny*nx

        # R_sp : sparse-Matrix des Randvektors r
        R_sp = spdiags(R.ravel(), 0, self.nxny, self.nxny)
        # R_sp_inv : invertierter r-Vektor (NICHT INVERSE!), entspricht Matlab diag(R==0)
        R_sp_inv = spdiags((R.ravel() == 0) * 1, 0, self.nxny, self.nxny)
        # Konvertierung des np.array Vektors r in sparse-Matrix (spalten)
        r = sparse.csc_matrix(R.reshape(self.nxny, 1))
        return r, R_sp, R_sp_inv

    def create_dirichlet(self, r, factor):
        # TODO: ausbessern -> bei Eingabe von Werten
        d = r*factor  # z.B. alle Randwerte = 0 für Wände, oder = 1 für Druckausgleichsfläche
        d = sparse.csc_matrix(d.reshape(self.nxny, 1))  # Sicherstellen, dass das korrekte Format vorliegt
        #a = d.toarray().reshape(self.ny, self.nx)
        return d

    def create_neumann_input_laplace(self):
        # TODO: verständlichere Variablennamen erstellen

        nm_links_range = np.arange(0, self.ny)
        nm_links_x = np.repeat(nm_links_range, self.ny)
        nm_links_y = np.tile(nm_links_range, self.ny)
        nm_rechts_range = np.arange((self.nxny - 1), self.nxny - (self.ny + 1), -1)
        nm_rechts_x = np.repeat(nm_rechts_range, self.ny)
        nm_rechts_y = np.tile(nm_rechts_range, self.ny)

        nm_oben_range = np.arange(0, self.nxny, self.ny)
        nm_oben_x = np.repeat(nm_oben_range, self.nx)
        nm_oben_y = np.tile(nm_oben_range, self.nx)

        nm_unten_range = np.arange(self.ny - 1, self.nxny, self.ny)
        nm_unten_x = np.repeat(nm_unten_range, self.nx)
        nm_unten_y = np.tile(nm_unten_range, self.nx)

        # schnittpunkte_x = [self.ny, self.nxny - (self.ny + 1)]  # Ecken ausbessern
        # schnittpunkte_y = [0, self.nxny - 1]

        i = np.concatenate([nm_links_x, nm_rechts_x, nm_oben_x, nm_unten_x])
        j = np.concatenate([nm_links_y, nm_rechts_y, nm_oben_y, nm_unten_y])
        return i, j

    def assign_neumann_laplace(self, i, j, M_r, factor, ignore_y_n: bool):
        """ Einbau der Neumann-Randbedingungen

            Für Beispiel i,j siehe "create_neumann_input_laplace():"
            Parameters:
                i   : Array mit Indizes für Zeilen-Zuweisung der Matrix (z.B. Zeile 0, Zeile 0, Zeile 1)
                j   :
                M_r :
                ignore_y_n  : "Ja", wenn keine Neumann-RB gesetzt werden sollen
        """
        if ignore_y_n is True:
            return M_r
        else:
            M_r = M_r.todok()
            M_r[i, j] *= factor  # Symmetrie-Bedingung

            schnittpunkte_x = [0, self.ny - 1, self.nxny - (self.ny - 1), self.nxny - 1]  # Ecken ausbessern
            schnittpunkte_y = [0, self.ny - 1, self.nxny - (self.ny - 1), self.nxny - 1]
            M_r[schnittpunkte_x, schnittpunkte_y] *= 1

        return M_r.tocsr()

    def assign_polynom_to_gradient(self, M_x, M_y):
        """ Ändert die Werte der Matrizen für den Einbau von Randbedingungen in die Gradientenmatrizen.

            Hier werden die Zentraldifferenzen an den Rändern durch Rückwärtsdifferenzen ersetzt um
            den Verust der Genauigkeit auszugleichen.
        """
        # TODO: sinnvollere Varaiblennamen erstellen; derzeit keine konsistente Beschreibung -> unverständlich

        M_x = M_x.todok()
        M_y = M_y.todok()

        # Matrix für Ableitung in X-Richtung
        # Koeffizienten durch Taylorreihenentwicklung ->
        # spaltenindex k = 1. Spalte in x-Richtung (bei nx=4,ny=5 z.B. #1)

        dx_k_j0x = np.arange(0, self.ny)  # j=0-Zeile dx -> -3 (ij)
        dx_k_j0y = dx_k_j0x  # j=0 Spalte dx -> -3 (ij)
        M_x[dx_k_j0x, dx_k_j0y] = -3 / (self.dx*2)
        dx_k_jnx = np.arange(self.nxny - self.ny, self.nxny, 1)  # j=n Zeile dx -> 3 (ij)
        dx_k_jny = dx_k_jnx  # j=n Spalte dx -> 3 (ij)
        M_x[dx_k_jnx, dx_k_jny] = 3 / (self.dx*2)
        # spaltenindex l = 2. Spalte in x-Richtung (bei nx=4,ny=5 z.B. #5)
        dx_l_j0x = dx_k_j0x  # j=0-Zeile dx -> -3 (ij+1)
        dx_l_j0y = dx_k_j0y + self.ny  # j=0 Spalte dx -> -3 (ij+1)
        M_x[dx_l_j0x, dx_l_j0y] = 4 / (self.dx*2)
        dx_l_jnx = dx_k_jnx  # j=n Zeile dx -> 3 (ij+1)
        dx_l_jny = dx_k_jny - self.ny  # j=n Spalte dx -> 3 (ij+1)
        M_x[dx_l_jnx, dx_l_jny] = -4 / (self.dx*2)
        # spaltenindex m = 3. Spalte in x-Richtung  (bei nx=4,ny=5 z.B. #9)
        dx_m_j0x = dx_k_j0x  # j=0-Zeile dx -> -3 (ij+2)
        dx_m_j0y = dx_l_j0y + self.ny  # j=0 Spalte dx -> -3 (ij+2)
        M_x[dx_m_j0x, dx_m_j0y] = -1 / (self.dx*2)
        dx_m_jnx = np.arange(self.nxny - self.ny, self.nxny, 1)  # j=n Zeile dx -> 3 (ij+2)
        dx_m_jny = dx_l_jny - self.ny  # j=n Spalte dx -> 3 (ij+2)
        M_x[dx_m_jnx, dx_m_jny] = 1 / (self.dx*2)

        # Matrix für Ableitung in y-Richtung
        # Spaltenname k = 1. Spalte in x-Richtung (bei nx=4,ny=5 z.B. #1, 2. = #5)
        # p für Normale an dem Rand in positive y-Richtung
        dy_p_k_zeilen = np.arange(0, self.nxny - (self.ny - 1), self.ny)  # Zeilenindizes -> Werte -3 (ij)
        dy_p_k_spalten = dy_p_k_zeilen  # Spaltenindizes -> Werte -3 (ij)
        # dy_p_l_zeilen = dy_p_k_zeilen  # Zeilenindizes -> Werte 4 (ij+1)
        dy_p_l_spalten = dy_p_k_zeilen + 1  # Spaltenindizes -> Werte 4 (ij+1)
        # dy_p_m_zeilen = dy_p_k_zeilen  # Zeilenindizes -> Werte -1 (ij+2)
        dy_p_m_spalten = dy_p_k_zeilen + 2  # Spaltenindizes -> Werte -1 (ij+2)
        dy_p_null_zeilen = np.arange(self.ny, self.nxny - (self.ny - 1), self.ny)  # Löschen alter Werte
        dy_p_null_spalten = np.arange(self.ny - 1, self.nxny - (self.ny - 1), self.ny)  # Löschen alter Werte
        M_y[dy_p_k_zeilen, dy_p_k_spalten] = 3 / (self.dy*2)
        M_y[dy_p_k_zeilen, dy_p_l_spalten] = -4 / (self.dy*2)
        M_y[dy_p_k_zeilen, dy_p_m_spalten] = 1 / (self.dy*2)
        M_y[dy_p_null_zeilen, dy_p_null_spalten] = 0

        # Matrix für Ableitung in y-Richtung
        # Spaltenname k = 1. Spalte in x-Richtung (bei nx=4,ny=5 z.B. #1, 2. = #5)
        # n für Normale an dem Rand in negative y-Richtung
        dy_n_k_zeilen = np.arange(self.ny - 1, self.nxny, self.ny)  # Zeilenindizes -> Werte -3 (ij)
        dy_n_k_spalten = dy_n_k_zeilen  # Spaltenindizes -> Werte -3 (ij)
        # dy_n_l_zeilen = dy_n_k_zeilen  # Zeilenindizes -> Werte 4 (ij+1)
        dy_n_l_spalten = dy_n_k_zeilen - 1  # Spaltenindizes -> Werte 4 (ij-1)
        # dy_n_m_zeilen = dy_n_k_zeilen  # Zeilenindizes -> Werte -1 (ij+2)
        dy_n_m_spalten = dy_n_k_zeilen - 2  # Spaltenindizes -> Werte -1 (ij-2)
        dy_n_null_zeilen = np.arange(self.ny - 1, self.nxny - (self.ny - 1), self.ny)  # Löschen alter Werte
        dy_n_null_spalten = np.arange(self.ny, self.nxny, self.ny)  # Löschen alter Werte
        M_y[dy_n_k_zeilen, dy_n_k_spalten] = -3 / (self.dy*2)
        M_y[dy_n_k_zeilen, dy_n_l_spalten] = 4 / (self.dy*2)
        M_y[dy_n_k_zeilen, dy_n_m_spalten] = -1 / (self.dy*2)
        M_y[dy_n_null_zeilen, dy_n_null_spalten] = 0

        return M_x.tocsr(), M_y.tocsr()

    def create_coeff_laplace_polynom(self):
        """ Erstellt eine Koeffizientenmatrix mit Vorwärts/ Rückwärts- Interpolation an den Rändern.

            Hier werden die Zentraldifferenzen an den Rändern durch Rückwärtsdifferenzen ersetzt um
            den Verust der Genauigkeit auszugleichen.

            WICHTIG! Aufgrund der Durchführbarkeit der Automatisierung werden hier zwei Matrizen erstellt und
            zusammengeführt um eine neue Matrix zu erstellen, anstatt die Werte einer vorher eingegebenen Matrix
            zu überschreiben!

            WICHTIG! Automatisierung für die Randerkennung aus der Randmatrix fehlt noch! Manuelle erstellung
            nur für rechtwinklige Systeme gültig.
        """
        # TODO: Automatische Erkennung der Ränder und deren Ausrichtung mittels Eingabe des Randvektors

        # Matrix für Ableitung in X-Richtung
        # Koeffizienten durch Taylorreihenentwicklung ->

        dx_haupt = np.ones(self.nxny)*(-2)  # Hauptdiagonale (HD)
        dx_haupt[:self.ny] = 1
        dx_haupt[self.nxny-self.ny:self.nxny] = 1

        dx_haupt_o = np.ones(self.nxny)  # Hauptdiagonale oberhalb HD
        dx_haupt_o[self.ny:2*self.ny] = -2
        dx_haupt_u = np.ones(self.nxny)  # Hauptdiagonale unterhalb HD
        dx_haupt_u[self.nxny-2*self.ny:self.nxny-self.ny] = -2

        dx_neben_o = np.zeros(self.nxny)  # Nebendiagonale oberhalb der HD
        dx_neben_o[2*self.ny:2*self.ny+(self.ny)] = 1
        dx_neben_u = np.zeros(self.nxny)  # Nebendiagonale unterhalb der HD
        dx_neben_u[self.nxny-3*self.ny:self.nxny-2*self.ny] = 1

        dx_data = [dx_haupt, dx_haupt_o, dx_haupt_u, dx_neben_o, dx_neben_u]
        dx_diags = [0, self.ny, -self.ny, 2*self.ny, -2*self.ny]

        dx_spdiags = spdiags(dx_data, dx_diags, self.nxny, self.nxny) / self.dx**2

        # Matrix für Ableitung in y-Richtung
        dy_haupt = np.ones(self.ny)*(-2)  # Hauptdiagonale (HD)
        dy_haupt[0] = 1
        dy_haupt[-1] = 1
        dy_haupt = np.tile(dy_haupt,self.nx)

        dy_haupt_o = np.ones(self.ny)  # Hauptdiagonale oberhalb HD
        dy_haupt_o[1] = -2
        dy_haupt_o[0] = 0
        dy_haupt_o = np.tile(dy_haupt_o, self.nx)

        dy_haupt_u = np.ones(self.ny)  # Hauptdiagonale unterhalb HD
        dy_haupt_u[-1] = 0
        dy_haupt_u[-2] = -2
        dy_haupt_u = np.tile(dy_haupt_u, self.nx)
        dy_haupt_u[-1] = -2

        dy_neben_o = np.zeros(self.ny)   # Nebendiagonale oberhalb der HD
        dy_neben_o[2] = 1
        dy_neben_o = np.tile(dy_neben_o, self.nx)

        dy_neben_u = np.zeros(self.ny)  # Nebendiagonale unterhalb der HD
        dy_neben_u[-3] = 1
        dy_neben_u = np.tile(dy_neben_u, self.nx)

        dy_data = [dy_haupt, dy_haupt_o, dy_haupt_u, dy_neben_o, dy_neben_u]
        dy_diags = [0, 1, -1, 2, -2]

        dy_spdiags = spdiags(dy_data, dy_diags, self.nxny, self.nxny) / self.dy**2
        dy_array = dy_spdiags.toarray()
        M = dx_spdiags + dy_spdiags
        M_array = M.toarray()

        return M.tocsr()

    def omega_0_conv_n1ert_to_csr(self, omega):
        print(type(omega), omega.shape)
        omega_0 = sparse.csr_matrix(omega)
        return omega_0.reshape(self.nxny, 1)

    def create_coeff_laplace_block(self):
        """ Funktion zur Erstellung der pentagiagonalen Koeffizientenmatrix
            ausgehend von der Poisson-Gleichung (d2p/dx2 + d2p/dy2) = omega_n

            !! im Test deutlich langsamer (bei 60x60 Zellen 380 ms vs. 36 ms), wird nicht weiter genutzt !!
        """
        diag_block = np.eye(self.ny) * (- 2 / self.dx ** 2 - 2 / self.dy ** 2)
        diag_block = diag_block + np.eye(self.ny, k=1) * (1 / self.dy ** 2)
        diag_block = diag_block + np.eye(self.ny, k=-1) * (1 / self.dy ** 2)
        M = np.kron(np.eye(self.nx), diag_block)
        M += np.eye(self.ny * self.nx, k=self.ny) * (1 / self.dx ** 2)
        M += np.eye(self.ny * self.nx, k=-self.ny) * (1 / self.dx ** 2)
        self.create_coeff_laplace()
        return M

    def create_coeff_laplace(self):
        """ Funktion zur Erstellung der pentagiagonalen Koeffizientenmatrix

            ausgehend von der Poisson-Gleichung (d2p/dx2 + d2p/dy2) = omega_n
            Verwendet sparse-Matrizen-Klasse von scipy
        """
        diag_haupt = np.ones(self.nxny).ravel() * (- 2 / (self.dx ** 2) - 2 / (self.dy ** 2))
        o = np.ones(self.ny)
        o[0] = 0
        diag_haupt_o = np.tile(o, self.nx) * (1 / (self.dy ** 2))
        u = np.ones(self.ny)
        u[-1] = 0
        diag_haupt_u = np.tile(u, self.nx) * (1 / (self.dy ** 2))
        diag_neben_o = np.ones(self.nxny) * (1 / (self.dx ** 2))
        diag_neben_u = np.ones(self.nxny) * (1 / (self.dx ** 2))
        data = np.array([diag_haupt, diag_haupt_o, diag_haupt_u, diag_neben_o, diag_neben_u])
        diags = np.array([0, 1, -1, self.ny, -self.ny])
        M = spdiags(data, diags, self.nxny, self.nxny)
        return M

    def create_coeff_gradient_dx(self):
        """ Funktion zur Erstellung der pentagiagonalen Koeffizientenmatrix

            ausgehend von der Poisson-Gleichung (d2p/dx2 + d2p/dy2) = omega_n
            Verwendet sparse-Matrizen-Klasse von scipy
        """
        diag_haupt = np.zeros(self.nxny).ravel()
        o = np.ones(self.nx)
        diag_haupt_o = np.tile(o, self.ny) * (1 / (self.dx * 2))
        u = np.ones(self.nx)
        diag_haupt_u = np.tile(u, self.ny) * (-1 / (self.dx * 2))
        data = np.array([diag_haupt, diag_haupt_o, diag_haupt_u])
        diags = np.array([0, self.ny, -self.ny])
        M_x = spdiags(data, diags, self.nxny, self.nxny)
        return M_x

    def create_coeff_gradient_dy(self):
        """ Funktion zur Erstellung der pentagiagonalen Koeffizientenmatrix

            ausgehend von der Poisson-Gleichung (d2p/dx2 + d2p/dy2) = omega_n
            Verwendet sparse-Matrizen-Klasse von scipy
        """
        diag_haupt = np.zeros(self.nxny).ravel()
        o = np.ones(self.nx)
        o[0] = 0
        diag_haupt_o = np.tile(o, self.ny) * (-1 / (self.dy * 2))
        u = np.ones(self.nx)
        u[-1] = 0
        diag_haupt_u = np.tile(u, self.ny) * (1 / (self.dy * 2))
        data = np.array([diag_haupt, diag_haupt_o, diag_haupt_u])
        diags = np.array([0, 1, -1])
        M_y = spdiags(data, diags, self.nxny, self.nxny)
        return M_y

    def assign_R_to_M(self, R_sp, R_sp_inv, M):
        """ Vorbereitung der Koeffizientenmatrix auf den Löservorgang

            Parameters:
                R_sp : sparse-Matrix des Randvektors r  [1,1,0,0,1,...]
                R_sp_inv : invertierter r-Vektor (NICHT INVERSE!), entspricht Matlab diag(R==0):
                            aus [1,1,0,0,1,...] wird [0,0,1,1,0,...]
                M : Matrix, welche angepasst werden soll
        """
        M_r = R_sp_inv.dot(M) + R_sp
        M_array = M_r.toarray()
        return M_r.tocsr()  # M_r für Koeff.Matrix mit Randvektor

    def assign_d_to_b(self, r, R_sp_inv, d, b):
        try:
            r_d = sparse.csc_matrix(r.multiply(d))
            b_r_d = r_d + R_sp_inv.dot(b)
        except:  # sollte nur passieren, wenn alle Elemente von d == 0
            r_d = sparse.csc_matrix(r.multiply(d))
            b_r_d = R_sp_inv.dot(b)  # Alternative zu matlab (R==0).*(-b)
            """
            a = R_sp_inv.dot(b)  # nach unten verschieben, wenn ok
            b_r_d = r_d + a"""
            """
            a_array = a.toarray()
            r_d_array = r_d.toarray()
            brd_array = b_r_d.toarray()
            """
        return b_r_d

    def solve_gradient_dx_dy(self, M_x, M_y, x):
        """Diskretisierung durch Multiplikation mit einer Koeffizientenmatrix.

            Parameters:
                M_x : Koeffizientenmatrix für die Ableitung nach dx
                M_y : Koeffizientenmatrix für die Ableitung nach dy
            Returns:
                u : Vektor für Geschwindigkeit in x-Richtung
                v : Vektor für Geschwindigkeit in y-Richtung
        """
        #print("M_x:")
        #self.analyse_fixpoint_attraction(M_x)
        #print("M_y:")
        #self.analyse_fixpoint_attraction(M_y)
        u = M_x.dot(x)
        v = M_y.dot(x)
        return sparse.csr_matrix(u), sparse.csr_matrix(v)

    def gradient_field(self,
                       M,  # Koeffizientenmatrix
                       x):
        """ Gradient in Abhängigkeit der eingegeben Matrix, siehe self.create_coeff_gradient.

            Parameters:
                M   : Koeffizientenmatrix, z.B: M_x oder M_y übergeben
                v   : Abzuleitende Funktion
        """
        return M.dot(x)

    def solve_poisson_spsolve(self, M_r, r, R_sp_inv, d, b):
        """Gesamtschrittverfahren mit Laplace-Matrix

           Parameters:
               M_r      : Koeffizientenmatrix in einem der sparse-Formate
               R        : Randvektor im csc oder csr-Format
               R_sp_inv : (R==0)-Vektor als sparse-Matrix in csr-Format
               b        : RHS, hier vorgegebene Funktion z.B. -omgea, psi, etc
        """

        self.analyse_fixpoint_attraction(M_r)  # anziehender oder abstossender Fixpunkt?

        b_r_d = self.assign_d_to_b(r, R_sp_inv, d, b)*((self.dx**2) * (self.dy**2) / (-2 * (self.dx**2 + self.dy**2)))

        a = sparse.linalg.spsolve(M_r, b_r_d)  # Lösungsvektor
        a = sparse.csc_matrix(a).reshape(self.nxny,1)
        # Randbedingungen wieder aufprägen
        #a = sparse.csc_matrix(R_sp_inv.dot(a)).transpose()

        probe = M_r.dot(a) - (b)
        print("Max. Diff. Probe", sparse.linalg.norm(probe, np.inf))  # M.*x_iter = b?
        print("max. / min. Werte", probe.max(), probe.min())

        return a

    def switcher(self, comparison):
        def poisson():
            b_coeff = (self.dx**2) * (self.dy**2) / (-2 * (self.dx**2 + self.dy**2))
            return b_coeff

        switch_b = {
            "poisson": poisson(),
            "laplace": 0,
            "gradient_X": 0,
            "gradient_y": 0,
            90: 0}  # nur zu Testzwecken

        return switch_b.get(comparison, "Bitte Eingabe prüfen")

    def solve_local_discret(self, M, x_start, b, formulation: str, conv_n1_target: float):
        """ Einschrittverfahren zentraler Differenzqupotient

            Lösung der Gleichung Mx = b iterativ, wobei M eine Koeffizientenmatrix ist

            Parameters:
                M           : Koeffizientenmatrix
                x_start     : Startwert fur x
                b           : Rechte Seite der Gleichung hier Spalten-Vektor
                formulation : mögliche Eingaben; "poisson","laplace","gradient_X","gradient_y":
        """
        # TODO: Lipschitzbedingung erfüllt? |f(t,y) - f(t,y_n)| <= L |y-y_n|
        # TODO: Relaxationsverfahren ergänzen

        conv_n1 = 1  # Konvergenzkriterium durch L2-Norm
        conv_n = conv_n1
        conv_iter = 0
        iteration = 0
        xdata = []
        ydata = []
        # Vorbereitung # TODO: ggf. auslagern?
        x_n = x_start.copy()  # Startwert für Omega
        norm = self.analyse_fixpoint_attraction(A)
        try:
            C = sparse.linalg.inv(D + L)
            spektral = self.analyse_spektralradius(C, A)
        except:
            C = lg.inv(D + L)
            spektral = self.analyse_spektralradius(C, A)
        if spektral > 1:
            relax_faktor = 1 / spektral  # Annäherung an einen Korrekten Relax.Faktor (funktioniert i.d.R. ganz i.O.)
            relax_faktor = 0.1
        else:
            relax_faktor = 1

        while conv_n1 > conv_n1_target:
            iteration += 1
            x_n1 = np.dot(np.dot(-C, U), x_n) + np.dot(C, b)  # xn+1=-C*Rxn+C*b
            try:
                conv_n1 = sparse.linalg.norm((x_n1 - x_n))
                if conv_n1 >= conv_n:
                    conv_iter += 1
                    relax_faktor /= conv_n1
                    print("Relaxationsparameter angepasst:", relax_faktor)
                    if conv_iter > 500 or np.round(relax_faktor, 5) == 0:
                        print("Keine weitere Anpassung möglich. Differentiation ist divergent!")
                        break
            except:
                conv_n1 = lg.norm((x_n1 - x_n), np.inf)
            x_n1 = x_n + relax_faktor * (x_n1 - x_n)  # Relaxationsverfahren
            x_n = x_n1.copy()
            conv_n = conv_n1.copy()
            # zur grafischen Darstellung Export der Daten
            print(iteration, conv_n1)
            xdata.append(iteration)
            ydata.append(conv_n1)
            if iteration == self.iter_inner:
                probe = A.dot(x_n1) - (b)
                print("Konvergenzkriterium nicht erreicht. Maximale Iteration", iteration, "Konv.=", conv_n1)
                print(sp.pprint(sp.Matrix(probe.toarray())))
                print("Max. Diff. Probe", sparse.linalg.norm(probe, np.inf))  # M.*x_iter = b?
                print("max. / min. Werte", probe.max(), probe.min())
                break
        probe = A.dot(x_n1) - (b)
        print("Max. Diff. Probe", sparse.linalg.norm(probe, np.inf))  # M.*x_iter = b?
        print("max. / min. Werte", probe.max(), probe.min())
        return x_n1

    def solve_poisson_iteration(self):


    def solve_jacobi(self, A, b, x_start, conv_n1_target):
        """ entspricht dem Jacobi-Lösungsverfahren mit Inversen Ax=b \n
            xn+1=-inv(D)*((L+R)*xn-b)
        """
        # TODO: fix

        try:
            A = A.tocsc()
        except:
            raise Warning("Keine Sparse-Matrix übergeben! Ineffiziente Berechnung!")

        L, U, D = self.lud_decomposition(A)  # Zerlegung A in  Links-, Rechts- und Diagonalmatrizen (LRD)
        conv_n1 = 1  # Konvergenzkriterium durch L2-Norm
        iteration = 0
        xdata = []
        ydata = []

        x_n = x_start.copy()  # Startwert für Omega
        norm = self.analyse_fixpoint_attraction(A)
        try:
            C = sparse.linalg.inv(D)
            spektral = self.analyse_spektralradius(C, A)
        except:
            C = lg.inv(D)
            spektral = self.analyse_spektralradius(C, A)
        if spektral > 1:
            relax_faktor = 1 / norm  # Annäherung an einen Korrekten Relax.Faktor (funktioniert i.d.R. ganz i.O.)
            relax_faktor = 1  # manuelles Setzen des Faktors
        else: relax_faktor = 1
        while conv_n1 > conv_n1_target:
            iteration += 1
            x_n1 = np.dot(-C, (np.dot((L + U), x_n) - b))
            try:
                conv_n1 = sparse.linalg.norm((x_n1 - x_n), 'fro')
                # conv_n1 = sparse.linalg.norm((psi - psi_n), np.inf)
            except:
                conv_n1 = lg.norm((x_n1 - x_n), np.inf)
            x_n1 = x_n + relax_faktor * (x_n1 - x_n)  # Relaxationsverfahren
            x_n = x_n1.copy()
            # zur grafischen Darstellung Export der Daten
            print(iteration, conv_n1)
            xdata.append(iteration)
            ydata.append(conv_n1)
            if iteration == self.iter_inner:
                probe = A.dot(x_n1) - (b)
                print("Konvergenzkriterium nicht erreicht. Maximale Iteration", iteration, "Konv.=", conv_n1)
                print(sp.pprint(sp.Matrix(probe.toarray())))
                print("Max. Diff. Probe", sparse.linalg.norm(probe, np.inf)) # M.*x_iter = b?
                print("max. / min. Werte", probe.max(), probe.min())
                break
        probe = A.dot(x_n1) - (b)
        print("Max. Diff. Probe", sparse.linalg.norm(probe, np.inf))  # M.*x_iter = b?
        print("max. / min. Werte", probe.max(), probe.min())
        return x_n1, xdata, ydata

    def solve_gausseidel(self, A, b, x_start, conv_n1_target):
        """ entspricht dem Jacobi-Lösungsverfahren mit Inversen Ax=b \n
            C = inv(D+L)
            xn+1=-C*Rxn+C*b
            siehe S.60 [1]
        """
        try:
            A = A.tocsc()
        except:
            raise Warning("Keine Sparse-Matrix übergeben! Ineffiziente Berechnung!")

        L, U, D = self.lud_decomposition(A)  # Zerlegung A in  Links-, Rechts- und Diagonalmatrizen (LRD)
        conv_n1 = 1  # Konvergenzkriterium durch L2-Norm
        conv_n = conv_n1
        conv_iter = 0
        iteration = 0
        xdata = []
        ydata = []
        # Vorbereitung # TODO: ggf. auslagern?
        x_n = x_start.copy()  # Startwert für Omega
        norm = self.analyse_fixpoint_attraction(A)
        try:
            C = sparse.linalg.inv(D + L)
            spektral = self.analyse_spektralradius(C, A)
        except:
            C = lg.inv(D + L)
            spektral = self.analyse_spektralradius(C, A)
        if spektral > 1:
            relax_faktor = 1 / spektral  # Annäherung an einen Korrekten Relax.Faktor (funktioniert i.d.R. ganz i.O.)
            relax_faktor = 0.1
        else: relax_faktor = 1

        while conv_n1 > conv_n1_target:
            iteration += 1
            x_n1 = np.dot(np.dot(-C, U), x_n) + np.dot(C, b)  # xn+1=-C*Rxn+C*b
            try:
                conv_n1 = sparse.linalg.norm((x_n1 - x_n))
                if conv_n1 >= conv_n:
                    conv_iter += 1
                    relax_faktor /= conv_n1
                    print("Relaxationsparameter angepasst:", relax_faktor)
                    if conv_iter > 500 or np.round(relax_faktor, 5) == 0:
                        print("Keine weitere Anpassung möglich. Differentiation ist divergent!")
                        break
            except:
                conv_n1 = lg.norm((x_n1 - x_n), np.inf)
            x_n1 = x_n + relax_faktor * (x_n1 - x_n)  # Relaxationsverfahren
            x_n = x_n1.copy()
            conv_n = conv_n1.copy()
            # zur grafischen Darstellung Export der Daten
            print(iteration, conv_n1)
            xdata.append(iteration)
            ydata.append(conv_n1)
            if iteration == self.iter_inner:
                probe = A.dot(x_n1) - (b)
                print("Konvergenzkriterium nicht erreicht. Maximale Iteration", iteration, "Konv.=", conv_n1)
                print(sp.pprint(sp.Matrix(probe.toarray())))
                print("Max. Diff. Probe", sparse.linalg.norm(probe, np.inf))  # M.*x_iter = b?
                print("max. / min. Werte", probe.max(), probe.min())
                break
        probe = A.dot(x_n1) - (b)
        print("Max. Diff. Probe", sparse.linalg.norm(probe, np.inf))  # M.*x_iter = b?
        print("max. / min. Werte", probe.max(), probe.min())
        """
        fig, ax = plt.subplots()
        fig.suptitle("Interner Numpy-Solver")
        ax.set(title='Konvergenz lokale Diskretisierung', xlabel='Iteration', ylabel='Konvergenzkriterium')
        ax.set_ylim(0, 10)
        konvergenz = ax.plot(xdata, ydata)"""
        return x_n1, xdata, ydata

    def solve_npsolve(self, A, b, x_start, conv_n1_target):
        """ entspricht dem Jacobi-Lösungsverfahren mit Inversen Ax=b \n
            xn+1=-inv(D)*((L+R)*xn-b)
        """
        # TODO: Sparse-Solver implementieren (gibt nzvals-Fehler für b)
        D = sparse.diags(A.diagonal())  # Diagonale Matrix (Werte auf der Hauptdiagonalen)

        try:
            A = A.tocsc()
        except:
            raise Warning("Keine Sparse-Matrix übergeben! Ineffiziente Berechnung!")

        L, R, D = self.lud_decomposition(A)  # Zerlegung A in  Links-, Rechts- und Diagonalmatrizen (LRD)
        conv_n1 = 1  # Konvergenzkriterium durch L2-Norm
        iteration = 0
        xdata = []
        ydata = []

        x_n = x_start.copy()  # Startwert für Omega
        self.analyse_fixpoint_attraction(A)
        relax_faktor = 1
        while conv_n1 > conv_n1_target:
            iteration += 1
            x_n1 = sparse.csc_matrix(sparse.linalg.spsolve(A, b)).reshape(self.nxny,1)
            try:
                conv_n1 = sparse.linalg.norm((x_n1 - x_n), np.inf)
                # conv_n1 = sparse.linalg.norm((psi - psi_n), np.inf)
            except:
                conv_n1 = lg.norm((x_n1 - x_n), np.inf)
            x_n1 = x_n + relax_faktor * (x_n1 - x_n)  # Relaxationsverfahren
            x_n = x_n1.copy()
            # zur grafischen Darstellung Export der Daten
            print(iteration, conv_n1)
            xdata.append(iteration)
            ydata.append(conv_n1)
            if iteration == self.iter_inner:
                probe = A.dot(x_n1) - (-b)
                print("Konvergenzkriterium nicht erreicht. Maximale Iteration", iteration, "Konv.=", conv_n1)
                print(sp.pprint(sp.Matrix(probe.toarray())))
                print("Max. Diff. Probe", sparse.linalg.norm(probe, np.inf))  # M.*x_iter = b?
                print("max. / min. Werte", probe.max(), probe.min())
                break
        probe = A.dot(x_n1) - (-b)
        print("Max. Diff. Probe", sparse.linalg.norm(probe, np.inf))  # M.*x_iter = b?
        print("max. / min. Werte", probe.max(), probe.min())
        return x_n1, xdata, ydata

    def lud_decomposition(self, A):
        # TODO: ggf. umbenennen in LUD-decomposition (links,rechts,diagonal)->(lower,upper,diagonal)
        try:
            L = sparse.tril(A, k=-1)  # untere/linke Dreiecksmatrix
            R = sparse.tril(A, k= 1)  # obere/rechte Dreiecksmatrix
            D = sparse.tril(A, k= 0)  # Diagonale Matrix (Werte auf der Hauptdiagonalen)
        except:
            L = np.tril(A, -1)  # untere/linke Dreiecksmatrix
            # sp.pprint(L)
            R = np.triu(A, 1)  # obere/rechte Dreiecksmatrix
            # sp.pprint(R)
            D = np.diag(np.diag(A))  # Diagonale Matrix (Werte auf der Hauptdiagonalen)
            # sp.pprint(D)
        return L, R, D

    def analyse_fixpoint_attraction(self, B):
        try:
            norm = sparse.linalg.norm(B, np.inf)
            print("Norm =", norm)
        except:
            print("Warnung: keine Sparse-Matrix eingegeben")
            k = np.ones(1)
            if type(B) != type(k):
                B = B.toarray()
            norm = lg.norm(B, np.inf)
            print("Norm =", norm)
        if norm < 1:
            print("anziehender Fixpunkt")
        else:
            print("abstossender Fixpunkt")
        return norm

    def analyse_spektralradius(self,
                               C,  # Verfahrensmatrix
                               A  # Matrix A bzw. M
                               ):
        """ Gibt den Spektralradius des angewendeten Verfahrens und der Matrix zurück.

            Parameters:
                C   : Verfahrensmatrix z.B: Gauss, Jacobi, QR, etc.
                A   : Matrix die geprüft werden soll (z.B: Koeff.matrix für Poisson)
        """
        try:
            I = sparse.csr_matrix(np.eye(self.nxny))
            spektral = sparse.linalg.norm(I - np.dot(A, C), np.inf)
        except:
            k = np.ones(1)
            if type(C) != type(k):
                print("C konvertiert")
                C = C.toarray()
            if type(A) != type(k):
                print("A konvertiert")
            spektral = np.linalg.norm(I - np.dot(A, C), np.inf)
        print("Spektralradius:", spektral)
        return spektral

    def solve_time_discret(self, M_r, M_x, M_y, u, v, omega, conv_n1_target):
        """ zeitliche Diskretisierung mittels Euler vorwärts/zentral
        (u_n+1 - u_n-1) / (2*dt) = rhs_n
        mit rhs_n als lokale Diskretisierung
        --> domega/dt = nue(d²omega/dx²+d²omega/dy²) - u*domega/dx - v*domega/dy
        """
        # TODO: performance-Optimierung notwendig
        # die Verwendung von b als 2. Ableitung ist hier nicht korrekt
        # TODO: CFL-Kriterium einbauen
        """
        # erster Durchlauf als Vorbereitung für LEAPFROG / mittlere Differenzschema
        domega_dx, domega_dy = self.solve_gradient_dx_dy(M_x, M_y, omega)
        laplace_omega = M_n.dot(omega)  # TODO: entfernen, wenn geprüft
        rhs_n = M_n.dot(omega) - u.dot(domega_dx) - v.dot(domega_dy)
        omega_n1 = rhs_n * self.dt + omega_n  # Berechnung omega für nächsten Zeitschritt
        conv_n1 = sparse.linalg.norm(x_n1 - x_n, np.inf)
        omega_n = omega_n1.copy()"""

        # grafische Darstellung
        data_x = []
        data_y = []
        fig, ax = plt.subplots()
        ax.set(title='Konvergenz zeitliche Diskretisierung', xlabel='Iteration', ylabel='Konvergenzkriterium')
        plot_conv_n1, = ax.plot(data_x, data_y)

        epsilon = 0.001  # Relaxationsfaktor
        omega_n = omega
        conv_n1_n = 1
        conv_n1_n1 = 1  # Variable zur Konvergenzüberwachung
        iteration = 0  # Variable zur Betrachtung des Iterationsverlaufs
        while conv_n1_n1 > conv_n1_target or iteration < self.iter_outer:
            # TODO: Ergänzung LEAPFROG
            # TODO: Fehlerbeseitigung Zeitverlauf
            if conv_n1_n1 > conv_n1_n:
                conv_n1_q = conv_n1_n1 / conv_n1_n
                epsilon = epsilon / (conv_n1_q*100)
            psi_n = self.solve_poisson_spsolve(M_r, r, R_sp_inv, d, b)

            domega_dx, domega_dy = self.solve_gradient_dx_dy(M_x, M_y, omega_n)  # lokale Diskretisierung do/dx+do/dy
            u, v = self.solve_gradient_dx_dy(M_x, M_y, psi_n)
            laplace_omega = M_n.dot(omega_n)  # TODO: entfernen, wenn geprüft
            a = M_n.dot(omega_n)
            b = - u.multiply(domega_dx)
            c = - v.multiply(domega_dy)
            rhs_n =  a + b + c   # Berechnung RHS (right hand side)
            omega_n1 = rhs_n * self.dt + omega_n  # Berechnung omega für nächsten Zeitschritt
            conv_n1_n1 = sparse.linalg.norm(omega_n1 - omega_n, np.inf)  # Berechnung der Norm als Konvergenzkriterium
            omega_n1 = omega_n + epsilon * (omega_n1 - omega_n)  # Relaxationsverfahren

            omega_n = omega_n1.copy()  # Lösungsvariablen weiter "schieben"
            conv_n1_n = conv_n1_n1.copy()

            iteration += 1
            time = iteration*self.dt

            data_x.append(iteration)  # für plot
            data_y.append(conv_n1_n1)  # für plot

            if iteration % 250 == 0:
                print(iteration, ";", "%f2" % time, "s;", conv_n1_n1)
                plot_conv_n1.set_data(data_x, data_y)  # update des plot
                ax.set_xlabel("Iteration:" + str(iteration))  # update des plot
                #self.plot_quiver(self.X_mesh, self.Y_mesh, u.toarray().reshape(self.ny, self.nx), v.toarray().reshape(self.ny, self.nx), "Vektorplot")
                self.solution_u.append(u)
                self.solution_v.append(v)
                self.solution_omega.append(omega_n1)
        plot_conv_n1.set_data(data_x, data_y)
        return omega_n1, u, v

    def create_nullvector(self):
        b = sparse.coo_matrix((self.nxny, 1))
        return sparse.csr_matrix(b)

    def preprocessing(self, assign_poly_y_n: bool, assign_R_y_n: bool, assign_N_y_n: bool, b):
        """ Funktion zur Vorbereitung der Matrizen, Randbedingungen usw. zur Berechnung.

            Parameters:
                assign_poly_y_n:   Soll die Polynominterpolation auf die Ränder andewendet werden? Bei einem nicht
                            rechtwinkligem Berechnungsgitter muss diese Funktion ggf. angepasst werden
                assign_R_y_n: Soll R komplett = 0 gesetzt werden?
                assign_N_y_n: Sollen die Neumann RB angewendet werden?
        """
        # Randbedingungen und Koeffizientenmatrizen erstellen
        self.create_geometry()
        r, R_sp, R_sp_inv = self.create_boundary(assign_R_y_n)
        d =self.create_dirichlet(r, 0)
        M_x = self.create_coeff_gradient_dx()
        M_y = self.create_coeff_gradient_dy()

        if assign_poly_y_n is True:  # zusätzliche Polynominterpolation an den Randpunkten?
            M_x, M_y = self.assign_polynom_to_gradient(M_x, M_y)
            M_r = self.create_coeff_laplace_polynom()
        else:
            M_r = self.create_coeff_laplace()

        # Randbedingungen aufprägen
        b = self.assign_d_to_b(r, R_sp_inv, d, b)
        if assign_R_y_n:
            M_x = self.assign_R_to_M(R_sp, R_sp_inv, M_x)
            M_y = self.assign_R_to_M(R_sp, R_sp_inv, M_y)
            M_r = self.assign_R_to_M(R_sp,R_sp_inv, M_r)
        if assign_N_y_n is True:
            i, j = self.create_neumann_input_laplace()
            M_r = self.assign_neumann_laplace(i, j, M_r, assign_N_y_n)

        return M_x, M_y, M_r, r, R_sp, R_sp_inv, d, b





if __name__ == "__main__":
    l = 1.0
    d = 1
    aufloes = 2
    nx = 21
    ny = 21.0
    test =  CfdSolver(d, l, aufloes, nx, ny, 60, 0.005, 1.e-02, 4000, 380)
    test.figure_init()

    # TODO: Pre- und Postprocessing
    # TODO: Berechnung
    ani = FuncAnimation(test.fig, test.animation_time, frames=8, init_func=test.animation_init, repeat=True,
                        interval=500, blit=True)

    plt.show()
