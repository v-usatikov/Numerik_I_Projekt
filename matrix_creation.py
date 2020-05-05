import numpy as np
from scipy.sparse import spdiags
from scipy import sparse

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

    dx_haupt = np.ones(self.nxny) * (-2)  # Hauptdiagonale (HD)
    dx_haupt[:self.ny] = 1
    dx_haupt[self.nxny - self.ny:self.nxny] = 1

    dx_haupt_o = np.ones(self.nxny)  # Hauptdiagonale oberhalb HD
    dx_haupt_o[self.ny:2 * self.ny] = -2
    dx_haupt_u = np.ones(self.nxny)  # Hauptdiagonale unterhalb HD
    dx_haupt_u[self.nxny - 2 * self.ny:self.nxny - self.ny] = -2

    dx_neben_o = np.zeros(self.nxny)  # Nebendiagonale oberhalb der HD
    dx_neben_o[2 * self.ny:2 * self.ny + self.ny] = 1
    dx_neben_u = np.zeros(self.nxny)  # Nebendiagonale unterhalb der HD
    dx_neben_u[self.nxny - 3 * self.ny:self.nxny - 2 * self.ny] = 1

    dx_data = [dx_haupt, dx_haupt_o, dx_haupt_u, dx_neben_o, dx_neben_u]
    dx_diags = [0, self.ny, -self.ny, 2 * self.ny, -2 * self.ny]

    dx_spdiags = spdiags(dx_data, dx_diags, self.nxny, self.nxny) / self.dx ** 2

    # Matrix für Ableitung in y-Richtung
    dy_haupt = np.ones(self.ny) * (-2)  # Hauptdiagonale (HD)
    dy_haupt[0] = 1
    dy_haupt[-1] = 1
    dy_haupt = np.tile(dy_haupt, self.nx)

    dy_haupt_o = np.ones(self.ny)  # Hauptdiagonale oberhalb HD
    dy_haupt_o[1] = -2
    dy_haupt_o[0] = 0
    dy_haupt_o = np.tile(dy_haupt_o, self.nx)

    dy_haupt_u = np.ones(self.ny)  # Hauptdiagonale unterhalb HD
    dy_haupt_u[-1] = 0
    dy_haupt_u[-2] = -2
    dy_haupt_u = np.tile(dy_haupt_u, self.nx)
    dy_haupt_u[-1] = -2

    dy_neben_o = np.zeros(self.ny)  # Nebendiagonale oberhalb der HD
    dy_neben_o[2] = 1
    dy_neben_o = np.tile(dy_neben_o, self.nx)

    dy_neben_u = np.zeros(self.ny)  # Nebendiagonale unterhalb der HD
    dy_neben_u[-3] = 1
    dy_neben_u = np.tile(dy_neben_u, self.nx)

    dy_data = [dy_haupt, dy_haupt_o, dy_haupt_u, dy_neben_o, dy_neben_u]
    dy_diags = [0, 1, -1, 2, -2]

    dy_spdiags = spdiags(dy_data, dy_diags, self.nxny, self.nxny) / self.dy ** 2

    M = dx_spdiags + dy_spdiags

    #dy_array = dy_spdiags.toarray()
    #dx_array = dx_spdiags.toarray()
    #M_array = M.toarray()
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
    # M_x_arra = M_x.toarray().reshape(self.nxny, self.nxny)
    return M_x


def create_coeff_gradient_dy(self):
    """ Funktion zur Erstellung der pentagiagonalen Koeffizientenmatrix

        ausgehend von der Poisson-Gleichung (d2p/dx2 + d2p/dy2) = omega_n
        Verwendet sparse-Matrizen-Klasse von scipy
    """
    diag_haupt = np.zeros(self.nxny).ravel()
    o = np.ones(self.nx)
    o[0] = 0
    diag_haupt_o = np.tile(o, self.ny) * (1 / (self.dy * 2))
    u = np.ones(self.nx)
    u[-1] = 0
    diag_haupt_u = np.tile(u, self.ny) * (-1 / (self.dy * 2))
    data = np.array([diag_haupt, diag_haupt_o, diag_haupt_u])
    diags = np.array([0, 1, -1])
    M_y = spdiags(data, diags, self.nxny, self.nxny)
    # M_y_arra = M_y.toarray().reshape(self.nxny,self.nxny)
    return M_y


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


def assign_neumann_to_laplace(self, i, j, M_r, factor, ignore_y_n: bool):
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


def assign_r_to_M(self, R_sp, R_sp_inv, M):
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

    r_d = r.multiply(d).toarray().ravel()

    b_r_d = np.add(r_d, R_sp_inv.dot(b))
    """
    a_array = a.toarray()
    r_d_array = r_d.toarray()
    brd_array = b_r_d.toarray()"""

    return b_r_d