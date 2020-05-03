import numpy as np
d = 1
l = 2
DT = 0.01  # Zeitschritt
NY = 101  # Große der Diskretisieren-Gitter für y-Achse
H = d/(NY-1)  # Schritt für x y
NX = int(l/H) + 1  # Große der Diskretisieren-Gitter für x-Achse
V0 = 0.1  # Eingang-Strom-Geschwindigkeit


def empty_F_array_with_BC():
    """Erstellt ein leeren F array mit GB I, II, und III für die weitere Berechnung."""
    array = np.zeros((NX, NY))
    # BC III
    for i in range(1, NY):
        array[0, i] = array[0, i-1] + H*V0
    # BC I
    for i in range(1, NX):
        array[:, -1] = array[0, -1] * np.ones(NX)


def empty_nabla_F_array_with_BC():
    """Erstellt ein leeren F array mit GB I, II, und III für die weitere Berechnung."""
    array = np.zeros((NX, NY))
    # BC I
    array[:, -1] = V0 * np.ones(NX)
    # BC II
    array[:, 0] = V0 * np.ones(NX)
    # BC I
    array[0, :] = V0 * np.ones(NY)


def in_objekt(i, j):
    """Zeigt ob der Punkt im Objekt ist."""
    a = 0.5
    b = 0.5
    r = 0.1
    x = H*i
    y = H*j
    return (x-a)**2 + (y-b)**2 <= r


def __l_border_der(f, fp1, fp2):
    return 1/(2*H) * (-3*f + 4*fp1 - fp2)


def __r_border_der(fm2, fm1, f):
    return 1/(2*H) * (fm2 - 4*fm1 + 3*f)


def __in_der(f, fp1):
    return 1/(2*H) * (fp1 - f)


def part_der_y(i, j, array):
    if j == 0:
        return __l_border_der(array[i, j], array[i, j+1], array[i, j+2])
    elif j == NY:
        return __r_border_der(array[i, j-2], array[i, j - 1], array[i, j])
    elif not in_objekt(i, j) and not in_objekt(i, j + 1):
        return __in_der(array[i, j], array[i, j + 1])
    elif not in_objekt(i, j) and in_objekt(i, j+1):
        return __r_border_der(array[i, j - 2], array[i, j - 1], array[i, j])
    elif in_objekt(i, j) and not in_objekt(i, j + 1):
        return __l_border_der(array[i, j], array[i, j+1], array[i, j+2])
    else:
        return None


def part_der_x(i, j, array):
    if i == 0:
        return __l_border_der(array[i, j], array[i+1, j], array[i+2, j])
    elif i == NX:
        return __r_border_der(array[i-2, j], array[i-1, j], array[i, j])
    elif not in_objekt(i, j) and not in_objekt(i + 1, j):
        return __in_der(array[i, j], array[i + 1, j])
    elif not in_objekt(i, j) and in_objekt(i + 1, j):
        return __r_border_der(array[i - 2, j], array[i - 1, j], array[i, j])
    elif in_objekt(i, j) and not in_objekt(i + 1, j):
        return __l_border_der(array[i, j], array[i+1, j], array[i+2, j])
    else:
        return None

