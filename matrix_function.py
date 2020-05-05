import numpy as np
from scipy.sparse import spdiags
from scipy import sparse
from scipy.sparse import linalg

def try_pos_definite(M):
    try: np.linalg.cholesky(M)
    except: print(Warning("Matrix nicht positiv definit. Keine Lösung garantiert"))

def jacobi_definite(M):
    return M.diagonal() is not 0