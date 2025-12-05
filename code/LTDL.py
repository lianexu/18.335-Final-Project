import numpy as np

# LTDL factorization algorithm and associated solvers
# Algorithms are from "Efficient Factorization of the 
# Joint Space Inertia Matric for Branched Kinematic Trees" by Roy Featherstone
# Also keep track of FLOPs

# --- H = LTDL FACTORIZATION --- #
def factor_LTDL(H, parent):
    """
    Featherstone's algorithm for factoring H = LTDL. 
    Accesses only the lower triangle of H. 
    The computed factors are stored in-place in H.

    Inputs: 
        H : NxN symmetric positive definite matrix to be factorized
        parent : N-element array representing kinematic tree structure
    Returns:
        L : unit lower-triangular matrix. Diagonal entries are 1.
        D : diagonal entries as a 1D array
        flops: number of matrix operations performed
    """
    # Symmetric condition: H == (H + H.T) / 2.0 
    flops = 0

    H = H.copy()
    n = H.shape[0]

    # factor H in-situ
    for k in range(n - 1, -1, -1):
        i = parent[k]
        while i != -1:
            h = H[k, i] / H[k, k]
            j = i
            flops += 1 # 1 Division 
            while j != -1:
                H[i, j] -= H[k, j] * h
                j = parent[j]
                flops += 2 # 1 Mult + 1 Subtraction
            H[k, i] = h
            i = parent[i]

    # extract D and L from H
    D = np.diag(H).copy()
    L = np.eye(n, dtype=H.dtype)
    for r in range(n):
        for c in range(r):
            L[r, c] = H[r, c]

    return L, D, flops

# --- SOLVE FORWARD DYNAMICS USING LTDL FACTORIZATION --- #
def compute_FD_from_LTDL(L, D, parent, rhs):
    y, flops1 = solve_L_negT(L, rhs.copy(), parent) # Backward solve
    z, flops2 = solve_diag(D, y) # Diagonal
    ddq, flops3 = solve_L_neg(L, z, parent) # Forward solve

    flops = flops1 + flops2 + flops3
    return ddq, flops

# --- HELPERS FOR MULTIPLYING A VECTOR BY A SPARSE TRIANGULAR FACTOR L --- #
def solve_L_negT(L, x, parent):
    """
    Computes L^-T * x
    Works in situ on the vector x, replacing it with L^-T * x
    """
    flops = 0

    n = len(x)
    for i in reversed(range(n)):
        x[i] = x[i]/L[i,i] # L[i,i] == 1, 0 FLOPS
        j = parent[i]
        while j != -1:
            x[j] -= L[i,j]*x[i]
            j = parent[j]
            flops += 2 # 1 Mult + 1 Subtraction
    return x, flops

def solve_L_neg(L, x, parent):
    """
    Computes L^-1 * x
    Works in situ on the vector x, replacing it with L^-1 * x
    """
    flops = 0

    n = len(x)
    for i in range(n):
        j = parent[i]
        while j != -1:
            x[i] -= L[i,j]*x[j]
            j = parent[j]
            flops += 2 # 1 Mult + 1 Subtraction
        x[i] = x[i]/L[i,i] # L[i,i] == 1, 0 FLOPS
    return x, flops

def solve_L_T(L, x, parent):
    """
    Computes L^T * x
    """
    flops = 0

    n = len(x)
    y = x.copy()
    for i in range(n):
        y[i] = L[i,i]*x[i] # L[i,i] == 1, 0 FLOPS
        j = parent[i]
        while j != -1:
            y[j] += L[i,j]*x[i]
            j = parent[j]
            flops += 2 # 1 Mult + 1 Addition
    return y, flops


def solve_L(L, x, parent):
    """
    Computes L * x
    """
    flops = 0

    n = len(x)
    y = x.copy()
    for i in reversed(range(n)):
        y[i] = L[i,i]*x[i] # L[i,i] == 1, 0 FLOPS
        j = parent[i]
        while j != -1:
            y[i] += L[i,j]*x[j]
            j = parent[j]
            flops += 2 # 1 Mult + 1 Addition
    return y, flops

def solve_diag(D, y):
    """
    Solves x = D^-1 * y for x where D is diagonal
    """
    flops = len(y) # 1 Division per entry
    x = y / D
    return x, flops
