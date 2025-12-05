import numpy as np

def LTDL_theo_flops(parent):
    """
    Calculates theoretical FLOPs for LTDL factorization based on Featherstone paper
    
    Theoretical FLOPs: 
    Factor: D1 (div) + D2 * 2 (m+a)
    Solve: n (div) + D1 * 4 (2 * (m+a))
    
    """
    n = len(parent)
    D1 = 0 # sum of ancestors
    D2 = 0 # sum of ancestors of ancestors

    for k in range(n - 1, -1, -1):
        i = parent[k]
        while i != -1:
            D1 += 1
            j = i
            while j != -1:
                D2 += 1
                j = parent[j]
            i = parent[i]

    flops_theo_fact = D1 * 1 + D2 * 2
    flops_theo_solve = n * 1 + D1 * 4

    return flops_theo_fact, flops_theo_solve

def naive_theo_flops(n):
    """
    Calculates theoretical FLOPs for naive (dense LTDL) factorization and linear system solve
    """
    flops_theo_fact = (n**3)/3 # LDL^T factorization
    flops_theo_solve = 2 * (n**2) # Forward + Backward substitution + Diagonal
    return flops_theo_fact, flops_theo_solve