import numpy as np
import matplotlib.pyplot as plt
from ltdl import factor_LTDL, compute_FD_from_LTDL

## BENCHMARK ACCURACY
def benchmark_accuracy():
    n = 30
    cond_numbers = []
    forward_errors_ltdl = []  
    backward_errors_ltdl = [] 
    
    for exponent in range(0, 16): 
        # scale = 10**exponent
        
        # Create Valid (symmetric positive definite) Ill-Conditioned H Matrix (with eigenvalues from 1 to 10^exponent)
        X = np.random.randn(n, n)
        Q, _ = np.linalg.qr(X)
        s = np.logspace(0, exponent, n) 
        H = Q @ np.diag(s) @ Q.T
        H = (H + H.T) / 2.0 # enforce symmetry 
        
        # Chain topology
        parent = np.arange(n) - 1
        parent[0] = -1
        
        rhs = np.random.rand(n)
        
        ddq_ref = np.linalg.pinv(H) @ rhs 
        # ddq_ref = np.linalg.solve(H, rhs)
        
        try:
            L, D, _ = factor_LTDL(H, parent)
            
            ddq_ltdl, _ = compute_FD_from_LTDL(L, D, parent, rhs)
            
            # Forward Error (should depend on the conditioning of the matrix): ||ddq_ltdl - ddq_ref|| / ||ddq_ref||
            forward_err = np.linalg.norm(ddq_ltdl - ddq_ref) / np.linalg.norm(ddq_ref)
            
            # Backward Error:
            # normalized by ||H||*||ddq_ltdl||
            res_norm = np.linalg.norm(H @ ddq_ltdl - rhs)
            H_norm = np.linalg.norm(H, ord=2) 
            ddq_norm = np.linalg.norm(ddq_ltdl, ord=2)

            backward_err = res_norm / (H_norm * ddq_norm)
            
        except np.linalg.LinAlgError:
            forward_err = np.nan
            backward_err = np.nan
            
        # Store
        cond = np.linalg.cond(H)
        cond_numbers.append(cond)
        forward_errors_ltdl.append(forward_err)
        backward_errors_ltdl.append(backward_err)
        
    return cond_numbers, forward_errors_ltdl, backward_errors_ltdl


def plot_benchmark_accuracy():
    conds, forward_errors, backward_errors = benchmark_accuracy()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    sort_idx = np.argsort(conds)
    conds = np.array(conds)[sort_idx]
    forward_errors = np.array(forward_errors)[sort_idx]
    backward_errors = np.array(backward_errors)[sort_idx]

    ax.loglog(conds, forward_errors, 'r-o', label=r'Forward Error $||\ddot{q} - \ddot{q}_{true}||$')
    ax.loglog(conds, backward_errors, 'b-s', label=r'Backward Error $||H \ddot{q} - (\tau - C)|| (scaled)$')


    ax.set_title('LTDL Solver Forward and Backward Error vs Condition Number')
    ax.set_xlabel('Condition Number $\kappa(H)$')
    ax.set_ylabel('Error')
    ax.legend()
    ax.grid(True, which="both", ls="-")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_benchmark_accuracy()
    