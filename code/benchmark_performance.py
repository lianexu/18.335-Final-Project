import numpy as np
import pinocchio as pin
from solve_humanoid import refactor_system_parents
from ltdl import factor_LTDL, compute_FD_from_LTDL
import matplotlib.pyplot as plt
from theoretical_flops import LTDL_theo_flops, naive_theo_flops


np.random.seed(42) 

# GENERATE KINEMATIC TREES
def generate_star_system(n):
    # Root (0) connected to Children (1..n-1) -> Humanoid/Branching Proxy
    parent = np.zeros(n, dtype=int)
    parent[0] = -1
    # Create Arrowhead Matrix
    H = np.eye(n) * 2.0
    H[0, 1:] = 0.5
    H[1:, 0] = 0.5
    H[0, 0] = n  # Dominant diagonal
    return H, parent

def generate_spider_system(n, legs):
    # Root (0) connected to Legs (1..legs), each leg is a chain of length n//legs
    parent = np.zeros(n, dtype=int)
    parent[0] = -1
    leg_length = (n - 1) // legs
    for leg in range(legs):
        for j in range(leg_length):
            idx = 1 + leg * leg_length + j
            if j == 0:
                parent[idx] = 0  # Connect to root
            else:
                parent[idx] = idx - 1  # Connect to previous in leg
    # Create Block-Diagonal Matrix with Coupling at Root
    H = np.eye(n) * 2.0
    for i in range(1, n):
        H[0, i] = 0.3
        H[i, 0] = 0.3
    H[0, 0] = legs * 2.0  # Dominant diagonal at root
    return H, parent

def generate_chain_system(n):
    # 0->1->2... -> Serial Manipulator Proxy
    parent = np.arange(n) - 1
    parent[0] = -1
    # Create Dense Matrix (Chain interactions fill everything)
    A = np.random.rand(n, n)
    H = A @ A.T + np.eye(n)
    return H, parent

def generate_humanoid_system():
    model = pin.buildSampleModelHumanoid()
    data = model.createData()

    print(f"Total DOFs: {model.nv} (6 Base + {model.nv-6} Actuated)")

    print(f"Generating random system H * q_dotdot = tau - C")

    # generate random state
    q = pin.neutral(model)
    q[7:] += np.random.uniform(-0.4, 0.4, size=model.nv - 6) # jitter the joints
    v = np.random.randn(model.nv)
    tau = np.random.randn(model.nv)

    # compute JSIM (H matrix) using CRBA
    pin.crba(model, data, q)
    H_full = (data.M + data.M.T) / 2.0  # Full 34x34 matrix
    
    # compute bias forces (C matrix) using RNEA
    C = pin.rnea(model, data, q, v, np.zeros(model.nv))
    
    # combine right hand side of equation
    rhs = tau - C

    # refactor tree data structure (parent array) to match Featherstone notation
    print(f"Original parents list: {list(model.parents)}")
    parent = refactor_system_parents(model)
    print(f"Refactored parents list: {list(parent)}")

    return H_full, parent, rhs, model.nv


def benchmark_general():
    sizes = np.arange(10, 150, 5)
    results = {
        'star_flops': [], 'naive_flops': [],
        'chain_flops': [],
    }

    for n in sizes:
        flops_theo_fact_naive, flops_theo_solve_naive = naive_theo_flops(n)
        results['naive_flops'].append(flops_theo_fact_naive + flops_theo_solve_naive)

        # Best case scenario for LTDL: Star
        H, parent = generate_star_system(n)
        L, D, flops_fact = factor_LTDL(H, parent)
        qdotdot, flops_solve = compute_FD_from_LTDL(L, D, parent, np.ones(n))
        results['star_flops'].append(flops_fact + flops_solve)
        

        # Worst case scenario for LTDL: Chain
        H, parent = generate_chain_system(n)
        L, D, flops_fact = factor_LTDL(H, parent)
        qdotdot, flops_solve = compute_FD_from_LTDL(L, D, parent, np.ones(n))
        results['chain_flops'].append(flops_fact + flops_solve)


    real_systems_algo = []
    real_systems_theo = []

    # Humanoid 
    H_full, parent, rhs, n_hum = generate_humanoid_system()
    L, D, flops_fact = factor_LTDL(H_full, parent)
    qdotdot, flops_solve = compute_FD_from_LTDL(L, D, parent, rhs)
    real_systems_algo.append((n_hum, flops_fact + flops_solve, "Humanoid (LTDL)"))

    flops_theo_fact_naive, flops_theo_solve_naive = naive_theo_flops(n_hum)
    real_systems_theo.append((n_hum, flops_theo_fact_naive + flops_theo_solve_naive, "Humanoid (Naive)"))

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))

    plt.plot(sizes, results['naive_flops'], 'r--', label='Naive (Cholesky) linear system solve FLOPS')
    plt.plot(sizes, results['chain_flops'], 'g-', alpha=0.5, label='Worst case: Chain Topology FLOPs with LTDL')
    plt.plot(sizes, results['star_flops'], 'b-', alpha=0.5, label='Best case: Star Topology FLOPs with LTDL')

    for dof, flops, name in real_systems_algo:
        if "Humanoid" in name:
            color = 'blue'
            marker = 'o'
        else:
            color = 'green'
            marker = 'o'
            
        plt.scatter(dof, flops, color=color, marker=marker, s=150, zorder=10, label=f"{name}")

    for dof, flops, name in real_systems_theo:
        if "Humanoid" in name:
            color = 'red' 
            marker = 's'  
        else:
            color = 'red' 
            marker = 's'  
            
        plt.scatter(dof, flops, color=color, marker=marker, s=150, zorder=9, facecolors='none', edgecolors=color, linewidth=2, label=f"{name}")

    plt.legend(loc='best')

    plt.title('FLOPs: Sparse vs Dense Topologies')
    plt.xlabel('Degrees of Freedom (N)')
    plt.ylabel('Total FLOPs (Factorize + Solve)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.3)

    plt.tight_layout()
    plt.show()


def benchmark_spider_variations():
    sizes = np.arange(20, 200, 10)

    leg_counts = [2, 4, 8, 16]
    
    results = {legs: [] for legs in leg_counts}
    results['chain'] = [] 
    results['star'] = [] 
    for n in sizes:
        H, parent = generate_chain_system(n)
        L, D, flops_fact = factor_LTDL(H, parent)
        _, flops_solve = compute_FD_from_LTDL(L, D, parent, np.ones(n))
        results['chain'].append(flops_fact + flops_solve)

        H, parent = generate_star_system(n)
        L, D, flops_fact = factor_LTDL(H, parent)
        _, flops_solve = compute_FD_from_LTDL(L, D, parent, np.ones(n))
        results['star'].append(flops_fact + flops_solve)

        for legs in leg_counts:
            H, parent = generate_spider_system(n, legs)
            
            L, D, flops_fact = factor_LTDL(H, parent)
            _, flops_solve = compute_FD_from_LTDL(L, D, parent, np.ones(n))
            
            results[legs].append(flops_fact + flops_solve)

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))

    plt.plot(sizes, results['chain'], 'k--', linewidth=2, alpha=0.6, label='Chain (1 Leg)')
    plt.plot(sizes, results['star'], 'k:', linewidth=2, alpha=0.6, label='Star (~N Legs)')

    colors = plt.cm.plasma(np.linspace(0, 0.8, len(leg_counts)))
    
    for idx, legs in enumerate(leg_counts):
        plt.plot(sizes, results[legs], '-o', color=colors[idx], label=f'Spider ({legs} Legs)')

    plt.title('Impact of Branching on LTDL FLOPs')
    plt.xlabel('Degrees of Freedom (N)')
    plt.ylabel('Total FLOPs (Factorize + Solve)')
    plt.yscale('log') # Log scale is crucial here
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    benchmark_general()
    # benchmark_spider_variations()