import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def get_branch_induced_sparsity_pattern(H, parent):
    n = H.shape[0]
    sparsity_pattern = np.zeros_like(H)
    
    for i in range(n):
        sparsity_pattern[i, i] = 1  # diagonal is always non-zero
        p = parent[i]
        while p != -1:
            sparsity_pattern[i, p] = 1
            sparsity_pattern[p, i] = 1
            p = parent[p]
    
    return sparsity_pattern

def print_sparsity_stats(A):
    n = A.shape[0]
    non_zeros = np.count_nonzero(np.abs(A))
    total_elements = n * n
    print(f"Matrix Size: {n}x{n}")
    print(f"Non-Zeros: {non_zeros}")
    print(f"Sparsity: {100 * non_zeros / total_elements:.1f}% filled")


def visualize_sparsity(A):
    """
    Visualize the sparsity pattern of any matrix A.
    """
    plt.figure(figsize=(6,6))
    plt.spy(A, markersize=4, color='black')
    plt.title(f"Sparsity Pattern (Non-Zeros)\nSize: {A.shape[0]}x{A.shape[1]}")
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.xlabel('Joint')
    plt.ylabel('Joint')
    plt.show()


def compare_branch_L_sparsity(H, L, L_naive, parent):
    n = H.shape[0]
    branch_sparsity_pattern = get_branch_induced_sparsity_pattern(H, parent)
    
    print("Branch-Induced Sparsity Pattern Stats:")
    print_sparsity_stats(branch_sparsity_pattern)    

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    
    # H matrix plot with branch sparsity pattern overlay
    h_nz = (H != 0)
    branch_zeros = (branch_sparsity_pattern == 0)
    other_zeros = (H == 0) & (~branch_zeros)
    image = np.ones((n, n, 3))
    image[other_zeros] = [0.9, 0.9, 0.9] 
    image[h_nz] = [0, 0, 0]
    ax1.imshow(image, interpolation='nearest')
    ax1.set_xticks(np.arange(0,n,5))
    ax1.set_yticks(np.arange(0,n,5))
    ax1.set_title(f"H matrix sparsity pattern")
    ax1.set_xlabel("Joint Index")
    ax1.set_ylabel("Joint Index")
    legend_elements = [
        Patch(facecolor='black', edgecolor='k', label='Non-Zero ($H_{ij} \\neq 0$)'),
        Patch(facecolor='white', edgecolor='k', label='Branch-Induced Zero'),
        Patch(facecolor='lightgray', edgecolor='k', label='Other Zero'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize='small')

    # L matrix
    l_nz = (L != 0)
    image = np.ones((n, n, 3))
    image[l_nz] = [0, 0, 0]
    ax2.imshow(image, interpolation='nearest')
    ax2.set_xticks(np.arange(0,n,5))
    ax2.set_yticks(np.arange(0,n,5))
    ax2.set_title(fr"L sparsity pattern ($L^TDL$)")
    ax2.set_xlabel("Joint Index")
    ax2.set_ylabel("Joint Index")


    # L_naive matrix
    lnaive_nz = (L_naive != 0)
    image = np.ones((n, n, 3))
    image[lnaive_nz] = [0, 0, 0]
    ax3.imshow(image, interpolation='nearest')
    ax3.set_xticks(np.arange(0,n,5))
    ax3.set_yticks(np.arange(0,n,5))
    ax3.set_title(fr"L sparsity pattern ($LDL^T$)")
    ax3.set_xlabel("Joint Index")
    ax3.set_ylabel("Joint Index")

    plt.tight_layout()
    plt.show()



# def compare_branch_L_sparsity(H, L, parent):
#     n = H.shape[0]
#     branch_sparsity_pattern = get_branch_induced_sparsity_pattern(H, parent)
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
    
#     ax1.spy(H, markersize=6, color='black')
#     ax1.set_title(f"Sparsity pattern of H, N={n})")
#     # ax1.grid(True, alpha=0.3)
#     ax1.set_xlabel('Joint')
#     ax1.set_ylabel('Joint')
    
#     ax2.spy(branch_sparsity_pattern, markersize=6, color='black')
#     ax2.set_title(f"Branch-Induced Sparsity\n(Kinematic Topology, N={n})")
#     # ax2.grid(True, alpha=0.3)
#     ax2.set_xlabel('Joint')

#     ax3.spy(L, markersize=6, color='black')
#     ax3.set_title(f"Sparsity of L Factor\n(LTDL Result, N={n})")
#     # ax3.grid(True, alpha=0.3)
#     ax3.set_xlabel('Joint')
#     ax3.set_ylabel('Joint')
    

#     plt.tight_layout()
#     plt.show()

def overlay_sparsity_patterns(H, L, parent):
    n = H.shape[0]
    branch_sparsity_pattern = get_branch_induced_sparsity_pattern(H, parent)
    
    h_nz = (H != 0)
    branch_zeros = (branch_sparsity_pattern == 0)
    other_zeros = (H == 0) & (~branch_zeros)
    
    # 3. Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    image = np.ones((n, n, 3))
    image[other_zeros] = [0.9, 0.9, 0.9] 
    image[h_nz] = [0, 0, 0]
    
    ax.imshow(image, interpolation='nearest')
    
    # Formatting
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_title(f"Sparsity of H with Branch-Induced Zeros Overlay (N={n})")
    ax.set_xlabel("Joint Index")
    ax.set_ylabel("Joint Index")

    plt.tight_layout()
    plt.show()

# def visualize_H(H):
#     n = H.shape[0]
    
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#     # make sparsity plot
#     ax1.spy(H, markersize=4, color='black')
#     ax1.set_title(f"Sparsity Pattern (Non-Zeros)\nSize: {n}x{n}")
#     ax1.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.3)
#     ax1.set_xticks(np.arange(0, n, 1))
#     ax1.set_yticks(np.arange(0, n, 1))
    
#     # make heatmap
#     im = ax2.imshow(np.log10(np.abs(H) + 1e-9), cmap='viridis', interpolation='nearest')
#     ax2.set_title("Log-Magnitude Heatmap")
#     plt.colorbar(im, ax=ax2, label="log10(|H|)")
    
#     plt.tight_layout()
#     plt.show()

#     # print sparsity stats
#     non_zeros = np.count_nonzero(np.abs(H))
#     total_elements = n * n
#     print(f"Matrix Size: {n}x{n}")
#     print(f"Non-Zeros: {non_zeros}")
#     print(f"Sparsity: {100 * non_zeros / total_elements:.1f}% filled")

