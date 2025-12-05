import numpy as np
import matplotlib.pyplot as plt

def visualize_H(H):
    n = H.shape[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # make sparsity plot
    ax1.spy(H, markersize=4, color='black')
    ax1.set_title(f"Sparsity Pattern (Non-Zeros)\nSize: {n}x{n}")
    ax1.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.set_xticks(np.arange(0, n, 1))
    ax1.set_yticks(np.arange(0, n, 1))
    
    # make heatmap
    im = ax2.imshow(np.log10(np.abs(H) + 1e-9), cmap='viridis', interpolation='nearest')
    ax2.set_title("Log-Magnitude Heatmap")
    plt.colorbar(im, ax=ax2, label="log10(|H|)")
    
    plt.tight_layout()
    plt.show()

    # print sparsity stats
    non_zeros = np.count_nonzero(np.abs(H))
    total_elements = n * n
    print(f"Matrix Size: {n}x{n}")
    print(f"Non-Zeros: {non_zeros}")
    print(f"Sparsity: {100 * non_zeros / total_elements:.1f}% filled")


if __name__ == "__main__":
    visualize_H("sample_model_humanoid_data.npz")