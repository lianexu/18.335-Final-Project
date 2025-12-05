import numpy as np
import pinocchio as pin
from ltdl import factor_LTDL, compute_FD_from_LTDL
from theoretical_flops import LTDL_theo_flops, naive_theo_flops
from visualize_H_structure import visualize_H

def refactor_system_parents(model):
    """
    Refactors parent / joint tree array to match Featherstone notation (including floating base).
    """
    parent = [0 for _ in range(model.nv)]

    # Iterate over all joints and edit their parents. Skip universe at index 0.
    for joint_idx in range(1, len(model.joints)):
        joint = model.joints[joint_idx]
        
        start_idx = joint.idx_v
        nv = joint.nv
        
        # If a joint has multiple DOFs, determine parent of the first DOF of the joint
        parent_joint_idx = model.parents[joint_idx]
        
        if parent_joint_idx == 0:
            # Parent is Universe means it is Root of the tree
            first_dof_parent = -1
        else:
            # Parent is the LAST DOF of the parent joint
            p_joint = model.joints[parent_joint_idx]
            first_dof_parent = p_joint.idx_v + p_joint.nv - 1

        # edit parents   
        parent[start_idx] = first_dof_parent
        
        for k in range(1, nv):
            parent[start_idx + k] = start_idx + k - 1
            
    return parent


if __name__ == "__main__":
    print("\n--- Testing Factorization and Solve with Sample Humanoid (34 DOF w/ Floating Base) ---")
    np.random.seed(42) 

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
    
    # H = LTDL factorization
    print(f"Factorizing {H_full.shape} matrix...")
    L, D, flops_factor_LTDL = factor_LTDL(H_full, parent)

    # Check H = LTDL reconstruction errors
    H_rec = L.T @ np.diag(D) @ L # reconstruct H from computed factors
    recon_err = np.linalg.norm(H_full - H_rec)
    print(f"H = LTDL reconstruction error: {recon_err:.3e}")

    # Solve for ddq (Forward Dynamics) using LTDL
    ddq_ltdl, flops_FD = compute_FD_from_LTDL(L, D, parent, rhs)

    # Solve for ddq using naive solver for ground truth reference 
    # ddq_ref = np.linalg.solve(H_full, rhs)
    ddq_ref = np.linalg.pinv(H_full) @ rhs

    # Check forward dynamics solution error
    sol_err = np.linalg.norm(ddq_ltdl - ddq_ref)
    print(f"FD Solution error:    {sol_err:.3e}")


    total_flops = flops_factor_LTDL + flops_FD
    flops_theo_fact_ltdl, flops_theo_solve_ltdl = LTDL_theo_flops(parent)
    flops_theo_fact_naive, flops_theo_solve_naive = naive_theo_flops(model.nv)

    print(f"Counted LTDL FLOPS: {total_flops} (Factorization: {flops_factor_LTDL} + Solve: {flops_FD})")
    print(f"Theoretical LTDL FLOPS: {flops_theo_fact_ltdl+flops_theo_solve_ltdl} (Factorization: {flops_theo_fact_ltdl} + Solve: {flops_theo_solve_ltdl})")
    print(f"Theoretical Naive FLOPS: {flops_theo_fact_naive+flops_theo_solve_naive} (Factorization: {flops_theo_fact_naive} + Solve: {flops_theo_solve_naive})")

    visualize_H(H_full)