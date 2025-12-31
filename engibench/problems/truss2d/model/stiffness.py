"""Utility methods for calculating the stiffness of a 2d truss structure."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse.linalg as sla
from scipy.sparse.linalg import ArpackError
from scipy.sparse.linalg import ArpackNoConvergence

from engibench.problems.truss2d.model import utils
from engibench.problems.truss2d.model.linear_solver import solve_spd_with_amg
from engibench.problems.truss2d.model.linear_solver import solve_with_spsolve

if TYPE_CHECKING:
    from engibench.problems.truss2d.model.conditions import Conditions


USE_ITERATIVE_SOLVER = False
EIGENVALUE_THRESHOLD = 1e-7


def validate_design(config: Conditions, design_rep: Any) -> bool:
    """Validates if the design representation is feasible.

    Args:
        config: The conditions object containing problem definition.
        design_rep: The design representation to evaluate.

    Returns:
        is_valid (bool): True if the design is valid, False otherwise.
    """
    bit_list, _, _, _ = utils.convert(config, design_rep)

    # 1. Validate at least one member
    if sum(bit_list) == 0:
        return False

    # 2. Validate no single node connections
    load_nodes = utils.get_load_nodes(config.load_conds[0])
    used_nodes_idx = utils.get_used_nodes(config, design_rep)
    fixed_nodes = utils.get_fully_fixed_nodes(config)
    free_used_nodes = [node for node in used_nodes_idx if node not in fixed_nodes]
    for node_idx in free_used_nodes:
        node_connections = utils.get_node_connections(config, design_rep, node_idx)
        if len(node_connections) <= 1:
            return False

    # 3. Validate all loaded nodes are used
    return all(ln in used_nodes_idx for ln in load_nodes)


def calculate_stiffness( # noqa: PLR0915 PLR0912 C901
    config: Conditions, design_rep: Any
) -> tuple[list[float], list[float], list[list[float]], list[float], list[list[float]]]:
    """Evaluates the design and returns physics metrics.

    Args:
        config: The conditions object containing problem definition.
        design_rep: The design representation to evaluate.

    Returns:
        stiffness (List[float]): Scalar stiffness per load case.
        compliance (List[float]): Compliance energy per load case.
        stresses (List[List[float]]): Normal stress for each member per load case.
        buckling_limits (List[float]): Critical Euler buckling stress for each member (static).
        deflections (List[List[float]]): Magnitude of deflection for every node in the problem per load case.
    """
    # Initialize empty returns
    num_load_cases = len(config.load_conds)
    if not design_rep or validate_design(config, design_rep) is False:
        return ([0.0] * num_load_cases, [0.0] * num_load_cases, [], [], [[0.0] * len(config.nodes)] * num_load_cases)

    # Get the correct design representation
    _, _, design, _ = utils.convert(config, design_rep)

    # --- 1. Geometry & Material Properties ---
    # Convert inputs to numpy for vectorized math
    all_nodes = np.array(config.nodes)  # Shape (N, 2)
    e = config.young_modulus
    r = config.member_radii
    a = np.pi * (r**2)
    # Moment of inertia for solid circle: I = (pi * r^4) / 4
    # Radius of gyration squared: k^2 = I/A = r^2 / 4
    # Euler Critical Stress: sigma_cr = (pi^2 * E) / (L/k)^2 = (pi^2 * E * r^2) / (4 * L^2)
    euler_const = (np.pi**2 * e * (r**2)) / 4.0

    # --- 2. Build Sparse Matrix Structure ---
    # Identify used nodes to reduce system size
    used_node_indices = np.unique(np.array(design).flatten())
    num_used = len(used_node_indices)

    # Mapping: Global Node Index -> Local Reduced Index
    old_to_new_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_node_indices)}

    # Pre-calculate Member Geometry
    # We store these to quickly calculate stress later without re-looping geometry
    member_data = []

    # Sparse Matrix triplets
    rows, cols, data = [], [], []

    # Critical buckling stress for each member (only depends on geometry, not load)
    buckling_limits = []

    for n1_old, n2_old in design:
        # Get coordinates
        p1 = all_nodes[n1_old]
        p2 = all_nodes[n2_old]
        diff = p2 - p1
        l = np.linalg.norm(diff)

        if l == 0:
            # Handle zero-length elements gracefully (though physically impossible)
            member_data.append({"valid": False})
            buckling_limits.append(0.0)
            continue

        # Direction Cosines
        c, s = diff / l

        # 1. Store data for Stress/Buckling calculations
        sigma_cr = euler_const / (l**2)
        buckling_limits.append(sigma_cr)

        member_data.append(
            {
                "valid": True,
                "n1_local": old_to_new_map[n1_old],
                "n2_local": old_to_new_map[n2_old],
                "cx": c,
                "cy": s,
                "L": l,
            }
        )

        # 2. Build Stiffness Matrix (Local -> Global)
        k_val = e * a / l
        cc, ss, cs = k_val * c * c, k_val * s * s, k_val * c * s

        n1, n2 = old_to_new_map[n1_old], old_to_new_map[n2_old]
        ix = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]

        # Standard 4x4 Truss Element addition
        # Node 1 (Top-Left)
        rows.extend([ix[0], ix[0], ix[1], ix[1]])
        cols.extend([ix[0], ix[1], ix[0], ix[1]])
        data.extend([cc, cs, cs, ss])
        # Node 2 (Bottom-Right)
        rows.extend([ix[2], ix[2], ix[3], ix[3]])
        cols.extend([ix[2], ix[3], ix[2], ix[3]])
        data.extend([cc, cs, cs, ss])
        rows.extend([ix[0], ix[0], ix[1], ix[1]])
        cols.extend([ix[2], ix[3], ix[2], ix[3]])
        data.extend([-cc, -cs, -cs, -ss])
        rows.extend([ix[2], ix[2], ix[3], ix[3]])
        cols.extend([ix[0], ix[1], ix[0], ix[1]])
        data.extend([-cc, -cs, -cs, -ss])

    # Assemble K
    k = coo_matrix((data, (rows, cols)), shape=(num_used * 2, num_used * 2)).tocsc()

    # --- 3. Apply Boundary Conditions ---
    all_dofs = np.array(config.nodes_dof)
    used_dofs = all_dofs[used_node_indices]

    # Check for stability (are there any fixed supports?)
    if np.all(used_dofs == 1):
        # Unstable / Floating
        return (
            [0.0] * num_load_cases,
            [0.0] * num_load_cases,
            [],
            buckling_limits,
            [[0.0] * len(all_nodes)] * num_load_cases,
        )

    dof_mask = used_dofs.flatten()  # 0=Fixed, 1=Free
    free_indices = np.where(dof_mask == 1)[0]
    k_free = k[free_indices, :][:, free_indices]

    # --- NEW: Stability & Rank Validation ---
    if k_free.shape[0] > 0:
        try:
            # Check the smallest eigenvalue (sigma=0 looks for values near 0)
            # k=1 means we only need the single smallest value
            min_ev = sla.eigsh(k_free, k=1, which="SM", return_eigenvectors=False, tol=1e-10)

            # If the smallest eigenvalue is effectively zero, it's a mechanism
            # 1e-7 is a safe threshold for floating point noise in truss systems
            if min_ev[0] < EIGENVALUE_THRESHOLD:
                return ([0.0] * num_load_cases, [0.0] * num_load_cases, [], buckling_limits,
                        [[0.0] * len(all_nodes)] * num_load_cases)

        except (ArpackError, ArpackNoConvergence, ValueError):
            # If the solver fails to converge, the matrix is likely extremely ill-conditioned
            return ([0.0] * num_load_cases, [0.0] * num_load_cases, [], buckling_limits, [[0.0] * len(all_nodes)] * num_load_cases)


    # --- 4. Solve for Each Load Case ---
    stiffness_res = []
    compliance_res = []
    stress_res = []
    deflection_res = []

    for load_case in config.load_conds:
        # A. Setup Load Vector
        full_load_array = np.array(load_case)

        # Validation: Loads on unused nodes?
        loaded_node_idxs = np.where(np.any(full_load_array != 0, axis=1))[0]
        if not np.all(np.isin(loaded_node_idxs, used_node_indices)):
            stiffness_res.append(0.0)
            compliance_res.append(0.0)
            stress_res.append([0.0] * len(design))
            deflection_res.append([0.0] * len(all_nodes))
            continue

        used_loads = full_load_array[used_node_indices].flatten()
        f_free = used_loads[free_indices]

        if np.sum(np.abs(f_free)) == 0:
            stiffness_res.append(0.0)
            compliance_res.append(0.0)
            stress_res.append([0.0] * len(design))
            deflection_res.append([0.0] * len(all_nodes))
            continue

        # B. Solve Linear System
        try:
            if USE_ITERATIVE_SOLVER is True:
                u_free = solve_spd_with_amg(k_free, f_free)
            else:
                u_free = solve_with_spsolve(k_free, f_free)
        except (RuntimeError, IndexError, TypeError):
            # Singular matrix or solver error
            stiffness_res.append(0.0)
            compliance_res.append(0.0)
            stress_res.append([0.0] * len(design))
            deflection_res.append([0.0] * len(all_nodes))
            continue

        # C. Reconstruct Full Displacement Vector (Used Nodes)
        u_used = np.zeros(num_used * 2)
        u_used[free_indices] = u_free

        # D. Calculate Stiffness & Compliance
        compliance = 0.5 * np.dot(f_free, u_free)
        compliance_res.append(compliance)

        # Stiffness: Sum(|F|) / Sum(|u|) at loaded nodes
        f_vecs = used_loads.reshape(-1, 2)
        u_vecs = u_used.reshape(-1, 2)
        f_mags = np.linalg.norm(f_vecs, axis=1)
        u_mags = np.linalg.norm(u_vecs, axis=1)
        load_mask = f_mags > 0
        total_f = np.sum(f_mags[load_mask])
        total_u = np.sum(u_mags[load_mask])
        stiffness_res.append(total_f / total_u if total_u > 0 else 0.0)

        # E. Calculate Member Stresses
        # Stress = E * Strain = E * (Delta_L / L)
        # Delta_L projected = (u2x - u1x)*cx + (u2y - u1y)*cy
        current_case_stresses = []
        for md in member_data:
            if not md["valid"]:
                current_case_stresses.append(0.0)
                continue

            # Indices in the u_used vector
            idx1 = md["n1_local"] * 2
            idx2 = md["n2_local"] * 2

            u1x, u1y = u_used[idx1], u_used[idx1 + 1]
            u2x, u2y = u_used[idx2], u_used[idx2 + 1]

            # Calculate axial deformation
            delta_l = (u2x - u1x) * md["cx"] + (u2y - u1y) * md["cy"]
            strain = delta_l / md["L"]
            stress = e * strain
            current_case_stresses.append(stress)

        stress_res.append(current_case_stresses)

        # F. Calculate Nodal Deflections (Global)
        # Map used node displacements back to the full list of original nodes
        full_node_deflections = [0.0] * len(all_nodes)
        for i, global_idx in enumerate(used_node_indices):
            mag = u_mags[i]
            full_node_deflections[global_idx] = mag

        deflection_res.append(full_node_deflections)

    return stiffness_res, compliance_res, stress_res, buckling_limits, deflection_res
