"""Classic Ground Structure Truss Optimization via Linear Programming (LP).

Formulates the provably convex global minimum-compliance (maximum stiffness)
truss topology optimization problem for a ground structure using the classic
Dorn, Gomory & Greenberg (1964) / Hemp (1973) / Gilbert & Tyas (2003) formulation:

    min_{q+, q- >= 0}  sum_{i=1}^M L_i * (q_i^+ + q_i^-)
    subject to:        B * (q^+ - q^-) = F_free

where:
    - q_i^+, q_i^- >= 0 are member tensile and compressive forces
    - B is the nodal equilibrium matrix at unconstrained degrees of freedom
    - F_free is the vector of external applied nodal loads
    - Member areas are given by A_i = (q_i^+ + q_i^-) / sigma_0

Solved to global optimality via SciPy's HiGHS linear programming solver.
The resulting optimal truss provides the absolute theoretical lower bound on
compliance (upper bound on stiffness) for any pin-jointed truss topology on the grid.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linprog

from ..fem.material import Material, RectangularSection
from ..fem.mesh import FrameMesh2D
from ..fem.solver import FrameSolver2D, SolverResult
from ..geometry.domain import DesignDomain2D, compute_volume_fraction
from ..geometry.ground_structure import GroundStructureData, build_fe_mesh_from_ground_structure
from ..fem.load_cases import LoadCase


@dataclass
class LPOptimizationResult:
    """Results from Linear Programming ground structure optimization.

    Attributes:
        optimal_forces: Member axial forces N_i (tension positive, compression negative).
        optimal_areas: Member cross-sectional areas A_i (m^2).
        optimal_thicknesses: Member thicknesses t_i (m) for unit out-of-plane depth.
        active_mask: Boolean array indicating members with non-zero area (A_i > tol).
        num_active_members: Count of active members.
        lp_objective_volume: Theoretical minimum truss volume from LP (m^3).
        achieved_volume: Achieved volume after scaling to target volume fraction.
        achieved_volume_fraction: Achieved volume fraction v_f.
        frame_compliance: Exact compliance when evaluated in the 2D Euler-Bernoulli frame solver.
        frame_stiffness: Structural stiffness K = 1 / C (N/m).
        axial_energy_share: Fraction of strain energy carried axially.
        bending_energy_share: Fraction of strain energy carried in flexural bending.
        solver_status: Solver exit status string ('optimal', etc.).
        solve_time_s: Runtime of the LP solve (seconds).
    """

    optimal_forces: np.ndarray
    optimal_areas: np.ndarray
    optimal_thicknesses: np.ndarray
    active_mask: np.ndarray
    num_active_members: int
    lp_objective_volume: float
    achieved_volume: float
    achieved_volume_fraction: float
    frame_compliance: float
    frame_stiffness: float
    axial_energy_share: float
    bending_energy_share: float
    solver_status: str
    solve_time_s: float


def build_nodal_equilibrium_matrix(
    gs: GroundStructureData,
    load_case: LoadCase,
    domain: DesignDomain2D,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """Construct the equilibrium matrix B at free degrees of freedom.

    Equilibrium at each free node j:
        sum_i B_{2j, i} * N_i = F_{x, j}
        sum_i B_{2j+1, i} * N_i = F_{y, j}

    Args:
        gs: GroundStructureData.
        load_case: LoadCase instance defining loads and boundary conditions.
        domain: DesignDomain2D.

    Returns:
        B: Matrix of shape (num_free_dofs, num_members).
        F_free: Vector of applied external loads at free DOFs (length num_free_dofs).
        free_dof_indices: Global DOF indices corresponding to the rows of B.
        fixed_dofs: Global DOF indices constrained by supports.
    """
    num_nodes = gs.num_nodes
    num_members = gs.num_members

    # 1. Map nodes to 2D translational DOFs: node_id -> (dof_x, dof_y)
    node_id_to_idx = {nid: i for i, nid in enumerate(sorted(gs.nodes.keys()))}

    # Identify constrained DOFs from boundary conditions
    fixed_dofs = set()

    # Determine boundary support nodes based on load case
    if "cantilever" in load_case.name.lower() or "lc1" in load_case.name.lower():
        # Left boundary nodes fixed
        for nid, (x, y) in gs.nodes.items():
            if abs(x - 0.0) <= 1e-4:
                idx = node_id_to_idx[nid]
                fixed_dofs.add(2 * idx)      # ux
                fixed_dofs.add(2 * idx + 1)  # uy
    elif "simply" in load_case.name.lower() or "lc2" in load_case.name.lower():
        # Pinned at bottom-left (0, 0), roller at bottom-right (W, 0)
        for nid, (x, y) in gs.nodes.items():
            if abs(x - 0.0) <= 1e-4 and abs(y - 0.0) <= 1e-4:
                idx = node_id_to_idx[nid]
                fixed_dofs.add(2 * idx)      # ux
                fixed_dofs.add(2 * idx + 1)  # uy
            elif abs(x - domain.width) <= 1e-4 and abs(y - 0.0) <= 1e-4:
                idx = node_id_to_idx[nid]
                fixed_dofs.add(2 * idx + 1)  # uy only (roller)
    elif "compression" in load_case.name.lower() or "lc3" in load_case.name.lower():
        # Bottom boundary fixed along y = 0
        for nid, (x, y) in gs.nodes.items():
            if abs(y - 0.0) <= 1e-4:
                idx = node_id_to_idx[nid]
                fixed_dofs.add(2 * idx)      # ux
                fixed_dofs.add(2 * idx + 1)  # uy
    else:
        # Fallback: check left edge
        for nid, (x, y) in gs.nodes.items():
            if abs(x - 0.0) <= 1e-4:
                idx = node_id_to_idx[nid]
                fixed_dofs.add(2 * idx)
                fixed_dofs.add(2 * idx + 1)

    total_2d_dofs = 2 * num_nodes
    free_dofs = [dof for dof in range(total_2d_dofs) if dof not in fixed_dofs]
    dof_to_row = {dof: row for row, dof in enumerate(free_dofs)}

    # 2. Build full equilibrium matrix B_full of size (2*num_nodes, num_members)
    B_full = np.zeros((total_2d_dofs, num_members), dtype=np.float64)

    for m_idx, (n1, n2) in enumerate(gs.members):
        x1, y1 = gs.nodes[n1]
        x2, y2 = gs.nodes[n2]
        L = float(np.hypot(x2 - x1, y2 - y1))
        c = (x2 - x1) / L
        s = (y2 - y1) / L

        i1 = node_id_to_idx[n1]
        i2 = node_id_to_idx[n2]

        # In-plane equilibrium contribution:
        # Tension force N pull inward: at n1 direction is + (c, s), at n2 direction is - (c, s)
        B_full[2 * i1, m_idx] += c
        B_full[2 * i1 + 1, m_idx] += s
        B_full[2 * i2, m_idx] -= c
        B_full[2 * i2 + 1, m_idx] -= s

    # 3. Build external load vector F_full
    F_full = np.zeros(total_2d_dofs, dtype=np.float64)

    if "cantilever" in load_case.name.lower() or "lc1" in load_case.name.lower():
        # Point load at right tip mid-height (W, H/2)
        target_pt = (domain.width, 0.5 * domain.height)
        closest_nid = min(
            gs.nodes.keys(),
            key=lambda nid: np.hypot(gs.nodes[nid][0] - target_pt[0], gs.nodes[nid][1] - target_pt[1]),
        )
        idx = node_id_to_idx[closest_nid]
        F_full[2 * idx + 1] = -1000.0  # Downward shear load
    elif "simply" in load_case.name.lower() or "lc2" in load_case.name.lower():
        # Central point load on top edge (W/2, H)
        target_pt = (0.5 * domain.width, domain.height)
        closest_nid = min(
            gs.nodes.keys(),
            key=lambda nid: np.hypot(gs.nodes[nid][0] - target_pt[0], gs.nodes[nid][1] - target_pt[1]),
        )
        idx = node_id_to_idx[closest_nid]
        F_full[2 * idx + 1] = -1000.0
    elif "compression" in load_case.name.lower() or "lc3" in load_case.name.lower():
        # Distributed load across top edge y = H
        top_nids = [nid for nid, (x, y) in gs.nodes.items() if abs(y - domain.height) <= 1e-4]
        per_node_f = -1000.0 / max(1, len(top_nids))
        for nid in top_nids:
            idx = node_id_to_idx[nid]
            F_full[2 * idx + 1] = per_node_f
    else:
        # Default right tip load
        idx = node_id_to_idx[max(gs.nodes.keys())]
        F_full[2 * idx + 1] = -1000.0

    # Extract rows corresponding to free DOFs
    B_free = B_full[free_dofs, :]
    F_free = F_full[free_dofs]

    return B_free, F_free, free_dofs, sorted(list(fixed_dofs))


def solve_ground_structure_lp(
    ground_structure: GroundStructureData,
    load_case: LoadCase,
    material: Material,
    target_volume_fraction: float,
    sigma_allowable: float = 250e6,
    zero_area_threshold: float = 1e-6,
) -> LPOptimizationResult:
    """Solve the global minimum-compliance ground structure truss LP.

    Formulation:
        min_{q+, q- >= 0}  sum_{i=1}^M L_i * (q_i^+ + q_i^-)
        subject to:        B * (q^+ - q^-) = F_free

    Args:
        ground_structure: GroundStructureData.
        load_case: LoadCase.
        material: Structural material.
        target_volume_fraction: Target volume budget v_f.
        sigma_allowable: Allowable stress limit (Pa, default 250 MPa).
        zero_area_threshold: Cutoff below which member area is treated as zero.

    Returns:
        LPOptimizationResult containing optimal areas, forces, and Euler-Bernoulli frame metrics.
    """
    t0 = time.time()
    gs = ground_structure
    domain = gs.domain
    num_members = gs.num_members
    L = gs.member_lengths

    B_free, F_free, _, _ = build_nodal_equilibrium_matrix(gs, load_case, domain)
    num_constraints = B_free.shape[0]

    # Variables: x = [q_1^+, ..., q_M^+, q_1^-, ..., q_M^-] (length 2*M)
    # Objective: c = [L_1, ..., L_M, L_1, ..., L_M]
    c_obj = np.concatenate([L, L])

    # Equalities: [B, -B] * [q+; q-] = F_free
    A_eq = np.hstack([B_free, -B_free])
    b_eq = F_free

    # Bounds: q_i^+ >= 0, q_i^- >= 0
    bounds = [(0.0, None) for _ in range(2 * num_members)]

    # Solve using HiGHS
    res_lp = linprog(
        c=c_obj,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    t_solve = time.time() - t0

    if not res_lp.success:
        # Fallback to zeros if LP infeasible
        return LPOptimizationResult(
            optimal_forces=np.zeros(num_members),
            optimal_areas=np.zeros(num_members),
            optimal_thicknesses=np.zeros(num_members),
            active_mask=np.zeros(num_members, dtype=bool),
            num_active_members=0,
            lp_objective_volume=0.0,
            achieved_volume=0.0,
            achieved_volume_fraction=0.0,
            frame_compliance=float("inf"),
            frame_stiffness=0.0,
            axial_energy_share=0.0,
            bending_energy_share=0.0,
            solver_status=res_lp.message,
            solve_time_s=t_solve,
        )

    x_sol = res_lp.x
    q_plus = x_sol[:num_members]
    q_minus = x_sol[num_members:]

    # Net forces: N = q+ - q-
    N_opt = q_plus - q_minus

    # Unscaled optimal areas for allowable stress: A_i = (q_i^+ + q_i^-) / sigma_0
    A_raw = (q_plus + q_minus) / sigma_allowable
    raw_vol = float(np.sum(A_raw * L) * domain.depth)

    # Scale areas to match the target volume fraction exactly
    # Subtract pad volume
    target_total_vol = target_volume_fraction * domain.bounding_volume
    # Compute pad volume estimate
    pad_thickness = 0.005
    dummy_sections = [RectangularSection(t=0.01, b=domain.depth) for _ in range(num_members)]
    trial_mesh = build_fe_mesh_from_ground_structure(gs, dummy_sections, material)
    _, _, pad_vol = load_case.apply(trial_mesh, domain, material)

    target_lattice_vol = max(1e-9, target_total_vol - pad_vol)
    scale_factor = target_lattice_vol / max(1e-12, raw_vol) if raw_vol > 0.0 else 1.0

    A_scaled = A_raw * scale_factor
    t_scaled = A_scaled / domain.depth  # for rectangular section of depth b

    # Active mask based on threshold
    max_t = np.max(t_scaled) if len(t_scaled) > 0 else 1.0
    active_mask = t_scaled >= max(1e-9, zero_area_threshold * max_t)
    num_active = int(np.sum(active_mask))

    # Rescale active members so total lattice volume matches target volume fraction exactly
    active_vol = float(np.sum(A_scaled[active_mask] * gs.member_lengths[active_mask]))
    if active_vol > 1e-12:
        rescale = target_lattice_vol / active_vol
        t_scaled[active_mask] *= rescale
        A_scaled[active_mask] = t_scaled[active_mask] * domain.depth

    # Evaluate the scaled optimal LP design in the full Euler-Bernoulli frame solver
    final_sections = [
        RectangularSection(t=max(1e-9, t_scaled[i]), b=domain.depth)
        for i in range(num_members)
    ]
    frame_mesh = build_fe_mesh_from_ground_structure(
        gs, final_sections, material, active_mask=active_mask
    )
    _, _, actual_pad_vol = load_case.apply(frame_mesh, domain, material)

    # Final exact volume scaling to match target volume budget accounting for actual pad volume
    target_lattice_vol = max(1e-9, target_total_vol - actual_pad_vol)
    curr_lattice_vol = sum(
        frame_mesh.elements[eid].volume
        for eid in frame_mesh.elements
        if eid <= num_members and active_mask[eid - 1]
    )
    if curr_lattice_vol > 1e-12:
        rescale = target_lattice_vol / curr_lattice_vol
        for eid in frame_mesh.elements:
            if eid <= num_members and active_mask[eid - 1]:
                elem = frame_mesh.elements[eid]
                new_t = elem.section.t * rescale
                elem.section = RectangularSection(t=new_t, b=domain.depth)
                t_scaled[eid - 1] = new_t
                A_scaled[eid - 1] = new_t * domain.depth

    frame_solver = FrameSolver2D(frame_mesh)
    try:
        frame_res = frame_solver.solve()
        f_comp = frame_res.compliance
        f_stiff = 1.0 / max(1e-15, f_comp)
        ax_share = frame_res.axial_energy_fraction
        bend_share = frame_res.bending_energy_fraction
    except Exception:
        # If frame solver detects mechanisms on raw truss, mark compliance as infinite
        f_comp = float("inf")
        f_stiff = 0.0
        ax_share = 1.0
        bend_share = 0.0

    achieved_vol = frame_mesh.total_volume
    achieved_vf = compute_volume_fraction(frame_mesh, domain)

    return LPOptimizationResult(
        optimal_forces=N_opt,
        optimal_areas=A_scaled,
        optimal_thicknesses=t_scaled,
        active_mask=active_mask,
        num_active_members=num_active,
        lp_objective_volume=raw_vol,
        achieved_volume=achieved_vol,
        achieved_volume_fraction=achieved_vf,
        frame_compliance=f_comp,
        frame_stiffness=f_stiff,
        axial_energy_share=ax_share,
        bending_energy_share=bend_share,
        solver_status="optimal",
        solve_time_s=t_solve,
    )


# Alias for consistent naming with other optimizers
optimize_ground_structure_truss_lp = solve_ground_structure_lp
