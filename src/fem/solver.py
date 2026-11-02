"""Sparse Finite Element solver for 2D Euler-Bernoulli frame structures."""

from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mesh import FrameMesh2D
from .elements import MemberInternalForces


class SingularMatrixError(Exception):
    """Raised when the structural stiffness matrix is singular or ill-conditioned.

    Indicates an unconstrained rigid-body mode, kinematic mechanism, or disconnected structure.
    """
    pass


@dataclass
class SolverResult:
    """Complete results of the 2D frame FE analysis.

    Attributes:
        mesh: Reference to the solved FrameMesh2D structure.
        displacements: Global displacement vector u (length 3 * num_nodes).
        reactions: Global nodal reactions vector R (length 3 * num_nodes).
        internal_forces: Dictionary mapping element_id to MemberInternalForces.
        compliance: Total structural compliance C = F^T @ u (Joules or N*m).
        total_volume: Total volume of members (m^3).
        total_mass: Total mass of members (kg).
        total_strain_energy: Total elastic strain energy U_total (Joules).
        total_axial_energy: Total axial strain energy U_axial (Joules).
        total_bending_energy: Total flexural strain energy U_bending (Joules).
        axial_energy_fraction: Ratio U_axial / U_total (dimensionless, [0, 1]).
        bending_energy_fraction: Ratio U_bending / U_total (dimensionless, [0, 1]).
    """

    mesh: FrameMesh2D
    displacements: np.ndarray
    reactions: np.ndarray
    internal_forces: Dict[int, MemberInternalForces]
    compliance: float
    total_volume: float
    total_mass: float
    total_strain_energy: float
    total_axial_energy: float
    total_bending_energy: float
    axial_energy_fraction: float
    bending_energy_fraction: float

    def get_nodal_displacement(self, node_id: int) -> Tuple[float, float, float]:
        """Return (ux, uy, theta) for a given node."""
        idx_ux, idx_uy, idx_th = self.mesh.get_dof_indices(node_id)
        return (
            float(self.displacements[idx_ux]),
            float(self.displacements[idx_uy]),
            float(self.displacements[idx_th]),
        )

    def get_nodal_reaction(self, node_id: int) -> Tuple[float, float, float]:
        """Return (Rx, Ry, Rm) for a given node."""
        idx_ux, idx_uy, idx_th = self.mesh.get_dof_indices(node_id)
        return (
            float(self.reactions[idx_ux]),
            float(self.reactions[idx_uy]),
            float(self.reactions[idx_th]),
        )

    def check_equilibrium(self) -> Dict[str, float]:
        """Verify global force and moment equilibrium across the entire structure.

        Returns a dictionary containing net forces, net moments, and applied vs reaction sums.
        """
        sum_fx_applied = sum(ld.fx for ld in self.mesh.point_loads)
        sum_fy_applied = sum(ld.fy for ld in self.mesh.point_loads)
        sum_m0_applied = 0.0
        for ld in self.mesh.point_loads:
            x, y = self.mesh.nodes[ld.node_id]
            # Cross product r x F = x * fy - y * fx + m
            sum_m0_applied += (x * ld.fy - y * ld.fx + ld.m)

        sum_rx = 0.0
        sum_ry = 0.0
        sum_m0_rx = 0.0
        for nid, (x, y) in self.mesh.nodes.items():
            rx, ry, rm = self.get_nodal_reaction(nid)
            sum_rx += rx
            sum_ry += ry
            sum_m0_rx += (x * ry - y * rx + rm)

        net_fx = sum_fx_applied + sum_rx
        net_fy = sum_fy_applied + sum_ry
        net_m0 = sum_m0_applied + sum_m0_rx

        total_applied_force = float(np.hypot(sum_fx_applied, sum_fy_applied))
        rel_err_fx = abs(net_fx) / max(1.0, abs(sum_fx_applied), total_applied_force)
        rel_err_fy = abs(net_fy) / max(1.0, abs(sum_fy_applied), total_applied_force)
        rel_err_m0 = abs(net_m0) / max(1.0, abs(sum_m0_applied))

        return {
            "sum_fx_applied": sum_fx_applied,
            "sum_fy_applied": sum_fy_applied,
            "sum_m0_applied": sum_m0_applied,
            "sum_rx": sum_rx,
            "sum_ry": sum_ry,
            "sum_m0_rx": sum_m0_rx,
            "net_fx": net_fx,
            "net_fy": net_fy,
            "net_m0": net_m0,
            "rel_err_fx": rel_err_fx,
            "rel_err_fy": rel_err_fy,
            "rel_err_m0": rel_err_m0,
        }


class FrameSolver2D:
    """Finite Element solver for 2D Euler-Bernoulli frame structures."""

    def __init__(self, mesh: FrameMesh2D) -> None:
        self.mesh = mesh

    def check_mesh_connectivity_and_supports(self) -> None:
        """Verify structural connectivity and planar rigid-body constraints.

        Multi-stage verification:
        1. Identifies isolated nodes with no incident members.
        2. Segregates the structural graph into connected components.
        3. For every connected component, checks that the prescribed boundary
           conditions span all 3 planar rigid-body modes (translations and rotation),
           preventing mechanisms and unconstrained motions.
        """
        if self.mesh.num_nodes == 0:
            raise ValueError("Mesh contains no nodes.")

        # Build adjacency graph
        adj: Dict[int, Set[int]] = {nid: set() for nid in self.mesh.nodes}
        for elem in self.mesh.elements.values():
            adj[elem.node_i].add(elem.node_j)
            adj[elem.node_j].add(elem.node_i)

        # Build node boundary condition map
        prescribed_by_node: Dict[int, Set[int]] = {nid: set() for nid in self.mesh.nodes}
        for bc in self.mesh.boundary_conditions:
            prescribed_by_node[bc.node_id].add(bc.dof_index)

        # Check for isolated nodes
        for nid, neighbors in adj.items():
            if len(neighbors) == 0:
                # If node has no members and not all 3 DOFs are fixed, it is a floating node
                if len(prescribed_by_node[nid]) < 3:
                    raise SingularMatrixError(
                        f"Floating unsupported node detected: Node {nid} is not connected "
                        f"to any element and has unconstrained degrees of freedom."
                    )

        # Decompose into connected components
        visited: Set[int] = set()
        components: List[List[int]] = []
        for nid in sorted(self.mesh.nodes.keys()):
            if nid not in visited:
                comp = []
                queue = [nid]
                visited.add(nid)
                while queue:
                    curr = queue.pop()
                    comp.append(curr)
                    for nbr in adj[curr]:
                        if nbr not in visited:
                            visited.add(nbr)
                            queue.append(nbr)
                components.append(comp)

        # Check rigid-body mode constraints per component
        for comp in components:
            if len(comp) == 1 and len(adj[comp[0]]) == 0:
                continue  # Fully constrained single node handled above

            comp_nodes = sorted(comp)
            comp_coords = np.array([self.mesh.nodes[n] for n in comp_nodes], dtype=np.float64)
            xc, yc = np.mean(comp_coords, axis=0)

            comp_node_map = {n: i for i, n in enumerate(comp_nodes)}
            n_comp_dofs = 3 * len(comp_nodes)

            # Assemble rigid-body mode vectors for this component:
            # Column 0: Translation in x [1, 0, 0, 1, 0, 0, ...]
            # Column 1: Translation in y [0, 1, 0, 0, 1, 0, ...]
            # Column 2: In-plane rotation about component centroid (xc, yc):
            #           ux = -(y - yc), uy = (x - xc), theta = 1
            V_rb = np.zeros((n_comp_dofs, 3), dtype=np.float64)
            for n in comp_nodes:
                idx = comp_node_map[n]
                x, y = self.mesh.nodes[n]
                V_rb[3 * idx, 0] = 1.0
                V_rb[3 * idx + 1, 1] = 1.0
                V_rb[3 * idx + 2, 2] = 1.0
                V_rb[3 * idx, 2] = -(y - yc)
                V_rb[3 * idx + 1, 2] = (x - xc)

            # Gather prescribed DOFs inside this component
            prescribed_comp_dofs: List[int] = []
            for n in comp_nodes:
                idx = comp_node_map[n]
                for dof_idx in prescribed_by_node[n]:
                    prescribed_comp_dofs.append(3 * idx + dof_idx)

            if len(prescribed_comp_dofs) < 3:
                raise SingularMatrixError(
                    f"Insufficient boundary conditions on structural component (nodes {comp_nodes}): "
                    f"Only {len(prescribed_comp_dofs)} DOFs are supported (at least 3 non-degenerate "
                    f"supports are required in 2D)."
                )

            # Check rank of the rigid-body modes projected onto the prescribed DOFs
            V_prescribed = V_rb[prescribed_comp_dofs, :]
            svals = np.linalg.svd(V_prescribed, compute_uv=False)
            rb_rank = int(np.sum(svals > 1e-8))

            if rb_rank < 3:
                raise SingularMatrixError(
                    f"Defective boundary conditions on structural component (nodes {comp_nodes}): "
                    f"Supports constrain only {rb_rank} of the 3 planar rigid-body modes "
                    f"(singular values: {svals}). The structure can move as a rigid body."
                )

    def assemble_global_stiffness(self) -> sp.csr_matrix:
        """Assemble the global structural stiffness matrix K into sparse CSR format."""
        num_dofs = self.mesh.num_dofs
        node_map = self.mesh.get_node_to_index_mapping()

        num_elements = self.mesh.num_elements
        if num_elements == 0:
            raise ValueError("Cannot assemble stiffness matrix: mesh contains no elements.")

        rows = np.empty(num_elements * 36, dtype=np.int32)
        cols = np.empty(num_elements * 36, dtype=np.int32)
        vals = np.empty(num_elements * 36, dtype=np.float64)

        offset = 0
        for elem in self.mesh.elements.values():
            idx_i = node_map[elem.node_i]
            idx_j = node_map[elem.node_j]

            elem_dofs = np.array(
                [
                    3 * idx_i,
                    3 * idx_i + 1,
                    3 * idx_i + 2,
                    3 * idx_j,
                    3 * idx_j + 1,
                    3 * idx_j + 2,
                ],
                dtype=np.int32,
            )

            k_elem = elem.global_stiffness_matrix()

            r_grid, c_grid = np.meshgrid(elem_dofs, elem_dofs, indexing="ij")
            rows[offset : offset + 36] = r_grid.ravel()
            cols[offset : offset + 36] = c_grid.ravel()
            vals[offset : offset + 36] = k_elem.ravel()
            offset += 36

        K_global = sp.coo_matrix((vals, (rows, cols)), shape=(num_dofs, num_dofs)).tocsr()
        return K_global

    def assemble_load_vector(self) -> np.ndarray:
        """Assemble the global nodal load vector F (length num_dofs)."""
        num_dofs = self.mesh.num_dofs
        node_map = self.mesh.get_node_to_index_mapping()
        F = np.zeros(num_dofs, dtype=np.float64)

        for ld in self.mesh.point_loads:
            idx = node_map[ld.node_id]
            F[3 * idx] += ld.fx
            F[3 * idx + 1] += ld.fy
            F[3 * idx + 2] += ld.m

        return F

    def solve(self) -> SolverResult:
        """Solve the linear structural system K @ u = F subject to boundary conditions.

        Employs a multi-stage verification pipeline:
        1. Graph connectivity and rigid-body support rank check.
        2. Diagonal stiffness equilibration (Jacobi preconditioning) to eliminate
           the dimensional disparity between axial and flexural stiffness.
        3. Equilibrated pivot inspection and mechanism detection.

        Returns:
            SolverResult containing displacements, reactions, compliance, and energy breakdown.

        Raises:
            SingularMatrixError: If the system is kinematically unconstrained, unstable,
                                 or disconnected.
        """
        # Stage 1: Structural connectivity and support constraints
        self.check_mesh_connectivity_and_supports()

        K = self.assemble_global_stiffness()
        F = self.assemble_load_vector()
        num_dofs = self.mesh.num_dofs
        node_map = self.mesh.get_node_to_index_mapping()

        # Identify prescribed DOFs from boundary conditions
        prescribed_dict: Dict[int, float] = {}
        for bc in self.mesh.boundary_conditions:
            idx = node_map[bc.node_id]
            global_dof = 3 * idx + bc.dof_index
            prescribed_dict[global_dof] = bc.value

        all_dofs = np.arange(num_dofs, dtype=np.int32)
        prescribed_dofs = np.array(sorted(prescribed_dict.keys()), dtype=np.int32)
        is_prescribed = np.zeros(num_dofs, dtype=bool)
        is_prescribed[prescribed_dofs] = True
        free_dofs = all_dofs[~is_prescribed]

        u = np.zeros(num_dofs, dtype=np.float64)
        for dof, val in prescribed_dict.items():
            u[dof] = val

        if len(free_dofs) > 0:
            K_ff = K[free_dofs, :][:, free_dofs].tocsr()
            K_fp = K[free_dofs, :][:, prescribed_dofs]
            u_p = np.array([prescribed_dict[d] for d in prescribed_dofs], dtype=np.float64)

            F_free = F[free_dofs]
            if len(prescribed_dofs) > 0:
                F_eff = F_free - K_fp @ u_p
            else:
                F_eff = F_free

            # Stage 2: Check for zero diagonal entries on free DOFs
            diag_entries = K_ff.diagonal()
            min_diag = float(np.min(diag_entries)) if len(diag_entries) > 0 else 0.0
            if min_diag <= 0.0:
                zero_free_idx = int(np.where(diag_entries <= 0.0)[0][0])
                zero_global_dof = int(free_dofs[zero_free_idx])
                node_idx = zero_global_dof // 3
                dof_type = ["ux", "uy", "theta"][zero_global_dof % 3]
                inv_node_map = {idx: nid for nid, idx in node_map.items()}
                nid = inv_node_map[node_idx]
                raise SingularMatrixError(
                    f"Zero stiffness detected on degree of freedom '{dof_type}' of node {nid}."
                )

            # Stage 3: Dimensionless diagonal equilibration (Jacobi scaling)
            # Normalizes axial (N/m) and rotational (N*m/rad) entries so all diagonal terms are 1.0
            d_scale = 1.0 / np.sqrt(diag_entries)
            D_mat = sp.diags(d_scale, format="csr")
            K_tilde = (D_mat @ K_ff @ D_mat).tocsc()
            F_tilde = d_scale * F_eff

            # Solve equilibrated system with SuperLU
            try:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")
                    lu = spla.splu(K_tilde)

                    # In equilibrated K_tilde, diagonal pivots are dimensionless.
                    # A near-zero pivot indicates an internal mechanism or zero-energy mode.
                    pivots = np.abs(lu.U.diagonal())
                    min_pivot = float(np.min(pivots))
                    max_pivot = float(np.max(pivots))

                    if min_pivot < 1e-10:
                        raise SingularMatrixError(
                            f"Kinematic mechanism detected: equilibrated pivot ({min_pivot:.3e}) "
                            f"is effectively zero (< 1e-10). The structure cannot support load."
                        )

                    u_tilde = lu.solve(F_tilde)

                    for w in caught_warnings:
                        if issubclass(w.category, (spla.MatrixRankWarning, UserWarning)):
                            raise SingularMatrixError(
                                f"Stiffness matrix rank deficiency warning: {w.message}"
                            )

            except (RuntimeError, spla.MatrixRankWarning, ValueError) as err:
                raise SingularMatrixError(
                    f"Structural stiffness matrix is singular or mechanism detected: {err}"
                ) from err

            # Unscale solution: u_f = D @ u_tilde
            u_f = d_scale * u_tilde

            if not np.all(np.isfinite(u_f)):
                raise SingularMatrixError(
                    "Solution contains NaN or Inf values. Structure has an unconstrained "
                    "rigid-body mode or kinematic mechanism."
                )

            u[free_dofs] = u_f

        # Compute reaction forces: R = K @ u - F
        R = K @ u - F

        # Compute internal forces and strain energy breakdown for each member
        internal_forces: Dict[int, MemberInternalForces] = {}
        total_u_axial = 0.0
        total_u_bending = 0.0

        for eid, elem in self.mesh.elements.items():
            idx_i = node_map[elem.node_i]
            idx_j = node_map[elem.node_j]
            elem_dofs = [
                3 * idx_i,
                3 * idx_i + 1,
                3 * idx_i + 2,
                3 * idx_j,
                3 * idx_j + 1,
                3 * idx_j + 2,
            ]
            u_elem = u[elem_dofs]
            forces = elem.compute_internal_forces(u_elem)
            internal_forces[eid] = forces
            total_u_axial += forces.strain_energy_axial
            total_u_bending += forces.strain_energy_bending

        total_strain_energy = total_u_axial + total_u_bending
        compliance = float(F @ u)

        if total_strain_energy > 1e-18:
            frac_axial = float(total_u_axial / total_strain_energy)
            frac_bending = float(total_u_bending / total_strain_energy)
        else:
            frac_axial = 0.0
            frac_bending = 0.0

        return SolverResult(
            mesh=self.mesh,
            displacements=u,
            reactions=R,
            internal_forces=internal_forces,
            compliance=compliance,
            total_volume=self.mesh.total_volume,
            total_mass=self.mesh.total_mass,
            total_strain_energy=total_strain_energy,
            total_axial_energy=total_u_axial,
            total_bending_energy=total_u_bending,
            axial_energy_fraction=frac_axial,
            bending_energy_fraction=frac_bending,
        )
