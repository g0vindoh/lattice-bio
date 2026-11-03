"""2D Solid Isotropic Material with Penalization (SIMP) Continuum Topology Optimization.

Implements density-based topology optimization on a 2D bilinear quad mesh (Q4 plane stress)
using the classic Bendsøe & Sigmund (2003) / Bourdin (2001) formulation:

    min_{x}      C(x) = U^T K(x) U = sum_{e=1}^N (E_min + x_e_tilde^p (E_0 - E_min)) u_e^T k_0 u_e
    subject to:  V(x) / V_0 = sum_{e=1}^N x_e_tilde / N <= v_f
                 0 <= x_e <= 1

with:
    - Standard density filtering to avoid checkerboarding and mesh dependency
    - Optimality Criteria (OC) update scheme with Lagrange multiplier bisection
    - Compatible with LC1 (Cantilever), LC2 (Simply Supported), LC3 (Compression Block)
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..geometry.domain import DesignDomain2D


@dataclass
class SIMPOptimizationResult:
    """Results from SIMP continuum topology optimization.

    Attributes:
        densities: 2D array of element physical densities (ny, nx) in [0, 1].
        filtered_densities: 2D array of filtered element densities (ny, nx).
        compliance: Final structural compliance C = U^T K U.
        volume_fraction: Achieved volume fraction.
        compliance_history: List of compliance values across iterations.
        change_history: List of maximum density changes across iterations.
        num_iterations: Number of iterations performed.
        converged: Whether convergence tolerance was met.
        solve_time_s: Total optimization runtime in seconds.
    """

    densities: np.ndarray
    filtered_densities: np.ndarray
    compliance: float
    volume_fraction: float
    compliance_history: List[float]
    change_history: List[float]
    num_iterations: int
    converged: bool
    solve_time_s: float


def compute_q4_element_stiffness(
    hx: float,
    hy: float,
    E: float = 1.0,
    nu: float = 0.3,
) -> np.ndarray:
    """Compute the 8x8 element stiffness matrix k_0 for a bilinear quad under plane stress.

    Nodes ordered counter-clockwise:
        3(-1, 1) --- 2(1, 1)
           |            |
        0(-1,-1) --- 1(1,-1)

    DOFs per node: [u_x, u_y].
    """
    # Plane stress constitutive matrix D
    factor = E / (1.0 - nu**2)
    D = factor * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, 0.5 * (1.0 - nu)],
    ])

    # 2x2 Gauss quadrature points and weights
    gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    k0 = np.zeros((8, 8), dtype=np.float64)

    a = hx / 2.0
    b = hy / 2.0

    for xi in gauss_pts:
        for eta in gauss_pts:
            # Shape function derivatives w.r.t natural coordinates xi, eta
            # N_i = 1/4 (1 + xi_i * xi) (1 + eta_i * eta)
            # dN/dxi, dN/deta
            dNdxi = 0.25 * np.array([
                -(1.0 - eta),
                 (1.0 - eta),
                 (1.0 + eta),
                -(1.0 + eta),
            ])
            dNdeta = 0.25 * np.array([
                -(1.0 - xi),
                -(1.0 + xi),
                 (1.0 + xi),
                 (1.0 - xi),
            ])

            # Jacobian J = [dx/dxi, dy/dxi; dx/deta, dy/deta]
            # For rectangular elements: dx/dxi = a, dy/deta = b, cross-terms = 0
            detJ = a * b
            invJ = np.array([[1.0 / a, 0.0], [0.0, 1.0 / b]])

            dNdx = invJ[0, 0] * dNdxi
            dNdy = invJ[1, 1] * dNdeta

            # Strain-displacement matrix B (3 x 8)
            B = np.zeros((3, 8), dtype=np.float64)
            for i in range(4):
                B[0, 2 * i] = dNdx[i]
                B[1, 2 * i + 1] = dNdy[i]
                B[2, 2 * i] = dNdy[i]
                B[2, 2 * i + 1] = dNdx[i]

            k0 += B.T @ D @ B * detJ

    return k0


class SIMPTopologyOptimizer:
    """2D SIMP Continuum Topology Optimizer using Optimality Criteria.

    Args:
        domain: DesignDomain2D specifying dimensions (width W, height H).
        nx: Number of elements along the horizontal direction.
        ny: Number of elements along the vertical direction.
        target_volume_fraction: Target volume fraction v_f in (0, 1].
        penalization: SIMP penalization power p (default 3.0).
        filter_radius: Filter radius r_min in spatial units (m).
        E0: Base material Young's modulus (Pa).
        Emin: Void material Young's modulus factor (default 1e-9 * E0).
        nu: Poisson's ratio (default 0.3).
        max_iterations: Maximum OC iterations (default 100).
        tolerance: Convergence tolerance on change in design variables (default 1e-3).
    """

    def __init__(
        self,
        domain: DesignDomain2D,
        nx: int = 60,
        ny: int = 20,
        target_volume_fraction: float = 0.3,
        penalization: float = 3.0,
        filter_radius: Optional[float] = None,
        E0: float = 1e9,
        Emin: float = 1e-9,
        nu: float = 0.3,
        max_iterations: int = 100,
        tolerance: float = 1e-3,
    ) -> None:
        self.domain = domain
        self.nx = int(nx)
        self.ny = int(ny)
        self.v_f = float(target_volume_fraction)
        self.p = float(penalization)
        self.E0 = float(E0)
        self.Emin = float(Emin)
        self.nu = float(nu)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)

        self.hx = domain.width / self.nx
        self.hy = domain.height / self.ny

        # Default filter radius: 1.5 * element size
        if filter_radius is None:
            self.r_min = 1.5 * max(self.hx, self.hy)
        else:
            self.r_min = float(filter_radius)

        self.num_elements = self.nx * self.ny
        self.num_nodes = (self.nx + 1) * (self.ny + 1)
        self.total_dofs = 2 * self.num_nodes

        # Precompute element stiffness
        self.k0 = compute_q4_element_stiffness(self.hx, self.hy, E=1.0, nu=self.nu)

        # Precompute DOF connectivity and filter matrix
        self._build_dof_indices()
        self._build_filter_weights()

    def _build_dof_indices(self) -> None:
        """Map each element to its 8 global DOFs."""
        self.elem_dofs = np.zeros((self.num_elements, 8), dtype=np.int64)
        for ely in range(self.ny):
            for elx in range(self.nx):
                eid = ely * self.nx + elx
                # Node numbering:
                # n3 --- n2
                # |       |
                # n0 --- n1
                n0 = ely * (self.nx + 1) + elx
                n1 = n0 + 1
                n2 = (ely + 1) * (self.nx + 1) + (elx + 1)
                n3 = (ely + 1) * (self.nx + 1) + elx

                nodes = [n0, n1, n2, n3]
                dofs = []
                for n in nodes:
                    dofs.extend([2 * n, 2 * n + 1])
                self.elem_dofs[eid, :] = dofs

    def _build_filter_weights(self) -> None:
        """Precompute sparse distance-based density filter matrix H."""
        # Centers of elements
        centers_x = np.zeros(self.num_elements, dtype=np.float64)
        centers_y = np.zeros(self.num_elements, dtype=np.float64)
        for ely in range(self.ny):
            for elx in range(self.nx):
                eid = ely * self.nx + elx
                centers_x[eid] = (elx + 0.5) * self.hx
                centers_y[eid] = (ely + 0.5) * self.hy

        i_list, j_list, w_list = [], [], []
        r = self.r_min
        # Search neighborhood bounds in element indices
        n_rad_x = int(np.ceil(r / self.hx))
        n_rad_y = int(np.ceil(r / self.hy))

        for ely in range(self.ny):
            for elx in range(self.nx):
                e1 = ely * self.nx + elx
                x1, y1 = centers_x[e1], centers_y[e1]

                for dy in range(-n_rad_y, n_rad_y + 1):
                    n_y = ely + dy
                    if not (0 <= n_y < self.ny):
                        continue
                    for dx in range(-n_rad_x, n_rad_x + 1):
                        n_x = elx + dx
                        if not (0 <= n_x < self.nx):
                            continue
                        e2 = n_y * self.nx + n_x
                        dist = np.hypot(centers_x[e2] - x1, centers_y[e2] - y1)
                        if dist < r:
                            weight = r - dist
                            i_list.append(e1)
                            j_list.append(e2)
                            w_list.append(weight)

        H = sp.coo_matrix(
            (w_list, (i_list, j_list)),
            shape=(self.num_elements, self.num_elements),
            dtype=np.float64,
        ).tocsr()

        # Row normalization: H_tilde = H / row_sum
        row_sum = np.array(H.sum(axis=1)).flatten()
        self.filter_weights = H
        self.filter_row_sum = np.maximum(row_sum, 1e-12)

    def filter_density(self, x: np.ndarray) -> np.ndarray:
        """Apply density filter: x_tilde = H * x / sum(H, axis=1)."""
        return self.filter_weights.dot(x) / self.filter_row_sum

    def setup_boundary_conditions(
        self,
        load_case_name: str = "cantilever",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate force vector F and fixed DOF indices for standard load cases.

        Args:
            load_case_name: 'cantilever' (LC1), 'simply_supported' (LC2), or 'compression' (LC3).

        Returns:
            F: Global force vector (length total_dofs).
            fixed_dofs: Array of fixed global DOF indices.
        """
        lc = load_case_name.lower()
        F = np.zeros(self.total_dofs, dtype=np.float64)
        fixed_dofs = []

        if "cantilever" in lc or "lc1" in lc:
            # Fixed at left boundary (x = 0): all nodes at elx = 0
            for ely in range(self.ny + 1):
                nid = ely * (self.nx + 1)
                fixed_dofs.extend([2 * nid, 2 * nid + 1])

            # Downward point load at right edge mid-height:
            mid_y = self.ny // 2
            load_node = mid_y * (self.nx + 1) + self.nx
            F[2 * load_node + 1] = -1.0

        elif "simply" in lc or "lc2" in lc:
            # Simply supported beam:
            # Bottom-left corner pinned: (0, 0)
            bl_node = 0
            fixed_dofs.extend([2 * bl_node, 2 * bl_node + 1])

            # Bottom-right corner roller: (nx, 0), vertical DOF fixed
            br_node = self.nx
            fixed_dofs.append(2 * br_node + 1)

            # Downward point load at top edge center
            mid_x = self.nx // 2
            top_mid_node = self.ny * (self.nx + 1) + mid_x
            F[2 * top_mid_node + 1] = -1.0

        elif "compression" in lc or "lc3" in lc:
            # Compression block:
            # Bottom edge fully fixed: ely = 0
            for elx in range(self.nx + 1):
                nid = elx
                fixed_dofs.extend([2 * nid, 2 * nid + 1])

            # Distributed downward load along top edge: ely = ny
            load_per_node = -1.0 / (self.nx + 1)
            for elx in range(self.nx + 1):
                nid = self.ny * (self.nx + 1) + elx
                F[2 * nid + 1] = load_per_node

        else:
            raise ValueError(f"Unknown SIMP load case: '{load_case_name}'")

        return F, np.array(sorted(list(set(fixed_dofs))), dtype=np.int64)

    def optimize(
        self,
        load_case_name: str = "cantilever",
    ) -> SIMPOptimizationResult:
        """Run SIMP topology optimization loop."""
        start_time = time.perf_counter()

        F, fixed_dofs = self.setup_boundary_conditions(load_case_name)
        all_dofs = np.arange(self.total_dofs, dtype=np.int64)
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

        # Initial uniform density equal to target volume fraction
        x = np.full(self.num_elements, self.v_f, dtype=np.float64)
        x_tilde = self.filter_density(x)

        compliance_history = []
        change_history = []
        converged = False

        # Precompute index arrays for sparse stiffness assembly
        iK = np.repeat(self.elem_dofs, 8, axis=1).flatten()
        jK = np.tile(self.elem_dofs, (1, 8)).flatten()

        U = np.zeros(self.total_dofs, dtype=np.float64)

        for iteration in range(1, self.max_iterations + 1):
            # 1. Compute element Young's moduli
            E_elements = self.Emin + (x_tilde**self.p) * (self.E0 - self.Emin)

            # 2. Assemble global stiffness matrix
            # sK: shape (num_elements, 8, 8) flattened
            sK = np.outer(E_elements, self.k0.flatten()).flatten()
            K = sp.coo_matrix((sK, (iK, jK)), shape=(self.total_dofs, self.total_dofs)).tocsc()

            # 3. Solve for displacements at free DOFs
            K_free = K[free_dofs, :][:, free_dofs]
            F_free = F[free_dofs]

            try:
                u_free = spla.spsolve(K_free, F_free)
                U[free_dofs] = u_free
                U[fixed_dofs] = 0.0
            except Exception as e:
                # Fallback if solver encounters conditioning trouble
                break

            # 4. Element compliance and sensitivities
            u_elem = U[self.elem_dofs]  # (num_elements, 8)
            # ce = u_e^T * k0 * u_e
            ce = np.sum((u_elem @ self.k0) * u_elem, axis=1)

            compliance = float(np.sum(E_elements * ce))
            compliance_history.append(compliance)

            # dC/d_xtilde = - p * xtilde^{p-1} * (E0 - Emin) * ce
            dc_dxtilde = -self.p * (x_tilde ** (self.p - 1)) * (self.E0 - self.Emin) * ce

            # Filter sensitivity: dc/dx = H^T * (dc_dxtilde / row_sum)
            dc_dx = self.filter_weights.T.dot(dc_dxtilde / self.filter_row_sum)

            # 5. Optimality Criteria (OC) update for x
            l1, l2 = 0.0, 1e9
            move = 0.2
            x_new = np.copy(x)

            # Bisection loop to find Lagrange multiplier lambda satisfying volume constraint
            while (l2 - l1) / max(1e-10, l1 + l2) > 1e-4:
                lmid = 0.5 * (l1 + l2)
                # B_e = - dc_dx / lmid
                # x_new = clamp(x * sqrt(max(0, -dc_dx / lmid)), x - move, x + move, 0, 1)
                Be = np.maximum(0.0, -dc_dx / max(1e-12, lmid))
                x_candidate = np.maximum(
                    0.0,
                    np.maximum(
                        x - move,
                        np.minimum(1.0, np.minimum(x + move, x * np.sqrt(Be))),
                    ),
                )
                x_candidate_tilde = self.filter_density(x_candidate)

                if np.mean(x_candidate_tilde) > self.v_f:
                    l1 = lmid
                else:
                    l2 = lmid
                x_new = x_candidate

            max_change = float(np.max(np.abs(x_new - x)))
            change_history.append(max_change)

            x = x_new
            x_tilde = self.filter_density(x)

            if max_change < self.tolerance:
                converged = True
                break

        solve_time = time.perf_counter() - start_time
        final_vf = float(np.mean(x_tilde))

        # Reshape to 2D matrices (ny, nx)
        densities_2d = x.reshape((self.ny, self.nx))
        filtered_2d = x_tilde.reshape((self.ny, self.nx))

        return SIMPOptimizationResult(
            densities=densities_2d,
            filtered_densities=filtered_2d,
            compliance=compliance_history[-1] if compliance_history else float("inf"),
            volume_fraction=final_vf,
            compliance_history=compliance_history,
            change_history=change_history,
            num_iterations=len(compliance_history),
            converged=converged,
            solve_time_s=solve_time,
        )
