"""Trabecular bone fabric tensor and stereological anisotropy metrics.

Implements the Cowin (1985) second-order fabric tensor H describing the orientation
distribution and structural anisotropy of lattice struts:
    H = (1 / sum(V_e)) * sum(V_e * (n_e (x) n_e))
where n_e = [cos(theta_e), sin(theta_e)]^T is the strut unit director.

Computes:
1. Fabric tensor H (2x2 symmetric, trace = 1.0).
2. Principal fabric eigenvalues (lambda_1 >= lambda_2, lambda_1 + lambda_2 = 1.0).
3. Principal orientation angle theta_fabric (degrees).
4. Fractional Anisotropy (FA index in [0, 1]).
5. Wolff's trajectorial alignment with global principal stress direction.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..fem.mesh import FrameMesh2D
from ..fem.solver import SolverResult


@dataclass
class FabricTensorResult:
    """Quantitative fabric tensor and stereological anisotropy metrics.

    Attributes:
        H: 2x2 symmetric fabric tensor matrix (normalized, trace = 1.0).
        eigenvalues: Sorted eigenvalues [lambda_1, lambda_2] with lambda_1 >= lambda_2.
        eigenvectors: 2x2 matrix where columns are the principal fabric directors.
        principal_angle_deg: Angle of the primary fabric orientation in [-90, 90] degrees.
        fractional_anisotropy: Fractional anisotropy index FA in [0, 1] (0 = isotropic, 1 = uniaxial).
        total_length: Total active strut length (m).
        total_volume: Total active strut volume (m^3).
        degree_of_anisotropy: Ratio lambda_1 / lambda_2 (DA >= 1.0).
    """

    H: np.ndarray
    eigenvalues: Tuple[float, float]
    eigenvectors: np.ndarray
    principal_angle_deg: float
    fractional_anisotropy: float
    total_length: float
    total_volume: float
    degree_of_anisotropy: float


def compute_lattice_fabric_tensor(
    mesh: FrameMesh2D,
    active_mask: Optional[np.ndarray] = None,
    weight_by_volume: bool = True,
) -> FabricTensorResult:
    """Compute the Cowin 2nd-order fabric tensor for a frame lattice.

    Args:
        mesh: FrameMesh2D containing elements.
        active_mask: Optional boolean array or list indicating active elements.
                     If None, all mesh elements are treated as active.
        weight_by_volume: If True, weights strut orientation dyads by element volume (A_e * L_e).
                          If False, weights by element length L_e.

    Returns:
        FabricTensorResult containing tensor H, eigenvalues, principal angle, and FA.
    """
    if mesh.num_elements == 0:
        raise ValueError("Cannot compute fabric tensor for an empty mesh.")

    H = np.zeros((2, 2), dtype=np.float64)
    total_weight = 0.0
    total_len = 0.0
    total_vol = 0.0

    elem_list = list(mesh.elements.values())
    for idx, elem in enumerate(elem_list):
        if active_mask is not None and not active_mask[idx]:
            continue

        x1, y1 = mesh.nodes[elem.node_i]
        x2, y2 = mesh.nodes[elem.node_j]
        dx = x2 - x1
        dy = y2 - y1
        L = float(np.hypot(dx, dy))
        if L < 1e-12:
            continue

        c = dx / L
        s = dy / L
        # Dyadic outer product n (x) n
        dyad = np.array([[c * c, c * s], [c * s, s * s]], dtype=np.float64)

        vol = elem.section.A * L
        weight = vol if weight_by_volume else L

        H += weight * dyad
        total_weight += weight
        total_len += L
        total_vol += vol

    if total_weight <= 0.0:
        raise ValueError("Total active lattice weight is zero; cannot normalize fabric tensor.")

    H /= total_weight

    # Spectral decomposition (symmetric 2x2)
    evals, evecs = np.linalg.eigh(H)
    # Sort descending: lambda_1 >= lambda_2
    order = np.argsort(evals)[::-1]
    lam1 = float(evals[order[0]])
    lam2 = float(evals[order[1]])
    v1 = evecs[:, order[0]]

    # Primary orientation angle in [-90, 90] degrees
    theta_rad = np.arctan2(v1[1], v1[0])
    theta_deg = float(np.degrees(theta_rad))
    if theta_deg > 90.0:
        theta_deg -= 180.0
    elif theta_deg < -90.0:
        theta_deg += 180.0

    # Fractional Anisotropy FA in 2D
    # FA = sqrt(2) * sqrt((lam1 - 0.5)^2 + (lam2 - 0.5)^2) / sqrt(lam1^2 + lam2^2)
    lam_mean = 0.5 * (lam1 + lam2)
    dev_sq = (lam1 - lam_mean) ** 2 + (lam2 - lam_mean) ** 2
    norm_sq = lam1 ** 2 + lam2 ** 2
    fa = float(np.sqrt(2.0 * dev_sq / max(norm_sq, 1e-15)))

    # Degree of Anisotropy DA = lambda_1 / lambda_2
    da = float(lam1 / max(lam2, 1e-12))

    return FabricTensorResult(
        H=H,
        eigenvalues=(lam1, lam2),
        eigenvectors=evecs[:, order],
        principal_angle_deg=theta_deg,
        fractional_anisotropy=np.clip(fa, 0.0, 1.0),
        total_length=total_len,
        total_volume=total_vol,
        degree_of_anisotropy=da,
    )


def compute_wolff_alignment_metric(
    fabric_result: FabricTensorResult,
    result: SolverResult,
) -> Dict[str, float]:
    """Compute the angular alignment between the lattice fabric tensor and the principal stress field.

    Quantifies Wolff's trajectorial hypothesis: living trabecular bone aligns its
    principal structural fabric with the trajectories of principal stress.

    Returns:
        Dictionary containing principal fabric angle, principal stress orientation,
        and absolute angular alignment error Delta_theta in degrees.
    """
    # Compute volume-weighted average principal stress orientation
    total_vol = 0.0
    avg_nx = 0.0
    avg_ny = 0.0

    for eid, elem in result.mesh.elements.items():
        forces = result.internal_forces[eid]
        # Element axial stress
        sigma = forces.axial_force / elem.section.A
        vol = elem.section.A * elem.length
        # Strut angle
        x1, y1 = result.mesh.nodes[elem.node_i]
        x2, y2 = result.mesh.nodes[elem.node_j]
        theta = np.arctan2(y2 - y1, x2 - x1)
        # Weight by strain energy and direction
        w = abs(forces.strain_energy_total)
        avg_nx += w * np.cos(2.0 * theta)
        avg_ny += w * np.sin(2.0 * theta)
        total_vol += w

    if total_vol > 0.0:
        stress_angle_deg = 0.5 * float(np.degrees(np.arctan2(avg_ny, avg_nx)))
    else:
        stress_angle_deg = 0.0

    fabric_angle_deg = fabric_result.principal_angle_deg
    # Angular difference accounting for 180-degree directional symmetry
    diff = abs(fabric_angle_deg - stress_angle_deg)
    while diff > 90.0:
        diff = abs(180.0 - diff)

    return {
        "fabric_principal_angle_deg": fabric_angle_deg,
        "stress_principal_angle_deg": stress_angle_deg,
        "wolff_alignment_discrepancy_deg": float(diff),
        "fractional_anisotropy": fabric_result.fractional_anisotropy,
    }
