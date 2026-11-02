"""Authoritative structural performance metrics for lattice evaluation.

Defines the mathematical formulations for:
- Compliance C = F^T @ u
- Structural Volume and Mass
- Volume Fraction v_f = V / V_domain
- Equivalent Stiffness K = 1 / C
- Specific Stiffness S = (1 / C) / Mass
- Axial and Bending Strain Energy Shares
- Maximum Combined Member Normal Stress and Yield Utilization
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

from ..fem.solver import SolverResult
from ..geometry.domain import DesignDomain2D


@dataclass(frozen=True)
class PerformanceMetrics:
    """Consolidated performance metrics for a solved structural design.

    Attributes:
        compliance: Compliance C = F^T @ u (Joules).
        stiffness: Equivalent stiffness K = 1 / C (N/m for unit load, or 1/J).
        mass: Total mass of members (kg).
        volume: Total volume of members (m^3).
        volume_fraction: Ratio of structural volume to bounding domain volume (dimensionless).
        specific_stiffness: Specific stiffness (1/C) / mass (1 / (J * kg)).
        axial_energy_share: Fraction of strain energy carried axially (dimensionless, [0, 1]).
        bending_energy_share: Fraction of strain energy carried in bending (dimensionless, [0, 1]).
        max_normal_stress: Maximum combined normal stress across all members (Pa).
        max_utilization: Peak stress divided by yield strength (dimensionless).
    """

    compliance: float
    stiffness: float
    mass: float
    volume: float
    volume_fraction: float
    specific_stiffness: float
    axial_energy_share: float
    bending_energy_share: float
    max_normal_stress: float
    max_utilization: float


def compute_compliance(result: SolverResult) -> float:
    """Compute structural compliance C = F^T @ u.

    In linear elasticity with fixed supports, compliance equals twice the total
    strain energy: C = 2 * U_total. Lower compliance indicates higher stiffness.
    """
    return float(result.compliance)


def compute_stiffness(compliance: float, reference_load: float = 1.0) -> float:
    """Compute equivalent global stiffness K = (reference_load^2) / C.

    For a unit load (reference_load = 1.0), this simplifies to K = 1 / C.
    """
    if compliance <= 1e-18:
        raise ValueError(f"Compliance must be positive and non-zero, got {compliance:.3e}.")
    return float((reference_load**2) / compliance)


def compute_specific_stiffness(compliance: float, mass: float) -> float:
    """Compute specific stiffness S = (1 / C) / mass.

    Quantifies the stiffness delivered per unit structural mass, the core metric
    for lightweight structural design.
    """
    if mass <= 1e-18:
        raise ValueError(f"Mass must be positive and non-zero, got {mass:.3e}.")
    k = compute_stiffness(compliance, reference_load=1.0)
    return float(k / mass)


def compute_energy_shares(result: SolverResult) -> Tuple[float, float]:
    """Compute the fraction of elastic strain energy carried axially vs in bending.

    Returns:
        (axial_share, bending_share) where both are in [0, 1] and sum to 1.0.
    """
    return (float(result.axial_energy_fraction), float(result.bending_energy_fraction))


def compute_member_peak_stress(
    result: SolverResult, yield_strength: float = 250e6
) -> Tuple[float, float, int]:
    """Compute the maximum combined normal stress and yield utilization across all members.

    Normal stress derivation for rectangular beam-column:
        - Axial stress: sigma_axial = |N| / A
        - Bending stress at outer fibers (y = +/- t/2):
          sigma_bending = |M_max| * (t / 2) / I = 6 * |M_max| / (b * t^2)
        - Peak normal stress: sigma_max = sigma_axial + sigma_bending
        - Utilization: eta = sigma_max / yield_strength

    Args:
        result: SolverResult instance.
        yield_strength: Material yield stress (Pa, default 250 MPa for mild steel).

    Returns:
        (max_stress, max_utilization, critical_element_id)
    """
    max_stress = 0.0
    crit_elem = -1

    for eid, forces in result.internal_forces.items():
        elem = result.mesh.elements[eid]
        A = elem.section.A
        I = elem.section.I

        # Axial normal stress
        sig_axial = abs(forces.axial_force) / A

        # Peak bending moment between node 1 and node 2
        m_max = max(abs(forces.bending_moment_1), abs(forces.bending_moment_2))

        # Distance to extreme fiber c = t / 2
        t = getattr(elem.section, "t", None)
        if t is None:
            t = np.sqrt(12.0 * I / A)
        c = t / 2.0

        sig_bend = m_max * c / I
        sig_combined = sig_axial + sig_bend

        if sig_combined > max_stress:
            max_stress = sig_combined
            crit_elem = eid

    utilization = float(max_stress / yield_strength) if yield_strength > 0 else 0.0
    return float(max_stress), float(utilization), crit_elem


def evaluate_design_metrics(
    result: SolverResult,
    domain: DesignDomain2D,
    yield_strength: float = 250e6,
) -> PerformanceMetrics:
    """Evaluate and compile all authoritative structural metrics for a solved design."""
    c = compute_compliance(result)
    k = compute_stiffness(c, reference_load=1.0)
    mass = result.total_mass
    vol = result.total_volume
    vf = float(vol / domain.bounding_volume)
    spec_k = compute_specific_stiffness(c, mass)
    ax_share, bend_share = compute_energy_shares(result)
    max_stress, max_util, _ = compute_member_peak_stress(result, yield_strength)

    return PerformanceMetrics(
        compliance=c,
        stiffness=k,
        mass=mass,
        volume=vol,
        volume_fraction=vf,
        specific_stiffness=spec_k,
        axial_energy_share=ax_share,
        bending_energy_share=bend_share,
        max_normal_stress=max_stress,
        max_utilization=max_util,
    )

