"""Euler strut buckling analysis for frame lattice structures.

Evaluates member-level Euler critical compressive loads and global lattice
buckling load factors (BLF) to identify premature elastic instability
prior to material yielding:
    P_cr = (pi^2 * E * I) / (k * L)^2
    lambda_e = P_cr / |N_e|  (for N_e < 0, compression)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..fem.mesh import FrameMesh2D
from ..fem.solver import SolverResult


@dataclass
class BucklingAnalysisResult:
    """Summary of lattice compressive buckling stability.

    Attributes:
        global_buckling_factor: Minimum buckling load factor lambda across all members.
                                lambda > 1.0 indicates elastic compressive stability under load.
                                lambda < 1.0 indicates premature elastic buckling occurs.
        critical_element_id: ID of the element governing the global buckling factor.
        critical_member_length: Length of the critical buckling member (m).
        critical_member_thickness: Thickness of the critical buckling member (m).
        critical_member_slenderness: Slenderness ratio L / t of the critical member.
        member_buckling_factors: Mapping of element_id -> buckling load factor lambda.
        num_compression_members: Total count of members carrying compressive axial force.
        num_buckling_critical: Count of members with lambda < 1.0.
        is_buckling_critical: True if global_buckling_factor < 1.0.
    """

    global_buckling_factor: float
    critical_element_id: Optional[int]
    critical_member_length: float
    critical_member_thickness: float
    critical_member_slenderness: float
    member_buckling_factors: Dict[int, float]
    num_compression_members: int
    num_buckling_critical: int
    is_buckling_critical: bool


def compute_euler_buckling_load(
    E: float,
    I: float,
    L: float,
    effective_length_factor: float = 1.0,
) -> float:
    """Compute the Euler critical buckling load for a prismatic column.

    Formula:
        P_cr = (pi^2 * E * I) / (k * L)^2

    Args:
        E: Young's modulus (Pa).
        I: Second moment of area (m^4).
        L: Unbraced member length (m).
        effective_length_factor: Effective length factor k (default 1.0 for pinned-pinned,
                                 0.7 for fixed-pinned, 0.5 for fixed-fixed).

    Returns:
        Critical compressive buckling load P_cr (N).
    """
    if L <= 0.0:
        raise ValueError(f"Member length must be positive, got {L}.")
    if effective_length_factor <= 0.0:
        raise ValueError(f"Effective length factor must be positive, got {effective_length_factor}.")

    L_eff = effective_length_factor * L
    return float((np.pi ** 2 * E * I) / (L_eff ** 2))


def evaluate_lattice_buckling(
    result: SolverResult,
    effective_length_factor: float = 1.0,
    compression_threshold: float = 1e-4,
) -> BucklingAnalysisResult:
    """Evaluate member-level and global Euler buckling load factors for a solved lattice.

    Args:
        result: SolverResult from FrameFESolver.solve().
        effective_length_factor: Column end restraint factor k (default 1.0).
        compression_threshold: Minimum compressive axial force magnitude (N) to evaluate.

    Returns:
        BucklingAnalysisResult containing global and per-member buckling factors.
    """
    mesh = result.mesh
    member_blf: Dict[int, float] = {}

    min_lambda = float("inf")
    crit_eid: Optional[int] = None
    crit_L = 0.0
    crit_t = 0.0
    crit_slender = 0.0
    num_comp = 0
    num_crit = 0

    for eid, elem in mesh.elements.items():
        forces = result.internal_forces[eid]
        # In our frame element sign convention, negative axial force indicates compression
        axial_N = forces.axial_force

        if axial_N < -compression_threshold:
            num_comp += 1
            P_comp = abs(axial_N)
            E = elem.material.E
            I = elem.section.I
            L = elem.length
            t = elem.section.t

            P_cr = compute_euler_buckling_load(E, I, L, effective_length_factor)
            blf = P_cr / P_comp
            member_blf[eid] = float(blf)

            if blf < 1.0:
                num_crit += 1

            if blf < min_lambda:
                min_lambda = blf
                crit_eid = eid
                crit_L = L
                crit_t = t
                crit_slender = L / t if t > 0.0 else float("inf")
        else:
            member_blf[eid] = float("inf")

    # If no members in compression (e.g., pure tensile loading), set safe BLF
    if num_comp == 0:
        min_lambda = float("inf")

    return BucklingAnalysisResult(
        global_buckling_factor=min_lambda,
        critical_element_id=crit_eid,
        critical_member_length=crit_L,
        critical_member_thickness=crit_t,
        critical_member_slenderness=crit_slender,
        member_buckling_factors=member_blf,
        num_compression_members=num_comp,
        num_buckling_critical=num_crit,
        is_buckling_critical=(min_lambda < 1.0),
    )
