"""Base classes and tracking data structures for structural optimizers."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from ..fem.solver import SolverResult
from ..fem.material import Material, RectangularSection
from ..fem.mesh import FrameMesh2D
from ..geometry.domain import DesignDomain2D, compute_volume_fraction
from ..geometry.ground_structure import GroundStructureData
from ..fem.load_cases import LoadCase


@dataclass
class OptimizationStep:
    """Snapshot of a single optimization iteration or generation.

    Attributes:
        step_index: Iteration or generation index.
        fe_evaluations: Cumulative count of FE solves performed up to this step.
        compliance: Compliance C = F^T @ u (Joules).
        stiffness: Equivalent stiffness K = 1 / C (N/m).
        volume: Structural volume (m^3).
        volume_fraction: Ratio V / V_domain (dimensionless).
        axial_energy_share: Fraction of strain energy carried axially.
        bending_energy_share: Fraction of strain energy carried in bending.
        num_active_members: Number of members with non-zero cross-section.
        active_mask: Boolean array indicating member presence.
        thicknesses: Array of member thickness values (meters).
        max_stress: Maximum combined normal stress (Pa).
    """

    step_index: int
    fe_evaluations: int
    compliance: float
    stiffness: float
    volume: float
    volume_fraction: float
    axial_energy_share: float
    bending_energy_share: float
    num_active_members: int
    active_mask: np.ndarray
    thicknesses: np.ndarray
    max_stress: float = 0.0


@dataclass
class OptimizationHistory:
    """Full optimization trajectory and final result container.

    Attributes:
        optimizer_name: Label ('RemodelingPruning' or 'GeneticAlgorithm').
        load_case_id: Load case identifier (e.g. 'LC1').
        target_volume_fraction: Target volume fraction budget.
        steps: List of OptimizationStep records along the trajectory.
        final_mesh: FrameMesh2D of the final optimized design.
        final_result: SolverResult of the final optimized design.
        total_fe_evaluations: Total number of FE solves invoked during search.
        runtime_seconds: Wall-clock execution time (seconds).
        converged: Boolean flag indicating successful convergence.
    """

    optimizer_name: str
    load_case_id: str
    target_volume_fraction: float
    steps: List[OptimizationStep] = field(default_factory=list)
    final_mesh: Optional[FrameMesh2D] = None
    final_result: Optional[SolverResult] = None
    total_fe_evaluations: int = 0
    runtime_seconds: float = 0.0
    converged: bool = False

    def to_dataframe(self) -> pd.DataFrame:
        """Convert step history into a pandas DataFrame."""
        rows = []
        for s in self.steps:
            rows.append(
                {
                    "step_index": s.step_index,
                    "fe_evaluations": s.fe_evaluations,
                    "compliance_J": s.compliance,
                    "stiffness_N_m": s.stiffness,
                    "volume_m3": s.volume,
                    "volume_fraction": s.volume_fraction,
                    "axial_energy_share": s.axial_energy_share,
                    "bending_energy_share": s.bending_energy_share,
                    "num_active_members": s.num_active_members,
                    "max_stress_Pa": s.max_stress,
                }
            )
        return pd.DataFrame(rows)

    @property
    def final_compliance(self) -> float:
        if self.final_result is not None:
            return float(self.final_result.compliance)
        return self.steps[-1].compliance if self.steps else float("nan")

    @property
    def final_stiffness(self) -> float:
        c = self.final_compliance
        return 1.0 / c if c > 1e-18 else float("nan")


class BaseOptimizer(ABC):
    """Abstract base class for 2D frame structural lattice optimizers."""

    def __init__(
        self,
        ground_structure: GroundStructureData,
        load_case: LoadCase,
        material: Material,
        target_volume_fraction: float,
        seed: int = 42,
    ) -> None:
        self.gs = ground_structure
        self.load_case = load_case
        self.material = material
        self.target_vf = float(target_volume_fraction)
        self.seed = int(seed)
        self.domain = ground_structure.domain

    @abstractmethod
    def optimize(self) -> OptimizationHistory:
        """Execute structural optimization and return full history trajectory."""
        pass

