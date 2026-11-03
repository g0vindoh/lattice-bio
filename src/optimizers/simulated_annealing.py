"""Simulated Annealing metaheuristic optimizer for 2D frame lattice design.

Implements discrete Simulated Annealing (Kirkpatrick et al., 1983) over member section levels:
1. State representation: discrete thickness levels per member, matching the GA search space.
2. Neighborhood move: perturbation of member thickness levels (addition, removal, resizing).
3. Volume constraint: volume repair operator to ensure strict fairness under equal material budget.
4. Metropolis acceptance criterion: P = exp(-Delta E / T) with geometric cooling schedule.
5. Feasibility enforcement: kinematic mechanisms or disconnected paths receive heavy compliance penalty.
"""

from __future__ import annotations
import math
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from .base import BaseOptimizer, OptimizationHistory, OptimizationStep
from ..fem.material import Material, RectangularSection
from ..fem.mesh import FrameMesh2D
from ..geometry.domain import DesignDomain2D, compute_volume_fraction
from ..geometry.ground_structure import GroundStructureData, build_fe_mesh_from_ground_structure
from ..fem.load_cases import LoadCase
from ..metrics.feasibility import check_design_feasibility
from ..metrics.metrics import (
    compute_compliance,
    compute_stiffness,
    compute_member_peak_stress,
)


class SimulatedAnnealingOptimizer(BaseOptimizer):
    """Simulated Annealing optimizer for 2D frame lattices.

    Args:
        ground_structure: GroundStructureData.
        load_case: LoadCase instance.
        material: Material properties.
        target_volume_fraction: Target volume fraction v_f.
        max_evaluations: Maximum FE solve budget (default 900, matching GA).
        initial_temperature: Starting temperature T_0 (default 10.0).
        cooling_rate: Multiplicative cooling factor alpha in (0, 1) (default 0.98).
        mutation_rate: Probability of perturbing any given member during a move (default 0.05).
        discrete_levels: Discrete candidate thickness levels in meters (0.0 is absent).
        seed: Random seed for exact reproducibility.
    """

    def __init__(
        self,
        ground_structure: GroundStructureData,
        load_case: LoadCase,
        material: Material,
        target_volume_fraction: float,
        max_evaluations: int = 900,
        initial_temperature: float = 10.0,
        cooling_rate: float = 0.98,
        mutation_rate: float = 0.05,
        discrete_levels: Optional[List[float]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(
            ground_structure=ground_structure,
            load_case=load_case,
            material=material,
            target_volume_fraction=target_volume_fraction,
            seed=seed,
        )
        self.max_evaluations = int(max_evaluations)
        self.initial_temperature = float(initial_temperature)
        self.cooling_rate = float(cooling_rate)
        self.mutation_rate = float(mutation_rate)

        if discrete_levels is None:
            # Matches GA discrete levels: 0.0 (absent) plus 6 thicknesses
            self.discrete_levels = np.array(
                [0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030], dtype=np.float64
            )
        else:
            self.discrete_levels = np.array(sorted(list(discrete_levels)), dtype=np.float64)

        self.num_levels = len(self.discrete_levels)
        self.rng = np.random.RandomState(self.seed)

    def _repair_volume(self, state: np.ndarray) -> np.ndarray:
        """Adjust candidate thickness levels so that total volume matches target budget."""
        repaired = np.copy(state)
        active_indices = np.where(repaired > 0)[0]

        if len(active_indices) == 0:
            active_indices = self.rng.choice(
                self.gs.num_members,
                size=max(5, self.gs.num_members // 5),
                replace=False,
            )
            repaired[active_indices] = self.rng.randint(1, self.num_levels, size=len(active_indices))

        target_vol = self.target_vf * self.domain.bounding_volume
        b = self.domain.depth

        for _ in range(50):
            thicknesses = self.discrete_levels[repaired]
            curr_vol = float(np.sum(b * thicknesses * self.gs.member_lengths))

            if abs(curr_vol - target_vol) / target_vol < 0.02:
                break

            if curr_vol > target_vol:
                cand = np.where(repaired > 1)[0]
                if len(cand) == 0:
                    cand = np.where(repaired > 0)[0]
                    if len(cand) <= 3:
                        break
                    idx = self.rng.choice(cand)
                    repaired[idx] = 0
                else:
                    idx = self.rng.choice(cand)
                    repaired[idx] -= 1
            else:
                cand = np.where(repaired < self.num_levels - 1)[0]
                if len(cand) == 0:
                    break
                idx = self.rng.choice(cand)
                repaired[idx] += 1

        return repaired

    def _evaluate(
        self,
        state: np.ndarray,
    ) -> Tuple[float, Optional[FrameMesh2D], Optional[object], np.ndarray, float]:
        """Evaluate candidate state using FrameSolver2D.

        Returns:
            compliance, mesh, solver_result, final_thicknesses, achieved_vf.
        """
        thicknesses = self.discrete_levels[state]
        b = self.domain.depth
        active_mask = state > 0

        if not np.any(active_mask):
            return 1e12, None, None, thicknesses, 0.0

        current_vol = float(np.sum(b * thicknesses * self.gs.member_lengths))
        target_vol = self.target_vf * self.domain.bounding_volume

        # Exact continuous rescaling to match target volume fraction precisely
        if current_vol > 1e-12:
            scale = target_vol / current_vol
            thicknesses = thicknesses * scale

        sections = [
            RectangularSection(t=max(1e-9, float(thicknesses[i])), b=float(b))
            for i in range(self.gs.num_members)
        ]

        try:
            mesh = build_fe_mesh_from_ground_structure(
                self.gs,
                sections=sections,
                material=self.material,
                active_mask=active_mask,
            )
            self.load_case.apply(mesh, self.domain, self.material)
            report, res = check_design_feasibility(mesh)

            if not report.is_feasible or res is None:
                return 1e12, None, None, thicknesses, 0.0

            c = float(compute_compliance(res))
            achieved_vf = float(mesh.total_volume / self.domain.bounding_volume)
            return c, mesh, res, thicknesses, achieved_vf

        except Exception:
            return 1e12, None, None, thicknesses, 0.0

    def optimize(self) -> OptimizationHistory:
        """Run Simulated Annealing optimization trajectory."""
        start_time = time.perf_counter()
        history = OptimizationHistory(
            optimizer_name="SimulatedAnnealing",
            load_case_id=self.load_case.name,
            target_volume_fraction=self.target_vf,
        )

        # 1. Initialize random state
        state = self.rng.randint(0, self.num_levels, size=self.gs.num_members)
        state = self._repair_volume(state)

        current_compliance, current_mesh, current_res, current_th, current_vf = self._evaluate(state)
        fe_evals = 1

        # Keep trying until initial state is feasible
        init_tries = 0
        while current_compliance >= 1e11 and init_tries < 50:
            init_tries += 1
            state = self.rng.randint(0, self.num_levels, size=self.gs.num_members)
            state = self._repair_volume(state)
            current_compliance, current_mesh, current_res, current_th, current_vf = self._evaluate(state)
            fe_evals += 1

        best_state = np.copy(state)
        best_compliance = current_compliance
        best_mesh = current_mesh
        best_res = current_res
        best_th = current_th
        best_vf = current_vf

        temperature = self.initial_temperature
        step_idx = 0

        # Record initial step
        if current_res is not None:
            active_m = current_th > 1e-8
            history.steps.append(
                OptimizationStep(
                    step_index=0,
                    fe_evaluations=fe_evals,
                    compliance=current_compliance,
                    stiffness=1.0 / max(1e-12, current_compliance),
                    volume=float(current_vf * self.domain.bounding_volume),
                    volume_fraction=current_vf,
                    axial_energy_share=float(current_res.axial_energy_fraction),
                    bending_energy_share=float(current_res.bending_energy_fraction),
                    num_active_members=int(np.sum(active_m)),
                    active_mask=active_m,
                    thicknesses=current_th,
                )
            )

        # Annealing loop
        while fe_evals < self.max_evaluations:
            step_idx += 1

            # Propose neighbor by mutating discrete thickness levels
            candidate_state = np.copy(state)
            mut_mask = self.rng.rand(self.gs.num_members) < self.mutation_rate
            if not np.any(mut_mask):
                mut_mask[self.rng.randint(0, self.gs.num_members)] = True

            for idx in np.where(mut_mask)[0]:
                delta = self.rng.choice([-1, 1])
                candidate_state[idx] = int(np.clip(candidate_state[idx] + delta, 0, self.num_levels - 1))

            candidate_state = self._repair_volume(candidate_state)
            cand_c, cand_mesh, cand_res, cand_th, cand_vf = self._evaluate(candidate_state)
            fe_evals += 1

            delta_e = cand_c - current_compliance

            # Metropolis acceptance criterion
            accept = False
            if delta_e <= 0.0:
                accept = True
            elif temperature > 1e-8:
                # Scaled Metropolis factor to prevent numerical overflow
                exponent = -delta_e / max(1e-12, temperature * max(1e-6, current_compliance))
                if exponent > -50.0:
                    prob = math.exp(exponent)
                    if self.rng.rand() < prob:
                        accept = True

            if accept:
                state = candidate_state
                current_compliance = cand_c
                current_mesh = cand_mesh
                current_res = cand_res
                current_th = cand_th
                current_vf = cand_vf

                if current_compliance < best_compliance:
                    best_compliance = current_compliance
                    best_state = np.copy(state)
                    best_mesh = current_mesh
                    best_res = current_res
                    best_th = current_th
                    best_vf = current_vf

            # Geometric cooling
            temperature *= self.cooling_rate

            # Record progress periodically (every 20 evals or on new best)
            if fe_evals % 20 == 0 or current_compliance == best_compliance:
                if best_res is not None:
                    active_m = best_th > 1e-8
                    history.steps.append(
                        OptimizationStep(
                            step_index=step_idx,
                            fe_evaluations=fe_evals,
                            compliance=best_compliance,
                            stiffness=1.0 / max(1e-12, best_compliance),
                            volume=float(best_vf * self.domain.bounding_volume),
                            volume_fraction=best_vf,
                            axial_energy_share=float(best_res.axial_energy_fraction),
                            bending_energy_share=float(best_res.bending_energy_fraction),
                            num_active_members=int(np.sum(active_m)),
                            active_mask=active_m,
                            thicknesses=best_th,
                        )
                    )

        runtime = time.perf_counter() - start_time
        history.final_mesh = best_mesh
        history.final_result = best_res
        history.total_fe_evaluations = fe_evals
        history.runtime_seconds = runtime
        history.converged = best_compliance < 1e11

        return history
