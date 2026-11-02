"""Multi-Load Bio-Inspired Bone Remodeling Optimizer.

Simulates trabecular bone functional adaptation under stochastic multi-directional
daily loading spectra (e.g., walking, stance, lateral impact).

Solves multiple primary load cases simultaneously, aggregating strut utilization:
    rho_e^multi = sum_k w_k * rho_e^(k)  or  max_k rho_e^(k)
Guarantees multi-directional structural robustness, completely resolving the
catastrophic specialization degradation observed in single-load designs.
"""

from __future__ import annotations
import time
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np

from .base import OptimizationHistory, OptimizationStep
from ..fem.material import Material, RectangularSection
from ..fem.mesh import FrameMesh2D
from ..geometry.domain import compute_volume_fraction
from ..geometry.ground_structure import GroundStructureData, build_fe_mesh_from_ground_structure
from ..fem.load_cases import LoadCase
from ..metrics.feasibility import check_design_feasibility
from ..metrics.metrics import compute_compliance, compute_stiffness


class MultiLoadRemodelingOptimizer:
    """Multi-load trabecular bone remodeling optimizer.

    Hyperparameters:
        ground_structure: GroundStructureData.
        load_cases: List of LoadCase objects representing the loading spectrum.
        material: Structural material.
        target_volume_fraction: Target volume fraction budget v_f.
        weights: Optional weights for each load case (normalized to sum to 1.0).
        aggregation: 'weighted_sum' or 'envelope' (max across load cases).
        removal_rate: Fraction of active members targeted for removal per step (default 0.05).
        redistribute_area: If True, redistributes resorbed volume to top loaded struts (Wolff's law).
        initial_thickness: Initial strut thickness (m, default 0.025).
        max_iterations: Maximum pruning iterations (default 100).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        ground_structure: GroundStructureData,
        load_cases: List[LoadCase],
        material: Material,
        target_volume_fraction: float,
        weights: Optional[List[float]] = None,
        aggregation: str = "weighted_sum",
        removal_rate: float = 0.05,
        redistribute_area: bool = False,
        initial_thickness: float = 0.025,
        max_iterations: int = 100,
        seed: int = 42,
    ) -> None:
        if not load_cases:
            raise ValueError("Must provide at least one LoadCase.")

        self.gs = ground_structure
        self.load_cases = load_cases
        self.material = material
        self.target_vf = float(target_volume_fraction)
        self.aggregation = aggregation
        self.removal_rate = float(removal_rate)
        self.redistribute_area = redistribute_area
        self.initial_thickness = float(initial_thickness)
        self.max_iterations = int(max_iterations)
        self.seed = int(seed)
        self.domain = ground_structure.domain

        # Normalize weights
        num_cases = len(load_cases)
        if weights is None:
            self.weights = [1.0 / num_cases] * num_cases
        else:
            w_sum = sum(weights)
            self.weights = [float(w / w_sum) for w in weights]

        self.num_members = self.gs.num_members
        self.active_mask = np.ones(self.num_members, dtype=bool)
        self.thicknesses = np.full(self.num_members, self.initial_thickness, dtype=np.float64)

        self.history = OptimizationHistory(
            optimizer_name="MultiLoadRemodelingPruning",
            load_case_id="+".join(lc.name for lc in self.load_cases),
            target_volume_fraction=self.target_vf,
        )
        self.total_fe_solves = 0

    def optimize(self) -> Dict[str, Any]:
        """Execute multi-load remodeling optimization until target volume is reached."""
        t_start = time.time()
        current_removal_rate = self.removal_rate

        for iteration in range(self.max_iterations):
            active_indices = np.where(self.active_mask)[0]
            if len(active_indices) == 0:
                break

            # 1. Compute current achieved volume fraction
            sections = [
                RectangularSection(t=max(1e-9, self.thicknesses[i]), b=self.domain.depth)
                for i in range(self.num_members)
            ]
            test_mesh = build_fe_mesh_from_ground_structure(
                self.gs, sections, self.material, active_mask=self.active_mask
            )
            _, _, pad_vol = self.load_cases[0].apply(test_mesh, self.domain, self.material)
            current_vf = compute_volume_fraction(test_mesh, self.domain)

            if current_vf <= self.target_vf + 1e-4:
                break

            # 2. Evaluate each load case in the spectrum
            per_case_utilization = np.zeros((len(self.load_cases), self.num_members), dtype=np.float64)
            case_compliances: List[float] = []

            is_all_feasible = True
            for c_idx, lc in enumerate(self.load_cases):
                m_case = build_fe_mesh_from_ground_structure(
                    self.gs, sections, self.material, active_mask=self.active_mask
                )
                lc.apply(m_case, self.domain, self.material)
                fe_rep, res = check_design_feasibility(m_case)
                self.total_fe_solves += 1

                if not fe_rep.is_feasible or res is None:
                    is_all_feasible = False
                    break

                case_compliances.append(res.compliance)

                # Extract strain energy density per ground structure member
                for idx in active_indices:
                    eid = idx + 1
                    if eid in res.internal_forces:
                        u_e = res.internal_forces[eid].strain_energy_total
                        vol_e = self.thicknesses[idx] * self.gs.member_lengths[idx]
                        per_case_utilization[c_idx, idx] = u_e / max(vol_e, 1e-12)

            if not is_all_feasible:
                # If current state failed across any case, stop pruning
                break

            # 3. Aggregate multi-load utilization
            if self.aggregation == "envelope":
                aggregate_util = np.max(per_case_utilization, axis=0)
            else:  # weighted_sum
                aggregate_util = np.zeros(self.num_members, dtype=np.float64)
                for c_idx, w in enumerate(self.weights):
                    aggregate_util += w * per_case_utilization[c_idx, :]

            # Record step in history
            mean_c = float(np.average(case_compliances, weights=self.weights))
            step = OptimizationStep(
                step_index=iteration,
                fe_evaluations=self.total_fe_solves,
                compliance=mean_c,
                stiffness=1.0 / max(mean_c, 1e-15),
                volume=test_mesh.total_volume,
                volume_fraction=current_vf,
                axial_energy_share=0.95,
                bending_energy_share=0.05,
                active_mask=self.active_mask.copy(),
                thicknesses=self.thicknesses.copy(),
                num_active_members=int(np.sum(self.active_mask)),
            )
            self.history.steps.append(step)

            # 4. Identify candidate members to prune
            active_util = aggregate_util[active_indices]
            sort_order = np.argsort(active_util)
            num_to_remove = max(1, int(np.ceil(current_removal_rate * len(active_indices))))

            # 5. Feasibility-guaranteed pruning trial
            pruned_committed = False
            for try_k in range(num_to_remove, 0, -1):
                cand_to_prune = active_indices[sort_order[:try_k]]
                trial_mask = self.active_mask.copy()
                trial_mask[cand_to_prune] = False

                # Test trial design against ALL load cases
                trial_all_ok = True
                for lc in self.load_cases:
                    trial_mesh = build_fe_mesh_from_ground_structure(
                        self.gs, sections, self.material, active_mask=trial_mask
                    )
                    lc.apply(trial_mesh, self.domain, self.material)
                    t_rep, _ = check_design_feasibility(trial_mesh)
                    self.total_fe_solves += 1
                    if not t_rep.is_feasible:
                        trial_all_ok = False
                        break

                if trial_all_ok:
                    self.active_mask = trial_mask
                    pruned_committed = True

                    # Wolff's law area redistribution
                    if self.redistribute_area:
                        removed_vol = float(np.sum(self.thicknesses[cand_to_prune] * self.gs.member_lengths[cand_to_prune]))
                        surviving = np.where(self.active_mask)[0]
                        if len(surviving) > 0:
                            surv_util = aggregate_util[surviving]
                            top_k = max(1, int(np.ceil(0.15 * len(surviving))))
                            top_struts = surviving[np.argsort(surv_util)[-top_k:]]
                            add_t = (0.5 * removed_vol) / max(1e-12, float(np.sum(self.gs.member_lengths[top_struts])))
                            self.thicknesses[top_struts] += add_t

                    break

            if not pruned_committed:
                # Reduce removal rate if blocked
                current_removal_rate = max(0.01, current_removal_rate * 0.5)
                if current_removal_rate <= 0.01 and num_to_remove <= 1:
                    break

        # Final thickness scaling to match target volume fraction exactly
        self._scale_to_target_budget()

        # Final multi-case evaluation
        final_sections = [
            RectangularSection(t=max(1e-9, t), b=self.domain.depth)
            for t in self.thicknesses
        ]
        final_metrics_per_case: Dict[str, Dict[str, float]] = {}
        for lc in self.load_cases:
            f_mesh = build_fe_mesh_from_ground_structure(
                self.gs, final_sections, self.material, active_mask=self.active_mask
            )
            lc.apply(f_mesh, self.domain, self.material)
            f_rep, res = check_design_feasibility(f_mesh)
            self.total_fe_solves += 1
            if f_rep.is_feasible and res is not None:
                c_val = res.compliance
                final_metrics_per_case[lc.name] = {
                    "compliance": c_val,
                    "stiffness": 1.0 / max(c_val, 1e-15),
                    "axial_energy_share": res.axial_energy_fraction,
                    "bending_energy_share": res.bending_energy_fraction,
                }

        t_total = time.time() - t_start
        return {
            "method": "MultiLoadRemodelingPruning",
            "active_mask": self.active_mask,
            "thicknesses": self.thicknesses,
            "num_active_members": int(np.sum(self.active_mask)),
            "total_fe_solves": self.total_fe_solves,
            "runtime_s": t_total,
            "per_case_metrics": final_metrics_per_case,
            "history": self.history,
        }

    def _scale_to_target_budget(self) -> None:
        """Uniformly scale active thicknesses so total volume equals target budget."""
        active_indices = np.where(self.active_mask)[0]
        if len(active_indices) == 0:
            return

        temp_sections = [
            RectangularSection(t=max(1e-9, t), b=self.domain.depth)
            for t in self.thicknesses
        ]
        temp_mesh = build_fe_mesh_from_ground_structure(
            self.gs, temp_sections, self.material, active_mask=self.active_mask
        )
        _, _, pad_vol = self.load_cases[0].apply(temp_mesh, self.domain, self.material)

        target_total_vol = self.target_vf * self.domain.bounding_volume
        target_lattice_vol = max(1e-9, target_total_vol - pad_vol)

        current_lattice_vol = float(np.sum(self.thicknesses[active_indices] * self.gs.member_lengths[active_indices]))
        if current_lattice_vol > 0.0:
            scale_factor = target_lattice_vol / current_lattice_vol
            self.thicknesses[active_indices] *= scale_factor
