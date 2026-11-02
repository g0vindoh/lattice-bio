"""Method A: Remodeling-inspired structural pruning optimizer for 2D lattices.

Bio-inspired structural optimization simulating cellular trabecular bone remodeling:
1. Starts with an over-connected ground structure.
2. Underloaded members (lowest strain energy density or stress) are progressively resorbed.
3. Vital load paths are protected from accidental removal via non-destructive connectivity checks.
4. Optionally redistributes resorbed volume to heavily loaded members (Wolff's law).
5. Halts when the target volume budget is achieved, scaling exactly to the volume fraction.
"""

from __future__ import annotations
import time
from typing import Dict, List, Optional, Set, Tuple
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


class RemodelingPruningOptimizer(BaseOptimizer):
    """Remodeling-inspired structural pruning optimizer.

    Hyperparameters:
        removal_rate: Fraction of active members targeted for removal each iteration (default 0.05).
        utilization_metric: 'strain_energy_density', 'strain_energy', 'axial_stress', or 'peak_stress'.
        redistribute_area: If True, redistributes a portion of resorbed volume to top loaded struts.
        initial_thickness: Initial thickness of candidate ground structure members (m, default 0.025).
        max_iterations: Maximum number of pruning iterations (default 100).
    """

    def __init__(
        self,
        ground_structure: GroundStructureData,
        load_case: LoadCase,
        material: Material,
        target_volume_fraction: float,
        removal_rate: float = 0.05,
        utilization_metric: str = "strain_energy_density",
        redistribute_area: bool = False,
        initial_thickness: float = 0.025,
        max_iterations: int = 100,
        seed: int = 42,
    ) -> None:
        super().__init__(
            ground_structure=ground_structure,
            load_case=load_case,
            material=material,
            target_volume_fraction=target_volume_fraction,
            seed=seed,
        )
        self.removal_rate = float(removal_rate)
        self.utilization_metric = str(utilization_metric).lower()
        self.redistribute_area = bool(redistribute_area)
        self.initial_thickness = float(initial_thickness)
        self.max_iterations = int(max_iterations)

        valid_metrics = ("strain_energy_density", "strain_energy", "axial_stress", "peak_stress")
        if self.utilization_metric not in valid_metrics:
            raise ValueError(
                f"Unknown utilization metric '{self.utilization_metric}'. "
                f"Must be one of {valid_metrics}."
            )

    def _compute_utilization(
        self,
        res: object,
        active_indices: List[int],
        thicknesses: np.ndarray,
    ) -> np.ndarray:
        """Calculate member utilization score phi_i for active members."""
        scores = np.zeros(len(active_indices), dtype=np.float64)

        for i, idx in enumerate(active_indices):
            eid = idx + 1
            forces = res.internal_forces[eid]
            t = thicknesses[idx]
            L = self.gs.member_lengths[idx]
            b = self.domain.depth
            vol = b * t * L

            if self.utilization_metric == "strain_energy_density":
                # Strain energy per unit volume: U_total / V
                scores[i] = forces.strain_energy_total / max(1e-12, vol)
            elif self.utilization_metric == "strain_energy":
                # Total strain energy
                scores[i] = forces.strain_energy_total
            elif self.utilization_metric == "axial_stress":
                # Axial normal stress: |N| / A
                scores[i] = abs(forces.axial_force) / max(1e-12, b * t)
            elif self.utilization_metric == "peak_stress":
                # Combined outer fiber stress: |N| / A + 6 * |M_max| / (b * t^2)
                m_max = max(abs(forces.bending_moment_1), abs(forces.bending_moment_2))
                sig_ax = abs(forces.axial_force) / max(1e-12, b * t)
                sig_bend = 6.0 * m_max / max(1e-12, b * (t**2))
                scores[i] = sig_ax + sig_bend

        return scores

    def optimize(self) -> OptimizationHistory:
        """Run remodeling-inspired iterative pruning."""
        start_time = time.time()
        num_members = self.gs.num_members

        # Compute initial thickness if needed to start from an over-connected, over-budget state
        V_target = self.target_vf * self.domain.bounding_volume
        L_total = self.gs.total_length
        b = self.domain.depth

        # Ensure initial structure starts over budget (e.g. at least 1.5x target budget)
        t_over_budget = 1.5 * V_target / max(1e-12, b * L_total)
        t_start = max(self.initial_thickness, t_over_budget)

        active_mask = np.ones(num_members, dtype=bool)
        thicknesses = np.full(num_members, t_start, dtype=np.float64)

        history = OptimizationHistory(
            optimizer_name="RemodelingPruning",
            load_case_id=self.load_case.id,
            target_volume_fraction=self.target_vf,
        )

        fe_evaluations = 0
        V_target = self.target_vf * self.domain.bounding_volume

        # Measure pad volume
        dummy_sec = RectangularSection(t=self.initial_thickness, b=self.domain.depth)
        trial_mesh = build_fe_mesh_from_ground_structure(
            self.gs, [dummy_sec] * num_members, self.material
        )
        _, _, pad_vol = self.load_case.apply(trial_mesh, self.domain, self.material)

        final_mesh = None
        final_result = None
        last_feasible_mask = active_mask.copy()
        last_feasible_thicknesses = thicknesses.copy()
        last_feasible_mesh = None
        last_feasible_result = None

        for iteration in range(self.max_iterations):
            # 1. Build FE mesh from current state
            sections = [
                RectangularSection(t=max(1e-9, thicknesses[i]), b=self.domain.depth)
                for i in range(num_members)
            ]
            mesh = build_fe_mesh_from_ground_structure(
                self.gs, sections, self.material, active_mask=active_mask
            )
            self.load_case.apply(mesh, self.domain, self.material)

            # 2. Feasibility and solve
            fe_evaluations += 1
            report, res = check_design_feasibility(mesh)

            if not report.is_feasible or res is None:
                # If infeasible, roll back to last feasible state
                active_mask = last_feasible_mask
                thicknesses = last_feasible_thicknesses
                final_mesh = last_feasible_mesh
                final_result = last_feasible_result
                break

            last_feasible_mask = active_mask.copy()
            last_feasible_thicknesses = thicknesses.copy()
            last_feasible_mesh = mesh
            last_feasible_result = res
            final_mesh = mesh
            final_result = res

            c = compute_compliance(res)
            k = compute_stiffness(c)
            curr_vol = mesh.total_volume
            curr_vf = curr_vol / self.domain.bounding_volume
            ax_share = res.axial_energy_fraction
            bend_share = res.bending_energy_fraction
            max_stress, _, _ = compute_member_peak_stress(res)

            step = OptimizationStep(
                step_index=iteration,
                fe_evaluations=fe_evaluations,
                compliance=c,
                stiffness=k,
                volume=curr_vol,
                volume_fraction=curr_vf,
                axial_energy_share=ax_share,
                bending_energy_share=bend_share,
                num_active_members=int(np.sum(active_mask)),
                active_mask=active_mask.copy(),
                thicknesses=thicknesses.copy(),
                max_stress=max_stress,
            )
            history.steps.append(step)

            # 3. Check termination: volume budget reached
            if curr_vol <= V_target * 1.001:
                # Scale thicknesses slightly to match volume budget exactly
                active_indices = np.where(active_mask)[0]
                active_lens = self.gs.member_lengths[active_indices]
                curr_active_vol = np.sum(thicknesses[active_indices] * active_lens) * self.domain.depth
                scale = max(0.1, (V_target - pad_vol) / max(1e-12, curr_active_vol))
                thicknesses[active_indices] *= scale
                history.converged = True
                break

            # 4. Prune least loaded members
            active_indices = np.where(active_mask)[0]
            scores = self._compute_utilization(res, active_indices, thicknesses)

            # Sort active members by utilization ascending
            sort_order = np.argsort(scores)
            sorted_active = active_indices[sort_order]

            # Number of members targeted for removal
            n_target_remove = max(1, int(np.ceil(self.removal_rate * len(active_indices))))

            # Tentatively prune while preserving load paths
            # Build current adjacency
            adj: Dict[int, Set[int]] = {nid: set() for nid in self.gs.nodes}
            for idx in active_indices:
                u, v = self.gs.members[idx]
                adj[u].add(v)
                adj[v].add(u)

            support_nodes = {bc.node_id for bc in mesh.boundary_conditions}
            loaded_nodes = {ld.node_id for ld in mesh.point_loads if abs(ld.fx) > 1e-6 or abs(ld.fy) > 1e-6}

            n_removed = 0
            pruned_vol = 0.0

            for candidate_idx in sorted_active:
                if n_removed >= n_target_remove:
                    break

                u_c, v_c = self.gs.members[candidate_idx]

                # Check if removing this member disconnects any loaded node
                adj[u_c].remove(v_c)
                adj[v_c].remove(u_c)

                # Breadth-first reachability from support nodes
                visited: Set[int] = set(support_nodes)
                queue = list(support_nodes)
                while queue:
                    curr = queue.pop(0)
                    for nbr in adj.get(curr, set()):
                        if nbr not in visited:
                            visited.add(nbr)
                            queue.append(nbr)

                if loaded_nodes.issubset(visited):
                    # Removal is safe: commit removal
                    active_mask[candidate_idx] = False
                    pruned_vol += (
                        thicknesses[candidate_idx]
                        * self.gs.member_lengths[candidate_idx]
                        * self.domain.depth
                    )
                    n_removed += 1
                else:
                    # Roll back removal
                    adj[u_c].add(v_c)
                    adj[v_c].add(u_c)

            # 5. Optional area redistribution (Wolff's law)
            if self.redistribute_area and n_removed > 0 and pruned_vol > 0.0:
                # Distribute 50% of resorbed volume to top 15% most utilized active members
                remaining_active = np.where(active_mask)[0]
                if len(remaining_active) > 0:
                    top_n = max(1, int(0.15 * len(remaining_active)))
                    # Find top utilized
                    remaining_scores = self._compute_utilization(
                        res, remaining_active, thicknesses
                    )
                    top_indices = remaining_active[np.argsort(remaining_scores)[-top_n:]]
                    top_lens = self.gs.member_lengths[top_indices]
                    bonus_t = (0.5 * pruned_vol) / (self.domain.depth * np.sum(top_lens))
                    thicknesses[top_indices] += bonus_t

            # If no member could be removed without breaking connectivity, scale thicknesses down
            if n_removed == 0:
                thicknesses[active_mask] *= 0.95

        # Final evaluation at converged state
        sections = [
            RectangularSection(t=max(1e-9, thicknesses[i]), b=self.domain.depth)
            for i in range(num_members)
        ]
        final_mesh = build_fe_mesh_from_ground_structure(
            self.gs, sections, self.material, active_mask=active_mask
        )
        self.load_case.apply(final_mesh, self.domain, self.material)
        fe_evaluations += 1
        _, final_res = check_design_feasibility(final_mesh)

        if final_res is not None:
            final_result = final_res
            c = compute_compliance(final_result)
            k = compute_stiffness(c)
            curr_vol = final_mesh.total_volume
            curr_vf = curr_vol / self.domain.bounding_volume
            ax_share = final_result.axial_energy_fraction
            bend_share = final_result.bending_energy_fraction
            max_stress, _, _ = compute_member_peak_stress(final_result)

            step = OptimizationStep(
                step_index=len(history.steps),
                fe_evaluations=fe_evaluations,
                compliance=c,
                stiffness=k,
                volume=curr_vol,
                volume_fraction=curr_vf,
                axial_energy_share=ax_share,
                bending_energy_share=bend_share,
                num_active_members=int(np.sum(active_mask)),
                active_mask=active_mask.copy(),
                thicknesses=thicknesses.copy(),
                max_stress=max_stress,
            )
            history.steps.append(step)

        history.final_mesh = final_mesh
        history.final_result = final_result
        history.total_fe_evaluations = fe_evaluations
        history.runtime_seconds = time.time() - start_time

        return history
