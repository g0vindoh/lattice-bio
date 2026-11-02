"""Method B: Genetic Algorithm optimizer over discrete member section levels.

Evolutionary structural optimization:
1. Chromosome: integer vector g of length M, representing discrete cross-section levels (0 = absent).
2. Volume Budget: enforced via a repair operator that downsizes/upsizes members to match the target volume fraction.
3. Feasibility Constraint: penalizes disconnected load paths or kinematic mechanisms with a large compliance penalty.
4. Evolutionary Operators: tournament selection, uniform crossover, creep/reset mutation, and elitism.
"""

from __future__ import annotations
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


class GeneticAlgorithmOptimizer(BaseOptimizer):
    """Genetic algorithm optimizer for 2D frame lattice design.

    Hyperparameters:
        pop_size: Population size (default 30).
        generations: Number of evolutionary generations (default 40).
        crossover_rate: Probability of crossover between parent pairs (default 0.8).
        mutation_rate: Per-gene mutation probability (default 0.02).
        tournament_size: Number of competitors in tournament selection (default 3).
        elite_count: Number of top-performing individuals preserved intact (default 2).
        discrete_levels: List of discrete thicknesses in meters (index 0 is absence: 0.0).
    """

    def __init__(
        self,
        ground_structure: GroundStructureData,
        load_case: LoadCase,
        material: Material,
        target_volume_fraction: float,
        pop_size: int = 30,
        generations: int = 40,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.02,
        tournament_size: int = 3,
        elite_count: int = 2,
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
        self.pop_size = int(pop_size)
        self.generations = int(generations)
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.tournament_size = int(tournament_size)
        self.elite_count = int(elite_count)

        if discrete_levels is None:
            # Default discrete section levels (m)
            self.discrete_levels = np.array(
                [0.0, 0.004, 0.008, 0.012, 0.016, 0.020, 0.025, 0.030],
                dtype=np.float64,
            )
        else:
            self.discrete_levels = np.array(sorted(discrete_levels), dtype=np.float64)

        self.num_levels = len(self.discrete_levels)
        self.rng = np.random.RandomState(self.seed)

    def _repair_volume(
        self, chromosome: np.ndarray, target_lattice_volume: float
    ) -> np.ndarray:
        """Repair operator: adjusts chromosome gene levels to satisfy the volume budget."""
        repaired = chromosome.copy()
        lengths = self.gs.member_lengths
        b = self.domain.depth

        # Current lattice volume
        curr_t = self.discrete_levels[repaired]
        curr_vol = float(np.sum(curr_t * lengths) * b)

        # 1. If over budget: downsize active members
        max_repair_iters = 1000
        it = 0
        while curr_vol > target_lattice_volume and it < max_repair_iters:
            active_idx = np.where(repaired > 0)[0]
            if len(active_idx) == 0:
                break
            # Randomly select active members to decrease level by 1
            chosen = self.rng.choice(active_idx)
            repaired[chosen] -= 1
            curr_t = self.discrete_levels[repaired]
            curr_vol = float(np.sum(curr_t * lengths) * b)
            it += 1

        # 2. If under budget: randomly promote active members up to limit
        it = 0
        while curr_vol < target_lattice_volume * 0.98 and it < max_repair_iters:
            promotable = np.where(repaired < self.num_levels - 1)[0]
            if len(promotable) == 0:
                break
            chosen = self.rng.choice(promotable)
            repaired[chosen] += 1
            new_t = self.discrete_levels[repaired]
            new_vol = float(np.sum(new_t * lengths) * b)
            if new_vol <= target_lattice_volume:
                curr_vol = new_vol
            else:
                repaired[chosen] -= 1
                break
            it += 1

        return repaired

    def _evaluate_individual(
        self, chromosome: np.ndarray
    ) -> Tuple[float, Optional[object], Optional[object]]:
        """Evaluate fitness of a chromosome. Returns (fitness, report, res)."""
        num_members = self.gs.num_members
        thicknesses = self.discrete_levels[chromosome]
        active_mask = chromosome > 0

        sections = [
            RectangularSection(t=max(1e-9, thicknesses[i]), b=self.domain.depth)
            for i in range(num_members)
        ]
        mesh = build_fe_mesh_from_ground_structure(
            self.gs, sections, self.material, active_mask=active_mask
        )
        self.load_case.apply(mesh, self.domain, self.material)

        report, res = check_design_feasibility(mesh)

        if report.is_feasible and res is not None:
            c = compute_compliance(res)
            fitness = 1.0 / max(1e-12, c)
            return fitness, report, res
        else:
            # Heavy penalty for infeasible designs
            penalty_c = 1e8 + len(report.unreachable_load_nodes) * 1e5
            if report.has_mechanism:
                penalty_c += 1e7
            fitness = 1.0 / penalty_c
            return fitness, report, None

    def optimize(self) -> OptimizationHistory:
        """Run genetic algorithm optimization."""
        start_time = time.time()
        num_members = self.gs.num_members

        # Calculate target volume budget
        V_target = self.target_vf * self.domain.bounding_volume
        dummy_sec = RectangularSection(t=0.01, b=self.domain.depth)
        trial_mesh = build_fe_mesh_from_ground_structure(
            self.gs, [dummy_sec] * num_members, self.material
        )
        _, _, pad_vol = self.load_case.apply(trial_mesh, self.domain, self.material)
        V_lattice_target = max(1e-12, V_target - pad_vol)

        # 1. Initialize population
        population: List[np.ndarray] = []
        for _ in range(self.pop_size):
            # Random initial gene levels biased toward active members
            chrom = self.rng.randint(0, self.num_levels, size=num_members)
            repaired_chrom = self._repair_volume(chrom, V_lattice_target)
            population.append(repaired_chrom)

        history = OptimizationHistory(
            optimizer_name="GeneticAlgorithm",
            load_case_id=self.load_case.id,
            target_volume_fraction=self.target_vf,
        )

        fe_evaluations = 0
        best_overall_fitness = -1.0
        best_overall_chrom = None
        best_overall_res = None
        best_overall_mesh = None

        for gen in range(self.generations):
            # Evaluate all individuals
            fitnesses = np.zeros(self.pop_size, dtype=np.float64)
            step_results = []

            for i, ind in enumerate(population):
                fit, report, res = self._evaluate_individual(ind)
                fe_evaluations += 1
                fitnesses[i] = fit
                step_results.append((ind, fit, report, res))

            # Rank population
            rank_indices = np.argsort(fitnesses)[::-1]  # Descending
            gen_best_idx = rank_indices[0]
            gen_best_ind, gen_best_fit, gen_best_rep, gen_best_res = step_results[gen_best_idx]

            if gen_best_fit > best_overall_fitness and gen_best_res is not None:
                best_overall_fitness = gen_best_fit
                best_overall_chrom = gen_best_ind.copy()
                best_overall_res = gen_best_res

                # Reconstruct best mesh
                t_best = self.discrete_levels[gen_best_ind]
                sec_best = [
                    RectangularSection(t=max(1e-9, t_best[j]), b=self.domain.depth)
                    for j in range(num_members)
                ]
                best_overall_mesh = build_fe_mesh_from_ground_structure(
                    self.gs, sec_best, self.material, active_mask=(gen_best_ind > 0)
                )
                self.load_case.apply(best_overall_mesh, self.domain, self.material)

            # Record generation snapshot
            if gen_best_res is not None:
                c_best = compute_compliance(gen_best_res)
                k_best = compute_stiffness(c_best)
                vol_best = gen_best_res.mesh.total_volume
                vf_best = vol_best / self.domain.bounding_volume
                ax_share = gen_best_res.axial_energy_fraction
                bend_share = gen_best_res.bending_energy_fraction
                max_s, _, _ = compute_member_peak_stress(gen_best_res)
            else:
                c_best = float("inf")
                k_best = 0.0
                vol_best = V_target
                vf_best = self.target_vf
                ax_share = 0.0
                bend_share = 0.0
                max_s = 0.0

            step = OptimizationStep(
                step_index=gen,
                fe_evaluations=fe_evaluations,
                compliance=c_best,
                stiffness=k_best,
                volume=vol_best,
                volume_fraction=vf_best,
                axial_energy_share=ax_share,
                bending_energy_share=bend_share,
                num_active_members=int(np.sum(gen_best_ind > 0)),
                active_mask=(gen_best_ind > 0),
                thicknesses=self.discrete_levels[gen_best_ind],
                max_stress=max_s,
            )
            history.steps.append(step)

            # 2. Reproduction: Elitism
            new_population: List[np.ndarray] = []
            for e in range(self.elite_count):
                new_population.append(population[rank_indices[e]].copy())

            # 3. Selection, Crossover & Mutation to fill new population
            while len(new_population) < self.pop_size:
                # Tournament selection for parent 1
                tourn1 = self.rng.choice(self.pop_size, size=self.tournament_size, replace=False)
                parent1_idx = tourn1[np.argmax(fitnesses[tourn1])]
                parent1 = population[parent1_idx]

                # Tournament selection for parent 2
                tourn2 = self.rng.choice(self.pop_size, size=self.tournament_size, replace=False)
                parent2_idx = tourn2[np.argmax(fitnesses[tourn2])]
                parent2 = population[parent2_idx]

                # Crossover
                if self.rng.rand() < self.crossover_rate:
                    mask = self.rng.rand(num_members) < 0.5
                    child = np.where(mask, parent1, parent2)
                else:
                    child = parent1.copy()

                # Mutation
                for g in range(num_members):
                    if self.rng.rand() < self.mutation_rate:
                        if self.rng.rand() < 0.7:
                            # Creep mutation (+/- 1 level)
                            delta = self.rng.choice([-1, 1])
                            child[g] = int(np.clip(child[g] + delta, 0, self.num_levels - 1))
                        else:
                            # Random reset mutation
                            child[g] = int(self.rng.randint(0, self.num_levels))

                # Volume Repair
                repaired_child = self._repair_volume(child, V_lattice_target)
                new_population.append(repaired_child)

            population = new_population

        history.final_mesh = best_overall_mesh
        history.final_result = best_overall_res
        history.total_fe_evaluations = fe_evaluations
        history.runtime_seconds = time.time() - start_time
        history.converged = bool(best_overall_fitness > 0.0)

        return history

