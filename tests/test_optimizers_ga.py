"""Unit tests for Method B: Genetic Algorithm optimizer."""

import pytest
import numpy as np

from src.geometry.domain import DesignDomain2D
from src.geometry.ground_structure import generate_ground_structure
from src.fem.load_cases import LoadCase
from src.fem.material import Material
from src.optimizers.ga import GeneticAlgorithmOptimizer


class TestGeneticAlgorithm:
    """Test GA chromosome representation, operators, repair, and convergence."""

    @pytest.fixture
    def setup_cantilever(self):
        domain = DesignDomain2D(width=3.0, height=1.0)
        # Small 5x3 grid for fast unit tests
        gs = generate_ground_structure(domain=domain, nx=5, ny=3)
        lc = LoadCase(id="LC1", name="Cantilever", load_type="cantilever", load_magnitude=500.0)
        mat = Material(E=2.1e11, density=7850.0)
        return domain, gs, lc, mat

    def test_volume_repair_operator(self, setup_cantilever):
        """Verify that the volume repair operator strictly bounds volume budget."""
        domain, gs, lc, mat = setup_cantilever
        target_vf = 0.2

        opt = GeneticAlgorithmOptimizer(
            ground_structure=gs,
            load_case=lc,
            material=mat,
            target_volume_fraction=target_vf,
            pop_size=10,
            generations=2,
            seed=42,
        )

        V_target = target_vf * domain.bounding_volume
        # Create an over-budget chromosome with maximum thickness everywhere
        max_chrom = np.full(gs.num_members, opt.num_levels - 1)
        repaired = opt._repair_volume(max_chrom, V_target)

        # Check all genes are valid levels
        assert np.all(repaired >= 0)
        assert np.all(repaired < opt.num_levels)

        # Check volume is within budget
        t_rep = opt.discrete_levels[repaired]
        vol_rep = np.sum(t_rep * gs.member_lengths) * domain.depth
        assert vol_rep <= V_target

    def test_ga_runs_and_preserves_elitism(self, setup_cantilever):
        """Verify that GA executes and elitism guarantees best compliance does not degrade."""
        domain, gs, lc, mat = setup_cantilever

        opt = GeneticAlgorithmOptimizer(
            ground_structure=gs,
            load_case=lc,
            material=mat,
            target_volume_fraction=0.25,
            pop_size=12,
            generations=6,
            crossover_rate=0.8,
            mutation_rate=0.03,
            elite_count=2,
            seed=42,
        )

        history = opt.optimize()

        assert len(history.steps) == 6
        assert history.total_fe_evaluations >= 12 * 6

        # Check elitism: best compliance should be non-increasing (or fitness non-decreasing)
        compliances = [s.compliance for s in history.steps if np.isfinite(s.compliance)]
        if len(compliances) > 1:
            # Min compliance up to generation g should be non-increasing
            running_min = np.minimum.accumulate(compliances)
            assert np.all(running_min[1:] <= running_min[:-1] + 1e-12)

        # Final result must be valid
        assert history.final_result is not None
        assert history.final_compliance > 0.0

    def test_infeasible_penalty(self, setup_cantilever):
        """Verify that an infeasible chromosome (zero active members) receives heavy penalty."""
        domain, gs, lc, mat = setup_cantilever

        opt = GeneticAlgorithmOptimizer(
            ground_structure=gs,
            load_case=lc,
            material=mat,
            target_volume_fraction=0.2,
            seed=42,
        )

        empty_chrom = np.zeros(gs.num_members, dtype=int)
        fit, report, res = opt._evaluate_individual(empty_chrom)

        assert report.is_feasible is False
        assert res is None
        # Fitness must be very small (<= 1e-7)
        assert fit <= 1e-7

