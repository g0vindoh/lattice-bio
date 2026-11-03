"""Tests for Simulated Annealing Metaheuristic Baseline."""

import numpy as np
import pytest

from src.fem.material import Material
from src.fem.load_cases import LoadCase
from src.geometry.domain import DesignDomain2D
from src.geometry.ground_structure import generate_ground_structure
from src.optimizers.simulated_annealing import SimulatedAnnealingOptimizer


class TestSimulatedAnnealingOptimizer:
    """Test suite for Simulated Annealing baseline."""

    @pytest.fixture
    def setup_model(self):
        domain = DesignDomain2D(width=2.0, height=1.0, depth=1.0)
        gs = generate_ground_structure(domain=domain, nx=5, ny=3, radius=1.2)
        mat = Material(name="StandardMat", E=1e9, density=1000.0)
        lc = LoadCase(id="LC1", name="Cantilever", load_type="cantilever")
        return domain, gs, mat, lc

    def test_sa_volume_repair(self, setup_model):
        domain, gs, mat, lc = setup_model
        opt = SimulatedAnnealingOptimizer(
            ground_structure=gs,
            load_case=lc,
            material=mat,
            target_volume_fraction=0.3,
            max_evaluations=10,
            seed=42,
        )

        state = np.ones(gs.num_members, dtype=int) * (opt.num_levels - 1)
        repaired = opt._repair_volume(state)

        assert len(repaired) == gs.num_members
        assert np.all(repaired >= 0)
        assert np.all(repaired < opt.num_levels)

    def test_sa_optimization_run(self, setup_model):
        domain, gs, mat, lc = setup_model
        opt = SimulatedAnnealingOptimizer(
            ground_structure=gs,
            load_case=lc,
            material=mat,
            target_volume_fraction=0.3,
            max_evaluations=25,
            initial_temperature=5.0,
            seed=123,
        )

        history = opt.optimize()

        assert history.optimizer_name == "SimulatedAnnealing"
        assert history.total_fe_evaluations >= 15
        assert len(history.steps) >= 1
        assert history.final_mesh is not None
        assert history.final_result is not None
        assert history.final_compliance < 1e11
        assert history.final_compliance > 0.0
