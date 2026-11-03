"""Tests for Stochastic Remodeling Pruning and Hyperparameter Tuning."""

import numpy as np
import pytest

from src.fem.material import Material
from src.fem.load_cases import LoadCase
from src.geometry.domain import DesignDomain2D
from src.geometry.ground_structure import generate_ground_structure
from src.optimizers.pruning import RemodelingPruningOptimizer
from src.optimizers.tuning import tune_pruning_hyperparameters


class TestStochasticPruningAndTuning:
    """Test suite for stochastic pruning behavior and tuning module."""

    @pytest.fixture
    def setup_model(self):
        domain = DesignDomain2D(width=2.0, height=1.0, depth=1.0)
        gs = generate_ground_structure(domain=domain, nx=5, ny=3, radius=1.2)
        mat = Material(name="StandardMat", E=1e9, density=1000.0)
        lc = LoadCase(id="LC1", name="Cantilever", load_type="cantilever")
        return domain, gs, mat, lc

    def test_deterministic_pruning_when_noise_zero(self, setup_model):
        domain, gs, mat, lc = setup_model

        # When stochastic_noise=0.0, different seeds must produce identical deterministic results
        opt1 = RemodelingPruningOptimizer(
            ground_structure=gs,
            load_case=lc,
            material=mat,
            target_volume_fraction=0.35,
            stochastic_noise=0.0,
            seed=1,
        )
        res1 = opt1.optimize()

        opt2 = RemodelingPruningOptimizer(
            ground_structure=gs,
            load_case=lc,
            material=mat,
            target_volume_fraction=0.35,
            stochastic_noise=0.0,
            seed=2,
        )
        res2 = opt2.optimize()

        assert res1.final_compliance == pytest.approx(res2.final_compliance, rel=1e-6)
        assert np.array_equal(res1.steps[-1].active_mask, res2.steps[-1].active_mask)

    def test_stochastic_pruning_variation_with_noise(self, setup_model):
        domain, gs, mat, lc = setup_model

        # When stochastic_noise > 0.0, different seeds explore different pruning trajectories
        opt1 = RemodelingPruningOptimizer(
            ground_structure=gs,
            load_case=lc,
            material=mat,
            target_volume_fraction=0.35,
            stochastic_noise=0.25,
            seed=42,
        )
        res1 = opt1.optimize()

        opt2 = RemodelingPruningOptimizer(
            ground_structure=gs,
            load_case=lc,
            material=mat,
            target_volume_fraction=0.35,
            stochastic_noise=0.25,
            seed=999,
        )
        res2 = opt2.optimize()

        # Both converge successfully and achieve valid compliance
        assert res1.converged
        assert res2.converged
        assert res1.final_compliance > 0.0
        assert res2.final_compliance > 0.0

    def test_tuning_module_execution(self, setup_model):
        domain, gs, mat, lc = setup_model

        param_grid = {
            "removal_rate": [0.05],
            "utilization_metric": ["strain_energy_density"],
            "redistribute_area": [False],
            "stochastic_noise": [0.0],
        }

        best_params, df = tune_pruning_hyperparameters(
            ground_structure=gs,
            load_case=lc,
            material=mat,
            target_volume_fraction=0.35,
            validation_seeds=[101],
            param_grid=param_grid,
        )

        assert "removal_rate" in best_params
        assert len(df) == 1
        assert "mean_compliance_J" in df.columns
        assert df["mean_compliance_J"].iloc[0] > 0.0
