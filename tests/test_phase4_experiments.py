"""Smoke and regression tests for Phase 4 experiment runners (E1 to E7)."""

import pytest
import os
import yaml
import numpy as np

from src.experiments.experiment_runner import (
    run_experiment_e1,
    run_experiment_e2,
    run_experiment_e3,
    run_experiment_e4,
    run_experiment_e5,
    run_experiment_e6,
    run_experiment_e7,
)


@pytest.fixture
def fast_phase4_cfg():
    """Create a lightweight config for fast test execution."""
    return {
        "seed": 42,
        "domain": {"width": 3.0, "height": 1.0, "depth": 1.0},
        "material": {"E": 2.1e11, "density": 7850.0, "yield_strength": 2.5e8},
        "pad_config": {"thickness": 0.005, "depth": 1.0, "attachment_tolerance": 1e-4},
        "default_ground_structure": {"nx": 5, "ny": 3, "radius": None},
        "e1_final_quality": {
            "volume_fractions": [0.2],
            "num_seeds_ga": 2,
            "num_seeds_voronoi": 2,
        },
        "e2_convergence": {"target_volume_fraction": 0.2, "load_cases": ["LC1"]},
        "e3_pareto": {"volume_fractions": [0.2, 0.3], "load_cases": ["LC1"]},
        "e4_robustness": {"target_volume_fraction": 0.2},
        "e5_kinematics": {"target_volume_fraction": 0.2},
        "e6_ablations": {
            "load_case": "LC1",
            "target_volume_fraction": 0.2,
            "pruning_removal_rates": [0.05],
            "pruning_utilization_metrics": ["strain_energy_density", "axial_stress"],
            "pruning_redistribute_area": [False],
        },
        "e7_scaling": {
            "grid_sizes": [[5, 3]],
            "target_volume_fraction": 0.2,
            "load_case": "LC1",
        },
        "pruning_defaults": {
            "removal_rate": 0.05,
            "utilization_metric": "strain_energy_density",
            "redistribute_area": False,
            "initial_thickness": 0.025,
            "max_iterations": 20,
        },
        "ga_defaults": {
            "pop_size": 4,
            "generations": 2,
            "crossover_rate": 0.8,
            "mutation_rate": 0.02,
            "tournament_size": 2,
            "elite_count": 1,
            "discrete_levels": [0.0, 0.01, 0.02],
        },
    }


class TestPhase4Experiments:
    """Validate execution pipelines for E1 through E7."""

    def test_e1_final_quality_runner(self, fast_phase4_cfg, tmp_path):
        """Test E1 final quality and statistical test generation."""
        df_e1, df_stat = run_experiment_e1(fast_phase4_cfg, out_dir=str(tmp_path))
        assert not df_e1.empty
        assert "compliance_J" in df_e1.columns
        assert not df_stat.empty
        assert "p_value_raw" in df_stat.columns

    def test_e3_pareto_runner(self, fast_phase4_cfg, tmp_path):
        """Test E3 Pareto front generation."""
        df_e3 = run_experiment_e3(fast_phase4_cfg, out_dir=str(tmp_path))
        assert not df_e3.empty
        assert "stiffness_N_m" in df_e3.columns
        assert "mass_kg" in df_e3.columns

    def test_e2_convergence_runner(self, fast_phase4_cfg, tmp_path):
        """Test E2 convergence tracking."""
        df_e2 = run_experiment_e2(fast_phase4_cfg, out_dir=str(tmp_path))
        assert not df_e2.empty
        assert "fe_evaluations" in df_e2.columns
        assert "compliance_J" in df_e2.columns

    def test_e4_robustness_runner(self, fast_phase4_cfg, tmp_path):
        """Test E4 cross-load evaluation."""
        df_e4 = run_experiment_e4(fast_phase4_cfg, out_dir=str(tmp_path))
        assert not df_e4.empty
        assert "degradation_ratio" in df_e4.columns

    def test_e5_kinematics_runner(self, fast_phase4_cfg, tmp_path):
        """Test E5 kinematics breakdown."""
        df_e5 = run_experiment_e5(fast_phase4_cfg, out_dir=str(tmp_path))
        assert not df_e5.empty
        assert "axial_energy_share" in df_e5.columns
        assert "kinematic_class" in df_e5.columns

    def test_e6_ablations_runner(self, fast_phase4_cfg, tmp_path):
        """Test E6 hyperparameter ablations."""
        df_e6 = run_experiment_e6(fast_phase4_cfg, out_dir=str(tmp_path))
        assert not df_e6.empty
        assert "ablation_group" in df_e6.columns

    def test_e7_scaling_runner(self, fast_phase4_cfg, tmp_path):
        """Test E7 scaling benchmarks."""
        df_e7 = run_experiment_e7(fast_phase4_cfg, out_dir=str(tmp_path))
        assert not df_e7.empty
        assert "single_fe_solve_time_s" in df_e7.columns
