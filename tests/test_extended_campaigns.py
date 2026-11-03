"""Smoke tests for new Phase A/B/C experiment campaign runners."""

import pytest
import os
import pandas as pd

from src.experiments.baselines_comparison import run_baselines_benchmark
from src.experiments.buckling_pareto import run_buckling_pareto_campaign
from src.experiments.multiload_spectra import run_multiload_spectra_campaign
from src.experiments.wolff_sweep import run_wolff_law_sweep
from src.experiments.grid_scaling_study import run_grid_scaling_study
from src.experiments.factorial_ablations import run_factorial_ablations_campaign
from src.experiments.stochastic_benchmark import run_stochastic_benchmark


class TestExtendedCampaignRunners:
    """Smoke test suite for advanced research experiment modules."""

    @pytest.fixture
    def test_results_dir(self, tmp_path):
        out = tmp_path / "test_results"
        out.mkdir()
        return str(out)

    def test_baselines_comparison_smoke(self, test_results_dir):
        # Run on LC1 with small parameters
        df = run_baselines_benchmark(
            config_path="configs/phase4_experiments.yaml",
            target_vf=0.3,
            out_dir=test_results_dir,
            seed=42,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "optimality_gap_percent" in df.columns
        assert os.path.exists(os.path.join(test_results_dir, "baselines_comparison.csv"))

    def test_buckling_pareto_smoke(self, test_results_dir):
        df = run_buckling_pareto_campaign(
            config_path="configs/phase4_experiments.yaml",
            target_vf=0.3,
            out_dir=test_results_dir,
            seed=42,
        )
        assert isinstance(df, pd.DataFrame)
        assert "global_buckling_factor" in df.columns
        assert os.path.exists(os.path.join(test_results_dir, "buckling_pareto.csv"))

    def test_multiload_spectra_smoke(self, test_results_dir):
        df = run_multiload_spectra_campaign(
            config_path="configs/phase4_experiments.yaml",
            target_vf=0.3,
            out_dir=test_results_dir,
            seed=42,
        )
        assert isinstance(df, pd.DataFrame)
        assert "condition_number_kappa" in df.columns
        assert "bounded_compliance_J" in df.columns
        assert os.path.exists(os.path.join(test_results_dir, "multiload_spectra_analysis.csv"))

    def test_wolff_sweep_smoke(self, test_results_dir):
        df = run_wolff_law_sweep(
            config_path="configs/phase4_experiments.yaml",
            out_dir=test_results_dir,
            seed=42,
        )
        assert isinstance(df, pd.DataFrame)
        assert "compliance_delta_percent" in pd.read_csv(os.path.join(test_results_dir, "wolff_sweep_summary.csv")).columns

    def test_grid_scaling_smoke(self, test_results_dir):
        df = run_grid_scaling_study(
            config_path="configs/phase4_experiments.yaml",
            target_vf=0.3,
            out_dir=test_results_dir,
            seed=42,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 8  # 5 resolutions + 3 aspect ratios
        assert os.path.exists(os.path.join(test_results_dir, "grid_scaling_study.csv"))

    def test_factorial_ablations_smoke(self, test_results_dir):
        df_abl, df_sed = run_factorial_ablations_campaign(
            config_path="configs/phase4_experiments.yaml",
            target_vf=0.3,
            out_dir=test_results_dir,
            seed=42,
        )
        assert len(df_abl) == 18  # 3 * 3 * 2 combinations
        assert len(df_sed) > 0
        assert os.path.exists(os.path.join(test_results_dir, "factorial_ablations.csv"))

    def test_stochastic_benchmark_smoke(self, test_results_dir):
        df_raw, df_tests = run_stochastic_benchmark(
            config_path="configs/phase4_experiments.yaml",
            target_vf=0.3,
            num_seeds=2,
            out_dir=test_results_dir,
        )
        assert len(df_raw) > 0
        assert len(df_tests) > 0
        assert os.path.exists(os.path.join(test_results_dir, "stochastic_benchmark.csv"))
        assert os.path.exists(os.path.join(test_results_dir, "hypothesis_tests_complete.csv"))
