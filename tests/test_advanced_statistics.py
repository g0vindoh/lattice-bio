"""Tests for Bootstrap Confidence Intervals and Statistical Power / MDE."""

import numpy as np
import pytest

from src.metrics.statistics import (
    bootstrap_median_ci,
    bootstrap_median_difference_ci,
    compute_minimum_detectable_effect,
    mann_whitney_u_test,
)


class TestAdvancedStatistics:
    """Test suite for bootstrap CIs, power, and MDE."""

    def test_bootstrap_median_ci(self):
        rng = np.random.RandomState(42)
        # Normal distribution with median 10.0
        data = rng.normal(loc=10.0, scale=1.0, size=50)

        med, ci_low, ci_high = bootstrap_median_ci(data, num_bootstraps=500, alpha=0.05, seed=123)

        assert ci_low <= med <= ci_high
        assert 9.0 <= med <= 11.0
        assert ci_high - ci_low < 1.5

    def test_bootstrap_median_difference_ci(self):
        rng = np.random.RandomState(42)
        # Group A centered at 20, Group B centered at 10 (diff ~ 10)
        sample_a = rng.normal(loc=20.0, scale=1.0, size=40)
        sample_b = rng.normal(loc=10.0, scale=1.0, size=40)

        diff, ci_low, ci_high = bootstrap_median_difference_ci(
            sample_a, sample_b, num_bootstraps=500, alpha=0.05, seed=123
        )

        assert ci_low <= diff <= ci_high
        assert 8.0 <= diff <= 12.0
        assert ci_low > 0.0  # Significantly positive difference

    def test_minimum_detectable_effect(self):
        mde_small = compute_minimum_detectable_effect(n1=5, n2=5)
        mde_large = compute_minimum_detectable_effect(n1=30, n2=30)

        # Larger sample sizes must yield smaller (more sensitive) MDE
        assert mde_large["mde_a12"] < mde_small["mde_a12"]
        assert mde_large["mde_rank_biserial"] < mde_small["mde_rank_biserial"]
        assert mde_large["mde_cohens_d"] < mde_small["mde_cohens_d"]
        assert mde_large["effective_sample_size"] == 15.0

    def test_mann_whitney_includes_cis_and_mde(self):
        a = [10.0, 11.0, 12.0, 10.5, 11.5]
        b = [20.0, 21.0, 22.0, 20.5, 21.5]

        res = mann_whitney_u_test(a, b, name_a="Pruning", name_b="GA")

        assert res.median_difference is not None
        assert res.ci_lower_diff is not None
        assert res.ci_upper_diff is not None
        assert res.minimum_detectable_effect is not None
        assert res.median_difference < 0.0  # Pruning is lower than GA
