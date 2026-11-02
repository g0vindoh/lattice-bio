"""Unit tests for statistical helpers: Mann-Whitney U, effect sizes, and Holm-Bonferroni correction."""

import pytest
import numpy as np

from src.metrics.statistics import (
    compute_distribution_summary,
    mann_whitney_u_test,
    apply_holm_bonferroni_correction,
)


class TestStatisticsHelpers:
    """Test non-parametric statistics, effect sizes, and multiple testing corrections."""

    def test_compute_distribution_summary(self):
        """Verify median, IQR, percentiles, mean and std."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        summary = compute_distribution_summary(data)

        assert summary.count == 7
        assert np.isclose(summary.median, 4.0)
        assert np.isclose(summary.mean, 4.0)
        assert np.isclose(summary.min_val, 1.0)
        assert np.isclose(summary.max_val, 7.0)
        assert summary.q25 < summary.median < summary.q75
        assert np.isclose(summary.iqr, summary.q75 - summary.q25)

    def test_mann_whitney_identical_distributions(self):
        """Identical distributions should yield non-significant p-value and near-zero effect size."""
        np.random.seed(42)
        sample_a = np.random.normal(10.0, 1.0, 50)
        sample_b = np.random.normal(10.0, 1.0, 50)

        res = mann_whitney_u_test(sample_a, sample_b, name_a="GroupA", name_b="GroupB")

        assert res.p_value > 0.05
        assert not res.is_significant_nominal
        # Vargha-Delaney A12 should be close to 0.5 (no stochastic dominance)
        assert np.isclose(res.vargha_delaney_a12, 0.5, atol=0.2)

    def test_mann_whitney_distinct_distributions(self):
        """Distinct distributions should yield highly significant p-value."""
        sample_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        sample_b = [10.0, 11.0, 12.0, 13.0, 14.0]

        res = mann_whitney_u_test(sample_a, sample_b, name_a="Low", name_b="High")

        assert res.p_value < 0.01
        assert res.is_significant_nominal
        # sample_a is completely smaller than sample_b: U_a = 0
        assert res.u_statistic == 0.0
        assert np.isclose(res.rank_biserial_r, 1.0)
        assert np.isclose(res.vargha_delaney_a12, 0.0)

    def test_holm_bonferroni_correction(self):
        """Verify step-down adjusted p-values and significance flags."""
        s_low1 = [1, 2, 3, 4, 5, 6]
        s_high1 = [10, 11, 12, 13, 14, 15]
        s_same1 = [5, 6, 7, 8, 9, 10]
        s_same2 = [5, 6, 7, 8, 9, 10]

        t1 = mann_whitney_u_test(s_low1, s_high1, "A", "B")  # p = 0.0022 < 0.05
        t2 = mann_whitney_u_test(s_same1, s_same2, "C", "D")  # p = 1.0 > 0.05

        corrected = apply_holm_bonferroni_correction([t1, t2], alpha=0.05)

        assert len(corrected) == 2
        # Monotonicity and correct flags
        assert corrected[0].adjusted_p_value >= corrected[0].p_value
        assert corrected[1].adjusted_p_value >= corrected[1].p_value
        assert corrected[0].is_significant_corrected
        assert not corrected[1].is_significant_corrected
