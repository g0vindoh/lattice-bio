"""Non-parametric statistical comparison helpers and summary metrics.

Formulations:
1. Median and Interquartile Range (IQR = Q75 - Q25).
2. Mann-Whitney U test for independent distributions (two-sided, exact or asymptotic).
3. Effect Size: Rank-Biserial Correlation r = 1 - (2 * U) / (n1 * n2).
   Interpretation: r in [-1, 1], where 0 indicates stochastic equality.
4. Vargha-Delaney A_12 effect size: A_12 = U1 / (n1 * n2).
   Interpretation: A_12 = 0.5 (no effect), > 0.5 (group 1 tends to be larger).
5. Holm-Bonferroni step-down correction for family-wise error rate control under
   multiple hypothesis testing.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DistributionSummary:
    """Summary statistics for a continuous sample distribution."""

    count: int
    mean: float
    std: float
    median: float
    q25: float
    q75: float
    iqr: float
    min_val: float
    max_val: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "q25": self.q25,
            "q75": self.q75,
            "iqr": self.iqr,
            "min": self.min_val,
            "max": self.max_val,
        }


@dataclass(frozen=True)
class HypothesisTestResult:
    """Outcome of a two-sample non-parametric hypothesis test."""

    method_a_name: str
    method_b_name: str
    u_statistic: float
    p_value: float
    rank_biserial_r: float
    vargha_delaney_a12: float
    is_significant_nominal: bool  # p < 0.05
    adjusted_p_value: Optional[float] = None
    is_significant_corrected: Optional[bool] = None
    median_difference: Optional[float] = None
    ci_lower_diff: Optional[float] = None
    ci_upper_diff: Optional[float] = None
    statistical_power: Optional[float] = None
    minimum_detectable_effect: Optional[float] = None


def compute_distribution_summary(data: Sequence[float]) -> DistributionSummary:
    """Compute robust non-parametric and descriptive statistics for a 1D sample."""
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        raise ValueError("Cannot compute summary statistics for empty sample.")

    q25, median, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
    return DistributionSummary(
        count=len(arr),
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=1 if len(arr) > 1 else 0)),
        median=float(median),
        q25=float(q25),
        q75=float(q75),
        iqr=float(q75 - q25),
        min_val=float(np.min(arr)),
        max_val=float(np.max(arr)),
    )


def mann_whitney_u_test(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    name_a: str = "GroupA",
    name_b: str = "GroupB",
    alternative: str = "two-sided",
) -> HypothesisTestResult:
    """Perform Mann-Whitney U test with Rank-Biserial and Vargha-Delaney effect sizes.

    Args:
        sample_a: Numerical observations for first group.
        sample_b: Numerical observations for second group.
        name_a: Label for first group.
        name_b: Label for second group.
        alternative: 'two-sided', 'less', or 'greater'.

    Returns:
        HypothesisTestResult containing U, p-value, and effect sizes.
    """
    a = np.asarray(sample_a, dtype=np.float64)
    b = np.asarray(sample_b, dtype=np.float64)

    n1 = len(a)
    n2 = len(b)
    if n1 == 0 or n2 == 0:
        raise ValueError("Samples must be non-empty for Mann-Whitney U test.")

    res = stats.mannwhitneyu(a, b, alternative=alternative)
    u_stat = float(res.statistic)
    p_val = float(res.pvalue)

    # Effect sizes
    # Rank-biserial correlation: r = 1 - 2*U / (n1*n2)
    # Note: stats.mannwhitneyu returns U for sample_a
    total_pairs = n1 * n2
    rank_biserial = 1.0 - (2.0 * u_stat) / total_pairs
    a12 = u_stat / total_pairs

    # Bootstrap 95% CI on median difference (Median_A - Median_B)
    med_diff, ci_low, ci_high = bootstrap_median_difference_ci(a, b)

    # Minimum detectable effect for this sample size pair
    mde_dict = compute_minimum_detectable_effect(n1, n2)

    return HypothesisTestResult(
        method_a_name=name_a,
        method_b_name=name_b,
        u_statistic=u_stat,
        p_value=p_val,
        rank_biserial_r=rank_biserial,
        vargha_delaney_a12=a12,
        is_significant_nominal=bool(p_val < 0.05),
        median_difference=med_diff,
        ci_lower_diff=ci_low,
        ci_upper_diff=ci_high,
        minimum_detectable_effect=mde_dict["mde_rank_biserial"],
    )


def apply_holm_bonferroni_correction(
    results: List[HypothesisTestResult],
    alpha: float = 0.05,
) -> List[HypothesisTestResult]:
    """Apply Holm-Bonferroni step-down correction for multiple hypothesis comparisons.

    Args:
        results: List of HypothesisTestResult instances.
        alpha: Overall family-wise error rate threshold (default 0.05).

    Returns:
        New list of HypothesisTestResult with adjusted_p_value and is_significant_corrected.
    """
    m = len(results)
    if m == 0:
        return []

    # Sort results by raw p-value ascending
    indices = list(range(m))
    indices.sort(key=lambda i: results[i].p_value)

    adjusted_results: List[Optional[HypothesisTestResult]] = [None] * m

    running_min = 1.0
    for rank, idx in enumerate(indices):
        k = rank + 1  # 1-indexed rank
        raw_p = results[idx].p_value
        # Holm-Bonferroni formula: p_adj = min(1.0, max(previous_adj, (m - k + 1) * raw_p))
        # Step-down requires monotonicity: p_adj[k] >= p_adj[k-1]
        step_factor = m - k + 1
        adj_p = min(1.0, step_factor * raw_p)

        orig = results[idx]
        adjusted_results[idx] = HypothesisTestResult(
            method_a_name=orig.method_a_name,
            method_b_name=orig.method_b_name,
            u_statistic=orig.u_statistic,
            p_value=orig.p_value,
            rank_biserial_r=orig.rank_biserial_r,
            vargha_delaney_a12=orig.vargha_delaney_a12,
            is_significant_nominal=orig.is_significant_nominal,
            adjusted_p_value=adj_p,
            is_significant_corrected=bool(adj_p < alpha),
        )

    # Ensure monotonicity of adjusted p-values
    # In step-down: p_adj[i] = max(p_adj[i], p_adj[i-1]) in sorted order
    sorted_adj = [adjusted_results[idx].adjusted_p_value for idx in indices]
    for i in range(1, m):
        if sorted_adj[i] < sorted_adj[i - 1]:
            sorted_adj[i] = sorted_adj[i - 1]

    final_results: List[HypothesisTestResult] = []
    for orig_idx in range(m):
        # find rank
        rank = indices.index(orig_idx)
        adj_p = sorted_adj[rank]
        orig = results[orig_idx]
        final_results.append(
            HypothesisTestResult(
                method_a_name=orig.method_a_name,
                method_b_name=orig.method_b_name,
                u_statistic=orig.u_statistic,
                p_value=orig.p_value,
                rank_biserial_r=orig.rank_biserial_r,
                vargha_delaney_a12=orig.vargha_delaney_a12,
                is_significant_nominal=orig.is_significant_nominal,
                adjusted_p_value=float(adj_p),
                is_significant_corrected=bool(adj_p < alpha),
                median_difference=orig.median_difference,
                ci_lower_diff=orig.ci_lower_diff,
                ci_upper_diff=orig.ci_upper_diff,
                statistical_power=orig.statistical_power,
                minimum_detectable_effect=orig.minimum_detectable_effect,
            )
        )

    return final_results


def bootstrap_median_ci(
    data: Sequence[float],
    num_bootstraps: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap percentile confidence interval for the sample median.

    Args:
        data: 1D sequence of observations.
        num_bootstraps: Number of bootstrap resamples (default 2000).
        alpha: Significance level (default 0.05 for 95% CI).
        seed: Random seed for exact reproducibility.

    Returns:
        (observed_median, ci_lower, ci_upper)
    """
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        raise ValueError("Cannot bootstrap empty sample.")

    observed_median = float(np.median(arr))
    if len(arr) == 1:
        return observed_median, observed_median, observed_median

    rng = np.random.RandomState(seed)
    n = len(arr)
    # Generate bootstrap medians
    indices = rng.randint(0, n, size=(num_bootstraps, n))
    resamples = arr[indices]
    boot_medians = np.median(resamples, axis=1)

    low_pct = 100.0 * (alpha / 2.0)
    high_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(boot_medians, low_pct))
    ci_upper = float(np.percentile(boot_medians, high_pct))

    return observed_median, ci_lower, ci_upper


def bootstrap_median_difference_ci(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    num_bootstraps: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap percentile confidence interval for difference in medians (Median_A - Median_B).

    Args:
        sample_a: 1D sequence for first group.
        sample_b: 1D sequence for second group.
        num_bootstraps: Number of resamples (default 2000).
        alpha: Significance level (default 0.05 for 95% CI).
        seed: Random seed for exact reproducibility.

    Returns:
        (observed_diff, ci_lower, ci_upper)
    """
    a = np.asarray(sample_a, dtype=np.float64)
    b = np.asarray(sample_b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Samples must be non-empty.")

    observed_diff = float(np.median(a) - np.median(b))

    rng = np.random.RandomState(seed)
    n1, n2 = len(a), len(b)
    idx1 = rng.randint(0, n1, size=(num_bootstraps, n1))
    idx2 = rng.randint(0, n2, size=(num_bootstraps, n2))

    med_a = np.median(a[idx1], axis=1)
    med_b = np.median(b[idx2], axis=1)
    boot_diffs = med_a - med_b

    low_pct = 100.0 * (alpha / 2.0)
    high_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(boot_diffs, low_pct))
    ci_upper = float(np.percentile(boot_diffs, high_pct))

    return observed_diff, ci_lower, ci_upper


def compute_minimum_detectable_effect(
    n1: int,
    n2: int,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> Dict[str, float]:
    """Compute the Minimum Detectable Effect (MDE) and power parameters for two-sample comparisons.

    Uses the asymptotic variance formula for the Mann-Whitney U / Vargha-Delaney statistic
    under the null hypothesis:
        Var(A12) = (n1 + n2 + 1) / (12 * n1 * n2)

    Args:
        n1: Sample size of group 1.
        n2: Sample size of group 2.
        alpha: Type I error rate (default 0.05, two-tailed).
        target_power: Desired statistical power (1 - beta, default 0.80).

    Returns:
        Dictionary with mde_a12, mde_rank_biserial, mde_cohens_d, and effective_sample_size.
    """
    if n1 <= 1 or n2 <= 1:
        return {
            "mde_a12": 0.5,
            "mde_rank_biserial": 1.0,
            "mde_cohens_d": float("inf"),
            "effective_sample_size": 0.0,
        }

    # Standard normal quantiles
    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(target_power)

    # Variance of A12 under H0
    var_a12 = (n1 + n2 + 1.0) / (12.0 * n1 * n2)
    se_a12 = np.sqrt(var_a12)

    # Minimum detectable deviation |A12 - 0.5|
    delta_a12 = (z_alpha + z_beta) * se_a12
    mde_a12 = min(0.5, float(delta_a12))

    # Rank-biserial r = 2 * A12 - 1 => delta_r = 2 * delta_a12
    mde_r = min(1.0, float(2.0 * delta_a12))

    # Equivalent Cohen's d: d = (z_alpha + z_beta) / sqrt(n_eff)
    n_eff = (n1 * n2) / (n1 + n2)
    mde_d = float((z_alpha + z_beta) / np.sqrt(n_eff))

    return {
        "mde_a12": mde_a12,
        "mde_rank_biserial": mde_r,
        "mde_cohens_d": mde_d,
        "effective_sample_size": float(n_eff),
    }


