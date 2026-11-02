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

    return HypothesisTestResult(
        method_a_name=name_a,
        method_b_name=name_b,
        u_statistic=u_stat,
        p_value=p_val,
        rank_biserial_r=rank_biserial,
        vargha_delaney_a12=a12,
        is_significant_nominal=bool(p_val < 0.05),
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
            )
        )

    return final_results

