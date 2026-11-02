"""Metrics, feasibility, and non-parametric statistical analysis module."""

from .feasibility import (
    SlendernessWarning,
    FeasibilityReport,
    check_design_feasibility,
)
from .metrics import (
    PerformanceMetrics,
    compute_compliance,
    compute_stiffness,
    compute_specific_stiffness,
    compute_energy_shares,
    compute_member_peak_stress,
    evaluate_design_metrics,
)
from .statistics import (
    DistributionSummary,
    HypothesisTestResult,
    compute_distribution_summary,
    mann_whitney_u_test,
    apply_holm_bonferroni_correction,
)

__all__ = [
    "SlendernessWarning",
    "FeasibilityReport",
    "check_design_feasibility",
    "PerformanceMetrics",
    "compute_compliance",
    "compute_stiffness",
    "compute_specific_stiffness",
    "compute_energy_shares",
    "compute_member_peak_stress",
    "evaluate_design_metrics",
    "DistributionSummary",
    "HypothesisTestResult",
    "compute_distribution_summary",
    "mann_whitney_u_test",
    "apply_holm_bonferroni_correction",
]
