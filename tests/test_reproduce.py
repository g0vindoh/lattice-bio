"""Tests for master reproduction CLI functions."""

import os
import pytest
from reproduce_all import get_git_commit_hash, run_figures


def test_get_git_commit_hash():
    """Verify git commit hash can be resolved."""
    h = get_git_commit_hash()
    assert isinstance(h, str)
    assert len(h) >= 4


def test_run_figures_execution():
    """Verify figure regeneration executes without raising exceptions."""
    run_figures()
    assert os.path.exists("figures/paper_fig1_overview.png")
    assert os.path.exists("figures/paper_fig2_optimized_topologies.png")
    assert os.path.exists("figures/paper_fig3_convergence.png")
    assert os.path.exists("figures/paper_fig4_distributions.png")
    assert os.path.exists("figures/paper_fig5_pareto.png")
    assert os.path.exists("figures/paper_fig6_kinematics_shares.png")
    assert os.path.exists("figures/paper_fig7_robustness_matrix.png")
    assert os.path.exists("figures/paper_fig8_scaling.png")
