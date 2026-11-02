"""Unit tests for Multi-Load Bio-Inspired Bone Remodeling Optimizer."""

import numpy as np
import pytest

from src.fem.material import Material
from src.fem.load_cases import load_cases_from_yaml
from src.geometry.domain import DesignDomain2D
from src.geometry.ground_structure import generate_ground_structure
from src.optimizers.multi_load import MultiLoadRemodelingOptimizer


class TestMultiLoadRemodeling:
    """Validation of multi-load remodeling optimization and robustness."""

    def test_multi_load_remodeling_convergence_and_feasibility(self):
        """Verify multi-load optimizer produces a feasible lattice for LC1 + LC2."""
        domain = DesignDomain2D(width=3.0, height=1.0)
        gs = generate_ground_structure(domain, nx=7, ny=3)
        lcs = load_cases_from_yaml("configs/phase2_baselines.yaml")
        mat = Material(E=210e9, density=7850)

        # Optimize simultaneously for LC1 (cantilever) and LC2 (3-point bending)
        opt = MultiLoadRemodelingOptimizer(
            ground_structure=gs,
            load_cases=[lcs["LC1"], lcs["LC2"]],
            material=mat,
            target_volume_fraction=0.25,
            removal_rate=0.10,
            seed=42,
        )

        res = opt.optimize()

        assert res["num_active_members"] < gs.num_members
        assert res["num_active_members"] > 5
        # Must be feasible and evaluated on BOTH load cases
        assert len(res["per_case_metrics"]) == 2
        compliances = [m["compliance"] for m in res["per_case_metrics"].values()]
        for c in compliances:
            assert 0.0 < c < 0.1
