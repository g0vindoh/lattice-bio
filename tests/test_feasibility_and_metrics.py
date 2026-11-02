"""Unit tests for feasibility verification and performance metrics modules."""

import pytest
import numpy as np

from src.geometry.domain import DesignDomain2D
from src.fem.material import Material, RectangularSection
from src.fem.mesh import FrameMesh2D
from src.fem.solver import FrameSolver2D
from src.metrics.feasibility import check_design_feasibility
from src.metrics.metrics import (
    compute_compliance,
    compute_stiffness,
    compute_specific_stiffness,
    compute_energy_shares,
    compute_member_peak_stress,
    evaluate_design_metrics,
)


class TestFeasibilityAndMetrics:
    """Test structural feasibility checks and metrics computations."""

    @pytest.fixture
    def cantilever_setup(self):
        mat = Material(E=2.1e11, density=7850.0)
        sec = RectangularSection(t=0.02, b=1.0)
        domain = DesignDomain2D(width=3.0, height=1.0)

        mesh = FrameMesh2D()
        mesh.add_node(1, 0.0, 0.0)
        mesh.add_node(2, 3.0, 0.0)
        mesh.add_element(1, 1, 2, mat, sec)
        mesh.add_fixed_support(1)
        mesh.add_load(2, fx=0.0, fy=-500.0)

        solver = FrameSolver2D(mesh)
        res = solver.solve()
        return mesh, res, domain

    def test_metrics_evaluations(self, cantilever_setup):
        mesh, res, domain = cantilever_setup

        c = compute_compliance(res)
        assert c > 0.0

        k = compute_stiffness(c)
        assert np.isclose(k, 1.0 / c, rtol=1e-14)

        mass = mesh.total_mass
        spec_k = compute_specific_stiffness(c, mass)
        assert np.isclose(spec_k, k / mass, rtol=1e-14)

        ax_share, bend_share = compute_energy_shares(res)
        assert np.isclose(ax_share + bend_share, 1.0, rtol=1e-12)
        assert np.isclose(bend_share, 1.0, atol=1e-12)  # Pure transverse cantilever

        max_stress, util, crit_id = compute_member_peak_stress(res, yield_strength=250e6)
        assert max_stress > 0.0
        assert util > 0.0
        assert crit_id == 1

        perf = evaluate_design_metrics(res, domain, yield_strength=250e6)
        assert perf.compliance == c
        assert perf.stiffness == k
        assert perf.mass == mass
        assert perf.specific_stiffness == spec_k

    def test_feasibility_check_valid_structure(self, cantilever_setup):
        mesh, _, _ = cantilever_setup
        report, res = check_design_feasibility(mesh)

        assert report.is_feasible is True
        assert report.is_connected is True
        assert report.has_load_path is True
        assert report.has_mechanism is False
        assert res is not None

    def test_feasibility_check_disconnected_load_path(self):
        """A loaded node disconnected from supports must fail feasibility."""
        mat = Material(E=2.1e11)
        sec = RectangularSection(t=0.02)

        mesh = FrameMesh2D()
        mesh.add_node(1, 0.0, 0.0)
        mesh.add_node(2, 1.0, 0.0)
        mesh.add_element(1, 1, 2, mat, sec)
        mesh.add_fixed_support(1)

        # Disconnected member carrying load
        mesh.add_node(3, 2.0, 0.0)
        mesh.add_node(4, 3.0, 0.0)
        mesh.add_element(2, 3, 4, mat, sec)
        mesh.add_load(4, fy=-100.0)

        report, res = check_design_feasibility(mesh)
        assert report.is_feasible is False
        assert report.has_load_path is False
        assert 4 in report.unreachable_load_nodes

    def test_slenderness_warnings(self):
        """Thick member with L/t < 10 must trigger slenderness warning."""
        mat = Material(E=2.1e11)
        # L = 1.0 m, t = 0.2 m -> L/t = 5 < 10
        sec_stocky = RectangularSection(t=0.2, b=1.0)

        mesh = FrameMesh2D()
        mesh.add_node(1, 0.0, 0.0)
        mesh.add_node(2, 1.0, 0.0)
        mesh.add_element(1, 1, 2, mat, sec_stocky)
        mesh.add_fixed_support(1)
        mesh.add_load(2, fy=-100.0)

        report, _ = check_design_feasibility(mesh)
        assert report.min_slenderness == 5.0
        assert report.has_slenderness_warnings is True
        assert len(report.slenderness_warnings) == 1
        assert report.slenderness_warnings[0].element_id == 1

