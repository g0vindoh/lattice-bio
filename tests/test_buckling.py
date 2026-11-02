"""Unit tests for Euler strut buckling analysis."""

import numpy as np
import pytest

from src.fem.material import Material, RectangularSection
from src.fem.mesh import FrameMesh2D, PointLoad
from src.fem.solver import FrameSolver2D
from src.metrics.buckling import (
    compute_euler_buckling_load,
    evaluate_lattice_buckling,
)


class TestEulerStrutBuckling:
    """Validation of Euler column buckling metrics and lattice stability."""

    def test_euler_buckling_analytical_formula(self):
        """Verify analytical Euler critical load against hand calculation."""
        E = 210e9
        t = 0.02
        b = 1.0
        I = (b * t ** 3) / 12.0  # (0.02^3)/12 = 6.66667e-7 m^4
        L = 1.0
        k = 1.0

        expected_P_cr = (np.pi ** 2 * E * I) / (k * L) ** 2
        computed_P_cr = compute_euler_buckling_load(E, I, L, effective_length_factor=k)

        assert abs(computed_P_cr - expected_P_cr) < 1e-6
        assert expected_P_cr > 1e6  # ~1.38 MN

    def test_compression_column_fe_buckling_factor(self):
        """Verify buckling load factor (BLF) on a solved compressive column."""
        mat = Material(E=210e9, density=7850)
        sec = RectangularSection(t=0.01)  # t = 10 mm, I = 8.333e-8
        mesh = FrameMesh2D()

        # Vertical column: node 1 at (0, 0), node 2 at (0, 1)
        mesh.add_node(1, 0.0, 0.0)
        mesh.add_node(2, 0.0, 1.0)
        mesh.add_element(1, 1, 2, mat, sec)

        # Fix base node 1
        mesh.add_boundary_condition(1, 0, 0.0)
        mesh.add_boundary_condition(1, 1, 0.0)
        mesh.add_boundary_condition(1, 2, 0.0)
        # Prevent horizontal sway at top node 2
        mesh.add_boundary_condition(2, 0, 0.0)

        # Theoretical P_cr
        P_cr = compute_euler_buckling_load(mat.E, sec.I, L=1.0, effective_length_factor=1.0)

        # Case A: Applied compressive load below P_cr (P = 0.5 * P_cr)
        mesh.point_loads = [PointLoad(node_id=2, fx=0.0, fy=-0.5 * P_cr)]
        solver = FrameSolver2D(mesh)
        res_subcritical = solver.solve()
        buckling_sub = evaluate_lattice_buckling(res_subcritical)

        assert buckling_sub.num_compression_members == 1
        assert pytest.approx(buckling_sub.global_buckling_factor, rel=1e-4) == 2.0
        assert not buckling_sub.is_buckling_critical

        # Case B: Applied compressive load above P_cr (P = 2.0 * P_cr)
        mesh.point_loads = [PointLoad(node_id=2, fx=0.0, fy=-2.0 * P_cr)]
        solver_super = FrameSolver2D(mesh)
        res_supercritical = solver_super.solve()
        buckling_super = evaluate_lattice_buckling(res_supercritical)

        assert pytest.approx(buckling_super.global_buckling_factor, rel=1e-4) == 0.5
        assert buckling_super.is_buckling_critical
        assert buckling_super.critical_element_id == 1

    def test_tensile_column_no_buckling(self):
        """Verify tensile loading produces infinite buckling factor and no critical flag."""
        mat = Material(E=210e9, density=7850)
        sec = RectangularSection(t=0.01)
        mesh = FrameMesh2D()
        mesh.add_node(1, 0.0, 0.0)
        mesh.add_node(2, 0.0, 1.0)
        mesh.add_element(1, 1, 2, mat, sec)
        mesh.add_boundary_condition(1, 0, 0.0)
        mesh.add_boundary_condition(1, 1, 0.0)
        mesh.add_boundary_condition(1, 2, 0.0)

        # Upward tensile load
        mesh.point_loads = [PointLoad(node_id=2, fx=0.0, fy=1000.0)]
        solver = FrameSolver2D(mesh)
        res = solver.solve()
        buckling = evaluate_lattice_buckling(res)

        assert buckling.num_compression_members == 0
        assert buckling.global_buckling_factor == float("inf")
        assert not buckling.is_buckling_critical
