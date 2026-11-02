"""Unit tests for Cowin trabecular fabric tensor and stereological anisotropy."""

import numpy as np
import pytest

from src.fem.material import Material, RectangularSection
from src.fem.mesh import FrameMesh2D
from src.geometry.domain import DesignDomain2D
from src.geometry.ground_structure import generate_ground_structure, build_fe_mesh_from_ground_structure
from src.metrics.fabric_tensor import (
    compute_lattice_fabric_tensor,
    FabricTensorResult,
)


class TestTrabecularFabricTensor:
    """Validation of Cowin second-order fabric tensor and anisotropy metrics."""

    def test_pure_horizontal_strut_anisotropy(self):
        """Pure horizontal struts must give H = [[1, 0], [0, 0]], FA = 1.0, theta = 0 deg."""
        mat = Material(E=210e9, density=7850)
        sec = RectangularSection(t=0.01)
        mesh = FrameMesh2D()
        mesh.add_node(1, 0.0, 0.0)
        mesh.add_node(2, 1.0, 0.0)
        mesh.add_element(1, 1, 2, mat, sec)

        res = compute_lattice_fabric_tensor(mesh)

        assert pytest.approx(res.H[0, 0], abs=1e-12) == 1.0
        assert pytest.approx(res.H[1, 1], abs=1e-12) == 0.0
        assert pytest.approx(res.H[0, 1], abs=1e-12) == 0.0
        assert pytest.approx(res.eigenvalues[0], abs=1e-12) == 1.0
        assert pytest.approx(res.eigenvalues[1], abs=1e-12) == 0.0
        assert pytest.approx(res.fractional_anisotropy, abs=1e-6) == 1.0
        assert pytest.approx(res.principal_angle_deg, abs=1e-4) == 0.0

    def test_pure_vertical_strut_anisotropy(self):
        """Pure vertical struts must give H = [[0, 0], [0, 1]], FA = 1.0, theta = 90 deg."""
        mat = Material(E=210e9, density=7850)
        sec = RectangularSection(t=0.01)
        mesh = FrameMesh2D()
        mesh.add_node(1, 0.0, 0.0)
        mesh.add_node(2, 0.0, 1.0)
        mesh.add_element(1, 1, 2, mat, sec)

        res = compute_lattice_fabric_tensor(mesh)

        assert pytest.approx(res.H[0, 0], abs=1e-12) == 0.0
        assert pytest.approx(res.H[1, 1], abs=1e-12) == 1.0
        assert pytest.approx(res.H[0, 1], abs=1e-12) == 0.0
        assert pytest.approx(res.fractional_anisotropy, abs=1e-6) == 1.0
        assert abs(abs(res.principal_angle_deg) - 90.0) < 1e-4

    def test_orthogonal_cross_is_isotropic(self):
        """Equal horizontal and vertical struts form an isotropic fabric (FA = 0.0, lambda = 0.5)."""
        mat = Material(E=210e9, density=7850)
        sec = RectangularSection(t=0.01)
        mesh = FrameMesh2D()
        # Center at (0, 0), cross of length 2 along x and y
        mesh.add_node(1, -1.0, 0.0)
        mesh.add_node(2, 1.0, 0.0)
        mesh.add_node(3, 0.0, -1.0)
        mesh.add_node(4, 0.0, 1.0)
        mesh.add_element(1, 1, 2, mat, sec)  # horizontal, L = 2
        mesh.add_element(2, 3, 4, mat, sec)  # vertical, L = 2

        res = compute_lattice_fabric_tensor(mesh)

        assert pytest.approx(res.H[0, 0], abs=1e-12) == 0.5
        assert pytest.approx(res.H[1, 1], abs=1e-12) == 0.5
        assert pytest.approx(res.H[0, 1], abs=1e-12) == 0.0
        assert pytest.approx(np.trace(res.H), abs=1e-12) == 1.0
        assert pytest.approx(res.fractional_anisotropy, abs=1e-6) == 0.0
        assert pytest.approx(res.degree_of_anisotropy, abs=1e-6) == 1.0

    def test_trace_normalization_and_ground_structure(self):
        """Fabric tensor trace must equal 1.0 across any complex lattice topology."""
        domain = DesignDomain2D(width=3.0, height=1.0)
        gs = generate_ground_structure(domain, nx=5, ny=3)
        mat = Material(E=210e9, density=7850)
        sec = RectangularSection(t=0.01)
        sections = [sec] * gs.num_members
        mesh = build_fe_mesh_from_ground_structure(gs, sections, mat)

        res = compute_lattice_fabric_tensor(mesh)

        assert pytest.approx(np.trace(res.H), abs=1e-12) == 1.0
        assert res.eigenvalues[0] >= res.eigenvalues[1] >= 0.0
        assert 0.0 <= res.fractional_anisotropy <= 1.0
