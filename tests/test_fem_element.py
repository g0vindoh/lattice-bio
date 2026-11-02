"""Unit tests for 2D Euler-Bernoulli frame elements and material definitions."""

import pytest
import numpy as np

from src.fem.material import Material, CrossSection, RectangularSection
from src.fem.elements import FrameElement2D


class TestMaterialAndSection:
    """Test material and cross-section property calculations and validations."""

    def test_material_validations(self):
        mat = Material(E=2.1e11, density=7850.0, name="Steel")
        assert mat.E == 2.1e11
        assert mat.density == 7850.0
        assert mat.name == "Steel"

        with pytest.raises(ValueError, match="Young's modulus"):
            Material(E=-1.0)
        with pytest.raises(ValueError, match="Young's modulus"):
            Material(E=0.0)
        with pytest.raises(ValueError, match="Density"):
            Material(E=1.0, density=0.0)

    def test_cross_section_validations(self):
        sec = CrossSection(A=0.01, I=1e-5)
        assert sec.A == 0.01
        assert sec.I == 1e-5

        with pytest.raises(ValueError, match="Cross-sectional area"):
            CrossSection(A=0.0, I=1e-5)
        with pytest.raises(ValueError, match="Second moment of area"):
            CrossSection(A=0.01, I=-1e-5)

    def test_rectangular_section_derivation(self):
        """Verify rectangular section formulas: A = b*t, I = (b*t^3)/12.

        For unit depth b = 1.0:
            A = t
            I = t^3 / 12
        """
        t = 0.03
        b = 1.0
        rect_unit = RectangularSection(t=t, b=b)
        assert rect_unit.t == t
        assert rect_unit.b == 1.0
        assert np.isclose(rect_unit.A, t, rtol=1e-14)
        assert np.isclose(rect_unit.I, (t**3) / 12.0, rtol=1e-14)

        # Non-unit depth test
        b_custom = 0.5
        rect_custom = RectangularSection(t=t, b=b_custom)
        assert np.isclose(rect_custom.A, b_custom * t, rtol=1e-14)
        assert np.isclose(rect_custom.I, b_custom * (t**3) / 12.0, rtol=1e-14)

        with pytest.raises(ValueError, match="Thickness"):
            RectangularSection(t=-0.01)
        with pytest.raises(ValueError, match="depth"):
            RectangularSection(t=0.01, b=0.0)


class TestFrameElement2D:
    """Test 2D frame element stiffness, kinematics, and internal force calculations."""

    @pytest.fixture
    def sample_element(self):
        mat = Material(E=2.0e11, density=7800.0)
        sec = RectangularSection(t=0.05, b=1.0)
        return FrameElement2D(
            element_id=1,
            node_i=10,
            node_j=20,
            coords_i=(0.0, 0.0),
            coords_j=(3.0, 4.0),
            material=mat,
            section=sec,
        )

    def test_element_geometry(self, sample_element):
        assert sample_element.length == 5.0
        assert np.isclose(sample_element.cos, 0.6, rtol=1e-14)
        assert np.isclose(sample_element.sin, 0.8, rtol=1e-14)
        expected_vol = sample_element.section.A * 5.0
        assert np.isclose(sample_element.volume, expected_vol, rtol=1e-14)
        expected_mass = 7800.0 * expected_vol
        assert np.isclose(sample_element.mass, expected_mass, rtol=1e-14)

    def test_zero_length_element_raises(self):
        mat = Material(E=1e9)
        sec = RectangularSection(t=0.01)
        with pytest.raises(ValueError, match="zero or near-zero length"):
            FrameElement2D(
                element_id=1,
                node_i=1,
                node_j=2,
                coords_i=(1.0, 2.0),
                coords_j=(1.0, 2.0),
                material=mat,
                section=sec,
            )

    def test_stiffness_symmetry(self, sample_element):
        k_local = sample_element.local_stiffness_matrix()
        assert k_local.shape == (6, 6)
        assert np.allclose(k_local, k_local.T, atol=1e-12)

        k_global = sample_element.global_stiffness_matrix()
        assert k_global.shape == (6, 6)
        assert np.allclose(k_global, k_global.T, atol=1e-12)

    def test_transformation_orthogonality(self, sample_element):
        T = sample_element.transformation_matrix()
        assert T.shape == (6, 6)
        # Orthogonal matrix satisfies T @ T.T = I
        identity = np.eye(6)
        assert np.allclose(T @ T.T, identity, atol=1e-14)
        assert np.allclose(T.T @ T, identity, atol=1e-14)

    def test_strain_energy_decomposition(self, sample_element):
        """Verify that total strain energy strictly equals axial + bending energy."""
        # Arbitrary non-trivial nodal displacement vector
        u_elem = np.array([0.001, -0.002, 0.0005, 0.003, 0.001, -0.0004], dtype=np.float64)
        forces = sample_element.compute_internal_forces(u_elem)

        # Global strain energy = 0.5 * u.T @ K_global @ u
        K_global = sample_element.global_stiffness_matrix()
        U_from_matrix = 0.5 * float(u_elem @ K_global @ u_elem)

        assert forces.strain_energy_axial >= 0.0
        assert forces.strain_energy_bending >= 0.0
        assert np.isclose(forces.strain_energy_total, U_from_matrix, rtol=1e-12)
        assert np.isclose(
            forces.strain_energy_total,
            forces.strain_energy_axial + forces.strain_energy_bending,
            rtol=1e-14,
        )

