"""Tests for 2D SIMP Continuum Topology Optimization Baseline."""

import numpy as np
import pytest

from src.geometry.domain import DesignDomain2D
from src.optimizers.simp import (
    SIMPTopologyOptimizer,
    compute_q4_element_stiffness,
)


class TestSIMPOptimizer:
    """Test suite for 2D SIMP topology optimization baseline."""

    def test_q4_stiffness_matrix_properties(self):
        hx, hy = 0.1, 0.05
        k0 = compute_q4_element_stiffness(hx, hy, E=1.0, nu=0.3)

        assert k0.shape == (8, 8)
        # Symmetry: k0 == k0.T
        assert np.allclose(k0, k0.T, atol=1e-10)

        # Eigenvalues: exactly 3 zero eigenvalues for rigid body modes (2 translations, 1 rotation)
        eigenvalues = np.linalg.eigvalsh(k0)
        max_eig = np.max(eigenvalues)
        normalized_eigs = eigenvalues / max_eig
        assert np.sum(np.isclose(normalized_eigs, 0.0, atol=1e-8)) == 3
        # All remaining 5 eigenvalues are strictly positive elastic deformation modes
        assert np.all(normalized_eigs[3:] > 1e-4)

    def test_simp_optimization_convergence(self):
        domain = DesignDomain2D(width=1.2, height=0.4, depth=1.0)
        optimizer = SIMPTopologyOptimizer(
            domain=domain,
            nx=12,
            ny=4,
            target_volume_fraction=0.35,
            penalization=3.0,
            max_iterations=6,
            tolerance=1e-4,
        )

        res = optimizer.optimize(load_case_name="cantilever")

        assert res.densities.shape == (4, 12)
        assert res.volume_fraction == pytest.approx(0.35, abs=0.03)
        assert res.compliance > 0.0
        assert len(res.compliance_history) >= 2
        # Density field should have evolved
        assert len(res.change_history) >= 1
        assert res.solve_time_s >= 0.0

    def test_simp_simply_supported_boundary(self):
        domain = DesignDomain2D(width=1.0, height=0.5, depth=1.0)
        optimizer = SIMPTopologyOptimizer(
            domain=domain,
            nx=10,
            ny=5,
            target_volume_fraction=0.4,
            max_iterations=3,
        )

        res = optimizer.optimize(load_case_name="simply_supported")
        assert res.compliance > 0.0
        assert np.all(res.densities >= 0.0)
        assert np.all(res.densities <= 1.0)
