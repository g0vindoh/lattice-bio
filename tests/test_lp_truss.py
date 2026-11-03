"""Tests for Classic Ground Structure Truss Optimization via Linear Programming."""

import numpy as np
import pytest

from src.fem.material import Material
from src.fem.load_cases import LoadCase
from src.geometry.domain import DesignDomain2D
from src.geometry.ground_structure import generate_ground_structure
from src.optimizers.lp_truss import (
    build_nodal_equilibrium_matrix,
    optimize_ground_structure_truss_lp,
)


class TestLPTrussOptimization:
    """Test suite for Ground Structure LP optimization."""

    @pytest.fixture
    def setup_domain_and_gs(self):
        domain = DesignDomain2D(width=3.0, height=1.0, depth=1.0)
        gs = generate_ground_structure(domain=domain, nx=7, ny=3, radius=1.5)
        mat = Material(name="StandardTestMat", E=1e9, density=1000.0)
        return domain, gs, mat

    def test_equilibrium_matrix_assembly(self, setup_domain_and_gs):
        domain, gs, mat = setup_domain_and_gs
        lc = LoadCase(id="LC1", name="Cantilever", load_type="cantilever")

        B, F_free, free_dofs, fixed_dofs = build_nodal_equilibrium_matrix(gs, lc, domain)

        assert B.shape[0] == len(free_dofs)
        assert B.shape[1] == gs.num_members
        assert len(F_free) == len(free_dofs)
        assert len(free_dofs) + len(fixed_dofs) == 2 * gs.num_nodes
        assert np.linalg.norm(F_free) > 0.0

    def test_lp_truss_cantilever_solve(self, setup_domain_and_gs):
        domain, gs, mat = setup_domain_and_gs
        lc = LoadCase(id="LC1", name="Cantilever", load_type="cantilever")

        result = optimize_ground_structure_truss_lp(
            ground_structure=gs,
            load_case=lc,
            material=mat,
            target_volume_fraction=0.3,
        )

        assert result.solver_status in ("optimal", "Optimization terminated successfully.")
        assert result.num_active_members > 0
        assert result.achieved_volume_fraction == pytest.approx(0.3, rel=1e-2)
        assert result.frame_compliance > 0.0
        assert result.frame_stiffness > 0.0
        # Since the LP finds an optimal truss, axial mechanics should heavily dominate
        assert result.axial_energy_share > 0.70
        assert result.solve_time_s >= 0.0

    def test_lp_truss_simply_supported_solve(self, setup_domain_and_gs):
        domain, gs, mat = setup_domain_and_gs
        lc = LoadCase(id="LC2", name="Simply Supported", load_type="simply_supported")

        result = optimize_ground_structure_truss_lp(
            ground_structure=gs,
            load_case=lc,
            material=mat,
            target_volume_fraction=0.25,
        )

        assert result.solver_status in ("optimal", "Optimization terminated successfully.")
        assert result.num_active_members > 0
        assert result.achieved_volume_fraction == pytest.approx(0.25, rel=1e-2)
        assert result.frame_compliance > 0.0
