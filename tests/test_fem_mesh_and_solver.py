"""Unit tests for FrameMesh2D, postprocessing, and solver metrics."""

import pytest
import numpy as np

from src.fem.material import Material, RectangularSection
from src.fem.mesh import FrameMesh2D, BoundaryCondition, PointLoad
from src.fem.solver import FrameSolver2D
from src.fem.postprocess import (
    member_results_to_dataframe,
    nodal_results_to_dataframe,
    summarize_results,
)


class TestMeshAndPostprocessing:
    """Test mesh operations, boundary condition assignment, and postprocessing outputs."""

    def test_mesh_construction(self):
        mat = Material(E=2.1e11, density=7850.0)
        sec = RectangularSection(t=0.02, b=1.0)

        mesh = FrameMesh2D()
        mesh.add_node(10, 0.0, 0.0)
        mesh.add_node(20, 1.0, 0.0)
        mesh.add_element(100, 10, 20, mat, sec)

        assert mesh.num_nodes == 2
        assert mesh.num_elements == 1
        assert mesh.num_dofs == 6
        assert np.isclose(mesh.total_volume, 0.02 * 1.0 * 1.0)
        assert np.isclose(mesh.total_mass, 7850.0 * 0.02 * 1.0 * 1.0)

    def test_duplicate_node_raises(self):
        mesh = FrameMesh2D()
        mesh.add_node(1, 0.0, 0.0)
        with pytest.raises(ValueError, match="already exists"):
            mesh.add_node(1, 1.0, 1.0)

    def test_invalid_dof_raises(self):
        mesh = FrameMesh2D()
        mesh.add_node(1, 0.0, 0.0)
        with pytest.raises(ValueError, match="Invalid DOF identifier"):
            mesh.add_boundary_condition(1, "invalid_dof")

    def test_postprocessing_dataframes(self):
        mat = Material(E=2.1e11, density=7850.0)
        sec = RectangularSection(t=0.02)

        mesh = FrameMesh2D()
        mesh.add_node(1, 0.0, 0.0)
        mesh.add_node(2, 2.0, 0.0)
        mesh.add_element(1, 1, 2, mat, sec)

        mesh.add_fixed_support(1)
        mesh.add_load(2, fx=1000.0, fy=500.0, m=100.0)

        solver = FrameSolver2D(mesh)
        res = solver.solve()

        df_members = member_results_to_dataframe(res)
        assert len(df_members) == 1
        assert "axial_force" in df_members.columns
        assert "bending_moment_1" in df_members.columns
        assert "strain_energy_axial" in df_members.columns
        assert "strain_energy_bending" in df_members.columns

        df_nodes = nodal_results_to_dataframe(res)
        assert len(df_nodes) == 2
        assert "ux" in df_nodes.columns
        assert "rx" in df_nodes.columns

        summary = summarize_results(res)
        assert summary["num_nodes"] == 2
        assert summary["num_elements"] == 1
        assert "compliance_J" in summary
        assert "axial_energy_fraction" in summary
        assert "bending_energy_fraction" in summary

