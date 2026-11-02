"""Unit tests for 3D-printable CAD / STL exporter."""

import os
import struct
import pytest

from src.fem.material import Material, RectangularSection
from src.fem.mesh import FrameMesh2D
from src.export.stl_exporter import export_lattice_to_stl


class TestSTLExporter:
    """Validation of watertight 3D solid STL generation for additive manufacturing."""

    def test_export_ascii_stl(self, tmp_path):
        """Verify ASCII STL file structure, facet counts, and syntax."""
        mesh = FrameMesh2D()
        mat = Material(E=210e9, density=7850)
        sec = RectangularSection(t=0.01)

        # Single member between (0, 0) and (1, 0)
        mesh.add_node(1, 0.0, 0.0)
        mesh.add_node(2, 1.0, 0.0)
        mesh.add_element(1, 1, 2, mat, sec)

        out_file = str(tmp_path / "test_strut.stl")
        n_facets = export_lattice_to_stl(mesh, out_file, out_of_plane_depth=0.01, binary=False)

        assert n_facets > 0
        assert os.path.exists(out_file)

        with open(out_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert lines[0].strip().startswith("solid")
        assert lines[-1].strip().startswith("endsolid")
        assert any("facet normal" in line for line in lines)
        assert any("vertex" in line for line in lines)

    def test_export_binary_stl(self, tmp_path):
        """Verify binary STL file header, facet count, and exact byte size."""
        mesh = FrameMesh2D()
        mat = Material(E=210e9, density=7850)
        sec = RectangularSection(t=0.01)

        # Triangle of 3 members
        mesh.add_node(1, 0.0, 0.0)
        mesh.add_node(2, 1.0, 0.0)
        mesh.add_node(3, 0.5, 0.866)
        mesh.add_element(1, 1, 2, mat, sec)
        mesh.add_element(2, 2, 3, mat, sec)
        mesh.add_element(3, 3, 1, mat, sec)

        out_bin = str(tmp_path / "test_triangle.stl")
        n_facets = export_lattice_to_stl(mesh, out_bin, out_of_plane_depth=0.01, binary=True)

        assert n_facets > 0
        assert os.path.exists(out_bin)

        # Binary STL exact size: 80 bytes header + 4 bytes facet count + 50 bytes per facet
        expected_size = 80 + 4 + n_facets * 50
        actual_size = os.path.getsize(out_bin)
        assert actual_size == expected_size

        with open(out_bin, "rb") as f:
            header = f.read(80)
            count = struct.unpack("<I", f.read(4))[0]

        assert count == n_facets
        assert b"Lattice-Bio" in header
