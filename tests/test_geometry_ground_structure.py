"""Unit tests for ground structure generator, duplicate filtering, and collinear removal."""

import pytest
import numpy as np

from src.geometry.domain import DesignDomain2D, compute_volume_fraction
from src.geometry.ground_structure import (
    generate_ground_structure,
    build_fe_mesh_from_ground_structure,
    is_node_strictly_between,
)
from src.fem.material import Material, RectangularSection


class TestGroundStructure:
    """Test candidate member generation and collinear filtering."""

    def test_hand_verifiable_3x3_grid(self):
        """Hand-checkable 3x3 grid member count.

        For a 3x3 grid with infinite radius (all pairwise connections considered):
        - 6 horizontal unit segments
        - 6 vertical unit segments
        - 8 diagonal (1x1) segments
        - 8 knight's move (2x1 and 1x2) segments
        Total valid members = 6 + 6 + 8 + 8 = 28.
        All span-2 horizontals (3), span-2 verticals (3), and corner-to-corner span-2 diagonals (2)
        must be filtered out because they pass collinearly through an intermediate node.
        """
        domain = DesignDomain2D(width=2.0, height=2.0)
        # Large radius to consider all node pairs
        gs = generate_ground_structure(domain=domain, nx=3, ny=3, radius=10.0)

        assert gs.num_nodes == 9
        assert gs.num_members == 28

        # Verify no member connects identical nodes
        for u, v in gs.members:
            assert u < v

        # Verify no duplicate members
        assert len(gs.members) == len(set(gs.members))

    def test_is_node_strictly_between(self):
        """Test collinear point detection."""
        p_a = (0.0, 0.0)
        p_b = (2.0, 0.0)

        assert is_node_strictly_between(p_a, p_b, (1.0, 0.0))
        assert not is_node_strictly_between(p_a, p_b, (0.0, 0.0))  # Endpoint A
        assert not is_node_strictly_between(p_a, p_b, (2.0, 0.0))  # Endpoint B
        assert not is_node_strictly_between(p_a, p_b, (3.0, 0.0))  # Outside segment
        assert not is_node_strictly_between(p_a, p_b, (1.0, 0.5))  # Off-line

    def test_standard_13x5_grid(self):
        """Verify the standard 13x5 ground structure generation."""
        domain = DesignDomain2D(width=3.0, height=1.0)
        gs = generate_ground_structure(domain=domain, nx=13, ny=5)

        assert gs.num_nodes == 65
        assert gs.num_members > 100

        # Verify no duplicates
        unique_members = set(gs.members)
        assert len(unique_members) == len(gs.members)

        # Verify that no member passes through an intermediate grid node
        node_coords = gs.nodes
        for n_a, n_b in gs.members:
            pa = node_coords[n_a]
            pb = node_coords[n_b]
            for n_c, pc in node_coords.items():
                if n_c != n_a and n_c != n_b:
                    assert not is_node_strictly_between(pa, pb, pc), (
                        f"Member ({n_a}, {n_b}) overlaps with node {n_c}."
                    )

    def test_build_fe_mesh_from_ground_structure(self):
        """Verify FE mesh construction from ground structure and section scaling."""
        domain = DesignDomain2D(width=3.0, height=1.0)
        gs = generate_ground_structure(domain=domain, nx=5, ny=3)
        mat = Material(E=2.1e11)
        sec = RectangularSection(t=0.01)

        mesh = build_fe_mesh_from_ground_structure(
            gs=gs,
            sections=[sec] * gs.num_members,
            material=mat,
        )

        assert mesh.num_nodes == gs.num_nodes
        assert mesh.num_elements == gs.num_members
        vf = compute_volume_fraction(mesh, domain)
        assert vf > 0.0

