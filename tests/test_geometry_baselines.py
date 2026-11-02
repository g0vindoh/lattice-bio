"""Unit tests for baseline generators, clipping, and exact volume fraction matching."""

import pytest
import numpy as np

from src.geometry.domain import DesignDomain2D, compute_volume_fraction
from src.geometry.ground_structure import (
    generate_ground_structure,
    build_fe_mesh_from_ground_structure,
)
from src.geometry.honeycomb import generate_honeycomb, build_fe_mesh_from_honeycomb
from src.geometry.voronoi import generate_voronoi_lattice, build_fe_mesh_from_voronoi
from src.geometry.pads import BoundaryPadConfig
from src.fem.load_cases import LoadCase
from src.fem.material import Material, RectangularSection


class TestBaselineGenerators:
    """Test geometric construction, clipping, and volume fraction matching for all baselines."""

    @pytest.fixture
    def domain(self):
        return DesignDomain2D(width=3.0, height=1.0)

    @pytest.fixture
    def material(self):
        return Material(E=2.1e11, density=7850.0)

    @pytest.mark.parametrize("target_vf", [0.2, 0.3, 0.4])
    def test_honeycomb_volume_fraction_matching(self, domain, material, target_vf):
        """Verify honeycomb wall thickness scaling matches target volume fraction exactly."""
        hc_data = generate_honeycomb(domain=domain, cell_size=0.25)

        # 1. Without pads
        mesh_no_pad, t_no_pad = build_fe_mesh_from_honeycomb(
            hc=hc_data,
            target_volume_fraction=target_vf,
            material=material,
            fixed_pad_volume=0.0,
        )
        vf_computed = compute_volume_fraction(mesh_no_pad, domain)
        assert np.isclose(vf_computed, target_vf, rtol=1e-6)

        # 2. With pads
        fixed_pad_vol = 0.05 * domain.bounding_volume  # 5% pad budget
        mesh_pad, t_pad = build_fe_mesh_from_honeycomb(
            hc=hc_data,
            target_volume_fraction=target_vf,
            material=material,
            fixed_pad_volume=fixed_pad_vol,
        )
        total_vol = mesh_pad.total_volume + fixed_pad_vol
        vf_with_pad = total_vol / domain.bounding_volume
        assert np.isclose(vf_with_pad, target_vf, rtol=1e-6)

    @pytest.mark.parametrize("target_vf", [0.2, 0.3, 0.4])
    def test_voronoi_volume_fraction_matching(self, domain, material, target_vf):
        """Verify Voronoi strut thickness scaling matches target volume fraction exactly."""
        vor_data = generate_voronoi_lattice(domain=domain, num_seeds=35, seed=123)

        mesh, t = build_fe_mesh_from_voronoi(
            vor_data=vor_data,
            target_volume_fraction=target_vf,
            material=material,
            fixed_pad_volume=0.0,
        )
        vf_computed = compute_volume_fraction(mesh, domain)
        assert np.isclose(vf_computed, target_vf, rtol=1e-6)

    @pytest.mark.parametrize("target_vf", [0.2, 0.3, 0.4])
    def test_ground_structure_uniform_volume_fraction_matching(self, domain, material, target_vf):
        """Verify uniformly thinned ground structure matches target volume fraction exactly."""
        gs = generate_ground_structure(domain=domain, nx=13, ny=5)

        # Thickness: t = V_target / L_total
        V_target = target_vf * domain.bounding_volume
        t_uniform = V_target / (domain.depth * gs.total_length)
        sec = RectangularSection(t=t_uniform, b=domain.depth)

        mesh = build_fe_mesh_from_ground_structure(
            gs=gs,
            sections=[sec] * gs.num_members,
            material=material,
        )
        vf_computed = compute_volume_fraction(mesh, domain)
        assert np.isclose(vf_computed, target_vf, rtol=1e-6)

    def test_all_nodes_within_domain_bounds(self, domain):
        """Verify clipping ensures all nodes lie within domain boundaries [0, W] x [0, H]."""
        hc = generate_honeycomb(domain=domain, cell_size=0.3)
        for nid, (x, y) in hc.nodes.items():
            assert domain.is_inside(x, y, tol=1e-5), f"Honeycomb node {nid} ({x}, {y}) outside bounds."

        vor = generate_voronoi_lattice(domain=domain, num_seeds=40, seed=42)
        for nid, (x, y) in vor.nodes.items():
            assert domain.is_inside(x, y, tol=1e-5), f"Voronoi node {nid} ({x}, {y}) outside bounds."

    def test_pad_attachment_for_all_load_cases(self, domain, material):
        """Verify pad attachment and load application for LC1, LC2, and LC3."""
        lc1 = LoadCase(id="LC1", name="Cantilever", load_type="cantilever")
        lc2 = LoadCase(id="LC2", name="SimplySupported", load_type="simply_supported")
        lc3 = LoadCase(id="LC3", name="CompressionBlock", load_type="compression_block")

        hc = generate_honeycomb(domain=domain, cell_size=0.3)

        for lc in [lc1, lc2, lc3]:
            mesh, _ = build_fe_mesh_from_honeycomb(hc, 0.2, material)
            pad_nodes, pad_elems, pad_vol = lc.apply(mesh, domain, material)
            assert len(pad_nodes) >= 0
            assert len(pad_elems) > 0
            assert pad_vol > 0.0
            assert len(mesh.boundary_conditions) > 0
            assert len(mesh.point_loads) > 0

