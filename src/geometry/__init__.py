"""Geometry generation module (domain, ground structures, honeycombs, Voronoi, pads)."""

from .domain import DesignDomain2D, compute_volume_fraction
from .ground_structure import (
    GroundStructureData,
    generate_ground_structure,
    build_fe_mesh_from_ground_structure,
)
from .honeycomb import (
    HoneycombData,
    generate_honeycomb,
    build_fe_mesh_from_honeycomb,
)
from .voronoi import (
    VoronoiData,
    generate_voronoi_lattice,
    build_fe_mesh_from_voronoi,
)
from .pads import (
    BoundaryPadConfig,
    attach_pads_and_bcs_lc1,
    attach_pads_and_bcs_lc2,
    attach_pads_and_bcs_lc3,
)

__all__ = [
    "DesignDomain2D",
    "compute_volume_fraction",
    "GroundStructureData",
    "generate_ground_structure",
    "build_fe_mesh_from_ground_structure",
    "HoneycombData",
    "generate_honeycomb",
    "build_fe_mesh_from_honeycomb",
    "VoronoiData",
    "generate_voronoi_lattice",
    "build_fe_mesh_from_voronoi",
    "BoundaryPadConfig",
    "attach_pads_and_bcs_lc1",
    "attach_pads_and_bcs_lc2",
    "attach_pads_and_bcs_lc3",
]
