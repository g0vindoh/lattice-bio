"""Regular hexagonal honeycomb lattice generator clipped to rectangular domain."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from .domain import DesignDomain2D
from ..fem.material import Material, RectangularSection
from ..fem.mesh import FrameMesh2D


def clip_segment_liang_barsky(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Clip a 2D line segment to a bounding box using the Liang-Barsky algorithm.

    Returns:
        Clipped ((x1, y1), (x2, y2)) or None if the segment lies entirely outside.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    p = [-dx, dx, -dy, dy]
    q = [p1[0] - xmin, xmax - p1[0], p1[1] - ymin, ymax - p1[1]]

    u1 = 0.0
    u2 = 1.0

    for pi, qi in zip(p, q):
        if abs(pi) < 1e-13:
            if qi < 0.0:
                return None
        else:
            t = qi / pi
            if pi < 0.0:
                if t > u2:
                    return None
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return None
                if t < u2:
                    u2 = t

    if u1 > u2:
        return None

    c1 = (float(p1[0] + u1 * dx), float(p1[1] + u1 * dy))
    c2 = (float(p1[0] + u2 * dx), float(p1[1] + u2 * dy))

    # Reject zero-length clipped remnants
    if np.hypot(c2[0] - c1[0], c2[1] - c1[1]) < 1e-6:
        return None

    return (c1, c2)


@dataclass
class HoneycombData:
    """Raw geometry data of a clipped regular hexagonal honeycomb.

    Attributes:
        nodes: Map of node_id -> (x, y) coordinates.
        members: List of unique member node-ID pairs (node_i, node_j).
        member_lengths: Array of member lengths (meters).
        cell_size: Characteristic cell wall length l (meters).
        domain: Associated DesignDomain2D.
    """

    nodes: Dict[int, Tuple[float, float]]
    members: List[Tuple[int, int]]
    member_lengths: np.ndarray
    cell_size: float
    domain: DesignDomain2D

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_members(self) -> int:
        return len(self.members)

    @property
    def total_length(self) -> float:
        return float(np.sum(self.member_lengths))


def generate_honeycomb(
    domain: DesignDomain2D,
    cell_size: float = 0.25,
) -> HoneycombData:
    """Generate a regular hexagonal honeycomb lattice tiled and clipped to the domain.

    Tessellation geometry:
        Hexagon cell wall length = cell_size = l.
        Horizontal column spacing = 1.5 * l.
        Vertical row spacing = sqrt(3) * l.
        Every vertex has degree 3 with interior angles of 120 degrees.

    Args:
        domain: DesignDomain2D bounding box [0, W] x [0, H].
        cell_size: Hexagon edge length l (meters). Must be positive.

    Returns:
        HoneycombData instance.
    """
    if cell_size <= 0.0:
        raise ValueError(f"cell_size must be positive, got {cell_size}.")

    l = float(cell_size)
    dx_col = 1.5 * l
    dy_row = np.sqrt(3.0) * l

    xmin, xmax, ymin, ymax = domain.bounds

    # Extend tile generation slightly beyond domain to ensure clean edge clipping
    margin = 2.0 * l
    cols = int(np.ceil((domain.width + 2.0 * margin) / dx_col)) + 2
    rows = int(np.ceil((domain.height + 2.0 * margin) / dy_row)) + 2

    start_x = -margin
    start_y = -margin

    raw_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    # Angle offsets for the 6 vertices of a regular hexagon
    angles = np.array([k * np.pi / 3.0 for k in range(6)])

    for col in range(cols):
        xc = start_x + col * dx_col
        y_shift = (col % 2) * (dy_row / 2.0)
        for row in range(rows):
            yc = start_y + row * dy_row + y_shift
            # Compute 6 vertices
            v_x = xc + l * np.cos(angles)
            v_y = yc + l * np.sin(angles)

            for k in range(6):
                p1 = (v_x[k], v_y[k])
                p2 = (v_x[(k + 1) % 6], v_y[(k + 1) % 6])
                raw_segments.append((p1, p2))

    # Clip all segments against domain bounding box
    clipped_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for p1, p2 in raw_segments:
        clipped = clip_segment_liang_barsky(p1, p2, xmin, xmax, ymin, ymax)
        if clipped is not None:
            clipped_segments.append(clipped)

    # Discretize and deduplicate vertices and edges
    # Round coordinates to tolerance grid to merge shared vertices
    vertex_map: Dict[Tuple[float, float], int] = {}
    nodes: Dict[int, Tuple[float, float]] = {}
    edges_set: Set[Tuple[int, int]] = set()
    node_id = 1

    def get_vertex_id(pt: Tuple[float, float]) -> int:
        nonlocal node_id
        # Snap to 5 decimal places (~10 microns)
        key = (round(pt[0], 5), round(pt[1], 5))
        if key not in vertex_map:
            vertex_map[key] = node_id
            nodes[node_id] = (float(pt[0]), float(pt[1]))
            node_id += 1
        return vertex_map[key]

    for c1, c2 in clipped_segments:
        v1 = get_vertex_id(c1)
        v2 = get_vertex_id(c2)
        if v1 != v2:
            edge = (min(v1, v2), max(v1, v2))
            edges_set.add(edge)

    members = sorted(edges_set)
    member_lengths = np.array(
        [
            np.hypot(
                nodes[n2][0] - nodes[n1][0],
                nodes[n2][1] - nodes[n1][1],
            )
            for n1, n2 in members
        ],
        dtype=np.float64,
    )

    return HoneycombData(
        nodes=nodes,
        members=members,
        member_lengths=member_lengths,
        cell_size=cell_size,
        domain=domain,
    )


def build_fe_mesh_from_honeycomb(
    hc: HoneycombData,
    target_volume_fraction: float,
    material: Material,
    fixed_pad_volume: float = 0.0,
) -> Tuple[FrameMesh2D, float]:
    """Construct a FrameMesh2D from HoneycombData with thickness scaled to target volume fraction.

    Volume Formula:
        V_target = target_volume_fraction * W * H * depth
        V_lattice = V_target - fixed_pad_volume
        t = V_lattice / (sum(L_i) * depth)

    Returns:
        (FrameMesh2D, scaled_thickness)
    """
    V_target = target_volume_fraction * hc.domain.bounding_volume
    V_lattice = max(1e-12, V_target - fixed_pad_volume)

    total_len = hc.total_length
    if total_len <= 1e-14:
        raise ValueError("Honeycomb has zero total length.")

    # With rectangular section of unit depth b = domain.depth:
    # A = b * t -> V = b * t * L_total -> t = V / (b * L_total)
    thickness = float(V_lattice / (hc.domain.depth * total_len))
    section = RectangularSection(t=thickness, b=hc.domain.depth)

    mesh = FrameMesh2D()
    for nid, (x, y) in hc.nodes.items():
        mesh.add_node(nid, x, y)

    for idx, (n1, n2) in enumerate(hc.members):
        mesh.add_element(
            element_id=idx + 1,
            node_i=n1,
            node_j=n2,
            material=material,
            section=section,
        )

    return mesh, thickness

