"""Ground structure candidate member generator with collinear overlap filtering."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np

from .domain import DesignDomain2D
from ..fem.material import Material, RectangularSection
from ..fem.mesh import FrameMesh2D


@dataclass
class GroundStructureData:
    """Represents a discrete ground structure network.

    Attributes:
        nodes: Map of node_id -> (x, y) coordinates.
        members: List of member tuples (node_i, node_j) with node_i < node_j.
        member_lengths: Array of member lengths (meters).
        nx: Number of grid nodes along x.
        ny: Number of grid nodes along y.
        radius: Candidate member connection radius (meters).
        domain: Associated DesignDomain2D.
    """

    nodes: Dict[int, Tuple[float, float]]
    members: List[Tuple[int, int]]
    member_lengths: np.ndarray
    nx: int
    ny: int
    radius: float
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


def is_node_strictly_between(
    p_a: Tuple[float, float],
    p_b: Tuple[float, float],
    p_c: Tuple[float, float],
    tol_cross: float = 1e-8,
    tol_alpha: float = 1e-5,
) -> bool:
    """Check if point C lies strictly in the interior of segment AB.

    Collinearity test: cross product (B - A) x (C - A) == 0.
    Interior test: parameter alpha = (C - A) . (B - A) / |B - A|^2 in (0, 1).
    """
    ux = p_b[0] - p_a[0]
    uy = p_b[1] - p_a[1]
    vx = p_c[0] - p_a[0]
    vy = p_c[1] - p_a[1]

    # Cross product in 2D
    cross = abs(ux * vy - uy * vx)
    u_sq = ux * ux + uy * uy
    if u_sq <= 1e-14:
        return False

    # Check perpendicular distance to line
    perp_dist = cross / np.sqrt(u_sq)
    if perp_dist > tol_cross:
        return False

    # Projection parameter
    alpha = (vx * ux + vy * uy) / u_sq
    return bool(tol_alpha < alpha < (1.0 - tol_alpha))


def generate_ground_structure(
    domain: DesignDomain2D,
    nx: int = 13,
    ny: int = 5,
    radius: Optional[float] = None,
) -> GroundStructureData:
    """Generate a regular grid ground structure without duplicate or collinear overlapping members.

    Args:
        domain: Rectangular design domain.
        nx: Number of grid points along x (default 13).
        ny: Number of grid points along y (default 5).
        radius: Connection radius. If None, defaults to include diagonals and
                knight's move neighbors (e.g. 1.5 * max(dx, dy) * sqrt(2)).

    Returns:
        GroundStructureData instance.
    """
    if nx < 2 or ny < 2:
        raise ValueError(f"Grid dimensions must be at least 2x2, got nx={nx}, ny={ny}.")

    dx = domain.width / (nx - 1)
    dy = domain.height / (ny - 1)

    if radius is None:
        # Default radius covers horizontal, vertical, diagonal, and knight's moves
        diag_step = np.hypot(dx, dy)
        radius = float(diag_step * 1.55)

    # 1. Create nodes
    nodes: Dict[int, Tuple[float, float]] = {}
    node_grid: Dict[Tuple[int, int], int] = {}
    node_id = 1
    for j in range(ny):
        y = j * dy
        for i in range(nx):
            x = i * dx
            nodes[node_id] = (x, y)
            node_grid[(i, j)] = node_id
            node_id += 1

    # 2. Candidate members
    raw_members: List[Tuple[int, int]] = []
    node_list = sorted(nodes.keys())
    n_total = len(node_list)

    # Convert coordinates to array for fast pairwise distance checking
    coords = np.array([nodes[nid] for nid in node_list], dtype=np.float64)

    for i in range(n_total):
        nid_a = node_list[i]
        p_a = coords[i]
        for j in range(i + 1, n_total):
            nid_b = node_list[j]
            p_b = coords[j]
            dist = float(np.hypot(p_b[0] - p_a[0], p_b[1] - p_a[1]))
            if dist <= radius + 1e-9:
                raw_members.append((nid_a, nid_b))

    # 3. Collinear filtering: remove any member AB if another grid node C lies strictly on AB
    filtered_members: List[Tuple[int, int]] = []
    member_lengths: List[float] = []

    for nid_a, nid_b in raw_members:
        p_a = nodes[nid_a]
        p_b = nodes[nid_b]

        has_intermediate_node = False
        for nid_c in node_list:
            if nid_c == nid_a or nid_c == nid_b:
                continue
            p_c = nodes[nid_c]
            if is_node_strictly_between(p_a, p_b, p_c):
                has_intermediate_node = True
                break

        if not has_intermediate_node:
            filtered_members.append((nid_a, nid_b))
            member_lengths.append(float(np.hypot(p_b[0] - p_a[0], p_b[1] - p_a[1])))

    return GroundStructureData(
        nodes=nodes,
        members=filtered_members,
        member_lengths=np.array(member_lengths, dtype=np.float64),
        nx=nx,
        ny=ny,
        radius=radius,
        domain=domain,
    )


def build_fe_mesh_from_ground_structure(
    gs: GroundStructureData,
    sections: Union[Dict[int, RectangularSection], List[RectangularSection]],
    material: Material,
    active_mask: Optional[np.ndarray] = None,
) -> FrameMesh2D:
    """Construct a FrameMesh2D from a ground structure and member cross-sections.

    Args:
        gs: GroundStructureData.
        sections: Map or list of RectangularSection for each member (1-indexed or 0-indexed).
        material: Structural material.
        active_mask: Optional boolean array indicating presence/absence of each member.

    Returns:
        FrameMesh2D instance.
    """
    mesh = FrameMesh2D()
    active_nodes: Set[int] = set()
    valid_elements: List[Tuple[int, int, int, RectangularSection]] = []

    for idx, (n_i, n_j) in enumerate(gs.members):
        if active_mask is not None and not active_mask[idx]:
            continue

        if isinstance(sections, dict):
            sec = sections[idx + 1] if (idx + 1) in sections else sections[idx]
        else:
            sec = sections[idx]

        if sec.t > 1e-9:
            active_nodes.add(n_i)
            active_nodes.add(n_j)
            valid_elements.append((idx + 1, n_i, n_j, sec))

    # Add only nodes that belong to active structural elements
    # If no elements are active, add all nodes so mesh is not empty
    if not active_nodes:
        for nid, (x, y) in gs.nodes.items():
            mesh.add_node(nid, x, y)
    else:
        for nid in sorted(active_nodes):
            x, y = gs.nodes[nid]
            mesh.add_node(nid, x, y)

    for elem_id, n_i, n_j, sec in valid_elements:
        mesh.add_element(
            element_id=elem_id,
            node_i=n_i,
            node_j=n_j,
            material=material,
            section=sec,
        )

    return mesh

