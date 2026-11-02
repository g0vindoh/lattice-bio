"""Voronoi lattice generator with domain clipping, edge merging, and volume scaling."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.spatial import Voronoi

from .domain import DesignDomain2D
from .honeycomb import clip_segment_liang_barsky
from ..fem.material import Material, RectangularSection
from ..fem.mesh import FrameMesh2D


@dataclass
class VoronoiData:
    """Raw geometry data of a clipped Voronoi cellular lattice.

    Attributes:
        nodes: Map of node_id -> (x, y) coordinates.
        members: List of unique member node-ID pairs (node_i, node_j).
        member_lengths: Array of member lengths (meters).
        seed: Random seed used for generation.
        num_seeds: Number of seed points in the domain.
        domain: Associated DesignDomain2D.
    """

    nodes: Dict[int, Tuple[float, float]]
    members: List[Tuple[int, int]]
    member_lengths: np.ndarray
    seed: int
    num_seeds: int
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


def generate_voronoi_lattice(
    domain: DesignDomain2D,
    num_seeds: int = 50,
    seed: int = 42,
    lloyd_iterations: int = 1,
    min_edge_length_factor: float = 0.05,
) -> VoronoiData:
    """Generate a random Voronoi lattice clipped to the domain with short-edge merging.

    Geometry Pipeline:
    1. Seed random points within [0, W] x [0, H] using numpy RandomState(seed).
    2. Mirror seeds across the 4 bounding edges to enforce perpendicular boundary closure.
    3. Optional Lloyd relaxation iterations to homogenize cell sizes (trabecular-like).
    4. Compute Voronoi ridge graph and clip ridges to domain with Liang-Barsky.
    5. Merge micro-edges shorter than min_edge_length_factor * mean_edge_length.

    Args:
        domain: Rectangular design domain.
        num_seeds: Number of interior Voronoi cell seeds.
        seed: Random seed for exact reproducibility.
        lloyd_iterations: Number of Lloyd relaxation centroid updates (default 1).
        min_edge_length_factor: Fraction of average edge length below which edges are collapsed.

    Returns:
        VoronoiData instance.
    """
    if num_seeds < 4:
        raise ValueError(f"num_seeds must be at least 4, got {num_seeds}.")

    rng = np.random.RandomState(seed)
    xmin, xmax, ymin, ymax = domain.bounds
    W = domain.width
    H = domain.height

    # Generate initial random interior points
    pts = rng.uniform(
        low=[0.02 * W, 0.02 * H],
        high=[0.98 * W, 0.98 * H],
        size=(num_seeds, 2),
    )

    # Optional Lloyd relaxation iterations
    for _ in range(lloyd_iterations):
        # Mirror points across boundaries for relaxation
        pts_mirrored = [pts]
        pts_mirrored.append(np.column_stack([-pts[:, 0], pts[:, 1]]))
        pts_mirrored.append(np.column_stack([2.0 * W - pts[:, 0], pts[:, 1]]))
        pts_mirrored.append(np.column_stack([pts[:, 0], -pts[:, 1]]))
        pts_mirrored.append(np.column_stack([pts[:, 0], 2.0 * H - pts[:, 1]]))
        all_pts = np.vstack(pts_mirrored)

        vor_temp = Voronoi(all_pts)
        new_pts = []
        for i in range(num_seeds):
            region_idx = vor_temp.point_region[i]
            region = vor_temp.regions[region_idx]
            if not region or -1 in region:
                new_pts.append(pts[i])
                continue
            poly = vor_temp.vertices[region]
            # Clip polygon vertices roughly to bounds
            px = np.clip(poly[:, 0], 0.0, W)
            py = np.clip(poly[:, 1], 0.0, H)
            cx = float(np.mean(px))
            cy = float(np.mean(py))
            new_pts.append([cx, cy])
        pts = np.array(new_pts)

    # Final Voronoi with mirrored boundary seeds
    pts_mirrored = [pts]
    pts_mirrored.append(np.column_stack([-pts[:, 0], pts[:, 1]]))
    pts_mirrored.append(np.column_stack([2.0 * W - pts[:, 0], pts[:, 1]]))
    pts_mirrored.append(np.column_stack([pts[:, 0], -pts[:, 1]]))
    pts_mirrored.append(np.column_stack([pts[:, 0], 2.0 * H - pts[:, 1]]))
    all_pts = np.vstack(pts_mirrored)

    vor = Voronoi(all_pts)

    # Collect finite ridges and clip to domain
    clipped_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for v1, v2 in vor.ridge_vertices:
        if v1 < 0 or v2 < 0:
            continue
        p1 = (float(vor.vertices[v1, 0]), float(vor.vertices[v1, 1]))
        p2 = (float(vor.vertices[v2, 0]), float(vor.vertices[v2, 1]))
        clipped = clip_segment_liang_barsky(p1, p2, xmin, xmax, ymin, ymax)
        if clipped is not None:
            clipped_segments.append(clipped)

    # Build unique node dictionary
    vertex_map: Dict[Tuple[float, float], int] = {}
    nodes: Dict[int, Tuple[float, float]] = {}
    node_id = 1

    def get_vertex_id(pt: Tuple[float, float]) -> int:
        nonlocal node_id
        key = (round(pt[0], 5), round(pt[1], 5))
        if key not in vertex_map:
            vertex_map[key] = node_id
            nodes[node_id] = (float(pt[0]), float(pt[1]))
            node_id += 1
        return vertex_map[key]

    raw_edges: Set[Tuple[int, int]] = set()
    for c1, c2 in clipped_segments:
        u = get_vertex_id(c1)
        v = get_vertex_id(c2)
        if u != v:
            raw_edges.add((min(u, v), max(u, v)))

    # Calculate average edge length to determine short-edge threshold
    raw_edge_list = list(raw_edges)
    lengths = [
        np.hypot(nodes[v][0] - nodes[u][0], nodes[v][1] - nodes[u][1])
        for u, v in raw_edge_list
    ]
    mean_length = float(np.mean(lengths)) if lengths else 1.0
    min_length_thresh = min_edge_length_factor * mean_length

    # Short edge merging via Union-Find (Disjoint-Set)
    parent = {nid: nid for nid in nodes}

    def find(i: int) -> int:
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int) -> None:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for u, v in raw_edge_list:
        dist = np.hypot(nodes[v][0] - nodes[u][0], nodes[v][1] - nodes[u][1])
        if dist < min_length_thresh:
            union(u, v)

    groups: Dict[int, List[int]] = {}
    for nid in nodes:
        root = find(nid)
        groups.setdefault(root, []).append(nid)

    final_nodes: Dict[int, Tuple[float, float]] = {}
    node_remap: Dict[int, int] = {}
    new_id = 1
    for root, cluster in sorted(groups.items()):
        coords = np.array([nodes[m] for m in cluster], dtype=np.float64)
        avg_coord = (float(np.mean(coords[:, 0])), float(np.mean(coords[:, 1])))
        final_nodes[new_id] = avg_coord
        for m in cluster:
            node_remap[m] = new_id
        new_id += 1

    final_edges: Set[Tuple[int, int]] = set()
    for u, v in raw_edge_list:
        nu = node_remap[u]
        nv = node_remap[v]
        if nu != nv:
            final_edges.add((min(nu, nv), max(nu, nv)))

    members = sorted(final_edges)
    member_lengths = np.array(
        [
            np.hypot(
                final_nodes[n2][0] - final_nodes[n1][0],
                final_nodes[n2][1] - final_nodes[n1][1],
            )
            for n1, n2 in members
        ],
        dtype=np.float64,
    )

    return VoronoiData(
        nodes=final_nodes,
        members=members,
        member_lengths=member_lengths,
        seed=seed,
        num_seeds=num_seeds,
        domain=domain,
    )


def build_fe_mesh_from_voronoi(
    vor_data: VoronoiData,
    target_volume_fraction: float,
    material: Material,
    fixed_pad_volume: float = 0.0,
) -> Tuple[FrameMesh2D, float]:
    """Construct a FrameMesh2D from VoronoiData with thickness scaled to target volume fraction.

    Volume Formula:
        V_target = target_volume_fraction * W * H * depth
        V_lattice = V_target - fixed_pad_volume
        t = V_lattice / (sum(L_i) * depth)

    Returns:
        (FrameMesh2D, scaled_thickness)
    """
    V_target = target_volume_fraction * vor_data.domain.bounding_volume
    V_lattice = max(1e-12, V_target - fixed_pad_volume)

    total_len = vor_data.total_length
    if total_len <= 1e-14:
        raise ValueError("Voronoi network has zero total length.")

    thickness = float(V_lattice / (vor_data.domain.depth * total_len))
    section = RectangularSection(t=thickness, b=vor_data.domain.depth)

    mesh = FrameMesh2D()
    for nid, (x, y) in vor_data.nodes.items():
        mesh.add_node(nid, x, y)

    for idx, (n1, n2) in enumerate(vor_data.members):
        mesh.add_element(
            element_id=idx + 1,
            node_i=n1,
            node_j=n2,
            material=material,
            section=section,
        )

    return mesh, thickness

