"""3D-Printable Solid CAD / STL Exporter for Frame Lattice Structures.

Converts 2D planar frame lattices (nodes, struts, cross-sections) into clean,
watertight 3D manifold STL meshes suitable for additive manufacturing (3D printing)
in slicing software (e.g. PrusaSlicer, Cura, Bambu Studio).

Generates:
1. 3D rectangular prism strut geometries with thickness t_e and out-of-plane depth D_z.
2. Polygonal cylindrical joint hubs at structural nodes ensuring seamless intersection.
3. Both ASCII STL and high-performance binary STL formats.
"""

from __future__ import annotations
import os
import struct
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from ..fem.mesh import FrameMesh2D


def _compute_facet_normal(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> np.ndarray:
    """Compute normalized outward unit normal vector for a triangular facet."""
    u = v2 - v1
    w = v3 - v1
    cross = np.cross(u, w)
    norm = np.linalg.norm(cross)
    return cross / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float64)


def export_lattice_to_stl(
    mesh: FrameMesh2D,
    file_path: str,
    out_of_plane_depth: float = 0.01,
    hub_radius_factor: float = 0.75,
    binary: bool = False,
    active_mask: Optional[np.ndarray] = None,
) -> int:
    """Export a 2D frame lattice to a 3D-printable solid STL file.

    Args:
        mesh: FrameMesh2D containing nodes, elements, and sections.
        file_path: Output path ending in .stl.
        out_of_plane_depth: Extrusion depth D_z in meters (default 0.01 m = 10 mm).
        hub_radius_factor: Joint hub radius multiplier relative to max incident strut thickness.
        binary: If True, writes compact binary STL; if False, writes standard ASCII STL.
        active_mask: Optional boolean mask of active elements.

    Returns:
        Total number of triangular facets generated in the STL mesh.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    depth = float(out_of_plane_depth)

    facets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # 1. Generate 3D prism facets for each active member
    elem_list = list(mesh.elements.values())
    node_max_thickness: Dict[int, float] = {nid: 0.0 for nid in mesh.nodes}

    for idx, elem in enumerate(elem_list):
        if active_mask is not None and not active_mask[idx]:
            continue

        x1, y1 = mesh.nodes[elem.node_i]
        x2, y2 = mesh.nodes[elem.node_j]
        dx = x2 - x1
        dy = y2 - y1
        L = float(np.hypot(dx, dy))
        if L < 1e-9:
            continue

        t = elem.section.t
        node_max_thickness[elem.node_i] = max(node_max_thickness[elem.node_i], t)
        node_max_thickness[elem.node_j] = max(node_max_thickness[elem.node_j], t)

        # In-plane unit normal
        nx = -dy / L
        ny = dx / L
        half_t = 0.5 * t

        # Bottom face vertices (z = 0)
        p1_b = np.array([x1 + half_t * nx, y1 + half_t * ny, 0.0])
        p2_b = np.array([x2 + half_t * nx, y2 + half_t * ny, 0.0])
        p3_b = np.array([x2 - half_t * nx, y2 - half_t * ny, 0.0])
        p4_b = np.array([x1 - half_t * nx, y1 - half_t * ny, 0.0])

        # Top face vertices (z = D_z)
        p1_t = np.array([x1 + half_t * nx, y1 + half_t * ny, depth])
        p2_t = np.array([x2 + half_t * nx, y2 + half_t * ny, depth])
        p3_t = np.array([x2 - half_t * nx, y2 - half_t * ny, depth])
        p4_t = np.array([x1 - half_t * nx, y1 - half_t * ny, depth])

        # 12 triangles per prism (6 faces x 2 triangles)
        # Bottom face (z=0, normal downward [0, 0, -1])
        facets.append((p1_b, p3_b, p2_b))
        facets.append((p1_b, p4_b, p3_b))

        # Top face (z=D_z, normal upward [0, 0, 1])
        facets.append((p1_t, p2_t, p3_t))
        facets.append((p1_t, p3_t, p4_t))

        # Side face 1 (p1 -> p2)
        facets.append((p1_b, p2_b, p2_t))
        facets.append((p1_b, p2_t, p1_t))

        # Side face 2 (p2 -> p3)
        facets.append((p2_b, p3_b, p3_t))
        facets.append((p2_b, p3_t, p2_t))

        # Side face 3 (p3 -> p4)
        facets.append((p3_b, p4_b, p4_t))
        facets.append((p3_b, p4_t, p3_t))

        # Side face 4 (p4 -> p1)
        facets.append((p4_b, p1_b, p1_t))
        facets.append((p4_b, p1_t, p4_t))

    # 2. Generate joint hubs at nodes to ensure watertight intersection
    n_hub_segments = 12
    hub_angles = np.linspace(0.0, 2.0 * np.pi, n_hub_segments, endpoint=False)

    for nid, (x, y) in mesh.nodes.items():
        t_max = node_max_thickness.get(nid, 0.0)
        if t_max <= 1e-9:
            continue

        r_hub = 0.5 * t_max * hub_radius_factor * 1.2
        hub_pts_b = [np.array([x + r_hub * np.cos(a), y + r_hub * np.sin(a), 0.0]) for a in hub_angles]
        hub_pts_t = [np.array([x + r_hub * np.cos(a), y + r_hub * np.sin(a), depth]) for a in hub_angles]

        center_b = np.array([x, y, 0.0])
        center_t = np.array([x, y, depth])

        for s in range(n_hub_segments):
            s_next = (s + 1) % n_hub_segments
            # Bottom cap
            facets.append((center_b, hub_pts_b[s_next], hub_pts_b[s]))
            # Top cap
            facets.append((center_t, hub_pts_t[s], hub_pts_t[s_next]))
            # Cylindrical walls
            facets.append((hub_pts_b[s], hub_pts_b[s_next], hub_pts_t[s_next]))
            facets.append((hub_pts_b[s], hub_pts_t[s_next], hub_pts_t[s]))

    total_facets = len(facets)

    # 3. Write STL file
    if binary:
        with open(file_path, "wb") as f:
            # 80-byte header
            header = b"Lattice-Bio 3D Printable Solid Mesh (Bone Remodeling)"
            header = header.ljust(80, b" ")
            f.write(header)
            # 4-byte unsigned int facet count
            f.write(struct.pack("<I", total_facets))
            for v1, v2, v3 in facets:
                norm = _compute_facet_normal(v1, v2, v3)
                data = struct.pack(
                    "<12fH",
                    norm[0], norm[1], norm[2],
                    v1[0], v1[1], v1[2],
                    v2[0], v2[1], v2[2],
                    v3[0], v3[1], v3[2],
                    0,
                )
                f.write(data)
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("solid lattice_bio_pruned\n")
            for v1, v2, v3 in facets:
                norm = _compute_facet_normal(v1, v2, v3)
                f.write(f"  facet normal {norm[0]:.6e} {norm[1]:.6e} {norm[2]:.6e}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
                f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
                f.write(f"      vertex {v3[0]:.6e} {v3[1]:.6e} {v3[2]:.6e}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            f.write("endsolid lattice_bio_pruned\n")

    return total_facets

