"""Load and support pad definition and application for standardized structural testing.

Fairness Rule Implementation:
To ensure an unbiased comparison across diverse lattice topologies (ground structures,
regular honeycombs, and stochastic Voronoi networks):
1. Loads and supports are NEVER applied to arbitrary topological nodes.
2. Every design is augmented with standardized 'load/support pad' members of fixed, thin
   thickness t_pad that interface the boundary supports and external load actuators.
3. The volume of these pad members V_pad is strictly identical or explicitly accounted
   for, counting toward the total volume budget for ALL designs:
       V_total = V_pad + V_lattice = target_volume_fraction * W * H * depth
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from .domain import DesignDomain2D
from ..fem.material import Material, RectangularSection
from ..fem.mesh import FrameMesh2D


@dataclass
class BoundaryPadConfig:
    """Configuration for standardized load and support pads.

    Attributes:
        pad_thickness: Cross-sectional thickness of pad members (m, default 0.005 m).
        pad_depth: Out-of-plane depth of pad members (m, default 1.0 m).
        attachment_tolerance: Distance tolerance for attaching lattice nodes to pads (m).
    """

    pad_thickness: float = 0.005
    pad_depth: float = 1.0
    attachment_tolerance: float = 1e-4


def attach_pads_and_bcs_lc1(
    mesh: FrameMesh2D,
    domain: DesignDomain2D,
    material: Material,
    pad_config: BoundaryPadConfig,
    load_magnitude: float = 1000.0,
) -> Tuple[List[int], List[int], float]:
    """Apply LC1 (Cantilever): Clamped left edge x=0, tip shear load at (W, H/2).

    Pad Architecture:
    - Support Pad: Vertical pad members along x=0 connecting left boundary nodes.
      All nodes on x=0 are clamped (ux=0, uy=0, theta=0).
    - Load Pad: Load application node at (W, H/2) connected to the right boundary nodes
      via a vertical distributing pad bracket. Load Fy = -load_magnitude is applied at (W, H/2).

    Returns:
        (pad_node_ids, pad_element_ids, pad_volume)
    """
    pad_sec = RectangularSection(t=pad_config.pad_thickness, b=pad_config.pad_depth)
    next_node_id = max(mesh.nodes.keys()) + 1 if mesh.nodes else 1
    next_elem_id = max(mesh.elements.keys()) + 1 if mesh.elements else 1

    pad_node_ids: List[int] = []
    pad_elem_ids: List[int] = []
    pad_volume = 0.0

    # 1. Left support pad: Find all nodes on x = 0
    left_nodes = [
        nid
        for nid, (x, y) in mesh.nodes.items()
        if abs(x - 0.0) <= pad_config.attachment_tolerance
    ]
    # Sort left nodes vertically by y
    left_nodes.sort(key=lambda nid: mesh.nodes[nid][1])

    # If top or bottom corner nodes on x=0 are missing, create anchor pad nodes
    if not left_nodes or abs(mesh.nodes[left_nodes[0]][1] - 0.0) > 1e-3:
        b_nid = next_node_id
        next_node_id += 1
        mesh.add_node(b_nid, 0.0, 0.0)
        left_nodes.insert(0, b_nid)
        pad_node_ids.append(b_nid)

    if abs(mesh.nodes[left_nodes[-1]][1] - domain.height) > 1e-3:
        t_nid = next_node_id
        next_node_id += 1
        mesh.add_node(t_nid, 0.0, domain.height)
        left_nodes.append(t_nid)
        pad_node_ids.append(t_nid)

    # Connect adjacent left boundary nodes with pad members and apply fixed support
    for i in range(len(left_nodes) - 1):
        n1 = left_nodes[i]
        n2 = left_nodes[i + 1]
        elem = mesh.add_element(next_elem_id, n1, n2, material, pad_sec)
        pad_elem_ids.append(next_elem_id)
        pad_volume += elem.volume
        next_elem_id += 1

    for nid in left_nodes:
        mesh.add_fixed_support(nid)

    # 2. Right load pad: Find right edge nodes near mid-height (W, H/2)
    right_mid_y = domain.height / 2.0
    right_nodes = [
        nid
        for nid, (x, y) in mesh.nodes.items()
        if abs(x - domain.width) <= pad_config.attachment_tolerance
    ]

    # Target load point node at (W, H/2)
    # Check if a node already exists at (W, H/2)
    load_nid = None
    for nid in right_nodes:
        if abs(mesh.nodes[nid][1] - right_mid_y) <= 1e-4:
            load_nid = nid
            break

    if load_nid is None:
        load_nid = next_node_id
        next_node_id += 1
        mesh.add_node(load_nid, domain.width, right_mid_y)
        pad_node_ids.append(load_nid)
        right_nodes.append(load_nid)

    right_nodes.sort(key=lambda nid: mesh.nodes[nid][1])

    # Connect adjacent right boundary nodes with pad members
    for i in range(len(right_nodes) - 1):
        n1 = right_nodes[i]
        n2 = right_nodes[i + 1]
        elem = mesh.add_element(next_elem_id, n1, n2, material, pad_sec)
        pad_elem_ids.append(next_elem_id)
        pad_volume += elem.volume
        next_elem_id += 1

    # Apply downward shear load at (W, H/2)
    mesh.add_load(load_nid, fx=0.0, fy=-float(load_magnitude), m=0.0)

    return pad_node_ids, pad_elem_ids, float(pad_volume)


def attach_pads_and_bcs_lc2(
    mesh: FrameMesh2D,
    domain: DesignDomain2D,
    material: Material,
    pad_config: BoundaryPadConfig,
    load_magnitude: float = 1000.0,
) -> Tuple[List[int], List[int], float]:
    """Apply LC2 (Simply Supported): Pin at (0, 0), roller at (W, 0), central load at (W/2, H).

    Returns:
        (pad_node_ids, pad_element_ids, pad_volume)
    """
    pad_sec = RectangularSection(t=pad_config.pad_thickness, b=pad_config.pad_depth)
    next_node_id = max(mesh.nodes.keys()) + 1 if mesh.nodes else 1
    next_elem_id = max(mesh.elements.keys()) + 1 if mesh.elements else 1

    pad_node_ids: List[int] = []
    pad_elem_ids: List[int] = []
    pad_volume = 0.0

    # 1. Bottom rail pad along y = 0
    bottom_nodes = [
        nid
        for nid, (x, y) in mesh.nodes.items()
        if abs(y - 0.0) <= pad_config.attachment_tolerance
    ]
    # Ensure (0,0) and (W,0) exist
    pin_nid = None
    roller_nid = None
    for nid in bottom_nodes:
        if abs(mesh.nodes[nid][0] - 0.0) <= 1e-4:
            pin_nid = nid
        if abs(mesh.nodes[nid][0] - domain.width) <= 1e-4:
            roller_nid = nid

    if pin_nid is None:
        pin_nid = next_node_id
        next_node_id += 1
        mesh.add_node(pin_nid, 0.0, 0.0)
        pad_node_ids.append(pin_nid)
        bottom_nodes.append(pin_nid)

    if roller_nid is None:
        roller_nid = next_node_id
        next_node_id += 1
        mesh.add_node(roller_nid, domain.width, 0.0)
        pad_node_ids.append(roller_nid)
        bottom_nodes.append(roller_nid)

    bottom_nodes.sort(key=lambda nid: mesh.nodes[nid][0])
    for i in range(len(bottom_nodes) - 1):
        n1 = bottom_nodes[i]
        n2 = bottom_nodes[i + 1]
        elem = mesh.add_element(next_elem_id, n1, n2, material, pad_sec)
        pad_elem_ids.append(next_elem_id)
        pad_volume += elem.volume
        next_elem_id += 1

    mesh.add_pinned_support(pin_nid)
    mesh.add_roller_x_support(roller_nid)

    # 2. Top load pad at (W/2, H)
    top_mid_x = domain.width / 2.0
    top_nodes = [
        nid
        for nid, (x, y) in mesh.nodes.items()
        if abs(y - domain.height) <= pad_config.attachment_tolerance
    ]
    load_nid = None
    for nid in top_nodes:
        if abs(mesh.nodes[nid][0] - top_mid_x) <= 1e-4:
            load_nid = nid
            break

    if load_nid is None:
        load_nid = next_node_id
        next_node_id += 1
        mesh.add_node(load_nid, top_mid_x, domain.height)
        pad_node_ids.append(load_nid)
        top_nodes.append(load_nid)

    top_nodes.sort(key=lambda nid: mesh.nodes[nid][0])
    for i in range(len(top_nodes) - 1):
        n1 = top_nodes[i]
        n2 = top_nodes[i + 1]
        elem = mesh.add_element(next_elem_id, n1, n2, material, pad_sec)
        pad_elem_ids.append(next_elem_id)
        pad_volume += elem.volume
        next_elem_id += 1

    mesh.add_load(load_nid, fx=0.0, fy=-float(load_magnitude), m=0.0)

    return pad_node_ids, pad_elem_ids, float(pad_volume)


def attach_pads_and_bcs_lc3(
    mesh: FrameMesh2D,
    domain: DesignDomain2D,
    material: Material,
    pad_config: BoundaryPadConfig,
    load_magnitude: float = 1000.0,
) -> Tuple[List[int], List[int], float]:
    """Apply LC3 (Compression Block): Fixed bottom edge y=0, distributed downward load on top y=H.

    Returns:
        (pad_node_ids, pad_element_ids, pad_volume)
    """
    pad_sec = RectangularSection(t=pad_config.pad_thickness, b=pad_config.pad_depth)
    next_node_id = max(mesh.nodes.keys()) + 1 if mesh.nodes else 1
    next_elem_id = max(mesh.elements.keys()) + 1 if mesh.elements else 1

    pad_node_ids: List[int] = []
    pad_elem_ids: List[int] = []
    pad_volume = 0.0

    # 1. Bottom support pad rail along y = 0
    bottom_nodes = [
        nid
        for nid, (x, y) in mesh.nodes.items()
        if abs(y - 0.0) <= pad_config.attachment_tolerance
    ]
    # Ensure (0,0) and (W,0) exist
    if not any(abs(mesh.nodes[nid][0] - 0.0) <= 1e-4 for nid in bottom_nodes):
        nid0 = next_node_id
        next_node_id += 1
        mesh.add_node(nid0, 0.0, 0.0)
        pad_node_ids.append(nid0)
        bottom_nodes.append(nid0)

    if not any(abs(mesh.nodes[nid][0] - domain.width) <= 1e-4 for nid in bottom_nodes):
        nidW = next_node_id
        next_node_id += 1
        mesh.add_node(nidW, domain.width, 0.0)
        pad_node_ids.append(nidW)
        bottom_nodes.append(nidW)

    bottom_nodes.sort(key=lambda nid: mesh.nodes[nid][0])
    for i in range(len(bottom_nodes) - 1):
        n1 = bottom_nodes[i]
        n2 = bottom_nodes[i + 1]
        elem = mesh.add_element(next_elem_id, n1, n2, material, pad_sec)
        pad_elem_ids.append(next_elem_id)
        pad_volume += elem.volume
        next_elem_id += 1

    for nid in bottom_nodes:
        mesh.add_fixed_support(nid)

    # 2. Top loading pad rail along y = H
    top_nodes = [
        nid
        for nid, (x, y) in mesh.nodes.items()
        if abs(y - domain.height) <= pad_config.attachment_tolerance
    ]
    if not any(abs(mesh.nodes[nid][0] - 0.0) <= 1e-4 for nid in top_nodes):
        nid0 = next_node_id
        next_node_id += 1
        mesh.add_node(nid0, 0.0, domain.height)
        pad_node_ids.append(nid0)
        top_nodes.append(nid0)

    if not any(abs(mesh.nodes[nid][0] - domain.width) <= 1e-4 for nid in top_nodes):
        nidW = next_node_id
        next_node_id += 1
        mesh.add_node(nidW, domain.width, domain.height)
        pad_node_ids.append(nidW)
        top_nodes.append(nidW)

    top_nodes.sort(key=lambda nid: mesh.nodes[nid][0])
    for i in range(len(top_nodes) - 1):
        n1 = top_nodes[i]
        n2 = top_nodes[i + 1]
        elem = mesh.add_element(next_elem_id, n1, n2, material, pad_sec)
        pad_elem_ids.append(next_elem_id)
        pad_volume += elem.volume
        next_elem_id += 1

    # Distribute downward load uniformly across all top nodes
    nodal_force = float(load_magnitude / len(top_nodes))
    for nid in top_nodes:
        mesh.add_load(nid, fx=0.0, fy=-nodal_force, m=0.0)

    return pad_node_ids, pad_elem_ids, float(pad_volume)

