"""Mesh and boundary condition management for 2D frame structures."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .material import Material, CrossSection
from .elements import FrameElement2D


@dataclass
class BoundaryCondition:
    """Prescribed nodal degree of freedom.

    Attributes:
        node_id: Target node identifier.
        dof_index: Degree of freedom index at this node:
                   0 for u_x (horizontal displacement),
                   1 for u_y (vertical displacement),
                   2 for theta (rotation).
        value: Prescribed displacement or rotation value (default 0.0).
    """

    node_id: int
    dof_index: int
    value: float = 0.0

    def __post_init__(self) -> None:
        if self.dof_index not in (0, 1, 2):
            raise ValueError(
                f"Invalid dof_index {self.dof_index}. Must be 0 (ux), 1 (uy), or 2 (theta)."
            )


@dataclass
class PointLoad:
    """Nodal applied point force and/or moment.

    Attributes:
        node_id: Target node identifier.
        fx: Global horizontal force (N).
        fy: Global vertical force (N).
        m: In-plane bending moment (N*m), positive counter-clockwise.
    """

    node_id: int
    fx: float = 0.0
    fy: float = 0.0
    m: float = 0.0


class FrameMesh2D:
    """Represents a 2D frame structure (mesh, materials, sections, boundary conditions, loads)."""

    def __init__(self) -> None:
        self.nodes: Dict[int, Tuple[float, float]] = {}
        self.elements: Dict[int, FrameElement2D] = {}
        self.boundary_conditions: List[BoundaryCondition] = []
        self.point_loads: List[PointLoad] = []

    def add_node(self, node_id: int, x: float, y: float) -> None:
        """Register a node with coordinates (x, y)."""
        node_id = int(node_id)
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists in mesh.")
        self.nodes[node_id] = (float(x), float(y))

    def add_element(
        self,
        element_id: int,
        node_i: int,
        node_j: int,
        material: Material,
        section: CrossSection,
    ) -> FrameElement2D:
        """Connect two nodes with a 2D frame element."""
        element_id = int(element_id)
        node_i = int(node_i)
        node_j = int(node_j)

        if element_id in self.elements:
            raise ValueError(f"Element {element_id} already exists in mesh.")
        if node_i not in self.nodes:
            raise KeyError(f"Start node {node_i} not defined in mesh.")
        if node_j not in self.nodes:
            raise KeyError(f"End node {node_j} not defined in mesh.")
        if node_i == node_j:
            raise ValueError(
                f"Element {element_id} connects node {node_i} to itself (zero length)."
            )

        elem = FrameElement2D(
            element_id=element_id,
            node_i=node_i,
            node_j=node_j,
            coords_i=self.nodes[node_i],
            coords_j=self.nodes[node_j],
            material=material,
            section=section,
        )
        self.elements[element_id] = elem
        return elem

    def add_boundary_condition(
        self, node_id: int, dof: Union[str, int], value: float = 0.0
    ) -> None:
        """Add a prescribed displacement or rotation to a node.

        Args:
            node_id: Node ID.
            dof: 'ux' (0), 'uy' (1), or 'theta' / 'm' / 'rot' (2).
            value: Prescribed value (default 0.0).
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not defined in mesh.")

        dof_map = {"ux": 0, "uy": 1, "theta": 2, "rot": 2, 0: 0, 1: 1, 2: 2}
        if dof not in dof_map:
            raise ValueError(
                f"Invalid DOF identifier '{dof}'. Valid: 'ux', 'uy', 'theta', 0, 1, 2."
            )

        bc = BoundaryCondition(node_id=node_id, dof_index=dof_map[dof], value=float(value))
        self.boundary_conditions.append(bc)

    def add_fixed_support(self, node_id: int) -> None:
        """Clamp/fix all DOFs at node (ux = 0, uy = 0, theta = 0)."""
        self.add_boundary_condition(node_id, 0, 0.0)
        self.add_boundary_condition(node_id, 1, 0.0)
        self.add_boundary_condition(node_id, 2, 0.0)

    def add_pinned_support(self, node_id: int) -> None:
        """Pin support at node (ux = 0, uy = 0, rotation free)."""
        self.add_boundary_condition(node_id, 0, 0.0)
        self.add_boundary_condition(node_id, 1, 0.0)

    def add_roller_x_support(self, node_id: int) -> None:
        """Roller allowing movement along X (uy = 0, ux and rotation free)."""
        self.add_boundary_condition(node_id, 1, 0.0)

    def add_roller_y_support(self, node_id: int) -> None:
        """Roller allowing movement along Y (ux = 0, uy and rotation free)."""
        self.add_boundary_condition(node_id, 0, 0.0)

    def add_load(
        self, node_id: int, fx: float = 0.0, fy: float = 0.0, m: float = 0.0
    ) -> None:
        """Apply external forces and/or moment to a node."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not defined in mesh.")
        load = PointLoad(node_id=node_id, fx=float(fx), fy=float(fy), m=float(m))
        self.point_loads.append(load)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_elements(self) -> int:
        return len(self.elements)

    @property
    def num_dofs(self) -> int:
        return 3 * self.num_nodes

    @property
    def total_volume(self) -> float:
        """Sum of member volumes: V = sum(A_e * L_e)."""
        return sum(elem.volume for elem in self.elements.values())

    @property
    def total_mass(self) -> float:
        """Sum of member masses: M = sum(rho_e * A_e * L_e)."""
        return sum(elem.mass for elem in self.elements.values())

    def get_node_to_index_mapping(self) -> Dict[int, int]:
        """Contiguous 0-based indexing for nodes sorted by node_id."""
        sorted_nids = sorted(self.nodes.keys())
        return {nid: idx for idx, nid in enumerate(sorted_nids)}

    def get_dof_indices(self, node_id: int) -> Tuple[int, int, int]:
        """Return the global DOF indices (dof_ux, dof_uy, dof_theta) for a given node_id."""
        node_map = self.get_node_to_index_mapping()
        idx = node_map[node_id]
        return (3 * idx, 3 * idx + 1, 3 * idx + 2)

