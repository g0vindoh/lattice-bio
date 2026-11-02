"""Structural feasibility verification module for lattice designs."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from ..fem.mesh import FrameMesh2D
from ..fem.solver import FrameSolver2D, SingularMatrixError, SolverResult


@dataclass
class SlendernessWarning:
    """Warning for a member where Euler-Bernoulli beam theory may be questionable."""

    element_id: int
    length: float
    thickness: float
    slenderness_ratio: float  # L / t


@dataclass
class FeasibilityReport:
    """Comprehensive feasibility diagnostics for a structural design.

    Attributes:
        is_feasible: True if design is mechanically feasible (no mechanisms, valid load paths).
        is_connected: True if all structural elements form a continuous path between loads and supports.
        has_load_path: True if every loaded node can reach at least one support node.
        has_mechanism: True if an unconstrained rigid-body mode or kinematic mechanism is detected.
        mechanism_error_msg: Detailed error description if a mechanism was found.
        slenderness_warnings: List of members with slenderness L/t < 10.0.
        min_slenderness: Minimum L/t across all members in the design.
        unreachable_load_nodes: Set of loaded nodes with no path to a support.
    """

    is_feasible: bool
    is_connected: bool
    has_load_path: bool
    has_mechanism: bool
    mechanism_error_msg: Optional[str] = None
    slenderness_warnings: List[SlendernessWarning] = field(default_factory=list)
    min_slenderness: float = 0.0
    unreachable_load_nodes: Set[int] = field(default_factory=set)

    @property
    def has_slenderness_warnings(self) -> bool:
        return len(self.slenderness_warnings) > 0

    def summary(self) -> str:
        status = "FEASIBLE" if self.is_feasible else "INFEASIBLE"
        lines = [f"[Feasibility Status: {status}]"]
        lines.append(f"  Connected Graph: {self.is_connected}")
        lines.append(f"  Load Paths Valid: {self.has_load_path}")
        lines.append(f"  Mechanism Detected: {self.has_mechanism}")
        if self.mechanism_error_msg:
            lines.append(f"  Mechanism Diagnostics: {self.mechanism_error_msg}")
        lines.append(f"  Min Slenderness L/t: {self.min_slenderness:.2f}")
        lines.append(f"  Slenderness Warnings (L/t < 10): {len(self.slenderness_warnings)}")
        if self.unreachable_load_nodes:
            lines.append(f"  Unreachable Load Nodes: {sorted(self.unreachable_load_nodes)}")
        return "\n".join(lines)


def check_design_feasibility(
    mesh: FrameMesh2D,
    slenderness_threshold: float = 10.0,
) -> Tuple[FeasibilityReport, Optional[SolverResult]]:
    """Perform a rigorous, non-destructive feasibility analysis on a frame mesh.

    Does not modify or repair the structure.

    Checks:
    1. Slenderness: flags members with L/t < 10 where Euler-Bernoulli shear neglects matter.
    2. Load Path: verifies graph reachability from all loaded nodes to at least one support node.
    3. Structural Mechanisms: runs FrameSolver2D to detect kinematic singularities, rigid-body
       modes, or unconstrained degrees of freedom.

    Args:
        mesh: FrameMesh2D instance with elements, boundary conditions, and loads.
        slenderness_threshold: Slenderness ratio L/t below which a warning is issued (default 10.0).

    Returns:
        (FeasibilityReport, Optional[SolverResult])
    """
    if mesh.num_elements == 0:
        return (
            FeasibilityReport(
                is_feasible=False,
                is_connected=False,
                has_load_path=False,
                has_mechanism=True,
                mechanism_error_msg="Mesh contains zero structural elements.",
            ),
            None,
        )

    # 1. Slenderness ratio evaluation
    slenderness_warnings: List[SlendernessWarning] = []
    min_slenderness = float("inf")

    for eid, elem in mesh.elements.items():
        # Rectangular section thickness t
        t = getattr(elem.section, "t", None)
        if t is None:
            # Fallback for generic cross section: t_eff = sqrt(12 * I / A)
            t = np.sqrt(12.0 * elem.section.I / elem.section.A)

        L = elem.length
        slenderness = float(L / t) if t > 0 else 0.0
        min_slenderness = min(min_slenderness, slenderness)

        if slenderness < slenderness_threshold:
            slenderness_warnings.append(
                SlendernessWarning(
                    element_id=eid,
                    length=L,
                    thickness=t,
                    slenderness_ratio=slenderness,
                )
            )

    if min_slenderness == float("inf"):
        min_slenderness = 0.0

    # 2. Graph Connectivity & Load Path Reachability
    adj: Dict[int, Set[int]] = {nid: set() for nid in mesh.nodes}
    for elem in mesh.elements.values():
        adj[elem.node_i].add(elem.node_j)
        adj[elem.node_j].add(elem.node_i)

    # Identify support nodes
    support_nodes: Set[int] = {bc.node_id for bc in mesh.boundary_conditions}

    # Identify loaded nodes (active non-zero forces or moments)
    loaded_nodes: Set[int] = {
        ld.node_id
        for ld in mesh.point_loads
        if abs(ld.fx) > 1e-12 or abs(ld.fy) > 1e-12 or abs(ld.m) > 1e-12
    }

    # Breadth-first search from all support nodes to find all reachable nodes
    reachable_from_supports: Set[int] = set()
    queue = list(support_nodes)
    for sn in queue:
        reachable_from_supports.add(sn)

    while queue:
        curr = queue.pop(0)
        for nbr in adj.get(curr, set()):
            if nbr not in reachable_from_supports:
                reachable_from_supports.add(nbr)
                queue.append(nbr)

    unreachable_loads = loaded_nodes - reachable_from_supports
    has_load_path = (len(unreachable_loads) == 0) and (len(support_nodes) > 0)

    # Global structural connectivity
    # Design is considered connected if all loaded and supported nodes belong to the same component
    is_connected = has_load_path

    # 3. Kinematic Mechanism / Singularity Check using FrameSolver2D
    solver = FrameSolver2D(mesh)
    solver_result = None
    has_mechanism = False
    mechanism_err = None

    try:
        solver_result = solver.solve()
    except SingularMatrixError as err:
        has_mechanism = True
        mechanism_err = str(err)
    except Exception as err:
        has_mechanism = True
        mechanism_err = f"Unexpected solver exception: {err}"

    is_feasible = bool(is_connected and has_load_path and not has_mechanism)

    report = FeasibilityReport(
        is_feasible=is_feasible,
        is_connected=is_connected,
        has_load_path=has_load_path,
        has_mechanism=has_mechanism,
        mechanism_error_msg=mechanism_err,
        slenderness_warnings=slenderness_warnings,
        min_slenderness=min_slenderness,
        unreachable_load_nodes=unreachable_loads,
    )

    return report, solver_result

