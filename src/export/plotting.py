"""Static visualization of lattice geometry and deformed state colored by strain energy."""

from __future__ import annotations
import os
from typing import Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from ..fem.mesh import FrameMesh2D
from ..fem.solver import SolverResult
from ..geometry.domain import DesignDomain2D


def plot_lattice_geometry(
    mesh: FrameMesh2D,
    domain: DesignDomain2D,
    title: str = "Lattice Geometry",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (9, 3.5),
) -> plt.Figure:
    """Plot the undeformed structural mesh geometry with domain boundary."""
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Draw domain boundary
    W, H = domain.width, domain.height
    ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], "k--", lw=1.2, alpha=0.5, label="Domain Boundary")

    # Draw members
    lines = []
    for elem in mesh.elements.values():
        x1, y1 = mesh.nodes[elem.node_i]
        x2, y2 = mesh.nodes[elem.node_j]
        lines.append([(x1, y1), (x2, y2)])

    lc = LineCollection(lines, colors="#2c3e50", linewidths=1.2, alpha=0.85)
    ax.add_collection(lc)

    # Draw nodes
    coords = np.array(list(mesh.nodes.values()))
    if len(coords) > 0:
        ax.scatter(coords[:, 0], coords[:, 1], color="#2980b9", s=10, zorder=3)

    # Highlight supported nodes
    sup_coords = np.array([mesh.nodes[bc.node_id] for bc in mesh.boundary_conditions])
    if len(sup_coords) > 0:
        ax.scatter(sup_coords[:, 0], sup_coords[:, 1], color="#e74c3c", marker="^", s=40, zorder=4, label="Supports")

    # Highlight loaded nodes
    for ld in mesh.point_loads:
        if abs(ld.fx) > 1e-6 or abs(ld.fy) > 1e-6:
            lx, ly = mesh.nodes[ld.node_id]
            ax.scatter([lx], [ly], color="#27ae60", marker="v", s=60, zorder=5, label="Applied Load")

    ax.set_aspect("equal")
    ax.set_xlim(-0.05 * W, 1.05 * W)
    ax.set_ylim(-0.1 * H, 1.1 * H)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig


def plot_deformed_shape_with_energy(
    result: SolverResult,
    domain: DesignDomain2D,
    title: str = "Deformed Configuration (Colored by Strain Energy)",
    save_path: Optional[str] = None,
    disp_scale: Optional[float] = None,
    figsize: Tuple[float, float] = (10, 3.8),
) -> plt.Figure:
    """Plot the deformed lattice configuration with members colored by strain energy."""
    mesh = result.mesh
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    W, H = domain.width, domain.height

    # Auto-scale displacement to ~8% of domain height if not specified
    if disp_scale is None:
        max_u = np.max(np.abs(result.displacements))
        disp_scale = float((0.08 * H) / max_u) if max_u > 1e-12 else 1.0

    # Collect undeformed reference lines
    undef_lines = []
    for elem in mesh.elements.values():
        x1, y1 = mesh.nodes[elem.node_i]
        x2, y2 = mesh.nodes[elem.node_j]
        undef_lines.append([(x1, y1), (x2, y2)])

    lc_undef = LineCollection(undef_lines, colors="#bdc3c7", linewidths=0.8, linestyle=":", alpha=0.6)
    ax.add_collection(lc_undef)

    # Collect deformed member lines and strain energy
    deformed_lines = []
    energies = []
    for eid, elem in mesh.elements.items():
        ux1, uy1, _ = result.get_nodal_displacement(elem.node_i)
        ux2, uy2, _ = result.get_nodal_displacement(elem.node_j)
        p1 = (mesh.nodes[elem.node_i][0] + disp_scale * ux1, mesh.nodes[elem.node_i][1] + disp_scale * uy1)
        p2 = (mesh.nodes[elem.node_j][0] + disp_scale * ux2, mesh.nodes[elem.node_j][1] + disp_scale * uy2)
        deformed_lines.append([p1, p2])
        energies.append(result.internal_forces[eid].strain_energy_total)

    energies = np.array(energies)
    lc_def = LineCollection(deformed_lines, cmap="plasma", linewidths=1.8)
    lc_def.set_array(energies)
    ax.add_collection(lc_def)

    cbar = fig.colorbar(lc_def, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Member Strain Energy (J)", fontsize=9)

    ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], "k--", lw=1.0, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_xlim(-0.05 * W, 1.05 * W)
    ax.set_ylim(-0.15 * H, 1.15 * H)
    sub = (
        f"Compliance: {result.compliance:.4e} J | "
        f"Axial: {result.axial_energy_fraction*100:.1f}% | "
        f"Bending: {result.bending_energy_fraction*100:.1f}%"
    )
    ax.set_title(f"{title}\n{sub}", fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig

