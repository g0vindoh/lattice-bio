"""Postprocessing utilities for 2D frame FE results."""

from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import numpy as np

from .solver import SolverResult


def member_results_to_dataframe(result: SolverResult) -> pd.DataFrame:
    """Convert per-member FE results into a pandas DataFrame.

    Columns include:
        element_id, node_i, node_j, length, axial_force, bending_moment_1,
        bending_moment_2, shear_force, strain_energy_axial, strain_energy_bending,
        strain_energy_total.
    """
    rows = []
    for eid, forces in result.internal_forces.items():
        elem = result.mesh.elements[eid]
        rows.append(
            {
                "element_id": eid,
                "node_i": elem.node_i,
                "node_j": elem.node_j,
                "length": elem.length,
                "axial_force": forces.axial_force,
                "bending_moment_1": forces.bending_moment_1,
                "bending_moment_2": forces.bending_moment_2,
                "shear_force": forces.shear_force,
                "strain_energy_axial": forces.strain_energy_axial,
                "strain_energy_bending": forces.strain_energy_bending,
                "strain_energy_total": forces.strain_energy_total,
            }
        )
    return pd.DataFrame(rows)


def nodal_results_to_dataframe(result: SolverResult) -> pd.DataFrame:
    """Convert nodal displacements and reactions into a pandas DataFrame.

    Columns include:
        node_id, x, y, ux, uy, theta, rx, ry, rm.
    """
    rows = []
    for nid, (x, y) in sorted(result.mesh.nodes.items()):
        ux, uy, th = result.get_nodal_displacement(nid)
        rx, ry, rm = result.get_nodal_reaction(nid)
        rows.append(
            {
                "node_id": nid,
                "x": x,
                "y": y,
                "ux": ux,
                "uy": uy,
                "theta": th,
                "rx": rx,
                "ry": ry,
                "rm": rm,
            }
        )
    return pd.DataFrame(rows)


def summarize_results(result: SolverResult) -> Dict[str, Any]:
    """Produce a high-level dictionary summary of structural metrics."""
    eq = result.check_equilibrium()
    return {
        "num_nodes": result.mesh.num_nodes,
        "num_elements": result.mesh.num_elements,
        "total_volume_m3": result.total_volume,
        "total_mass_kg": result.total_mass,
        "compliance_J": result.compliance,
        "total_strain_energy_J": result.total_strain_energy,
        "axial_strain_energy_J": result.total_axial_energy,
        "bending_strain_energy_J": result.total_bending_energy,
        "axial_energy_fraction": result.axial_energy_fraction,
        "bending_energy_fraction": result.bending_energy_fraction,
        "equilibrium_net_fx_N": eq["net_fx"],
        "equilibrium_net_fy_N": eq["net_fy"],
        "equilibrium_net_m0_Nm": eq["net_m0"],
    }

