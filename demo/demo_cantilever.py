"""Demonstration of 2D Frame FE Solver: Cantilever and Portal Frame Analysis.

Shows how to:
1. Define materials and rectangular cross-sections (with derived A and I).
2. Build 2D frame mesh geometry with nodes and elements.
3. Apply boundary conditions and loads.
4. Solve for nodal displacements, reactions, and per-member internal forces.
5. Inspect the axial vs bending strain energy partition and check equilibrium.
"""

from __future__ import annotations
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fem.material import Material, RectangularSection
from src.fem.mesh import FrameMesh2D
from src.fem.solver import FrameSolver2D
from src.fem.postprocess import member_results_to_dataframe, summarize_results


def run_demo():
    print("=== Demo 1: Slender Cantilever Beam under Tip Shear ===")
    mat = Material(E=2.1e11, density=7850.0, name="Steel")
    # Section of thickness 0.02 m and unit depth 1.0 m -> A = 0.02 m^2, I = (0.02)^3 / 12 = 6.6667e-7 m^4
    sec = RectangularSection(t=0.02, b=1.0)
    print(f"Material: {mat.name}, E = {mat.E:.2e} Pa, rho = {mat.density} kg/m^3")
    print(f"Section: thickness t = {sec.t} m, depth b = {sec.b} m")
    print(f"Derived properties: Area A = {sec.A:.4e} m^2, Second Moment I = {sec.I:.4e} m^4\n")

    mesh = FrameMesh2D()
    mesh.add_node(1, 0.0, 0.0)
    mesh.add_node(2, 3.0, 0.0)
    mesh.add_element(1, 1, 2, mat, sec)

    mesh.add_fixed_support(1)
    mesh.add_load(2, fx=0.0, fy=-500.0, m=0.0)

    solver = FrameSolver2D(mesh)
    res = solver.solve()

    ux_tip, uy_tip, th_tip = res.get_nodal_displacement(2)
    print(f"Tip Displacement: ux = {ux_tip:.6e} m, uy = {uy_tip:.6e} m, theta = {th_tip:.6e} rad")
    rx1, ry1, rm1 = res.get_nodal_reaction(1)
    print(f"Root Reactions:   Rx = {rx1:.2f} N, Ry = {ry1:.2f} N, Rm = {rm1:.2f} N*m")
    print(f"Total Compliance: C = F^T u = {res.compliance:.6e} J")
    print(f"Axial Strain Energy:   {res.total_axial_energy:.6e} J ({res.axial_energy_fraction * 100:.2f}%)")
    print(f"Bending Strain Energy: {res.total_bending_energy:.6e} J ({res.bending_energy_fraction * 100:.2f}%)\n")

    print("Per-member force breakdown:")
    df_members = member_results_to_dataframe(res)
    print(df_members.to_string(index=False))

    summary = summarize_results(res)
    print("\nGlobal Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    run_demo()
