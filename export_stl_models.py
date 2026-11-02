"""Export 3D-printable STL solid models of optimal lattice designs."""

import os
from src.geometry.domain import DesignDomain2D
from src.geometry.ground_structure import generate_ground_structure
from src.fem.load_cases import load_cases_from_yaml
from src.fem.material import Material
from src.optimizers.pruning import RemodelingPruningOptimizer
from src.optimizers.multi_load import MultiLoadRemodelingOptimizer
from src.export.stl_exporter import export_lattice_to_stl


def export_all_paper_stl_models(out_dir: str = "stl"):
    """Export 3D-printable watertight STL models for the paper's key designs."""
    os.makedirs(out_dir, exist_ok=True)
    domain = DesignDomain2D(width=3.0, height=1.0)
    gs = generate_ground_structure(domain, nx=13, ny=5)
    lcs = load_cases_from_yaml("configs/phase2_baselines.yaml")
    mat = Material(E=210e9, density=7850)

    # 1. Optimal Cantilever Truss (LC1)
    print("Generating 3D STL for Method A (LC1 Cantilever)...")
    opt1 = RemodelingPruningOptimizer(
        ground_structure=gs,
        load_case=lcs["LC1"],
        material=mat,
        target_volume_fraction=0.20,
        removal_rate=0.05,
    )
    hist1 = opt1.optimize()
    stl_lc1 = os.path.join(out_dir, "lattice_pruned_LC1_cantilever.stl")
    n1 = export_lattice_to_stl(hist1.final_mesh, stl_lc1, out_of_plane_depth=0.02, binary=True)
    print(f"Exported: {stl_lc1} ({n1} facets)")

    # 2. Optimal 3-Point Bending Arch (LC2)
    print("Generating 3D STL for Method A (LC2 3-Point Arch)...")
    opt2 = RemodelingPruningOptimizer(
        ground_structure=gs,
        load_case=lcs["LC2"],
        material=mat,
        target_volume_fraction=0.20,
        removal_rate=0.05,
    )
    hist2 = opt2.optimize()
    stl_lc2 = os.path.join(out_dir, "lattice_pruned_LC2_bending_arch.stl")
    n2 = export_lattice_to_stl(hist2.final_mesh, stl_lc2, out_of_plane_depth=0.02, binary=True)
    print(f"Exported: {stl_lc2} ({n2} facets)")

    # 3. Multi-Load Robust Resilient Lattice (LC1 + LC2)
    print("Generating 3D STL for Multi-Load Robust Lattice (LC1 + LC2)...")
    opt_multi = MultiLoadRemodelingOptimizer(
        ground_structure=gs,
        load_cases=[lcs["LC1"], lcs["LC2"]],
        material=mat,
        target_volume_fraction=0.20,
        removal_rate=0.05,
    )
    res_multi = opt_multi.optimize()
    from src.geometry.ground_structure import build_fe_mesh_from_ground_structure
    from src.fem.material import RectangularSection
    secs = [RectangularSection(t=max(1e-9, t)) for t in res_multi["thicknesses"]]
    mesh_multi = build_fe_mesh_from_ground_structure(gs, secs, mat, active_mask=res_multi["active_mask"])
    stl_multi = os.path.join(out_dir, "lattice_multiload_robust.stl")
    n_multi = export_lattice_to_stl(mesh_multi, stl_multi, out_of_plane_depth=0.02, binary=True)
    print(f"Exported: {stl_multi} ({n_multi} facets)")


if __name__ == "__main__":
    export_all_paper_stl_models()
