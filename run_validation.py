"""Reproducible Phase 1 FE Validation Runner.

Loads test parameters from a YAML configuration file, executes each of the 7 required
analytical validation benchmarks plus mechanism detection, computes analytical vs numerical
metrics, and exports a verifiable summary table.

Adheres strictly to Ground Rule 2:
"Never fabricate, hand-edit, or smooth results. Every number must come from a script
that can be re-run from a YAML config and a seed."
"""

from __future__ import annotations
import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd

from src.fem.material import Material, RectangularSection
from src.fem.elements import FrameElement2D
from src.fem.mesh import FrameMesh2D
from src.fem.solver import FrameSolver2D, SingularMatrixError


def run_phase1_validation(config_path: str = "configs/phase1_validation.yaml") -> pd.DataFrame:
    """Execute all Phase 1 validation tests using parameters from config_path."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    mat_cfg = cfg["material"]
    sec_cfg = cfg["section"]
    test_cfg = cfg["test_cases"]
    rel_tol = float(cfg["tolerances"]["relative"])

    default_mat = Material(E=float(mat_cfg["E"]), density=float(mat_cfg["density"]), name=mat_cfg["name"])
    default_sec = RectangularSection(t=float(sec_cfg["thickness"]), b=float(sec_cfg["depth"]))

    results_table = []

    # -------------------------------------------------------------------------
    # Test 1: Axial Bar
    # -------------------------------------------------------------------------
    t1_cfg = test_cfg["axial_bar"]
    L1 = float(t1_cfg["length"])
    P1 = float(t1_cfg["axial_load"])

    mesh1 = FrameMesh2D()
    mesh1.add_node(1, 0.0, 0.0)
    mesh1.add_node(2, L1, 0.0)
    mesh1.add_element(1, 1, 2, default_mat, default_sec)
    mesh1.add_fixed_support(1)
    mesh1.add_load(2, fx=P1, fy=0.0, m=0.0)

    res1 = FrameSolver2D(mesh1).solve()
    ux1, _, _ = res1.get_nodal_displacement(2)
    expected_ux1 = (P1 * L1) / (default_mat.E * default_sec.A)
    err1 = abs(ux1 - expected_ux1) / abs(expected_ux1)
    pass1 = bool(err1 <= rel_tol)

    results_table.append(
        {
            "Test ID": "Test 1",
            "Description": "Axial bar tip displacement: PL / (EA)",
            "Metric": "Tip displacement ux (m)",
            "Expected": expected_ux1,
            "Computed": ux1,
            "Relative Error": err1,
            "Tolerance": rel_tol,
            "Status": "PASS" if pass1 else "FAIL",
        }
    )

    # -------------------------------------------------------------------------
    # Test 2: Cantilever with tip load
    # -------------------------------------------------------------------------
    t2_cfg = test_cfg["cantilever"]
    L2 = float(t2_cfg["length"])
    P2 = float(t2_cfg["tip_load"])

    mesh2 = FrameMesh2D()
    mesh2.add_node(1, 0.0, 0.0)
    mesh2.add_node(2, L2, 0.0)
    mesh2.add_element(1, 1, 2, default_mat, default_sec)
    mesh2.add_fixed_support(1)
    mesh2.add_load(2, fx=0.0, fy=P2, m=0.0)

    res2 = FrameSolver2D(mesh2).solve()
    _, uy2, th2 = res2.get_nodal_displacement(2)
    expected_uy2 = (P2 * (L2**3)) / (3.0 * default_mat.E * default_sec.I)
    err2 = abs(uy2 - expected_uy2) / abs(expected_uy2)
    pass2 = bool(err2 <= rel_tol)

    results_table.append(
        {
            "Test ID": "Test 2",
            "Description": "Cantilever tip deflection: PL^3 / (3EI)",
            "Metric": "Tip deflection uy (m)",
            "Expected": expected_uy2,
            "Computed": uy2,
            "Relative Error": err2,
            "Tolerance": rel_tol,
            "Status": "PASS" if pass2 else "FAIL",
        }
    )

    # -------------------------------------------------------------------------
    # Test 3: Simply Supported Beam with central load
    # -------------------------------------------------------------------------
    t3_cfg = test_cfg["simply_supported"]
    L3 = float(t3_cfg["length"])
    P3 = float(t3_cfg["center_load"])

    mesh3 = FrameMesh2D()
    mesh3.add_node(1, 0.0, 0.0)
    mesh3.add_node(2, L3 / 2.0, 0.0)
    mesh3.add_node(3, L3, 0.0)
    mesh3.add_element(1, 1, 2, default_mat, default_sec)
    mesh3.add_element(2, 2, 3, default_mat, default_sec)
    mesh3.add_pinned_support(1)
    mesh3.add_roller_x_support(3)
    mesh3.add_load(2, fx=0.0, fy=P3, m=0.0)

    res3 = FrameSolver2D(mesh3).solve()
    _, uy3, _ = res3.get_nodal_displacement(2)
    expected_uy3 = (P3 * (L3**3)) / (48.0 * default_mat.E * default_sec.I)
    err3 = abs(uy3 - expected_uy3) / abs(expected_uy3)
    pass3 = bool(err3 <= rel_tol)

    results_table.append(
        {
            "Test ID": "Test 3",
            "Description": "Simply supported beam central deflection: PL^3 / (48EI)",
            "Metric": "Midpoint deflection uy (m)",
            "Expected": expected_uy3,
            "Computed": uy3,
            "Relative Error": err3,
            "Tolerance": rel_tol,
            "Status": "PASS" if pass3 else "FAIL",
        }
    )

    # -------------------------------------------------------------------------
    # Test 4: Unconstrained element rigid-body zero-energy modes
    # -------------------------------------------------------------------------
    elem4 = FrameElement2D(
        element_id=1,
        node_i=1,
        node_j=2,
        coords_i=(0.0, 0.0),
        coords_j=(3.0, 4.0),
        material=default_mat,
        section=default_sec,
    )
    K4 = elem4.global_stiffness_matrix()
    eigvals4 = np.linalg.eigvalsh(K4)
    max_eig4 = float(eigvals4[-1])
    zero_eigs4 = eigvals4[:3]
    max_zero_ratio4 = float(np.max(np.abs(zero_eigs4)) / max_eig4)
    expected_zero_modes = 3
    num_detected_zeros = int(np.sum(np.abs(eigvals4) / max_eig4 < 1e-12))
    pass4 = (num_detected_zeros == expected_zero_modes)

    results_table.append(
        {
            "Test ID": "Test 4",
            "Description": "Unconstrained element rigid-body modes (multiplicity 3)",
            "Metric": "Zero-energy modes count (ratio < 1e-12)",
            "Expected": expected_zero_modes,
            "Computed": num_detected_zeros,
            "Relative Error": max_zero_ratio4,
            "Tolerance": 1e-10,
            "Status": "PASS" if pass4 else "FAIL",
        }
    )

    # -------------------------------------------------------------------------
    # Test 5: Symmetric Structure under Symmetric Load
    # -------------------------------------------------------------------------
    t5_cfg = test_cfg["symmetric_frame"]
    W5 = float(t5_cfg["width"])
    H5 = float(t5_cfg["height"])
    P5 = float(t5_cfg["vertical_load"])

    mesh5 = FrameMesh2D()
    mesh5.add_node(1, 0.0, 0.0)
    mesh5.add_node(2, 0.0, H5)
    mesh5.add_node(3, W5 / 2.0, H5)
    mesh5.add_node(4, W5, H5)
    mesh5.add_node(5, W5, 0.0)

    mesh5.add_element(1, 1, 2, default_mat, default_sec)
    mesh5.add_element(2, 2, 3, default_mat, default_sec)
    mesh5.add_element(3, 3, 4, default_mat, default_sec)
    mesh5.add_element(4, 4, 5, default_mat, default_sec)

    mesh5.add_fixed_support(1)
    mesh5.add_fixed_support(5)
    mesh5.add_load(3, fx=0.0, fy=P5, m=0.0)

    res5 = FrameSolver2D(mesh5).solve()
    ux_mid5, _, th_mid5 = res5.get_nodal_displacement(3)
    ux_left5, uy_left5, th_left5 = res5.get_nodal_displacement(2)
    ux_right5, uy_right5, th_right5 = res5.get_nodal_displacement(4)

    sym_err_ux_knee = abs(ux_left5 + ux_right5) / abs(ux_left5)
    sym_err_uy_knee = abs(uy_left5 - uy_right5) / abs(uy_left5)
    pass5 = bool(
        abs(ux_mid5) < 1e-12
        and abs(th_mid5) < 1e-12
        and sym_err_ux_knee <= rel_tol
        and sym_err_uy_knee <= rel_tol
    )

    results_table.append(
        {
            "Test ID": "Test 5",
            "Description": "Symmetric portal frame under symmetric load: knee ux asymmetry",
            "Metric": "Relative knee asymmetry |ux_L + ux_R| / |ux_L|",
            "Expected": 0.0,
            "Computed": sym_err_ux_knee,
            "Relative Error": sym_err_ux_knee,
            "Tolerance": rel_tol,
            "Status": "PASS" if pass5 else "FAIL",
        }
    )

    # -------------------------------------------------------------------------
    # Test 6: Mesh Refinement
    # -------------------------------------------------------------------------
    t6_cfg = test_cfg["mesh_refinement"]
    L6 = float(t6_cfg["length"])
    P6 = float(t6_cfg["tip_load"])
    levels6 = t6_cfg["refinement_levels"]

    expected_uy6 = (P6 * (L6**3)) / (3.0 * default_mat.E * default_sec.I)
    max_refine_err = 0.0

    for n_elem in levels6:
        m6 = FrameMesh2D()
        for idx, x in enumerate(np.linspace(0.0, L6, n_elem + 1)):
            m6.add_node(idx + 1, x, 0.0)
        for i in range(n_elem):
            m6.add_element(i + 1, i + 1, i + 2, default_mat, default_sec)
        m6.add_fixed_support(1)
        m6.add_load(n_elem + 1, fx=0.0, fy=P6, m=0.0)
        r6 = FrameSolver2D(m6).solve()
        _, uy_n, _ = r6.get_nodal_displacement(n_elem + 1)
        err_n = abs(uy_n - expected_uy6) / abs(expected_uy6)
        max_refine_err = max(max_refine_err, err_n)

    pass6 = bool(max_refine_err <= rel_tol)
    results_table.append(
        {
            "Test ID": "Test 6",
            "Description": "Mesh refinement invariance (1 to 32 elements): tip deflection",
            "Metric": "Max relative error across 6 refinement levels",
            "Expected": expected_uy6,
            "Computed": uy_n,
            "Relative Error": max_refine_err,
            "Tolerance": rel_tol,
            "Status": "PASS" if pass6 else "FAIL",
        }
    )

    # -------------------------------------------------------------------------
    # Test 7: Equilibrium
    # -------------------------------------------------------------------------
    t7_cfg = test_cfg["equilibrium_multi_member"]
    mesh7 = FrameMesh2D()
    mesh7.add_node(1, 0.0, 0.0)
    mesh7.add_node(2, 2.0, 1.5)
    mesh7.add_node(3, 4.0, 0.0)
    mesh7.add_node(4, 5.0, float(t7_cfg["height"]))
    mesh7.add_node(5, float(t7_cfg["span"]) + 1.0, 0.0)

    mesh7.add_element(1, 1, 2, default_mat, default_sec)
    mesh7.add_element(2, 2, 3, default_mat, default_sec)
    mesh7.add_element(3, 2, 4, default_mat, default_sec)
    mesh7.add_element(4, 4, 5, default_mat, default_sec)

    mesh7.add_fixed_support(1)
    mesh7.add_pinned_support(3)
    mesh7.add_roller_x_support(5)

    mesh7.add_load(2, fx=float(t7_cfg["load_fx"]), fy=float(t7_cfg["load_fy"]), m=float(t7_cfg["load_moment"]))
    mesh7.add_load(4, fx=-500.0, fy=-1000.0, m=-250.0)

    res7 = FrameSolver2D(mesh7).solve()
    eq7 = res7.check_equilibrium()
    max_eq_rel_err = max(eq7["rel_err_fx"], eq7["rel_err_fy"], eq7["rel_err_m0"])
    pass7 = bool(max_eq_rel_err <= rel_tol)

    results_table.append(
        {
            "Test ID": "Test 7",
            "Description": "Global equilibrium: reactions sum to applied loads (Fx, Fy, M0)",
            "Metric": "Max relative equilibrium error",
            "Expected": 0.0,
            "Computed": max_eq_rel_err,
            "Relative Error": max_eq_rel_err,
            "Tolerance": rel_tol,
            "Status": "PASS" if pass7 else "FAIL",
        }
    )

    # -------------------------------------------------------------------------
    # Test 8: Singular Matrix / Mechanism Detection
    # -------------------------------------------------------------------------
    mesh8 = FrameMesh2D()
    mesh8.add_node(1, 0.0, 0.0)
    mesh8.add_node(2, 2.0, 0.0)
    mesh8.add_element(1, 1, 2, default_mat, default_sec)
    mesh8.add_pinned_support(1)  # Only 1 pinned support -> rotational mechanism
    mesh8.add_load(2, fy=-100.0)

    detected_singular = False
    try:
        FrameSolver2D(mesh8).solve()
    except SingularMatrixError:
        detected_singular = True

    results_table.append(
        {
            "Test ID": "Test 8",
            "Description": "Singular matrix / mechanism detection (rotational mechanism)",
            "Metric": "SingularMatrixError raised (1=True, 0=False)",
            "Expected": 1.0,
            "Computed": 1.0 if detected_singular else 0.0,
            "Relative Error": 0.0 if detected_singular else 1.0,
            "Tolerance": 0.0,
            "Status": "PASS" if detected_singular else "FAIL",
        }
    )

    df = pd.DataFrame(results_table)

    # Export to results
    os.makedirs("results", exist_ok=True)
    csv_path = "results/phase1_validation_summary.csv"
    md_path = "results/phase1_validation_summary.md"
    df.to_csv(csv_path, index=False)

    # Format numbers cleanly for Markdown table
    df_md = df.copy()
    for col in ["Expected", "Computed", "Relative Error", "Tolerance"]:
        df_md[col] = df_md[col].apply(
            lambda v: f"{v:.6e}" if isinstance(v, (int, float, np.floating)) and abs(v) > 0 and (abs(v) < 1e-3 or abs(v) >= 1e4)
            else (f"{v:.6f}" if isinstance(v, (int, float, np.floating)) and abs(v) > 0
            else ("0.0" if isinstance(v, (int, float, np.floating)) and v == 0.0 else str(v)))
        )

    with open(md_path, "w") as f:
        f.write("# Phase 1 FE Solver Validation Summary\n\n")
        f.write(f"Configuration: `{config_path}`\n\n")
        f.write(df_md.to_markdown(index=False))
        f.write("\n")

    print(f"\n[Validation Runner] Successfully exported results to {csv_path} and {md_path}\n")
    print(df_md.to_string(index=False))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 1 FE validation tests.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase1_validation.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()
    run_phase1_validation(args.config)
