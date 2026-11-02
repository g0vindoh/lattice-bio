"""Phase 2 controlled baseline comparison pipeline.

Evaluates three structural baselines:
1. Uniformly thinned ground structure (13x5 grid with full diagonals)
2. Regular hexagonal honeycomb
3. Random Voronoi lattice (evaluated over 20 random seeds, reporting median and IQR)

Across three load cases:
- LC1: Cantilever (tip shear at right mid-height)
- LC2: Simply Supported (three-point bending)
- LC3: Compression Block (distributed top load)

At volume fractions: 0.2, 0.3, 0.4.
"""

from __future__ import annotations
import os
import yaml
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..geometry.domain import DesignDomain2D, compute_volume_fraction
from ..geometry.ground_structure import (
    generate_ground_structure,
    build_fe_mesh_from_ground_structure,
)
from ..geometry.honeycomb import generate_honeycomb, build_fe_mesh_from_honeycomb
from ..geometry.voronoi import generate_voronoi_lattice, build_fe_mesh_from_voronoi
from ..fem.load_cases import LoadCase, load_cases_from_yaml
from ..fem.material import Material, RectangularSection
from ..metrics.feasibility import check_design_feasibility
from ..metrics.metrics import evaluate_design_metrics
from ..export.plotting import (
    plot_lattice_geometry,
    plot_deformed_shape_with_energy,
)


def evaluate_single_design(
    baseline_type: str,
    load_case: LoadCase,
    domain: DesignDomain2D,
    material: Material,
    target_vf: float,
    geom_data: object,
    seed: Optional[int] = None,
) -> Tuple[Dict, object, object]:
    """Build and evaluate a single design configuration."""
    # 1. Determine pad volume using a trial mesh
    if baseline_type == "ground_structure":
        dummy_sec = RectangularSection(t=0.01, b=domain.depth)
        trial_mesh = build_fe_mesh_from_ground_structure(
            geom_data, [dummy_sec] * geom_data.num_members, material
        )
    elif baseline_type == "honeycomb":
        trial_mesh, _ = build_fe_mesh_from_honeycomb(geom_data, target_vf, material)
    elif baseline_type == "voronoi":
        trial_mesh, _ = build_fe_mesh_from_voronoi(geom_data, target_vf, material)
    else:
        raise ValueError(f"Unknown baseline: {baseline_type}")

    _, _, pad_vol = load_case.apply(trial_mesh, domain, material)

    # 2. Build final mesh with exact remaining volume allocation
    V_target = target_vf * domain.bounding_volume
    V_lattice = max(1e-12, V_target - pad_vol)

    if baseline_type == "ground_structure":
        t_strut = V_lattice / (domain.depth * geom_data.total_length)
        sec = RectangularSection(t=t_strut, b=domain.depth)
        mesh = build_fe_mesh_from_ground_structure(
            geom_data, [sec] * geom_data.num_members, material
        )
    elif baseline_type == "honeycomb":
        mesh, t_strut = build_fe_mesh_from_honeycomb(
            geom_data, target_vf, material, fixed_pad_volume=pad_vol
        )
    elif baseline_type == "voronoi":
        mesh, t_strut = build_fe_mesh_from_voronoi(
            geom_data, target_vf, material, fixed_pad_volume=pad_vol
        )

    # 3. Apply load case and boundary conditions to final mesh
    load_case.apply(mesh, domain, material)

    # 4. Feasibility and solve
    report, res = check_design_feasibility(mesh)

    record = {
        "baseline": baseline_type,
        "load_case": load_case.id,
        "target_vf": target_vf,
        "seed": seed if seed is not None else 0,
        "is_feasible": report.is_feasible,
        "has_mechanism": report.has_mechanism,
        "has_load_path": report.has_load_path,
        "min_slenderness": report.min_slenderness,
        "slenderness_warnings_count": len(report.slenderness_warnings),
        "strut_thickness_m": t_strut,
        "pad_volume_m3": pad_vol,
        "pad_volume_fraction": pad_vol / domain.bounding_volume,
    }

    if res is not None:
        perf = evaluate_design_metrics(res, domain)
        record.update(
            {
                "compliance_J": perf.compliance,
                "stiffness_N_m": perf.stiffness,
                "mass_kg": perf.mass,
                "volume_m3": perf.volume,
                "achieved_vf": perf.volume_fraction,
                "specific_stiffness": perf.specific_stiffness,
                "axial_energy_share": perf.axial_energy_share,
                "bending_energy_share": perf.bending_energy_share,
                "max_stress_Pa": perf.max_normal_stress,
                "max_utilization": perf.max_utilization,
            }
        )
    else:
        record.update(
            {
                "compliance_J": np.nan,
                "stiffness_N_m": np.nan,
                "mass_kg": np.nan,
                "volume_m3": np.nan,
                "achieved_vf": np.nan,
                "specific_stiffness": np.nan,
                "axial_energy_share": np.nan,
                "bending_energy_share": np.nan,
                "max_stress_Pa": np.nan,
                "max_utilization": np.nan,
            }
        )

    return record, mesh, res


def run_all_phase2_baselines(
    config_path: str = "configs/phase2_baselines.yaml",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete experimental matrix across all baselines, load cases, and volume fractions."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    domain = DesignDomain2D(
        width=float(cfg["domain"]["width"]),
        height=float(cfg["domain"]["height"]),
        depth=float(cfg["domain"]["depth"]),
    )
    material = Material(
        E=float(cfg["material"]["E"]),
        density=float(cfg["material"]["density"]),
        name=cfg["material"]["name"],
    )
    load_cases = load_cases_from_yaml(config_path)
    volume_fractions = [float(v) for v in cfg["volume_fractions"]]
    num_vor_seeds = int(cfg["voronoi"].get("num_seeds_eval", 20))

    # Pre-generate ground structure and honeycomb geometries
    gs_cfg = cfg["ground_structure"]
    gs_data = generate_ground_structure(
        domain=domain,
        nx=int(gs_cfg["nx"]),
        ny=int(gs_cfg["ny"]),
        radius=float(gs_cfg["radius"]) if gs_cfg["radius"] is not None else None,
    )

    hc_cfg = cfg["honeycomb"]
    hc_data = generate_honeycomb(
        domain=domain,
        cell_size=float(hc_cfg["cell_size"]),
    )

    # Save undeformed geometry plots
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    dummy_sec = RectangularSection(t=0.01, b=domain.depth)
    mesh_gs_plot = build_fe_mesh_from_ground_structure(
        gs_data, [dummy_sec] * gs_data.num_members, material
    )
    plot_lattice_geometry(
        mesh_gs_plot, domain, "Baseline 1: Uniform Ground Structure (13x5 Grid)", "figures/geometry_ground_structure.png"
    )

    mesh_hc_plot, _ = build_fe_mesh_from_honeycomb(hc_data, 0.2, material)
    plot_lattice_geometry(
        mesh_hc_plot, domain, "Baseline 2: Regular Honeycomb Lattice", "figures/geometry_honeycomb.png"
    )

    sample_vor = generate_voronoi_lattice(
        domain=domain,
        num_seeds=int(cfg["voronoi"]["num_seeds"]),
        seed=1,
        lloyd_iterations=int(cfg["voronoi"]["lloyd_iterations"]),
    )
    mesh_vor_plot, _ = build_fe_mesh_from_voronoi(sample_vor, 0.2, material)
    plot_lattice_geometry(
        mesh_vor_plot, domain, "Baseline 3: Voronoi Lattice (Seed 1)", "figures/geometry_voronoi_seed1.png"
    )

    all_records = []
    saved_deformed_plots = set()

    print("\n--- Starting Phase 2 Baseline Evaluations ---")
    for lc_key, lc in load_cases.items():
        for vf in volume_fractions:
            print(f"Running {lc_key} at volume fraction {vf:.2f}...")

            # 1. Ground Structure
            rec_gs, mesh_gs, res_gs = evaluate_single_design(
                "ground_structure", lc, domain, material, vf, gs_data
            )
            all_records.append(rec_gs)

            # 2. Honeycomb
            rec_hc, mesh_hc, res_hc = evaluate_single_design(
                "honeycomb", lc, domain, material, vf, hc_data
            )
            all_records.append(rec_hc)

            # 3. Voronoi across 20 seeds
            for s in range(1, num_vor_seeds + 1):
                vor_data_s = generate_voronoi_lattice(
                    domain=domain,
                    num_seeds=int(cfg["voronoi"]["num_seeds"]),
                    seed=s,
                    lloyd_iterations=int(cfg["voronoi"]["lloyd_iterations"]),
                )
                rec_vor, mesh_vor, res_vor = evaluate_single_design(
                    "voronoi", lc, domain, material, vf, vor_data_s, seed=s
                )
                all_records.append(rec_vor)

                # Save deformed figures for vf=0.2 on seed 1
                if vf == 0.2 and s == 1:
                    plot_deformed_shape_with_energy(
                        res_gs, domain, f"Ground Structure Deformed ({lc_key})", f"figures/deformed_gs_{lc_key}_vf0.2.png"
                    )
                    plot_deformed_shape_with_energy(
                        res_hc, domain, f"Honeycomb Deformed ({lc_key})", f"figures/deformed_hc_{lc_key}_vf0.2.png"
                    )
                    plot_deformed_shape_with_energy(
                        res_vor, domain, f"Voronoi (Seed 1) Deformed ({lc_key})", f"figures/deformed_vor_{lc_key}_vf0.2.png"
                    )

    df_all = pd.DataFrame(all_records)
    csv_all_path = "results/phase2_baseline_all_runs.csv"
    df_all.to_csv(csv_all_path, index=False)
    print(f"Exported all {len(df_all)} runs to {csv_all_path}")

    # Build statistical summary table (reporting Voronoi Median and IQR)
    summary_rows = []
    for lc_key in load_cases.keys():
        for vf in volume_fractions:
            # GS
            sub_gs = df_all[(df_all["baseline"] == "ground_structure") & (df_all["load_case"] == lc_key) & (df_all["target_vf"] == vf)]
            row_gs = sub_gs.iloc[0]
            summary_rows.append(
                {
                    "Baseline": "Ground Structure (Uniform)",
                    "Load Case": lc_key,
                    "Target VF": vf,
                    "Compliance (J)": f"{row_gs['compliance_J']:.4e}",
                    "Stiffness (N/m)": f"{row_gs['stiffness_N_m']:.4e}",
                    "Specific Stiffness (1/(J*kg))": f"{row_gs['specific_stiffness']:.4e}",
                    "Axial Energy Share (%)": f"{row_gs['axial_energy_share']*100:.1f}%",
                    "Bending Energy Share (%)": f"{row_gs['bending_energy_share']*100:.1f}%",
                    "Feasible": "YES" if row_gs["is_feasible"] else "NO",
                    "Min Slenderness L/t": f"{row_gs['min_slenderness']:.1f}",
                    "Slenderness Warnings": row_gs["slenderness_warnings_count"],
                }
            )

            # Honeycomb
            sub_hc = df_all[(df_all["baseline"] == "honeycomb") & (df_all["load_case"] == lc_key) & (df_all["target_vf"] == vf)]
            row_hc = sub_hc.iloc[0]
            summary_rows.append(
                {
                    "Baseline": "Honeycomb (Regular Hex)",
                    "Load Case": lc_key,
                    "Target VF": vf,
                    "Compliance (J)": f"{row_hc['compliance_J']:.4e}",
                    "Stiffness (N/m)": f"{row_hc['stiffness_N_m']:.4e}",
                    "Specific Stiffness (1/(J*kg))": f"{row_hc['specific_stiffness']:.4e}",
                    "Axial Energy Share (%)": f"{row_hc['axial_energy_share']*100:.1f}%",
                    "Bending Energy Share (%)": f"{row_hc['bending_energy_share']*100:.1f}%",
                    "Feasible": "YES" if row_hc["is_feasible"] else "NO",
                    "Min Slenderness L/t": f"{row_hc['min_slenderness']:.1f}",
                    "Slenderness Warnings": row_hc["slenderness_warnings_count"],
                }
            )

            # Voronoi (20 seeds)
            sub_vor = df_all[(df_all["baseline"] == "voronoi") & (df_all["load_case"] == lc_key) & (df_all["target_vf"] == vf)]
            c_med = sub_vor["compliance_J"].median()
            c_iqr = sub_vor["compliance_J"].quantile(0.75) - sub_vor["compliance_J"].quantile(0.25)
            k_med = sub_vor["stiffness_N_m"].median()
            spk_med = sub_vor["specific_stiffness"].median()
            spk_iqr = sub_vor["specific_stiffness"].quantile(0.75) - sub_vor["specific_stiffness"].quantile(0.25)
            ax_med = sub_vor["axial_energy_share"].median()
            bend_med = sub_vor["bending_energy_share"].median()
            all_feas = "YES" if sub_vor["is_feasible"].all() else "SOME_FAIL"
            min_slend = sub_vor["min_slenderness"].min()
            total_slend_warn = sub_vor["slenderness_warnings_count"].sum()

            summary_rows.append(
                {
                    "Baseline": f"Voronoi (20 Seeds: Median [IQR])",
                    "Load Case": lc_key,
                    "Target VF": vf,
                    "Compliance (J)": f"{c_med:.4e} [{c_iqr:.2e}]",
                    "Stiffness (N/m)": f"{k_med:.4e}",
                    "Specific Stiffness (1/(J*kg))": f"{spk_med:.4e} [{spk_iqr:.2e}]",
                    "Axial Energy Share (%)": f"{ax_med*100:.1f}%",
                    "Bending Energy Share (%)": f"{bend_med*100:.1f}%",
                    "Feasible": all_feas,
                    "Min Slenderness L/t": f"{min_slend:.1f}",
                    "Slenderness Warnings": f"{total_slend_warn} across 20 seeds",
                }
            )

    df_summary = pd.DataFrame(summary_rows)
    csv_sum_path = "results/phase2_baseline_summary.csv"
    md_sum_path = "results/phase2_baseline_summary.md"
    df_summary.to_csv(csv_sum_path, index=False)

    with open(md_sum_path, "w") as f:
        f.write("# Phase 2 Baseline Results Summary\n\n")
        f.write("Evaluation across LC1 (Cantilever), LC2 (Simply Supported), LC3 (Compression Block).\n\n")
        f.write(df_summary.to_markdown(index=False))
        f.write("\n")

    print(f"Exported summary table to {csv_sum_path} and {md_sum_path}\n")

    # Generate summary comparison figure
    generate_comparison_plots(df_all)

    return df_all, df_summary


def generate_comparison_plots(df_all: pd.DataFrame) -> None:
    """Generate comparative performance charts across baselines and load cases."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=300)
    lcs = ["LC1", "LC2", "LC3"]
    titles = ["LC1 (Cantilever)", "LC2 (Simply Supported)", "LC3 (Compression Block)"]

    for idx, (lc, title) in enumerate(zip(lcs, titles)):
        ax = axes[idx]
        sub = df_all[df_all["load_case"] == lc]

        # Ground structure
        gs_sub = sub[sub["baseline"] == "ground_structure"].sort_values("target_vf")
        ax.plot(gs_sub["target_vf"], gs_sub["compliance_J"], "o-", color="#2980b9", lw=2, label="Ground Structure")

        # Honeycomb
        hc_sub = sub[sub["baseline"] == "honeycomb"].sort_values("target_vf")
        ax.plot(hc_sub["target_vf"], hc_sub["compliance_J"], "s--", color="#e67e22", lw=2, label="Honeycomb")

        # Voronoi median & IQR
        vor_sub = sub[sub["baseline"] == "voronoi"]
        vfs = sorted(vor_sub["target_vf"].unique())
        vor_med = [vor_sub[vor_sub["target_vf"] == v]["compliance_J"].median() for v in vfs]
        vor_q25 = [vor_sub[vor_sub["target_vf"] == v]["compliance_J"].quantile(0.25) for v in vfs]
        vor_q75 = [vor_sub[vor_sub["target_vf"] == v]["compliance_J"].quantile(0.75) for v in vfs]

        ax.plot(vfs, vor_med, "^-.", color="#27ae60", lw=2, label="Voronoi (Median)")
        ax.fill_between(vfs, vor_q25, vor_q75, color="#27ae60", alpha=0.2, label="Voronoi (IQR)")

        ax.set_yscale("log")
        ax.set_xlabel("Volume Fraction")
        ax.set_ylabel("Compliance C (J, log scale)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        if idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = "figures/baseline_compliance_vs_vf.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Generated comparative performance plot: {fig_path}")

