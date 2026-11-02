"""Publication-grade figure generation for paper artifacts (Fig 1 - Fig 8).

Generates static, high-resolution figures matching the paper outline:
- Fig 1: Lattice generation domain & baselines overview (Ground structure, Honeycomb, Voronoi)
- Fig 2: Example optimized designs for each method and load case (2x3 grid)
- Fig 3: Convergence dynamics (Best compliance vs FE evaluations)
- Fig 4: Performance distributions (Boxplots of specific stiffness across seeds)
- Fig 5: Pareto frontiers (Stiffness vs Mass across volume budgets)
- Fig 6: Kinematic strain energy shares (Axial vs Bending breakdown)
- Fig 7: Cross-load generalization & robustness matrix
- Fig 8: Computational scaling with grid size
"""

from __future__ import annotations
import os
import json
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection

from ..geometry.domain import DesignDomain2D
from ..geometry.ground_structure import generate_ground_structure
from ..geometry.honeycomb import generate_honeycomb
from ..geometry.voronoi import generate_voronoi_lattice
from ..fem.material import Material


# Global publication style settings
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.titlesize": 12,
        "lines.linewidth": 1.75,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

PALETTE = {
    "RemodelingPruning": "#2b5c8f",     # Deep biological blue
    "GeneticAlgorithm": "#d95f02",      # Evolutionary amber
    "Honeycomb": "#7570b3",             # Hexagonal purple
    "Voronoi": "#1b9e77",               # Trabecular emerald
    "UniformGroundStructure": "#666666", # Neutral grey
}


def plot_fig1_overview(out_path: str = "figures/paper_fig1_overview.png") -> None:
    """Plot Figure 1: Lattice architecture overview across candidates and baselines."""
    domain = DesignDomain2D(width=3.0, height=1.0)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

    # Panel A: Ground Structure
    gs = generate_ground_structure(domain, nx=13, ny=5, radius=0.6)
    ax_a = axes[0]
    ax_a.plot([0, 3, 3, 0, 0], [0, 0, 1, 1, 0], "k--", lw=1.2, alpha=0.5)
    lines_a = [
        [(gs.nodes[m[0]][0], gs.nodes[m[0]][1]), (gs.nodes[m[1]][0], gs.nodes[m[1]][1])]
        for m in gs.members
    ]
    ax_a.add_collection(LineCollection(lines_a, colors="#2c3e50", linewidths=0.9, alpha=0.7))
    coords_a = np.array(list(gs.nodes.values()))
    ax_a.scatter(coords_a[:, 0], coords_a[:, 1], color="#2980b9", s=14, zorder=3)
    ax_a.set_title(f"(a) Ground Structure ({len(gs.members)} Members)", fontweight="bold")
    ax_a.set_aspect("equal")
    ax_a.set_xlim(-0.1, 3.1)
    ax_a.set_ylim(-0.1, 1.1)
    ax_a.grid(True, linestyle=":", alpha=0.4)

    # Panel B: Hexagonal Honeycomb
    hc = generate_honeycomb(domain, cell_size=0.35)
    ax_b = axes[1]
    ax_b.plot([0, 3, 3, 0, 0], [0, 0, 1, 1, 0], "k--", lw=1.2, alpha=0.5)
    lines_b = [
        [(hc.nodes[m[0]][0], hc.nodes[m[0]][1]), (hc.nodes[m[1]][0], hc.nodes[m[1]][1])]
        for m in hc.members
    ]
    ax_b.add_collection(LineCollection(lines_b, colors="#7570b3", linewidths=1.6, alpha=0.85))
    coords_b = np.array(list(hc.nodes.values()))
    ax_b.scatter(coords_b[:, 0], coords_b[:, 1], color="#5e3c99", s=12, zorder=3)
    ax_b.set_title(f"(b) Regular Honeycomb ({len(hc.members)} Members)", fontweight="bold")
    ax_b.set_aspect("equal")
    ax_b.set_xlim(-0.1, 3.1)
    ax_b.set_ylim(-0.1, 1.1)
    ax_b.grid(True, linestyle=":", alpha=0.4)

    # Panel C: Stochastic Voronoi Lattice
    vor = generate_voronoi_lattice(domain, num_seeds=45, lloyd_iterations=3, seed=42)
    ax_c = axes[2]
    ax_c.plot([0, 3, 3, 0, 0], [0, 0, 1, 1, 0], "k--", lw=1.2, alpha=0.5)
    lines_c = [
        [(vor.nodes[m[0]][0], vor.nodes[m[0]][1]), (vor.nodes[m[1]][0], vor.nodes[m[1]][1])]
        for m in vor.members
    ]
    ax_c.add_collection(LineCollection(lines_c, colors="#1b9e77", linewidths=1.4, alpha=0.85))
    coords_c = np.array(list(vor.nodes.values()))
    ax_c.scatter(coords_c[:, 0], coords_c[:, 1], color="#006837", s=10, zorder=3)
    ax_c.set_title(f"(c) Trabecular Voronoi ({len(vor.members)} Members)", fontweight="bold")
    ax_c.set_aspect("equal")
    ax_c.set_xlim(-0.1, 3.1)
    ax_c.set_ylim(-0.1, 1.1)
    ax_c.grid(True, linestyle=":", alpha=0.4)

    for ax in axes:
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated Paper Figure 1: {out_path}")


def plot_fig2_optimized_topologies(
    json_path: str = "demo/data/lattice_data.json",
    out_path: str = "figures/paper_fig2_optimized_topologies.png",
) -> None:
    """Plot Figure 2: Optimized lattice topologies (2x3 grid: Pruning vs GA across LC1, LC2, LC3)."""
    if not os.path.exists(json_path):
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lcs = ["LC1", "LC2", "LC3"]
    lc_titles = [
        "LC1: Cantilever Tip Shear",
        "LC2: Simply Supported 3-Point",
        "LC3: Compression Block",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 6.2))
    cmap = plt.get_cmap("plasma")

    for col, (lc, lc_name) in enumerate(zip(lcs, lc_titles)):
        scenario_key = f"{lc}_vf20"
        if scenario_key not in data.get("scenarios", {}):
            continue
        sc = data["scenarios"][scenario_key]

        # Row 0: Pruning
        frame_p = sc["remodeling_pruning"][-1]
        ax_p = axes[0, col]
        _draw_lattice_frame(ax_p, frame_p, cmap)
        ax_p.set_title(f"Method A (Pruning) - {lc_name}\nK = {frame_p['stiffness']:.1f} N/m", fontweight="bold", fontsize=9.5)

        # Row 1: GA
        frame_ga = sc["genetic_algorithm"][-1]
        ax_ga = axes[1, col]
        _draw_lattice_frame(ax_ga, frame_ga, cmap)
        ax_ga.set_title(f"Method B (GA) - {lc_name}\nK = {frame_ga['stiffness']:.1f} N/m", fontweight="bold", fontsize=9.5)

    for r in range(2):
        for c in range(3):
            axes[r, c].set_aspect("equal")
            axes[r, c].set_xlim(-0.1, 3.1)
            axes[r, c].set_ylim(-0.1, 1.1)
            axes[r, c].grid(True, linestyle=":", alpha=0.35)
            if r == 1:
                axes[r, c].set_xlabel("x (m)")
            if c == 0:
                axes[r, c].set_ylabel("y (m)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated Paper Figure 2: {out_path}")


def _draw_lattice_frame(ax: plt.Axes, frame: Dict, cmap) -> None:
    """Helper to draw frame members colored by strain energy with thickness."""
    ax.plot([0, 3, 3, 0, 0], [0, 0, 1, 1, 0], "k--", lw=0.8, alpha=0.4)

    lines = []
    colors = []
    lws = []
    for m in frame.get("members", []):
        lines.append([(m["x1"], m["y1"]), (m["x2"], m["y2"])])
        norm_e = m.get("normalized_energy", 0.0)
        colors.append(cmap(np.clip(norm_e, 0.0, 1.0)))
        t_mm = m.get("thickness", 0.01) * 1000.0
        lw = np.clip(t_mm * 0.12, 0.8, 3.5)
        lws.append(lw)

    lc = LineCollection(lines, colors=colors, linewidths=lws, alpha=0.9)
    ax.add_collection(lc)

    # Supports & loads
    nodes_dict = frame.get("nodes", {})
    if isinstance(nodes_dict, list):
        nodes_list = nodes_dict
    else:
        nodes_list = list(nodes_dict.values())

    for n in nodes_list:
        x, y = n["x"], n["y"]
        if abs(x) < 1e-4:  # Left boundary supports
            ax.scatter([x], [y], color="#e74c3c", marker="^", s=22, zorder=4)
        if abs(x - 3.0) < 1e-4 and abs(y - 0.5) < 1e-4:  # Cantilever tip load
            ax.scatter([x], [y], color="#27ae60", marker="v", s=35, zorder=5)


def plot_fig3_convergence(
    csv_path: str = "results/e2_convergence.csv",
    out_path: str = "figures/paper_fig3_convergence.png",
) -> None:
    """Plot convergence curves: Best-so-far compliance vs FE evaluations."""
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    lcs = ["LC1", "LC2", "LC3"]
    titles = [
        "LC1: Cantilever Tip Shear",
        "LC2: Simply Supported 3-Point",
        "LC3: Compression Block",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)

    for idx, (lc, title) in enumerate(zip(lcs, titles)):
        ax = axes[idx]
        sub = df[df["load_case"] == lc]

        for method in ["RemodelingPruning", "GeneticAlgorithm"]:
            m_sub = sub[sub["method"] == method].sort_values("fe_evaluations")
            if not m_sub.empty:
                label = "Method A: Pruning" if "Pruning" in method else "Method B: GA"
                marker = "o" if "Pruning" in method else "s"
                color = PALETTE.get(method, "#333333")
                ax.plot(
                    m_sub["fe_evaluations"],
                    m_sub["compliance_J"],
                    marker=marker,
                    markersize=4,
                    color=color,
                    label=label,
                )

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("FE Solver Evaluations (log scale)")
        if idx == 0:
            ax.set_ylabel("Compliance C (J, log scale)")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated Paper Figure 3: {out_path}")


def plot_fig4_distributions(
    csv_path: str = "results/e1_final_quality.csv",
    out_path: str = "figures/paper_fig4_distributions.png",
) -> None:
    """Plot boxplots of specific stiffness across stochastic seeds."""
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    lcs = ["LC1", "LC2", "LC3"]
    titles = [
        "LC1: Cantilever",
        "LC2: 3-Point Bending",
        "LC3: Compression Block",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    methods_order = [
        "RemodelingPruning",
        "GeneticAlgorithm",
        "UniformGroundStructure",
        "Voronoi",
        "Honeycomb",
    ]
    labels = ["Pruning (A)", "GA (B)", "Uniform GS", "Voronoi", "Honeycomb"]

    for idx, (lc, title) in enumerate(zip(lcs, titles)):
        ax = axes[idx]
        sub = df[(df["load_case"] == lc) & (df["target_vf"] == 0.20)]

        data_to_plot = []
        for m in methods_order:
            vals = sub[sub["method"] == m]["specific_stiffness"].dropna().values
            data_to_plot.append(vals if len(vals) > 0 else [0.0])

        box = ax.boxplot(
            data_to_plot,
            patch_artist=True,
            tick_labels=labels,
            medianprops=dict(color="black", linewidth=1.5),
        )

        for patch, m in zip(box["boxes"], methods_order):
            patch.set_facecolor(PALETTE.get(m, "#bbbbbb"))
            patch.set_alpha(0.85)

        ax.set_yscale("log")
        ax.set_title(f"{title} (vf = 0.20)", fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Specific Stiffness (1/C)/mass [log]")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated Paper Figure 4: {out_path}")


def plot_fig5_pareto(
    csv_path: str = "results/e3_pareto_fronts.csv",
    out_path: str = "figures/paper_fig5_pareto.png",
) -> None:
    """Plot Pareto fronts: Stiffness vs Achieved Mass across volume fractions."""
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    lcs = ["LC1", "LC2", "LC3"]
    titles = [
        "LC1: Cantilever",
        "LC2: 3-Point Bending",
        "LC3: Compression Block",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    for idx, (lc, title) in enumerate(zip(lcs, titles)):
        ax = axes[idx]
        sub = df[df["load_case"] == lc]

        for method in ["RemodelingPruning", "GeneticAlgorithm", "Voronoi", "Honeycomb"]:
            m_sub = sub[sub["method"] == method].sort_values("mass_kg")
            if not m_sub.empty:
                label = "Pruning" if "Pruning" in method else ("GA" if "Genetic" in method else method)
                ax.plot(
                    m_sub["mass_kg"],
                    m_sub["stiffness_N_m"],
                    "o-",
                    color=PALETTE.get(method, "#333333"),
                    label=label,
                    lw=1.8,
                    markersize=5,
                )

        ax.set_yscale("log")
        ax.set_xlabel("Lattice Mass (kg)")
        if idx == 0:
            ax.set_ylabel("Structural Stiffness (N/m, log scale)")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated Paper Figure 5: {out_path}")


def plot_fig6_kinematics(
    csv_path: str = "results/e5_axial_bending_shares.csv",
    out_path: str = "figures/paper_fig6_kinematics_shares.png",
) -> None:
    """Plot stacked bar chart of axial vs bending strain energy share."""
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    sub = df[df["load_case"] == "LC1"].copy()
    if sub.empty:
        sub = df.copy()

    methods = sub["method"].tolist()
    labels = [
        "Pruning\n(Bio)",
        "GA\n(Evol)",
        "Honeycomb\n(Hex)",
        "Voronoi\n(Trabec)",
    ]
    axial = sub["axial_energy_share"].values * 100.0
    bending = sub["bending_energy_share"].values * 100.0

    x = np.arange(len(methods))
    width = 0.55

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    p1 = ax.bar(x, axial, width, label="Axial (Stretch-Dominated)", color="#1b7837", alpha=0.9)
    p2 = ax.bar(x, bending, width, bottom=axial, label="Bending (Flexure-Dominated)", color="#762a83", alpha=0.9)

    ax.set_ylabel("Share of Total Strain Energy (%)", fontweight="bold")
    ax.set_title("Deformation Kinematics Breakdown (LC1 Cantilever, vf = 0.20)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Annotate percentages
    for i in range(len(methods)):
        if axial[i] > 10:
            ax.text(x[i], axial[i] / 2, f"{axial[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        if bending[i] > 10:
            ax.text(x[i], axial[i] + bending[i] / 2, f"{bending[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated Paper Figure 6: {out_path}")


def plot_fig7_robustness_matrix(
    csv_path: str = "results/e4_cross_load_robustness.csv",
    out_path: str = "figures/paper_fig7_robustness_matrix.png",
) -> None:
    """Plot cross-load degradation heatmap matrix."""
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    lcs = ["LC1", "LC2", "LC3"]
    matrix = np.zeros((3, 3))

    for i, opt_lc in enumerate(lcs):
        for j, eval_lc in enumerate(lcs):
            sub = df[(df["optimized_for"] == opt_lc) & (df["evaluated_on"] == eval_lc)]
            if not sub.empty:
                val = sub["degradation_ratio"].values[0] if "degradation_ratio" in sub.columns else sub["compliance_degradation_ratio"].values[0]
                matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(np.log10(matrix), cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(lcs, fontweight="bold")
    ax.set_yticklabels(lcs, fontweight="bold")
    ax.set_xlabel("Evaluated Load Case", fontweight="bold")
    ax.set_ylabel("Optimized Load Case", fontweight="bold")
    ax.set_title("Cross-Load Specialization Gap\n(log10 Compliance Degradation Ratio)", fontweight="bold")

    for i in range(3):
        for j in range(3):
            val = matrix[i, j]
            text = f"{val:.1f}x" if val < 100 else f"{val:.0f}x"
            color = "white" if np.log10(val) > 2.0 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontweight="bold", fontsize=10)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(Degradation Ratio)", fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated Paper Figure 7: {out_path}")


def plot_fig8_scaling(
    csv_path: str = "results/e7_scaling.csv",
    out_path: str = "figures/paper_fig8_scaling.png",
) -> None:
    """Plot computational scaling curves: Runtime and FE solve time vs members."""
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)

    fig, ax1 = plt.subplots(figsize=(6.5, 4.0))

    ax1.plot(
        df["num_members"],
        df["total_optimization_time_s"],
        "o-",
        color="#2b5c8f",
        label="Optimization Runtime (s)",
        lw=2,
    )
    ax1.set_xlabel("Number of Ground Structure Candidate Members", fontweight="bold")
    ax1.set_ylabel("Total Optimization Runtime (s)", color="#2b5c8f", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#2b5c8f")
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(
        df["num_members"],
        df["single_fe_solve_time_s"] * 1000.0,
        "s--",
        color="#d95f02",
        label="Single FE Solve (ms)",
        lw=2,
    )
    ax2.set_ylabel("Single FE Solve Time (ms)", color="#d95f02", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#d95f02")

    ax1.set_title("Computational Scaling vs Network Density (Pruning)", fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated Paper Figure 8: {out_path}")


def generate_all_paper_figures() -> None:
    """Generate all publication paper figures (Fig 1 - Fig 8) from result CSVs."""
    plot_fig1_overview()
    plot_fig2_optimized_topologies()
    plot_fig3_convergence()
    plot_fig4_distributions()
    plot_fig5_pareto()
    plot_fig6_kinematics()
    plot_fig7_robustness_matrix()
    plot_fig8_scaling()
