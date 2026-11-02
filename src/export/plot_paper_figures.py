"""Publication-grade figure generation for paper artifacts (Fig 1 - Fig 8).

Generates static, high-resolution figures matching the paper outline:
- Fig 1: Concept & lattice architecture overview
- Fig 3: Convergence dynamics (Best compliance vs FE evaluations)
- Fig 4: Performance distributions (Boxplots of specific stiffness across seeds)
- Fig 5: Pareto frontiers (Stiffness vs Mass across volume budgets)
- Fig 6: Kinematic strain energy shares (Axial vs Bending breakdown)
- Fig 7: Cross-load generalization & robustness matrix
- Fig 8: Computational scaling with grid size
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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
    sub = df[np.isclose(df["target_vf"], 0.20)]
    lcs = ["LC1", "LC2", "LC3"]
    titles = [
        "LC1 (Cantilever)",
        "LC2 (Simply Supported)",
        "LC3 (Compression Block)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    methods_order = [
        "RemodelingPruning",
        "GeneticAlgorithm",
        "UniformGroundStructure",
        "Voronoi",
        "Honeycomb",
    ]
    labels_order = [
        "Method A\n(Pruning)",
        "Method B\n(GA)",
        "Uniform\nGS",
        "Voronoi\n(Trabecular)",
        "Hex\nHoneycomb",
    ]

    for idx, (lc, title) in enumerate(zip(lcs, titles)):
        ax = axes[idx]
        lc_sub = sub[sub["load_case"] == lc]

        data_to_plot = []
        box_colors = []
        for m in methods_order:
            vals = lc_sub[lc_sub["method"] == m]["specific_stiffness"].dropna().values
            data_to_plot.append(vals if len(vals) > 0 else [0.0])
            box_colors.append(PALETTE.get(m, "#555555"))

        bp = ax.boxplot(
            data_to_plot,
            patch_artist=True,
            tick_labels=labels_order,
            widths=0.6,
            medianprops=dict(color="black", linewidth=2.0),
        )

        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor("#333333")

        ax.set_yscale("log")
        if idx == 0:
            ax.set_ylabel("Specific Stiffness (N/(m·kg), log scale)")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated Paper Figure 4: {out_path}")


def plot_fig5_pareto(
    csv_path: str = "results/e3_pareto_fronts.csv",
    out_path: str = "figures/paper_fig5_pareto.png",
) -> None:
    """Plot Pareto frontiers: Stiffness vs Mass across volume fractions."""
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    lcs = ["LC1", "LC2", "LC3"]
    titles = [
        "LC1: Cantilever",
        "LC2: Simply Supported",
        "LC3: Compression Block",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    for idx, (lc, title) in enumerate(zip(lcs, titles)):
        ax = axes[idx]
        sub = df[df["load_case"] == lc]

        for method in ["RemodelingPruning", "Voronoi", "Honeycomb"]:
            m_sub = sub[sub["method"] == method].sort_values("mass_kg")
            if not m_sub.empty:
                label = (
                    "Method A: Pruning"
                    if "Pruning" in method
                    else ("Voronoi" if "Voronoi" in method else "Honeycomb")
                )
                marker = "o" if "Pruning" in method else ("^" if "Voronoi" in method else "s")
                color = PALETTE.get(method, "#333333")
                ax.plot(
                    m_sub["mass_kg"],
                    m_sub["stiffness_N_m"],
                    marker=marker,
                    markersize=5,
                    color=color,
                    label=label,
                )

        ax.set_yscale("log")
        ax.set_xlabel("Total Structural Mass (kg)")
        if idx == 0:
            ax.set_ylabel("Stiffness K = 1/C (N/m, log scale)")
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
    """Plot stacked bar chart of Axial vs Bending strain energy shares."""
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    lcs = ["LC1", "LC2", "LC3"]
    methods = ["RemodelingPruning", "GeneticAlgorithm", "Voronoi", "Honeycomb"]
    m_labels = ["Pruning", "GA", "Voronoi", "Honeycomb"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)

    for idx, lc in enumerate(lcs):
        ax = axes[idx]
        sub = df[df["load_case"] == lc]

        axial_shares = []
        bending_shares = []
        for m in methods:
            row = sub[sub["method"] == m]
            if not row.empty:
                ax_val = float(row["axial_energy_share"].iloc[0]) * 100.0
                bend_val = float(row["bending_energy_share"].iloc[0]) * 100.0
            else:
                ax_val, bend_val = 0.0, 0.0
            axial_shares.append(ax_val)
            bending_shares.append(bend_val)

        x_indices = np.arange(len(methods))
        ax.bar(
            x_indices,
            axial_shares,
            label="Axial (Stretch)" if idx == 0 else "",
            color="#2b5c8f",
            width=0.55,
            edgecolor="#222222",
        )
        ax.bar(
            x_indices,
            bending_shares,
            bottom=axial_shares,
            label="Bending (Flexure)" if idx == 0 else "",
            color="#e67e22",
            width=0.55,
            edgecolor="#222222",
        )

        ax.set_xticks(x_indices)
        ax.set_xticklabels(m_labels, rotation=20, ha="right")
        ax.set_title(f"Load Case {lc}", fontweight="bold")
        ax.set_ylim([0, 100])
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        if idx == 0:
            ax.set_ylabel("Strain Energy Share (%)")
            ax.legend(loc="lower left", frameon=True, facecolor="white")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated Paper Figure 6: {out_path}")


def plot_fig7_robustness_matrix(
    csv_path: str = "results/e4_cross_load_robustness.csv",
    out_path: str = "figures/paper_fig7_robustness_matrix.png",
) -> None:
    """Plot cross-load-case degradation heatmap."""
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    lcs = ["LC1", "LC2", "LC3"]

    matrix = np.full((3, 3), np.nan)
    for i, opt_lc in enumerate(lcs):
        for j, eval_lc in enumerate(lcs):
            row = df[(df["optimized_for"] == opt_lc) & (df["evaluated_on"] == eval_lc)]
            if not row.empty and not pd.isna(row["degradation_ratio"].iloc[0]):
                matrix[i, j] = float(row["degradation_ratio"].iloc[0])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=1.0, vmax=max(5.0, np.nanmax(matrix)))

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(["LC1 (Cantilever)", "LC2 (SS Beam)", "LC3 (Compression)"])
    ax.set_yticklabels(["Opt LC1", "Opt LC2", "Opt LC3"])
    ax.set_xlabel("Evaluated Load Case", fontweight="bold")
    ax.set_ylabel("Optimized-For Load Case", fontweight="bold")
    ax.set_title("Cross-Load Degradation Ratio (C_cross / C_native)", fontweight="bold")

    for i in range(3):
        for j in range(3):
            val = matrix[i, j]
            text = f"{val:.2f}x" if not np.isnan(val) else "FAIL"
            color = "white" if val > 2.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Degradation Factor", rotation=270, labelpad=15)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated Paper Figure 7: {out_path}")


def plot_fig8_scaling(
    csv_path: str = "results/e7_scaling.csv",
    out_path: str = "figures/paper_fig8_scaling.png",
) -> None:
    """Plot scaling runtime vs number of structural members and nodes."""
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
    """Generate all publication paper figures from result CSVs."""
    plot_fig3_convergence()
    plot_fig4_distributions()
    plot_fig5_pareto()
    plot_fig6_kinematics()
    plot_fig7_robustness_matrix()
    plot_fig8_scaling()

