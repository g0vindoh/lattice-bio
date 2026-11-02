"""Reproducible Phase 3 script: Remodeling Pruning vs Genetic Algorithm comparison.

Runs Method A and Method B across load cases LC1, LC2, and LC3 at target volume fraction 0.2.
Exports trajectory metrics, comparison tables, and static deformed visualizations.
"""

from __future__ import annotations
import os
import sys
import yaml
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.geometry.domain import DesignDomain2D
from src.geometry.ground_structure import generate_ground_structure
from src.fem.load_cases import load_cases_from_yaml
from src.fem.material import Material
from src.optimizers.pruning import RemodelingPruningOptimizer
from src.optimizers.ga import GeneticAlgorithmOptimizer
from src.export.plotting import plot_deformed_shape_with_energy


def run_phase3_comparison(config_path: str = "configs/phase3_optimizers.yaml") -> Tuple[pd.DataFrame, Dict]:
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
        yield_strength=float(cfg["material"]["yield_strength"]),
        name=cfg["material"]["name"],
    )
    load_cases = load_cases_from_yaml(config_path)
    target_vf = float(cfg["target_volume_fraction"])

    gs_cfg = cfg["ground_structure"]
    gs = generate_ground_structure(
        domain=domain,
        nx=int(gs_cfg["nx"]),
        ny=int(gs_cfg["ny"]),
        radius=float(gs_cfg["radius"]) if gs_cfg["radius"] is not None else None,
    )

    pruning_cfg = cfg["pruning"]
    ga_cfg = cfg["genetic_algorithm"]

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    summary_records = []
    trajectories = {}

    print(f"\n================================================================================")
    print(f" Phase 3: Remodeling-Inspired Pruning vs Genetic Algorithm Optimization")
    print(f" Target Volume Fraction: {target_vf:.2f} | Ground Structure: {gs.num_nodes} nodes, {gs.num_members} members")
    print(f"================================================================================\n")

    for lc_key, lc in load_cases.items():
        print(f"--- Running Load Case {lc_key} ({lc.name}) ---")

        # 1. Method A: Remodeling-Inspired Pruning
        print(f"  [Method A] Running Remodeling-Inspired Pruning...")
        opt_pruning = RemodelingPruningOptimizer(
            ground_structure=gs,
            load_case=lc,
            material=material,
            target_volume_fraction=target_vf,
            removal_rate=float(pruning_cfg["removal_rate"]),
            utilization_metric=pruning_cfg["utilization_metric"],
            redistribute_area=bool(pruning_cfg["redistribute_area"]),
            initial_thickness=float(pruning_cfg["initial_thickness"]),
            max_iterations=int(pruning_cfg["max_iterations"]),
            seed=int(cfg["seed"]),
        )
        h_pruning = opt_pruning.optimize()
        res_p = h_pruning.final_result
        trajectories[f"Pruning_{lc_key}"] = h_pruning

        summary_records.append(
            {
                "Method": "Method A: Remodeling Pruning",
                "Load Case": lc_key,
                "Target VF": target_vf,
                "Achieved VF": h_pruning.steps[-1].volume_fraction if h_pruning.steps else target_vf,
                "Compliance (J)": h_pruning.final_compliance,
                "Stiffness (N/m)": h_pruning.final_stiffness,
                "FE Solves": h_pruning.total_fe_evaluations,
                "Runtime (s)": h_pruning.runtime_seconds,
                "Active Members": h_pruning.steps[-1].num_active_members if h_pruning.steps else 0,
                "Axial Share (%)": f"{res_p.axial_energy_fraction * 100:.1f}%" if res_p else "N/A",
                "Bending Share (%)": f"{res_p.bending_energy_fraction * 100:.1f}%" if res_p else "N/A",
            }
        )

        if res_p is not None:
            plot_deformed_shape_with_energy(
                res_p,
                domain,
                f"Method A: Remodeling Pruning ({lc_key})",
                f"figures/phase3_pruned_deformed_{lc_key}.png",
            )

        # 2. Method B: Genetic Algorithm
        print(f"  [Method B] Running Genetic Algorithm (Generations: {ga_cfg['generations']}, Pop: {ga_cfg['pop_size']})...")
        opt_ga = GeneticAlgorithmOptimizer(
            ground_structure=gs,
            load_case=lc,
            material=material,
            target_volume_fraction=target_vf,
            pop_size=int(ga_cfg["pop_size"]),
            generations=int(ga_cfg["generations"]),
            crossover_rate=float(ga_cfg["crossover_rate"]),
            mutation_rate=float(ga_cfg["mutation_rate"]),
            tournament_size=int(ga_cfg["tournament_size"]),
            elite_count=int(ga_cfg["elite_count"]),
            discrete_levels=[float(x) for x in ga_cfg["discrete_levels"]],
            seed=int(cfg["seed"]),
        )
        h_ga = opt_ga.optimize()
        res_ga = h_ga.final_result
        trajectories[f"GA_{lc_key}"] = h_ga

        summary_records.append(
            {
                "Method": "Method B: Genetic Algorithm",
                "Load Case": lc_key,
                "Target VF": target_vf,
                "Achieved VF": h_ga.steps[-1].volume_fraction if h_ga.steps else target_vf,
                "Compliance (J)": h_ga.final_compliance,
                "Stiffness (N/m)": h_ga.final_stiffness,
                "FE Solves": h_ga.total_fe_evaluations,
                "Runtime (s)": h_ga.runtime_seconds,
                "Active Members": h_ga.steps[-1].num_active_members if h_ga.steps else 0,
                "Axial Share (%)": f"{res_ga.axial_energy_fraction * 100:.1f}%" if res_ga else "N/A",
                "Bending Share (%)": f"{res_ga.bending_energy_fraction * 100:.1f}%" if res_ga else "N/A",
            }
        )

        if res_ga is not None:
            plot_deformed_shape_with_energy(
                res_ga,
                domain,
                f"Method B: Genetic Algorithm ({lc_key})",
                f"figures/phase3_ga_deformed_{lc_key}.png",
            )

    df_summary = pd.DataFrame(summary_records)
    csv_path = "results/phase3_optimizers_summary.csv"
    md_path = "results/phase3_optimizers_summary.md"
    df_summary.to_csv(csv_path, index=False)

    with open(md_path, "w") as f:
        f.write("# Phase 3 Optimization Results: Pruning vs Genetic Algorithm\n\n")
        f.write(f"Target Volume Fraction: {target_vf:.2f}\n\n")
        f.write(df_summary.to_markdown(index=False))
        f.write("\n")

    print(f"\nSuccessfully exported results to {csv_path} and {md_path}")
    print("\n" + df_summary.to_string(index=False))

    # Generate Convergence Plots
    plot_convergence_curves(trajectories)

    return df_summary, trajectories


def plot_convergence_curves(trajectories: Dict[str, OptimizationHistory]) -> None:
    """Plot convergence curves of compliance vs FE evaluations for LC1, LC2, and LC3."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=300)
    lcs = ["LC1", "LC2", "LC3"]
    titles = ["LC1 (Cantilever)", "LC2 (Simply Supported)", "LC3 (Compression Block)"]

    for idx, (lc, title) in enumerate(zip(lcs, titles)):
        ax = axes[idx]
        h_p = trajectories.get(f"Pruning_{lc}")
        h_ga = trajectories.get(f"GA_{lc}")

        if h_p and h_p.steps:
            df_p = h_p.to_dataframe()
            ax.plot(
                df_p["fe_evaluations"],
                df_p["compliance_J"],
                "o-",
                color="#e74c3c",
                lw=2,
                label="Method A: Pruning",
            )

        if h_ga and h_ga.steps:
            df_ga = h_ga.to_dataframe()
            # Cumulative minimum compliance for GA
            cum_min_c = np.minimum.accumulate(df_ga["compliance_J"])
            ax.plot(
                df_ga["fe_evaluations"],
                cum_min_c,
                "s--",
                color="#2980b9",
                lw=2,
                label="Method B: GA (Best-so-far)",
            )

        ax.set_yscale("log")
        ax.set_xlabel("FE Evaluations")
        ax.set_ylabel("Compliance C (J, log scale)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        if idx == 0:
            ax.legend(fontsize=9)

    plt.tight_layout()
    fig_path = "figures/phase3_convergence_comparison.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Generated convergence trajectory plot: {fig_path}")


if __name__ == "__main__":
    run_phase3_comparison()

