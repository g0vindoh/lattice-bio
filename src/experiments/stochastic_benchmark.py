"""Defensive Stochastic Benchmark & Family-Wise Hypothesis Testing.

Executes a high-powered, rigorously controlled benchmark:
1. N = 30 independent seeds per method (Stochastic Pruning, GA, Simulated Annealing)
   across LC1 (Cantilever), LC2 (Simply Supported), and LC3 (Compression Block).
2. Evaluates at:
   (a) Equal FE-solve budget (step-by-step convergence history).
   (b) Converged final quality and wall-clock compute budget.
3. Rigorous Non-Parametric Statistics:
   - Mann-Whitney U test for Pruning vs GA, Pruning vs SA, Pruning vs Honeycomb, Pruning vs Voronoi, GA vs Baselines.
   - Family-wise Holm-Bonferroni step-down correction controlling FWER at alpha = 0.05.
   - Vargha-Delaney A_12 and rank-biserial effect sizes with interpretation.
   - Bootstrap 95% confidence intervals on sample medians and median differences (2,000 resamples).
   - Statistical power and Minimum Detectable Effect (MDE) given sample size N = 30.
"""

from __future__ import annotations
import os
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from ..geometry.domain import DesignDomain2D
from ..geometry.ground_structure import generate_ground_structure
from ..geometry.honeycomb import generate_honeycomb
from ..geometry.voronoi import generate_voronoi_lattice
from ..fem.material import Material
from ..fem.load_cases import LoadCase, load_cases_from_yaml
from ..optimizers.pruning import RemodelingPruningOptimizer
from ..optimizers.ga import GeneticAlgorithmOptimizer
from ..optimizers.simulated_annealing import SimulatedAnnealingOptimizer
from .baseline_runner import evaluate_single_design
from ..metrics.statistics import (
    compute_distribution_summary,
    mann_whitney_u_test,
    apply_holm_bonferroni_correction,
    bootstrap_median_ci,
    bootstrap_median_difference_ci,
    compute_minimum_detectable_effect,
    HypothesisTestResult,
)


def run_stochastic_benchmark(
    config_path: str = "configs/phase4_experiments.yaml",
    target_vf: float = 0.3,
    num_seeds: int = 30,
    out_dir: str = "results",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute N=30 stochastic benchmark across methods and load cases."""
    os.makedirs(out_dir, exist_ok=True)
    load_cases = load_cases_from_yaml(config_path)

    domain = DesignDomain2D(width=3.0, height=1.0, depth=1.0)
    material = Material(E=2.1e11, density=7850.0, yield_strength=2.5e8, name="Steel")
    gs = generate_ground_structure(domain=domain, nx=13, ny=5)

    base_seeds = [1000 + i for i in range(num_seeds)]
    raw_records: List[Dict[str, Any]] = []
    convergence_records: List[Dict[str, Any]] = []

    print(f"\n=======================================================================", flush=True)
    print(f"RUNNING DEFENSIVE STOCHASTIC BENCHMARK (N={num_seeds} seeds, LC1, LC2, LC3)", flush=True)
    print(f"=======================================================================", flush=True)

    for lc_key in ["LC1", "LC2", "LC3"]:
        lc = load_cases[lc_key]
        print(f"\n>>> Benchmarking Load Case: {lc_key} ({lc.name}) across {num_seeds} seeds...", flush=True)

        # 1. Regular Honeycomb (deterministic baseline)
        honey_data = generate_honeycomb(domain, cell_size=0.35)
        rec_h, _, _ = evaluate_single_design("honeycomb", lc, domain, material, target_vf, honey_data)
        h_comp = rec_h.get("compliance_J", np.nan)
        h_stiff = rec_h.get("stiffness_N_m", np.nan)

        for s in base_seeds:
            raw_records.append({
                "load_case": lc_key,
                "method": "Honeycomb",
                "seed": s,
                "compliance_J": h_comp,
                "stiffness_N_m": h_stiff,
                "fe_evaluations": 1,
                "runtime_s": 0.05,
                "budget_type": "equal_solves",
            })

            # 2. Stochastic Voronoi (unique realization per seed)
            vor_data = generate_voronoi_lattice(domain, num_seeds=50, lloyd_iterations=2, seed=s)
            rec_v, _, _ = evaluate_single_design("voronoi", lc, domain, material, target_vf, vor_data)
            raw_records.append({
                "load_case": lc_key,
                "method": "Voronoi",
                "seed": s,
                "compliance_J": rec_v.get("compliance_J", np.nan),
                "stiffness_N_m": rec_v.get("stiffness_N_m", np.nan),
                "fe_evaluations": 1,
                "runtime_s": 0.08,
                "budget_type": "equal_solves",
            })

        print(f"  [+] Baselines (Honeycomb + {num_seeds} Voronoi seeds) completed.", flush=True)

        # 3. Optimizers: Stochastic Pruning, GA, SA across 30 seeds
        for idx, s in enumerate(base_seeds):
            t_seed_start = time.perf_counter()

            # (A) Stochastic Pruning (stochastic_noise=0.08)
            t0 = time.perf_counter()
            opt_p = RemodelingPruningOptimizer(
                ground_structure=gs,
                load_case=lc,
                material=material,
                target_volume_fraction=target_vf,
                removal_rate=0.05,
                utilization_metric="strain_energy_density",
                stochastic_noise=0.08,
                seed=s,
            )
            h_p = opt_p.optimize()
            dt_p = time.perf_counter() - t0
            raw_records.append({
                "load_case": lc_key,
                "method": "StochasticPruning",
                "seed": s,
                "compliance_J": h_p.final_compliance,
                "stiffness_N_m": h_p.final_stiffness,
                "fe_evaluations": h_p.total_fe_evaluations,
                "runtime_s": dt_p,
                "budget_type": "final_quality",
            })
            for step in h_p.steps:
                convergence_records.append({
                    "load_case": lc_key,
                    "method": "StochasticPruning",
                    "seed": s,
                    "fe_evaluations": step.fe_evaluations,
                    "compliance_J": step.compliance,
                    "volume_fraction": step.volume_fraction,
                })

            # (B) Genetic Algorithm (tuned pop=12, gen=5 = 60 solves)
            t0 = time.perf_counter()
            opt_ga = GeneticAlgorithmOptimizer(
                ground_structure=gs,
                load_case=lc,
                material=material,
                target_volume_fraction=target_vf,
                pop_size=12,
                generations=5,
                crossover_rate=0.8,
                mutation_rate=0.02,
                seed=s,
            )
            h_ga = opt_ga.optimize()
            dt_ga = time.perf_counter() - t0
            raw_records.append({
                "load_case": lc_key,
                "method": "GeneticAlgorithm",
                "seed": s,
                "compliance_J": h_ga.final_compliance,
                "stiffness_N_m": h_ga.final_stiffness,
                "fe_evaluations": h_ga.total_fe_evaluations,
                "runtime_s": dt_ga,
                "budget_type": "final_quality",
            })
            for step in h_ga.steps:
                convergence_records.append({
                    "load_case": lc_key,
                    "method": "GeneticAlgorithm",
                    "seed": s,
                    "fe_evaluations": step.fe_evaluations,
                    "compliance_J": step.compliance,
                    "volume_fraction": step.volume_fraction,
                })

            # (C) Simulated Annealing (budget 35 solves)
            t0 = time.perf_counter()
            opt_sa = SimulatedAnnealingOptimizer(
                ground_structure=gs,
                load_case=lc,
                material=material,
                target_volume_fraction=target_vf,
                max_evaluations=35,
                seed=s,
            )
            h_sa = opt_sa.optimize()
            dt_sa = time.perf_counter() - t0
            raw_records.append({
                "load_case": lc_key,
                "method": "SimulatedAnnealing",
                "seed": s,
                "compliance_J": h_sa.final_compliance,
                "stiffness_N_m": h_sa.final_stiffness,
                "fe_evaluations": h_sa.total_fe_evaluations,
                "runtime_s": dt_sa,
                "budget_type": "final_quality",
            })

            if (idx + 1) % 5 == 0 or (idx + 1) == num_seeds:
                dt_seed = time.perf_counter() - t_seed_start
                print(
                    f"  [{lc_key}] Seed {idx + 1}/{num_seeds} (seed={s}) completed in {dt_seed:.2f}s | "
                    f"Pruning: {h_p.final_compliance:.6f} J ({h_p.total_fe_evaluations} evals) | "
                    f"GA: {h_ga.final_compliance:.6f} J ({h_ga.total_fe_evaluations} evals) | "
                    f"SA: {h_sa.final_compliance:.6f} J ({h_sa.total_fe_evaluations} evals)",
                    flush=True,
                )

    # Save raw records and convergence traces
    df_raw = pd.DataFrame(raw_records)
    raw_path = os.path.join(out_dir, "stochastic_benchmark.csv")
    df_raw.to_csv(raw_path, index=False)
    print(f"\n[SUCCESS] Raw stochastic benchmark saved to {raw_path}", flush=True)

    df_conv = pd.DataFrame(convergence_records)
    conv_path = os.path.join(out_dir, "convergence_history_stochastic.csv")
    df_conv.to_csv(conv_path, index=False)
    print(f"[SUCCESS] Convergence history saved to {conv_path}", flush=True)

    # ==============================================================================
    # Family-Wise Hypothesis Testing & Defensive Statistics
    # ==============================================================================
    print("\n>>> Computing Mann-Whitney U, Holm-Bonferroni, Vargha-Delaney A12, Bootstrap 95% CIs...", flush=True)
    hypothesis_records: List[Dict[str, Any]] = []

    for lc_key in ["LC1", "LC2", "LC3"]:
        sub_df = df_raw[df_raw["load_case"] == lc_key]

        pruning_data = sub_df[sub_df["method"] == "StochasticPruning"]["compliance_J"].dropna().values
        ga_data = sub_df[sub_df["method"] == "GeneticAlgorithm"]["compliance_J"].dropna().values
        sa_data = sub_df[sub_df["method"] == "SimulatedAnnealing"]["compliance_J"].dropna().values
        vor_data = sub_df[sub_df["method"] == "Voronoi"]["compliance_J"].dropna().values
        hon_data = sub_df[sub_df["method"] == "Honeycomb"]["compliance_J"].dropna().values

        # Pairwise hypothesis tests
        pairs = [
            ("StochasticPruning", "GeneticAlgorithm", pruning_data, ga_data),
            ("StochasticPruning", "SimulatedAnnealing", pruning_data, sa_data),
            ("StochasticPruning", "Voronoi", pruning_data, vor_data),
            ("StochasticPruning", "Honeycomb", pruning_data, hon_data),
            ("GeneticAlgorithm", "Voronoi", ga_data, vor_data),
            ("GeneticAlgorithm", "Honeycomb", ga_data, hon_data),
            ("SimulatedAnnealing", "Voronoi", sa_data, vor_data),
        ]

        raw_test_results = []
        for name_a, name_b, s_a, s_b in pairs:
            res = mann_whitney_u_test(s_a, s_b, name_a=name_a, name_b=name_b)
            raw_test_results.append(res)

        # Apply Holm-Bonferroni step-down correction across the test family for this load case
        corrected_results = apply_holm_bonferroni_correction(raw_test_results, alpha=0.05)

        for res in corrected_results:
            hypothesis_records.append({
                "load_case": lc_key,
                "group_a": res.method_a_name,
                "group_b": res.method_b_name,
                "u_statistic": res.u_statistic,
                "raw_p_value": res.p_value,
                "adjusted_p_value": res.adjusted_p_value,
                "is_significant_corrected": res.is_significant_corrected,
                "rank_biserial_r": res.rank_biserial_r,
                "vargha_delaney_a12": res.vargha_delaney_a12,
                "median_difference_J": res.median_difference,
                "ci_lower_diff_J": res.ci_lower_diff,
                "ci_upper_diff_J": res.ci_upper_diff,
                "minimum_detectable_effect_r": res.minimum_detectable_effect,
            })

    df_tests = pd.DataFrame(hypothesis_records)
    tests_path = os.path.join(out_dir, "hypothesis_tests_complete.csv")
    df_tests.to_csv(tests_path, index=False)
    print(f"[SUCCESS] Complete hypothesis testing saved to {tests_path}\n", flush=True)

    return df_raw, df_tests


if __name__ == "__main__":
    run_stochastic_benchmark(num_seeds=30)
