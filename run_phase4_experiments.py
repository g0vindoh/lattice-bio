"""Master executable runner for Phase 4 Experimental Campaign (E1 to E7).

Runs all research experiments, verifies data integrity, applies non-parametric
statistical corrections, generates publication figures, and compiles the
authoritative summary report.
"""

from __future__ import annotations
import os
import sys
import time
import yaml
import pandas as pd

from src.experiments.experiment_runner import (
    run_experiment_e1,
    run_experiment_e2,
    run_experiment_e3,
    run_experiment_e4,
    run_experiment_e5,
    run_experiment_e6,
    run_experiment_e7,
)
from src.export.plot_paper_figures import generate_all_paper_figures


def main() -> None:
    start_total = time.time()
    config_path = "configs/phase4_experiments.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("=" * 80)
    print(" PHASE 4: FULL EXPERIMENTAL CAMPAIGN (E1 to E7)")
    print(f" Config: {config_path} | Seed: {cfg.get('seed', 42)}")
    print(f" Python: {sys.version.split()[0]} on {sys.platform}")
    print("=" * 80)

    # 1. Run E1
    t0 = time.time()
    df_e1, df_stat = run_experiment_e1(cfg, out_dir="results")
    print(f"-> E1 completed in {time.time() - t0:.2f} s")

    # 2. Run E2
    t0 = time.time()
    df_e2 = run_experiment_e2(cfg, out_dir="results")
    print(f"-> E2 completed in {time.time() - t0:.2f} s")

    # 3. Run E3
    t0 = time.time()
    df_e3 = run_experiment_e3(cfg, out_dir="results")
    print(f"-> E3 completed in {time.time() - t0:.2f} s")

    # 4. Run E4
    t0 = time.time()
    df_e4 = run_experiment_e4(cfg, out_dir="results")
    print(f"-> E4 completed in {time.time() - t0:.2f} s")

    # 5. Run E5
    t0 = time.time()
    df_e5 = run_experiment_e5(cfg, out_dir="results")
    print(f"-> E5 completed in {time.time() - t0:.2f} s")

    # 6. Run E6
    t0 = time.time()
    df_e6 = run_experiment_e6(cfg, out_dir="results")
    print(f"-> E6 completed in {time.time() - t0:.2f} s")

    # 7. Run E7
    t0 = time.time()
    df_e7 = run_experiment_e7(cfg, out_dir="results")
    print(f"-> E7 completed in {time.time() - t0:.2f} s")

    # 8. Generate Publication Paper Figures
    print("\n>>> Generating Publication Paper Figures...")
    generate_all_paper_figures()

    # 9. Generate Executive Markdown Summary Report
    print("\n>>> Compiling Authoritative Paper Summary Report...")
    summary_path = "results/phase4_paper_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Phase 4 Experimental Campaign: Final Paper Benchmark Summary\n\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')} | Total Runtime: {time.time() - start_total:.1f} s\n\n")

        f.write("## 1. Experiment E1: Statistical Performance Summary (vf = 0.20)\n\n")
        sub_e1 = df_e1[df_e1["target_vf"] == 0.20]
        agg_e1 = sub_e1.groupby(["load_case", "method"]).agg(
            median_compliance=("compliance_J", "median"),
            iqr_compliance=("compliance_J", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            median_stiffness=("stiffness_N_m", "median"),
            median_specific_stiffness=("specific_stiffness", "median"),
            evaluations=("fe_solves", "median"),
            runtime=("runtime_s", "median"),
        ).reset_index()
        f.write(agg_e1.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 2. Hypothesis Testing: Mann-Whitney U Tests with Holm-Bonferroni Correction\n\n")
        f.write(df_stat.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 3. Experiment E4: Cross-Load Generalization Degradation Matrix\n\n")
        piv_e4 = df_e4.pivot(index="optimized_for", columns="evaluated_on", values="degradation_ratio")
        f.write(piv_e4.to_markdown())
        f.write("\n\n")

        f.write("## 4. Experiment E5: Kinematics (Axial vs Bending Energy Shares at vf = 0.20)\n\n")
        f.write(df_e5.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 5. Experiment E6: Ablation Studies (LC1, vf = 0.20)\n\n")
        f.write(df_e6.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 6. Experiment E7: Computational Scaling Benchmark\n\n")
        f.write(df_e7.to_markdown(index=False))
        f.write("\n")

    print(f"Successfully generated summary report: {summary_path}")
    print(f"\n================================================================================")
    print(f" Phase 4 Complete! Total Execution Time: {time.time() - start_total:.2f} s")
    print(f"================================================================================\n")


if __name__ == "__main__":
    main()

