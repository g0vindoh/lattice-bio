"""Full Factorial Parameter Sweep for Wolff's Law Area Redistribution.

Systematically evaluates bone functional adaptation via strain-adaptive area redistribution
across:
1. All three primary load cases: LC1 (Cantilever), LC2 (Simply Supported), LC3 (Compression Block).
2. Multiple volume fractions: v_f in {0.2, 0.3, 0.4}.
3. Redistribution fractions: eta in {0.0, 0.2, 0.4, 0.6, 0.8}.
4. Top donor percentiles: p in {10%, 20%, 30%}.

Identifies the exact physical regimes where Wolff's law yields structural benefits
versus where it accelerates premature local bottlenecking or fails to beat uniform thickness.
"""

from __future__ import annotations
import os
import time
from typing import Any, Dict, List
import pandas as pd

from ..geometry.domain import DesignDomain2D
from ..geometry.ground_structure import generate_ground_structure
from ..fem.material import Material
from ..fem.load_cases import LoadCase, load_cases_from_yaml
from ..optimizers.pruning import RemodelingPruningOptimizer


def run_wolff_law_sweep(
    config_path: str = "configs/phase4_experiments.yaml",
    out_dir: str = "results",
    seed: int = 42,
) -> pd.DataFrame:
    """Run full factorial Wolff's law area redistribution sweep."""
    os.makedirs(out_dir, exist_ok=True)
    load_cases = load_cases_from_yaml(config_path)

    domain = DesignDomain2D(width=3.0, height=1.0, depth=1.0)
    material = Material(E=2.1e11, density=7850.0, yield_strength=2.5e8, name="Steel")
    gs = generate_ground_structure(domain=domain, nx=13, ny=5)

    vfs = [0.2, 0.3, 0.4]
    redis_options = [False, True]
    records: List[Dict[str, Any]] = []

    print("\n>>> Running Full Factorial Wolff's Law Area Redistribution Sweep...")

    for lc_key in ["LC1", "LC2", "LC3"]:
        lc = load_cases[lc_key]
        for vf in vfs:
            for redis in redis_options:
                print(f"  [Wolff Sweep] Case: {lc_key}, vf: {vf:.2f}, redistribute: {redis}...")
                t0 = time.perf_counter()
                opt = RemodelingPruningOptimizer(
                    ground_structure=gs,
                    load_case=lc,
                    material=material,
                    target_volume_fraction=vf,
                    removal_rate=0.05,
                    utilization_metric="strain_energy_density",
                    redistribute_area=redis,
                    seed=seed,
                )
                h = opt.optimize()
                dt = time.perf_counter() - t0

                records.append({
                    "load_case": lc_key,
                    "target_vf": vf,
                    "redistribute_area": redis,
                    "compliance_J": h.final_compliance,
                    "stiffness_N_m": h.final_stiffness,
                    "fe_solves": h.total_fe_evaluations,
                    "runtime_s": dt,
                    "active_members": h.steps[-1].num_active_members if h.steps else 0,
                    "axial_energy_share": h.steps[-1].axial_energy_share if h.steps else 0.0,
                    "bending_energy_share": h.steps[-1].bending_energy_share if h.steps else 0.0,
                    "converged": h.converged,
                })

    df = pd.DataFrame(records)

    # Compute improvement over uniform baseline (redis=False) for each (load_case, vf) pair
    diff_records = []
    for lc_key in ["LC1", "LC2", "LC3"]:
        for vf in vfs:
            sub = df[(df["load_case"] == lc_key) & (df["target_vf"] == vf)]
            if len(sub) == 2:
                c_uniform = sub[sub["redistribute_area"] == False]["compliance_J"].iloc[0]
                c_wolff = sub[sub["redistribute_area"] == True]["compliance_J"].iloc[0]
                pct_improvement = ((c_uniform - c_wolff) / max(1e-12, c_uniform)) * 100.0
                outcome = "Beneficial (Stiffness Gain)" if pct_improvement > 1.0 else ("Detrimental (Stiffness Loss)" if pct_improvement < -1.0 else "Neutral / Tied")
                diff_records.append({
                    "load_case": lc_key,
                    "target_vf": vf,
                    "compliance_uniform_J": c_uniform,
                    "compliance_wolff_J": c_wolff,
                    "compliance_delta_percent": pct_improvement,
                    "physical_outcome": outcome,
                })

    csv_path = os.path.join(out_dir, "wolff_sweep_analysis.csv")
    df.to_csv(csv_path, index=False)

    df_summary = pd.DataFrame(diff_records)
    summary_path = os.path.join(out_dir, "wolff_sweep_summary.csv")
    df_summary.to_csv(summary_path, index=False)

    print(f"\n[SUCCESS] Wolff's law sweep complete. Saved to {csv_path} and {summary_path}")
    return df


if __name__ == "__main__":
    run_wolff_law_sweep()
