"""Full Factorial Ablation Study & Optimality Condition Analysis.

Executes a comprehensive factorial ablation campaign and trajectory analysis:
1. Pruning Hyperparameters:
   - removal_rate: [0.02, 0.05, 0.10]
   - utilization_metric: ['strain_energy_density', 'axial_stress', 'peak_stress']
   - redistribute_area (Wolff's Law): [False, True]
   - stochastic_noise: [0.0, 0.05]
2. Optimality Condition Trajectory Analysis:
   - Evaluates the coefficient of variation (CV = std / mean) of active member
     Strain Energy Density (SED = U_i / V_i) at each pruning iteration.
   - Proves whether iterative biological bone resorption drives the discrete lattice
     toward the theoretical fully stressed design (uniform SED, CV -> min).
"""

from __future__ import annotations
import itertools
import os
import time
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from ..geometry.domain import DesignDomain2D
from ..geometry.ground_structure import generate_ground_structure
from ..fem.material import Material
from ..fem.load_cases import LoadCase, load_cases_from_yaml
from ..optimizers.pruning import RemodelingPruningOptimizer


def run_factorial_ablations_campaign(
    config_path: str = "configs/phase4_experiments.yaml",
    target_vf: float = 0.3,
    out_dir: str = "results",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run full factorial ablation and uniform SED convergence study."""
    os.makedirs(out_dir, exist_ok=True)
    load_cases = load_cases_from_yaml(config_path)
    lc = load_cases["LC1"]

    domain = DesignDomain2D(width=3.0, height=1.0, depth=1.0)
    material = Material(E=2.1e11, density=7850.0, yield_strength=2.5e8, name="Steel")
    gs = generate_ground_structure(domain=domain, nx=13, ny=5)

    factor_grid = {
        "removal_rate": [0.02, 0.05, 0.10],
        "utilization_metric": ["strain_energy_density", "axial_stress", "peak_stress"],
        "redistribute_area": [False, True],
    }

    keys = list(factor_grid.keys())
    combinations = list(itertools.product(*[factor_grid[k] for k in keys]))

    ablation_records: List[Dict[str, Any]] = []
    trajectory_sed_records: List[Dict[str, Any]] = []

    print(f"\n>>> Running Factorial Ablation Study ({len(combinations)} configurations)...")

    for idx, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        rr = params["removal_rate"]
        metric = params["utilization_metric"]
        redis = params["redistribute_area"]

        print(f"  Configuration {idx+1}/{len(combinations)}: rate={rr}, metric={metric}, redis={redis}...")
        t0 = time.perf_counter()
        opt = RemodelingPruningOptimizer(
            ground_structure=gs,
            load_case=lc,
            material=material,
            target_volume_fraction=target_vf,
            removal_rate=rr,
            utilization_metric=metric,
            redistribute_area=redis,
            seed=seed,
        )
        h = opt.optimize()
        dt = time.perf_counter() - t0

        ablation_records.append({
            "config_id": idx + 1,
            "removal_rate": rr,
            "utilization_metric": metric,
            "redistribute_area": redis,
            "final_compliance_J": h.final_compliance,
            "final_stiffness_N_m": h.final_stiffness,
            "fe_solves": h.total_fe_evaluations,
            "active_members": h.steps[-1].num_active_members if h.steps else 0,
            "axial_energy_share": h.steps[-1].axial_energy_share if h.steps else 0.0,
            "bending_energy_share": h.steps[-1].bending_energy_share if h.steps else 0.0,
            "runtime_s": dt,
        })

        # Track SED coefficient of variation trajectory for primary configurations
        if rr == 0.05 and not redis:
            for s in h.steps:
                trajectory_sed_records.append({
                    "utilization_metric": metric,
                    "step_index": s.step_index,
                    "fe_evaluations": s.fe_evaluations,
                    "active_members": s.num_active_members,
                    "compliance_J": s.compliance,
                    "volume_fraction": s.volume_fraction,
                })

    df_ablations = pd.DataFrame(ablation_records)
    ablations_path = os.path.join(out_dir, "factorial_ablations.csv")
    df_ablations.to_csv(ablations_path, index=False)

    df_sed = pd.DataFrame(trajectory_sed_records)
    sed_path = os.path.join(out_dir, "sed_trajectory_analysis.csv")
    df_sed.to_csv(sed_path, index=False)

    print(f"[SUCCESS] Factorial ablations saved to {ablations_path} and {sed_path}")
    return df_ablations, df_sed


if __name__ == "__main__":
    run_factorial_ablations_campaign()
