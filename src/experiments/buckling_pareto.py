"""Buckling and Stress Realism Campaign: Compliance vs. Buckling Pareto Frontiers.

Evaluates structural behavior when compressive Euler member buckling instability
and material yield constraints are enforced:
1. Sweeps buckling penalty weights from 0.0 (unconstrained) to 10.0 (strongly buckling-constrained).
2. Evaluates global buckling load factors (BLF lambda_min) and peak normal stresses.
3. Quantifies the trade-off frontier between compliance and elastic compressive stability.
"""

from __future__ import annotations
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
from ..optimizers.ga import GeneticAlgorithmOptimizer
from ..metrics.buckling import evaluate_lattice_buckling
from ..metrics.metrics import compute_member_peak_stress


def run_buckling_pareto_campaign(
    config_path: str = "configs/phase4_experiments.yaml",
    target_vf: float = 0.3,
    out_dir: str = "results",
    seed: int = 42,
) -> pd.DataFrame:
    """Run buckling Pareto frontier analysis across penalty weights.

    Args:
        config_path: Path to configuration YAML.
        target_vf: Target volume fraction.
        out_dir: Destination directory for CSV outputs.
        seed: Random seed.

    Returns:
        DataFrame containing Pareto frontier points.
    """
    os.makedirs(out_dir, exist_ok=True)
    load_cases = load_cases_from_yaml(config_path)

    domain = DesignDomain2D(width=3.0, height=1.0, depth=1.0)
    material = Material(E=2.1e11, density=7850.0, yield_strength=2.5e8, name="Steel")
    gs = generate_ground_structure(domain=domain, nx=13, ny=5)

    penalty_weights = [0.0, 0.5, 1.0, 2.5, 5.0, 10.0]
    records: List[Dict[str, Any]] = []

    for lc_key in ["LC1", "LC2"]:
        lc = load_cases[lc_key]
        print(f"\n[Buckling Pareto] Analyzing {lc_key} ({lc.name}) across {len(penalty_weights)} penalty weights...")

        # 1. Pruning under varying buckling penalties
        for w in penalty_weights:
            print(f"  -> Remodeling Pruning with buckling penalty weight = {w}...")
            t0 = time.perf_counter()
            opt_p = RemodelingPruningOptimizer(
                ground_structure=gs,
                load_case=lc,
                material=material,
                target_volume_fraction=target_vf,
                removal_rate=0.05,
                utilization_metric="strain_energy_density",
                buckling_penalty_weight=w,
                yield_stress_limit=material.yield_strength,
                seed=seed,
            )
            h_p = opt_p.optimize()
            dt_p = time.perf_counter() - t0

            if h_p.final_result is not None:
                blf_data = evaluate_lattice_buckling(h_p.final_result)
                max_stress, _, _ = compute_member_peak_stress(h_p.final_result)
                records.append({
                    "load_case": lc_key,
                    "method": "RemodelingPruning",
                    "buckling_penalty_weight": w,
                    "compliance_J": h_p.final_compliance,
                    "stiffness_N_m": h_p.final_stiffness,
                    "global_buckling_factor": blf_data.global_buckling_factor,
                    "is_buckling_critical": blf_data.is_buckling_critical,
                    "num_buckling_critical_struts": blf_data.num_buckling_critical,
                    "critical_element_id": blf_data.critical_element_id,
                    "critical_slenderness_L_t": blf_data.critical_member_slenderness,
                    "max_stress_Pa": max_stress,
                    "stress_safety_factor": material.yield_strength / max(1e-12, max_stress),
                    "active_members": h_p.steps[-1].num_active_members if h_p.steps else 0,
                    "axial_energy_share": h_p.steps[-1].axial_energy_share if h_p.steps else np.nan,
                    "runtime_s": dt_p,
                })

        # 2. GA under baseline vs buckling-penalized search
        for w in [0.0, 2.0, 5.0]:
            print(f"  -> Genetic Algorithm with buckling penalty weight = {w}...")
            t0 = time.perf_counter()
            opt_ga = GeneticAlgorithmOptimizer(
                ground_structure=gs,
                load_case=lc,
                material=material,
                target_volume_fraction=target_vf,
                pop_size=20,
                generations=20,
                buckling_penalty_weight=w,
                yield_stress_limit=material.yield_strength,
                seed=seed,
            )
            h_ga = opt_ga.optimize()
            dt_ga = time.perf_counter() - t0

            if h_ga.final_result is not None:
                blf_data = evaluate_lattice_buckling(h_ga.final_result)
                max_stress, _, _ = compute_member_peak_stress(h_ga.final_result)
                records.append({
                    "load_case": lc_key,
                    "method": "GeneticAlgorithm",
                    "buckling_penalty_weight": w,
                    "compliance_J": h_ga.final_compliance,
                    "stiffness_N_m": h_ga.final_stiffness,
                    "global_buckling_factor": blf_data.global_buckling_factor,
                    "is_buckling_critical": blf_data.is_buckling_critical,
                    "num_buckling_critical_struts": blf_data.num_buckling_critical,
                    "critical_element_id": blf_data.critical_element_id,
                    "critical_slenderness_L_t": blf_data.critical_member_slenderness,
                    "max_stress_Pa": max_stress,
                    "stress_safety_factor": material.yield_strength / max(1e-12, max_stress),
                    "active_members": h_ga.steps[-1].num_active_members if h_ga.steps else 0,
                    "axial_energy_share": h_ga.steps[-1].axial_energy_share if h_ga.steps else np.nan,
                    "runtime_s": dt_ga,
                })

    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "buckling_pareto.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[SUCCESS] Buckling Pareto campaign complete. Saved to {csv_path}")
    return df


if __name__ == "__main__":
    run_buckling_pareto_campaign()
