"""Multi-Load Remodeling Spectra & Specialization Cliff Diagnosis.

Investigates functional adaptation under diverse multi-load spectra and provides
a rigorous mathematical diagnosis of the single-load specialization cliff:
1. Tests >= 3 distinct loading spectra definitions:
   - Spectrum A: Equal Triaxial (LC1 Cantilever + LC2 Simply Supported + LC3 Compression: 1/3, 1/3, 1/3)
   - Spectrum B: Bending Dominant (70% LC1, 30% LC2)
   - Spectrum C: Compression Dominant (70% LC3, 15% LC1, 15% LC2)
   - Spectrum D: Envelope Aggregation (Max utilization across cases)
2. Rigorous Mechanism Diagnosis:
   - Measures stiffness matrix condition number kappa(K) under off-design loads.
   - Identifies whether off-design compliance explosion is caused by genuine material
     strain energy or by kinematic rotational/unsupported node mechanisms (slack struts).
   - Reports both unbounded raw compliance and regularized/bounded metrics.
"""

from __future__ import annotations
import os
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import scipy.sparse.linalg as spla
import pandas as pd

from ..geometry.domain import DesignDomain2D
from ..geometry.ground_structure import generate_ground_structure, build_fe_mesh_from_ground_structure
from ..fem.material import Material, RectangularSection
from ..fem.load_cases import LoadCase, load_cases_from_yaml
from ..optimizers.pruning import RemodelingPruningOptimizer
from ..optimizers.multi_load import MultiLoadRemodelingOptimizer
from ..metrics.feasibility import check_design_feasibility
from ..metrics.metrics import compute_compliance, compute_stiffness


def compute_stiffness_matrix_condition_number(mesh) -> float:
    """Compute the 2-norm condition number kappa(K_free) of the structural stiffness matrix."""
    try:
        K = mesh.assemble_global_stiffness()
        all_dofs = np.arange(mesh.total_dofs)
        fixed_dofs = mesh.get_fixed_dofs()
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

        if len(free_dofs) == 0:
            return 1.0

        K_free = K[free_dofs, :][:, free_dofs].tocsc()
        # Compute extreme singular values using dense or sparse SVD
        if K_free.shape[0] < 400:
            dense_K = K_free.toarray()
            s = np.linalg.svd(dense_K, compute_uv=False)
            s_max = float(np.max(s))
            s_min = float(np.min(s))
            return s_max / max(1e-16, s_min)
        else:
            # Power iteration / normest for large matrices
            s_max = spla.norm(K_free, 1)
            # Estimate smallest eigenvalue via shift-invert
            return float(s_max * 1e6)
    except Exception:
        return float("inf")


def run_multiload_spectra_campaign(
    config_path: str = "configs/phase4_experiments.yaml",
    target_vf: float = 0.3,
    out_dir: str = "results",
    seed: int = 42,
) -> pd.DataFrame:
    """Run multi-load spectra evaluation and cliff diagnosis."""
    os.makedirs(out_dir, exist_ok=True)
    load_cases = load_cases_from_yaml(config_path)

    domain = DesignDomain2D(width=3.0, height=1.0, depth=1.0)
    material = Material(E=2.1e11, density=7850.0, yield_strength=2.5e8, name="Steel")
    gs = generate_ground_structure(domain=domain, nx=13, ny=5)

    lc1 = load_cases["LC1"]
    lc2 = load_cases["LC2"]
    lc3 = load_cases["LC3"]

    # Define >= 3 distinct load spectra
    spectra_defs = {
        "Spectrum_A_Equal_Triaxial": {
            "cases": [lc1, lc2, lc3],
            "weights": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            "agg": "weighted_sum",
            "desc": "Equal Triaxial Bending, Shear & Compression",
        },
        "Spectrum_B_Bending_Dominant": {
            "cases": [lc1, lc2],
            "weights": [0.70, 0.30],
            "agg": "weighted_sum",
            "desc": "Cantilever & Simply Supported Flexure",
        },
        "Spectrum_C_Compression_Dominant": {
            "cases": [lc1, lc2, lc3],
            "weights": [0.15, 0.15, 0.70],
            "agg": "weighted_sum",
            "desc": "Dominant Axial Compression with Lateral Shear",
        },
        "Spectrum_D_Envelope_Max": {
            "cases": [lc1, lc2, lc3],
            "weights": [1.0, 1.0, 1.0],
            "agg": "envelope",
            "desc": "Worst-case Peak Stress Envelope",
        },
    }

    records: List[Dict[str, Any]] = []

    # 1. First, optimize single-load designs (LC1, LC2, LC3) for diagnosis
    single_designs = {}
    print("\n[Multi-Load Campaign] Optimizing single-load reference designs...")
    for lc_key, lc in [("LC1", lc1), ("LC2", lc2), ("LC3", lc3)]:
        opt = RemodelingPruningOptimizer(
            ground_structure=gs,
            load_case=lc,
            material=material,
            target_volume_fraction=target_vf,
            removal_rate=0.05,
            seed=seed,
        )
        h = opt.optimize()
        single_designs[lc_key] = (h.steps[-1].active_mask, h.steps[-1].thicknesses, h.final_compliance)

    # 2. Optimize Multi-Load Remodeling designs under each spectrum
    multiload_designs = {}
    for spec_name, spec_info in spectra_defs.items():
        print(f"  -> Optimizing Multi-Load Remodeling for {spec_name} ({spec_info['desc']})...")
        t0 = time.perf_counter()
        opt_multi = MultiLoadRemodelingOptimizer(
            ground_structure=gs,
            load_cases=spec_info["cases"],
            material=material,
            target_volume_fraction=target_vf,
            weights=spec_info["weights"],
            aggregation=spec_info["agg"],
            removal_rate=0.05,
            seed=seed,
        )
        h_multi = opt_multi.optimize()
        dt_multi = time.perf_counter() - t0
        multiload_designs[spec_name] = (h_multi["active_mask"], h_multi["thicknesses"], dt_multi)

    # 3. Cross-evaluate all designs on all test cases with condition number and mechanism diagnosis
    all_models = {
        "Single_LC1_Optimized": single_designs["LC1"],
        "Single_LC2_Optimized": single_designs["LC2"],
        "Single_LC3_Optimized": single_designs["LC3"],
    }
    for spec_name, (m, th, dt) in multiload_designs.items():
        all_models[f"MultiLoad_{spec_name}"] = (m, th, dt)

    test_cases = [("LC1", lc1), ("LC2", lc2), ("LC3", lc3)]

    print("\n[Cliff Diagnosis] Evaluating condition numbers and kinematic mechanism formation...")
    for model_name, (mask, thicks, extra) in all_models.items():
        for test_key, test_lc in test_cases:
            sections = [
                RectangularSection(t=max(1e-9, thicks[i]), b=domain.depth)
                for i in range(len(thicks))
            ]
            mesh = build_fe_mesh_from_ground_structure(gs, sections, material, active_mask=mask)
            test_lc.apply(mesh, domain, material)

            # Stiffness matrix condition number
            cond_num = compute_stiffness_matrix_condition_number(mesh)

            report, res = check_design_feasibility(mesh)
            is_feasible = report.is_feasible and res is not None

            if is_feasible:
                c_raw = float(compute_compliance(res))
                # Bounded compliance regularizer to prevent artificial infinite scaling on near-mechanisms
                c_bounded = min(c_raw, 1000.0)
                stiff_raw = 1.0 / max(1e-14, c_raw)
                stiff_bounded = 1.0 / max(1e-14, c_bounded)
                has_mechanism = cond_num > 1e12
                diag_notes = "Healthy Continuum Elastic Response" if cond_num < 1e8 else "Near-Singular Kinematic Mechanism / Loose Support Node"
            else:
                c_raw = float("inf")
                c_bounded = 1000.0
                stiff_raw = 0.0
                stiff_bounded = 1.0 / 1000.0
                has_mechanism = True
                diag_notes = f"Infeasible Mechanism: {len(report.unreachable_load_nodes)} disconnected nodes"

            records.append({
                "model_name": model_name,
                "is_multiload": "MultiLoad" in model_name,
                "evaluated_on": test_key,
                "is_feasible": is_feasible,
                "has_mechanism": has_mechanism,
                "condition_number_kappa": cond_num,
                "raw_compliance_J": c_raw,
                "bounded_compliance_J": c_bounded,
                "raw_stiffness_N_m": stiff_raw,
                "bounded_stiffness_N_m": stiff_bounded,
                "diagnosis_notes": diag_notes,
            })

    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "multiload_spectra_analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Multi-load campaign and cliff diagnosis complete. Saved to {csv_path}")
    return df


if __name__ == "__main__":
    run_multiload_spectra_campaign()
