"""JSON Replay Exporter for Interactive Static Web Demo (Phase 5).

Exports high-fidelity structural lattice replay snapshots for:
- Method A: Bio-Inspired Remodeling Pruning (per-iteration topology, stresses, deformation)
- Method B: Genetic Algorithm (per-generation best individual topology, stresses, deformation)
- Baselines: Regular Hexagonal Honeycomb, Random Voronoi, and Uniform Ground Structure

Format is JSON-schema compliant for standalone web clients (Canvas 2D / WebGL)
with zero backend requirements.
"""

from __future__ import annotations
import os
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..geometry.domain import DesignDomain2D
from ..geometry.ground_structure import (
    generate_ground_structure,
    build_fe_mesh_from_ground_structure,
    GroundStructureData,
)
from ..geometry.honeycomb import generate_honeycomb, build_fe_mesh_from_honeycomb
from ..geometry.voronoi import generate_voronoi_lattice, build_fe_mesh_from_voronoi
from ..fem.material import Material, RectangularSection
from ..fem.load_cases import LoadCase, load_cases_from_yaml
from ..metrics.feasibility import check_design_feasibility
from ..metrics.metrics import (
    evaluate_design_metrics,
    compute_compliance,
    compute_stiffness,
    compute_member_peak_stress,
)
from ..optimizers.pruning import RemodelingPruningOptimizer
from ..optimizers.ga import GeneticAlgorithmOptimizer
from ..experiments.baseline_runner import evaluate_single_design


def _extract_frame_data(
    mesh: Any,
    result: Optional[Any],
    step_index: int,
    fe_evals: int,
    domain: DesignDomain2D,
) -> Dict[str, Any]:
    """Extract full node positions, displacements, and member stress/energy metrics."""
    nodes_data = {}
    for nid, (x, y) in mesh.nodes.items():
        if result is not None and nid in result.mesh.nodes:
            ux, uy, theta = result.get_nodal_displacement(nid)
        else:
            ux, uy, theta = 0.0, 0.0, 0.0

        nodes_data[str(nid)] = {
            "x": float(x),
            "y": float(y),
            "ux": float(ux),
            "uy": float(uy),
            "theta": float(theta),
        }

    members_data = []
    max_energy = 1e-12
    max_peak_stress = 1e-12

    # First pass to compute normalizers
    if result is not None:
        for elem_id, elem in mesh.elements.items():
            if elem_id in result.internal_forces:
                ef = result.internal_forces[elem_id]
                max_energy = max(max_energy, ef.strain_energy_total)
                m_max = max(abs(ef.bending_moment_1), abs(ef.bending_moment_2))
                sig = abs(ef.axial_force) / max(1e-9, elem.section.A) + 6.0 * m_max / max(
                    1e-9, elem.section.b * (elem.section.t**2)
                )
                max_peak_stress = max(max_peak_stress, sig)

    for elem_id, elem in mesh.elements.items():
        x1, y1 = mesh.nodes[elem.node_i]
        x2, y2 = mesh.nodes[elem.node_j]

        if result is not None and elem_id in result.internal_forces:
            ef = result.internal_forces[elem_id]
            se_tot = float(ef.strain_energy_total)
            se_ax = float(ef.strain_energy_axial)
            se_bend = float(ef.strain_energy_bending)
            ax_force = float(ef.axial_force)
            m1 = float(ef.bending_moment_1)
            m2 = float(ef.bending_moment_2)
            m_max = max(abs(m1), abs(m2))
            sig = abs(ax_force) / max(1e-9, elem.section.A) + 6.0 * m_max / max(
                1e-9, elem.section.b * (elem.section.t**2)
            )
        else:
            se_tot, se_ax, se_bend = 0.0, 0.0, 0.0
            ax_force, m1, m2, sig = 0.0, 0.0, 0.0, 0.0

        members_data.append(
            {
                "id": elem_id,
                "node_i": elem.node_i,
                "node_j": elem.node_j,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "thickness": float(elem.section.t),
                "length": float(elem.length),
                "strain_energy": se_tot,
                "normalized_energy": float(se_tot / max_energy),
                "axial_force": ax_force,
                "moment_1": m1,
                "moment_2": m2,
                "peak_stress": float(sig),
                "normalized_stress": float(sig / max_peak_stress),
            }
        )

    # Compute global metrics
    if result is not None:
        c = float(compute_compliance(result))
        k = float(compute_stiffness(c))
        ax_share = float(result.axial_energy_fraction)
        bend_share = float(result.bending_energy_fraction)
        vol = float(mesh.total_volume)
        vf = float(vol / domain.bounding_volume)
    else:
        c, k, ax_share, bend_share, vol, vf = 1e8, 1e-8, 0.5, 0.5, 0.0, 0.0

    return {
        "step_index": step_index,
        "fe_evaluations": fe_evals,
        "compliance": c,
        "stiffness": k,
        "volume": vol,
        "volume_fraction": vf,
        "axial_energy_share": ax_share,
        "bending_energy_share": bend_share,
        "num_active_members": len(members_data),
        "nodes": nodes_data,
        "members": members_data,
    }


def export_demo_dataset(
    config_path: str = "configs/phase4_experiments.yaml",
    out_path: str = "demo/data/lattice_data.json",
    seed: int = 42,
    load_cases_filter: Optional[List[str]] = None,
    vfs_filter: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Generate complete replay dataset for the static web demo."""
    domain = DesignDomain2D(width=3.0, height=1.0, depth=1.0)
    material = Material(E=2.1e11, density=7850.0, yield_strength=2.5e8)
    all_load_cases = load_cases_from_yaml(config_path)
    if load_cases_filter is not None:
        load_cases = {k: v for k, v in all_load_cases.items() if k in load_cases_filter}
    else:
        load_cases = all_load_cases

    gs = generate_ground_structure(domain, 13, 5)

    vfs = vfs_filter if vfs_filter is not None else [0.20, 0.30, 0.40]

    demo_data: Dict[str, Any] = {
        "metadata": {
            "title": "Bone Remodeling vs Evolution: Lightweight 2D Lattice Benchmark",
            "domain": {"width": domain.width, "height": domain.height, "depth": domain.depth},
            "material": {"E": material.E, "density": material.density, "yield_strength": material.yield_strength},
            "load_cases": {k: {"id": lc.id, "name": lc.name, "magnitude": lc.load_magnitude} for k, lc in load_cases.items()},
            "volume_fractions": vfs,
        },
        "scenarios": {},
    }

    for lc_key, lc in load_cases.items():
        for vf in vfs:
            scenario_key = f"{lc_key}_vf{int(vf * 100)}"
            print(f"Generating demo replay for {scenario_key}...")

            # 1. Method A: Remodeling Pruning Replay Frames
            opt_p = RemodelingPruningOptimizer(
                ground_structure=gs,
                load_case=lc,
                material=material,
                target_volume_fraction=vf,
                removal_rate=0.05,
                utilization_metric="strain_energy_density",
                redistribute_area=False,
                initial_thickness=0.025,
                max_iterations=80,
                seed=seed,
            )
            # Trace step by step to extract full FE solutions per iteration
            num_members = gs.num_members
            t_start = max(0.025, 1.5 * (vf * domain.bounding_volume) / (domain.depth * gs.total_length))
            active_mask = np.ones(num_members, dtype=bool)
            thicknesses = np.full(num_members, t_start, dtype=np.float64)

            pruning_frames = []
            fe_counter = 0

            # Measure pad volume
            dummy_sec = RectangularSection(t=0.01, b=domain.depth)
            trial_mesh = build_fe_mesh_from_ground_structure(gs, [dummy_sec] * num_members, material)
            _, _, pad_vol = lc.apply(trial_mesh, domain, material)
            V_target = vf * domain.bounding_volume

            for it in range(80):
                sections = [RectangularSection(t=max(1e-9, thicknesses[i]), b=domain.depth) for i in range(num_members)]
                mesh = build_fe_mesh_from_ground_structure(gs, sections, material, active_mask=active_mask)
                lc.apply(mesh, domain, material)
                fe_counter += 1
                report, res = check_design_feasibility(mesh)
                if not report.is_feasible or res is None:
                    break

                # Extract frame
                frame = _extract_frame_data(mesh, res, it, fe_counter, domain)
                pruning_frames.append(frame)

                curr_vol = mesh.total_volume
                if curr_vol <= V_target * 1.001:
                    # Final scaled frame
                    active_indices = np.where(active_mask)[0]
                    active_lens = gs.member_lengths[active_indices]
                    curr_active_vol = np.sum(thicknesses[active_indices] * active_lens) * domain.depth
                    scale = max(0.1, (V_target - pad_vol) / max(1e-12, curr_active_vol))
                    thicknesses[active_indices] *= scale

                    sec_final = [RectangularSection(t=max(1e-9, thicknesses[i]), b=domain.depth) for i in range(num_members)]
                    final_mesh = build_fe_mesh_from_ground_structure(gs, sec_final, material, active_mask=active_mask)
                    lc.apply(final_mesh, domain, material)
                    fe_counter += 1
                    _, res_final = check_design_feasibility(final_mesh)
                    if res_final is not None:
                        frame_f = _extract_frame_data(final_mesh, res_final, it + 1, fe_counter, domain)
                        pruning_frames.append(frame_f)
                    break

                # Pruning step
                active_indices = np.where(active_mask)[0]
                scores = opt_p._compute_utilization(res, active_indices, thicknesses)
                sorted_active = active_indices[np.argsort(scores)]
                n_remove = max(1, int(np.ceil(0.05 * len(active_indices))))

                adj = {nid: set() for nid in gs.nodes}
                for idx in active_indices:
                    u, v = gs.members[idx]
                    adj[u].add(v)
                    adj[v].add(u)
                support_nodes = {bc.node_id for bc in mesh.boundary_conditions}
                loaded_nodes = {ld.node_id for ld in mesh.point_loads if abs(ld.fx) > 1e-6 or abs(ld.fy) > 1e-6}
                n_rem = 0
                for cand in sorted_active:
                    if n_rem >= n_remove:
                        break
                    u_c, v_c = gs.members[cand]
                    adj[u_c].remove(v_c)
                    adj[v_c].remove(u_c)
                    visited = set(support_nodes)
                    queue = list(support_nodes)
                    while queue:
                        curr = queue.pop(0)
                        for nbr in adj.get(curr, set()):
                            if nbr not in visited:
                                visited.add(nbr)
                                queue.append(nbr)
                    if loaded_nodes.issubset(visited):
                        active_mask[cand] = False
                        n_rem += 1
                    else:
                        adj[u_c].add(v_c)
                        adj[v_c].add(u_c)

            # 2. Method B: Genetic Algorithm Replay Frames (sample key generations)
            opt_ga = GeneticAlgorithmOptimizer(
                ground_structure=gs,
                load_case=lc,
                material=material,
                target_volume_fraction=vf,
                pop_size=30,
                generations=30,
                crossover_rate=0.8,
                mutation_rate=0.02,
                tournament_size=3,
                elite_count=2,
                discrete_levels=[0.0, 0.004, 0.008, 0.012, 0.016, 0.020, 0.025, 0.030],
                seed=seed,
            )
            # Custom step-by-step extraction of best individual mesh
            V_lattice_target = max(1e-12, V_target - pad_vol)
            pop = [opt_ga._repair_volume(opt_ga.rng.randint(0, opt_ga.num_levels, size=num_members), V_lattice_target) for _ in range(30)]

            ga_frames = []
            ga_evals = 0
            best_fit_global = -1.0
            best_mesh_global = None
            best_res_global = None

            for gen in range(30):
                fits = np.zeros(30)
                step_res = []
                for i_ind, ind in enumerate(pop):
                    fit, rep, r_solve = opt_ga._evaluate_individual(ind)
                    ga_evals += 1
                    fits[i_ind] = fit
                    step_res.append((ind, fit, rep, r_solve))

                rank_idx = np.argsort(fits)[::-1]
                g_best_ind, g_best_fit, g_best_rep, g_best_res = step_res[rank_idx[0]]

                if g_best_fit > best_fit_global and g_best_res is not None:
                    best_fit_global = g_best_fit
                    best_res_global = g_best_res
                    t_b = opt_ga.discrete_levels[g_best_ind]
                    sec_b = [RectangularSection(t=max(1e-9, t_b[j]), b=domain.depth) for j in range(num_members)]
                    best_mesh_global = build_fe_mesh_from_ground_structure(gs, sec_b, material, active_mask=(g_best_ind > 0))
                    lc.apply(best_mesh_global, domain, material)

                # Record frame at every generation or every 2 generations
                if best_mesh_global is not None and best_res_global is not None:
                    g_frame = _extract_frame_data(best_mesh_global, best_res_global, gen, ga_evals, domain)
                    ga_frames.append(g_frame)

                # Reproduction
                new_pop = [pop[rank_idx[0]].copy(), pop[rank_idx[1]].copy()]
                while len(new_pop) < 30:
                    cand_a = opt_ga.rng.choice(30, size=3, replace=False)
                    p1 = pop[cand_a[np.argmax(fits[cand_a])]]
                    cand_b = opt_ga.rng.choice(30, size=3, replace=False)
                    p2 = pop[cand_b[np.argmax(fits[cand_b])]]
                    # Crossover
                    child = p1.copy()
                    if opt_ga.rng.random() < 0.8:
                        mask = opt_ga.rng.random(num_members) < 0.5
                        child[mask] = p2[mask]
                    # Mutation
                    for m_idx in range(num_members):
                        if opt_ga.rng.random() < 0.02:
                            child[m_idx] = opt_ga.rng.randint(0, opt_ga.num_levels)
                    child = opt_ga._repair_volume(child, V_lattice_target)
                    new_pop.append(child)
                pop = new_pop

            # 3. Baselines: Regular Honeycomb, Voronoi, Uniform Ground Structure
            hc_data = generate_honeycomb(domain, cell_size=0.35)
            _, m_hc, res_hc = evaluate_single_design("honeycomb", lc, domain, material, vf, hc_data)
            hc_frame = _extract_frame_data(m_hc, res_hc, 0, 1, domain)

            vor_data = generate_voronoi_lattice(domain, num_seeds=50, lloyd_iterations=2, seed=seed)
            _, m_vor, res_vor = evaluate_single_design("voronoi", lc, domain, material, vf, vor_data, seed=seed)
            vor_frame = _extract_frame_data(m_vor, res_vor, 0, 1, domain)

            _, m_ugs, res_ugs = evaluate_single_design("ground_structure", lc, domain, material, vf, gs, seed=seed)
            ugs_frame = _extract_frame_data(m_ugs, res_ugs, 0, 1, domain)

            demo_data["scenarios"][scenario_key] = {
                "load_case": lc_key,
                "volume_fraction": vf,
                "remodeling_pruning": pruning_frames,
                "genetic_algorithm": ga_frames,
                "baselines": {
                    "honeycomb": hc_frame,
                    "voronoi": vor_frame,
                    "uniform_gs": ugs_frame,
                },
            }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(demo_data, f, indent=2)

    print(f"Successfully exported interactive demo dataset to {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")
    return demo_data
