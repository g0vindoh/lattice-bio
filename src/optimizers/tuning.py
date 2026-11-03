"""Hyperparameter tuning and validation module for lattice optimizers.

Provides systematic search across hyperparameter combinations for both
GeneticAlgorithmOptimizer and RemodelingPruningOptimizer on a held-out set
of validation seeds/instances, ensuring equal and unbiased tuning effort
prior to benchmark evaluation.
"""

from __future__ import annotations
import itertools
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .ga import GeneticAlgorithmOptimizer
from .pruning import RemodelingPruningOptimizer
from ..fem.material import Material
from ..geometry.ground_structure import GroundStructureData
from ..fem.load_cases import LoadCase


@dataclass
class TuningTrialResult:
    """Result of a single hyperparameter configuration trial.

    Attributes:
        optimizer_name: 'GeneticAlgorithm' or 'RemodelingPruning'.
        params: Dictionary of tested hyperparameters.
        mean_compliance: Mean compliance achieved across validation seeds.
        std_compliance: Standard deviation of compliance across validation seeds.
        mean_runtime_s: Mean execution time per seed.
        mean_fe_evals: Mean number of FE evaluations invoked.
        trials_count: Number of validation seeds tested.
    """

    optimizer_name: str
    params: Dict[str, Any]
    mean_compliance: float
    std_compliance: float
    mean_runtime_s: float
    mean_fe_evals: float
    trials_count: int


def tune_pruning_hyperparameters(
    ground_structure: GroundStructureData,
    load_case: LoadCase,
    material: Material,
    target_volume_fraction: float = 0.3,
    validation_seeds: Optional[List[int]] = None,
    param_grid: Optional[Dict[str, List[Any]]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Tune RemodelingPruning hyperparameters over held-out validation seeds.

    Args:
        ground_structure: GroundStructureData.
        load_case: LoadCase.
        material: Material.
        target_volume_fraction: Target volume fraction.
        validation_seeds: List of seeds for validation (default [1001, 1002, 1003]).
        param_grid: Dictionary of parameter choices to explore.

    Returns:
        best_params: Dict of best performing hyperparameters.
        results_df: DataFrame of all tested configurations and scores.
    """
    if validation_seeds is None:
        validation_seeds = [1001, 1002, 1003]

    if param_grid is None:
        param_grid = {
            "removal_rate": [0.03, 0.05, 0.08],
            "utilization_metric": ["strain_energy_density", "axial_stress"],
            "redistribute_area": [False, True],
            "stochastic_noise": [0.0, 0.05],
        }

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[k] for k in keys]))

    trial_records = []
    best_c = float("inf")
    best_params = {}

    for combo in combinations:
        params = dict(zip(keys, combo))
        compliances = []
        runtimes = []
        evals_list = []

        for s in validation_seeds:
            opt = RemodelingPruningOptimizer(
                ground_structure=ground_structure,
                load_case=load_case,
                material=material,
                target_volume_fraction=target_volume_fraction,
                removal_rate=params["removal_rate"],
                utilization_metric=params["utilization_metric"],
                redistribute_area=params["redistribute_area"],
                stochastic_noise=params["stochastic_noise"],
                seed=s,
            )
            history = opt.optimize()
            compliances.append(history.final_compliance)
            runtimes.append(history.runtime_seconds)
            evals_list.append(history.total_fe_evaluations)

        mean_c = float(np.mean(compliances))
        std_c = float(np.std(compliances))
        mean_rt = float(np.mean(runtimes))
        mean_ev = float(np.mean(evals_list))

        record = {
            **params,
            "mean_compliance_J": mean_c,
            "std_compliance_J": std_c,
            "mean_runtime_s": mean_rt,
            "mean_fe_evals": mean_ev,
        }
        trial_records.append(record)

        if mean_c < best_c:
            best_c = mean_c
            best_params = params

    return best_params, pd.DataFrame(trial_records)


def tune_ga_hyperparameters(
    ground_structure: GroundStructureData,
    load_case: LoadCase,
    material: Material,
    target_volume_fraction: float = 0.3,
    validation_seeds: Optional[List[int]] = None,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    generations_budget: int = 15,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Tune GeneticAlgorithm hyperparameters over held-out validation seeds.

    Args:
        ground_structure: GroundStructureData.
        load_case: LoadCase.
        material: Material.
        target_volume_fraction: Target volume fraction.
        validation_seeds: List of seeds for validation.
        param_grid: Dictionary of parameter choices to explore.
        generations_budget: Generations to evaluate per trial.

    Returns:
        best_params: Dict of best performing hyperparameters.
        results_df: DataFrame of all tested configurations and scores.
    """
    if validation_seeds is None:
        validation_seeds = [1001, 1002]

    if param_grid is None:
        param_grid = {
            "pop_size": [20, 30],
            "crossover_rate": [0.7, 0.9],
            "mutation_rate": [0.02, 0.05],
            "tournament_size": [2, 3],
        }

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[k] for k in keys]))

    trial_records = []
    best_c = float("inf")
    best_params = {}

    for combo in combinations:
        params = dict(zip(keys, combo))
        compliances = []
        runtimes = []
        evals_list = []

        for s in validation_seeds:
            opt = GeneticAlgorithmOptimizer(
                ground_structure=ground_structure,
                load_case=load_case,
                material=material,
                target_volume_fraction=target_volume_fraction,
                pop_size=params["pop_size"],
                generations=generations_budget,
                crossover_rate=params["crossover_rate"],
                mutation_rate=params["mutation_rate"],
                tournament_size=params["tournament_size"],
                seed=s,
            )
            history = opt.optimize()
            compliances.append(history.final_compliance)
            runtimes.append(history.runtime_seconds)
            evals_list.append(history.total_fe_evaluations)

        mean_c = float(np.mean(compliances))
        std_c = float(np.std(compliances))
        mean_rt = float(np.mean(runtimes))
        mean_ev = float(np.mean(evals_list))

        record = {
            **params,
            "mean_compliance_J": mean_c,
            "std_compliance_J": std_c,
            "mean_runtime_s": mean_rt,
            "mean_fe_evals": mean_ev,
        }
        trial_records.append(record)

        if mean_c < best_c:
            best_c = mean_c
            best_params = params

    return best_params, pd.DataFrame(trial_records)
