"""Unit tests for Method A: Remodeling-inspired pruning optimizer."""

import pytest
import numpy as np

from src.geometry.domain import DesignDomain2D, compute_volume_fraction
from src.geometry.ground_structure import generate_ground_structure
from src.fem.load_cases import LoadCase
from src.fem.material import Material
from src.optimizers.pruning import RemodelingPruningOptimizer


class TestRemodelingPruning:
    """Test pruning rules, convergence, and volume targeting."""

    @pytest.fixture
    def setup_cantilever(self):
        domain = DesignDomain2D(width=3.0, height=1.0)
        gs = generate_ground_structure(domain=domain, nx=7, ny=3)
        lc = LoadCase(id="LC1", name="Cantilever", load_type="cantilever", load_magnitude=500.0)
        mat = Material(E=2.1e11, density=7850.0)
        return domain, gs, lc, mat

    def test_pruning_volume_convergence(self, setup_cantilever):
        """Verify that pruning progressively reduces volume down to target budget."""
        domain, gs, lc, mat = setup_cantilever
        target_vf = 0.25

        optimizer = RemodelingPruningOptimizer(
            ground_structure=gs,
            load_case=lc,
            material=mat,
            target_volume_fraction=target_vf,
            removal_rate=0.08,
            utilization_metric="strain_energy_density",
            redistribute_area=False,
            initial_thickness=0.02,
            max_iterations=40,
            seed=42,
        )

        history = optimizer.optimize()

        # Must record multiple steps
        assert len(history.steps) > 1
        assert history.total_fe_evaluations > 1

        # Final volume fraction must match target budget
        final_vf = history.steps[-1].volume_fraction
        assert np.isclose(final_vf, target_vf, rtol=0.05)

        # Monotonic decrease in number of active members
        active_counts = [s.num_active_members for s in history.steps]
        assert active_counts[-1] < active_counts[0]

        # Feasibility must be maintained
        assert history.final_result is not None
        assert history.final_result.compliance > 0.0

    @pytest.mark.parametrize("metric", ["strain_energy_density", "axial_stress", "peak_stress"])
    def test_pruning_utilization_metrics(self, setup_cantilever, metric):
        """Verify that all configurable utilization metrics execute successfully."""
        domain, gs, lc, mat = setup_cantilever

        opt = RemodelingPruningOptimizer(
            ground_structure=gs,
            load_case=lc,
            material=mat,
            target_volume_fraction=0.3,
            removal_rate=0.1,
            utilization_metric=metric,
            max_iterations=5,
            seed=42,
        )
        history = opt.optimize()
        assert len(history.steps) >= 2

    def test_pruning_area_redistribution_wolffs_law(self, setup_cantilever):
        """Verify that Wolff's law area redistribution runs and thickens remaining struts."""
        domain, gs, lc, mat = setup_cantilever

        opt = RemodelingPruningOptimizer(
            ground_structure=gs,
            load_case=lc,
            material=mat,
            target_volume_fraction=0.25,
            removal_rate=0.1,
            redistribute_area=True,
            initial_thickness=0.02,
            max_iterations=10,
            seed=42,
        )
        history = opt.optimize()
        assert len(history.steps) >= 2
        # Check that top member thickness increased beyond initial thickness
        final_thicknesses = history.steps[-1].thicknesses
        assert np.max(final_thicknesses) > 0.02

