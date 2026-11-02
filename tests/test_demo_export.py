"""Unit tests for demo JSON serialization and replay frames."""

import pytest
import os
import json
import numpy as np

from src.geometry.domain import DesignDomain2D
from src.fem.material import Material
from src.fem.load_cases import load_cases_from_yaml
from src.geometry.ground_structure import generate_ground_structure
from src.export.json_exporter import export_demo_dataset


class TestDemoJSONExport:
    """Validate JSON replay generation and data consistency."""

    def test_json_structure_and_schema(self, tmp_path):
        """Test minimal dataset generation and verify all required keys."""
        out_file = str(tmp_path / "test_demo.json")

        # Run fast test on LC1 at vf=0.20
        data = export_demo_dataset(
            config_path="configs/phase4_experiments.yaml",
            out_path=out_file,
            seed=42,
            load_cases_filter=["LC1"],
            vfs_filter=[0.20],
        )

        assert os.path.exists(out_file)
        with open(out_file, "r") as f:
            loaded = json.load(f)

        assert "metadata" in loaded
        assert "scenarios" in loaded

        # Pick first scenario
        first_scen_key = list(loaded["scenarios"].keys())[0]
        scen = loaded["scenarios"][first_scen_key]

        assert "remodeling_pruning" in scen
        assert "genetic_algorithm" in scen
        assert "baselines" in scen

        pruning_frames = scen["remodeling_pruning"]
        assert len(pruning_frames) > 0

        first_frame = pruning_frames[0]
        assert "compliance" in first_frame
        assert "stiffness" in first_frame
        assert "volume_fraction" in first_frame
        assert "members" in first_frame
        assert "nodes" in first_frame

        # Check member schema
        m0 = first_frame["members"][0]
        assert "x1" in m0 and "y1" in m0 and "x2" in m0 and "y2" in m0
        assert "thickness" in m0
        assert "strain_energy" in m0
        assert "peak_stress" in m0
