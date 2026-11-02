"""Load case definitions and configuration loader."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import yaml

from ..geometry.domain import DesignDomain2D
from ..geometry.pads import (
    BoundaryPadConfig,
    attach_pads_and_bcs_lc1,
    attach_pads_and_bcs_lc2,
    attach_pads_and_bcs_lc3,
)
from .material import Material
from .mesh import FrameMesh2D


@dataclass(frozen=True)
class LoadCase:
    """Represents a standardized structural load case.

    Attributes:
        id: Load case identifier (e.g. 'LC1', 'LC2', 'LC3').
        name: Descriptive label.
        load_type: 'cantilever', 'simply_supported', or 'compression_block'.
        load_magnitude: Total external load magnitude in Newtons (default 1000.0).
        pad_config: BoundaryPadConfig instance.
    """

    id: str
    name: str
    load_type: str
    load_magnitude: float = 1000.0
    pad_config: BoundaryPadConfig = field(default_factory=BoundaryPadConfig)

    def apply(
        self, mesh: FrameMesh2D, domain: DesignDomain2D, material: Material
    ) -> Tuple[List[int], List[int], float]:
        """Apply this load case's boundary conditions, pads, and loads to a mesh.

        Returns:
            (pad_node_ids, pad_element_ids, pad_volume)
        """
        lt = self.load_type.lower()
        if lt in ("cantilever", "lc1"):
            return attach_pads_and_bcs_lc1(
                mesh=mesh,
                domain=domain,
                material=material,
                pad_config=self.pad_config,
                load_magnitude=self.load_magnitude,
            )
        elif lt in ("simply_supported", "lc2"):
            return attach_pads_and_bcs_lc2(
                mesh=mesh,
                domain=domain,
                material=material,
                pad_config=self.pad_config,
                load_magnitude=self.load_magnitude,
            )
        elif lt in ("compression_block", "compression", "lc3"):
            return attach_pads_and_bcs_lc3(
                mesh=mesh,
                domain=domain,
                material=material,
                pad_config=self.pad_config,
                load_magnitude=self.load_magnitude,
            )
        else:
            raise ValueError(f"Unknown load case type: '{self.load_type}'")


def load_cases_from_yaml(config_path: str) -> Dict[str, LoadCase]:
    """Load load cases from a YAML configuration file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    lc_cfg = cfg.get("load_cases", {})
    pad_cfg_dict = cfg.get("pad_config", {})
    pad_config = BoundaryPadConfig(
        pad_thickness=float(pad_cfg_dict.get("thickness", 0.005)),
        pad_depth=float(pad_cfg_dict.get("depth", 1.0)),
        attachment_tolerance=float(pad_cfg_dict.get("attachment_tolerance", 1e-4)),
    )

    load_cases: Dict[str, LoadCase] = {}
    for lc_key, data in lc_cfg.items():
        lc = LoadCase(
            id=lc_key,
            name=data.get("name", lc_key),
            load_type=data.get("type", lc_key),
            load_magnitude=float(data.get("magnitude", 1000.0)),
            pad_config=pad_config,
        )
        load_cases[lc_key] = lc

    return load_cases
