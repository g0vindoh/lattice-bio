"""Design domain specifications and volume fraction conventions."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

from ..fem.mesh import FrameMesh2D


@dataclass(frozen=True)
class DesignDomain2D:
    """Rectangular 2D design domain with unit out-of-plane depth.

    Attributes:
        width: Domain width W (x-extent, meters).
        height: Domain height H (y-extent, meters).
        depth: Out-of-plane depth b (meters, default 1.0).
    """

    width: float = 3.0
    height: float = 1.0
    depth: float = 1.0

    def __post_init__(self) -> None:
        if self.width <= 0.0 or self.height <= 0.0 or self.depth <= 0.0:
            raise ValueError(
                f"Domain dimensions must be positive, got W={self.width}, "
                f"H={self.height}, depth={self.depth}."
            )

    @property
    def aspect_ratio(self) -> float:
        """Domain aspect ratio W / H."""
        return float(self.width / self.height)

    @property
    def bounding_volume(self) -> float:
        """Total bounding volume: V_domain = W * H * depth."""
        return float(self.width * self.height * self.depth)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return (xmin, xmax, ymin, ymax)."""
        return (0.0, self.width, 0.0, self.height)

    def is_inside(self, x: float, y: float, tol: float = 1e-9) -> bool:
        """Check if point (x, y) lies inside or on the domain boundary."""
        return bool(
            -tol <= x <= self.width + tol and -tol <= y <= self.height + tol
        )


def compute_volume_fraction(
    mesh: FrameMesh2D, domain: DesignDomain2D
) -> float:
    """Compute the structural volume fraction v_f = V_total / V_domain.

    Defined once as the shared authoritative calculation across the entire codebase.

    Formula:
        V_total = sum_{i=1}^M (A_i * L_i)
        v_f = V_total / (W * H * depth)

    Args:
        mesh: FrameMesh2D instance.
        domain: DesignDomain2D instance.

    Returns:
        Volume fraction v_f (dimensionless float).
    """
    total_volume = mesh.total_volume
    return float(total_volume / domain.bounding_volume)

