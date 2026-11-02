"""Material and cross-section definitions for 2D Euler-Bernoulli frame elements.

Modeling Assumptions:
1. Material is linear elastic, isotropic, and homogeneous.
2. In classical 1D Euler-Bernoulli beam theory, the axial stiffness depends on E * A,
   and flexural stiffness depends on E * I. Poisson's ratio does not enter the 1D governing
   equations.
3. Cross-section is rectangular with in-plane width (thickness) `t` and out-of-plane
   depth `b` (defaulting to unit depth b = 1.0).
4. Second moment of area I is computed about the centroidal bending axis perpendicular
   to the plane of deformation:
       A = b * t
       I = (b * t^3) / 12
   When b = 1.0 (unit out-of-plane depth):
       A = t
       I = t^3 / 12
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Material:
    """Isotropic linear elastic material properties.

    Attributes:
        E: Young's modulus (Pa), must be positive.
        density: Mass density (kg/m^3), must be positive. Default is 1.0.
        yield_strength: Material yield stress (Pa), must be positive. Default is 250 MPa.
        name: Optional label or identifier for the material.
    """

    E: float
    density: float = 1.0
    yield_strength: float = 250e6
    name: str = "LinearElastic"

    def __post_init__(self) -> None:
        if self.E <= 0.0:
            raise ValueError(f"Young's modulus E must be positive, got {self.E}.")
        if self.density <= 0.0:
            raise ValueError(f"Density must be positive, got {self.density}.")
        if self.yield_strength <= 0.0:
            raise ValueError(f"Yield strength must be positive, got {self.yield_strength}.")


@dataclass(frozen=True)
class CrossSection:
    """General cross-section with explicitly specified area and second moment of area.

    Attributes:
        A: Cross-sectional area (m^2), must be positive.
        I: Second moment of area (m^4) for in-plane bending, must be positive.
    """

    A: float
    I: float

    def __post_init__(self) -> None:
        if self.A <= 0.0:
            raise ValueError(f"Cross-sectional area A must be positive, got {self.A}.")
        if self.I <= 0.0:
            raise ValueError(f"Second moment of area I must be positive, got {self.I}.")


@dataclass(frozen=True)
class RectangularSection(CrossSection):
    """Rectangular beam cross-section.

    Derivation of Area A and Second Moment of Area I:
    For a rectangle of in-plane thickness/width `t` (parallel to the bending deformation)
    and out-of-plane depth `b` (perpendicular to the 2D deformation plane):
        - Area: A = b * t
        - Second moment of area about centroidal axis (neutral axis):
          I = \\int_{-t/2}^{t/2} y^2 (b \\, dy) = b * [y^3 / 3]_{-t/2}^{t/2} = (b * t^3) / 12

    For unit out-of-plane depth (b = 1.0):
        - A = 1.0 * t = t
        - I = (1.0 * t^3) / 12 = t^3 / 12

    Attributes:
        t: In-plane width/thickness (m), must be positive.
        b: Out-of-plane depth (m), must be positive. Default is 1.0 (unit depth).
    """

    t: float
    b: float = 1.0

    def __init__(self, t: float, b: float = 1.0) -> None:
        if t <= 0.0:
            raise ValueError(f"Thickness t must be positive, got {t}.")
        if b <= 0.0:
            raise ValueError(f"Out-of-plane depth b must be positive, got {b}.")
        object.__setattr__(self, "t", float(t))
        object.__setattr__(self, "b", float(b))
        # Derived geometric properties
        area = float(b * t)
        inertia = float(b * (t**3) / 12.0)
        object.__setattr__(self, "A", area)
        object.__setattr__(self, "I", inertia)

