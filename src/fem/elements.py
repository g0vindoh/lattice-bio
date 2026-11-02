"""2D Euler-Bernoulli frame element formulation.

Contains the 2D 2-node frame element with 3 degrees of freedom per node:
(u_x, u_y, theta_z), supporting both axial and bending deformations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .material import Material, CrossSection


@dataclass(frozen=True)
class MemberInternalForces:
    """Internal forces and strain energy breakdown for an element.

    Attributes:
        axial_force: Axial normal force (N). Positive = tension, Negative = compression.
        bending_moment_1: End moment at node 1 (N*m), counter-clockwise in local coords.
        bending_moment_2: End moment at node 2 (N*m), counter-clockwise in local coords.
        shear_force: Transverse shear force V1' in local coordinates (N).
        strain_energy_axial: Axial strain energy U_axial (Joules).
        strain_energy_bending: Flexural strain energy U_bending (Joules).
        strain_energy_total: Total strain energy U_total = U_axial + U_bending (Joules).
    """

    axial_force: float
    bending_moment_1: float
    bending_moment_2: float
    shear_force: float
    strain_energy_axial: float
    strain_energy_bending: float
    strain_energy_total: float


class FrameElement2D:
    """2D Euler-Bernoulli 2-node beam-column (frame) element.

    Degrees of freedom:
        Local DOFs:  [u1', v1', theta1', u2', v2', theta2']
        Global DOFs: [u_x1, u_y1, theta1, u_x2, u_y2, theta2]

    Kinematic and mechanical assumptions:
    1. Small displacements, small rotations (linear elasticity).
    2. Straight, prismatic member of constant cross-section (A, I, E).
    3. Euler-Bernoulli hypothesis: cross-sections remain plane and normal to the
       deformed beam axis (shear deformation neglected, valid for slender members).
    4. Uncoupled axial and bending response in local coordinates (first-order linear theory).
    """

    def __init__(
        self,
        element_id: int,
        node_i: int,
        node_j: int,
        coords_i: Tuple[float, float],
        coords_j: Tuple[float, float],
        material: Material,
        section: CrossSection,
    ) -> None:
        self.element_id = int(element_id)
        self.node_i = int(node_i)
        self.node_j = int(node_j)
        self.coords_i = (float(coords_i[0]), float(coords_i[1]))
        self.coords_j = (float(coords_j[0]), float(coords_j[1]))
        self.material = material
        self.section = section

        dx = self.coords_j[0] - self.coords_i[0]
        dy = self.coords_j[1] - self.coords_i[1]
        length = float(np.hypot(dx, dy))

        if length <= 1e-14:
            raise ValueError(
                f"Element {element_id} has zero or near-zero length ({length:.3e} m)."
            )

        self.length = length
        self.cos = dx / length
        self.sin = dy / length

    @property
    def volume(self) -> float:
        """Total volume of the element: V = A * L."""
        return float(self.section.A * self.length)

    @property
    def mass(self) -> float:
        """Total mass of the element: M = rho * A * L."""
        return float(self.material.density * self.section.A * self.length)

    def local_stiffness_matrix(self) -> np.ndarray:
        """Compute the 6x6 local stiffness matrix K_e'.

        Local DOFs order: [u1', v1', theta1', u2', v2', theta2']
        """
        L = self.length
        E = self.material.E
        A = self.section.A
        I = self.section.I

        ka = E * A / L
        kb1 = 12.0 * E * I / (L**3)
        kb2 = 6.0 * E * I / (L**2)
        kb3 = 4.0 * E * I / L
        kb4 = 2.0 * E * I / L

        k_local = np.array(
            [
                [ka, 0.0, 0.0, -ka, 0.0, 0.0],
                [0.0, kb1, kb2, 0.0, -kb1, kb2],
                [0.0, kb2, kb3, 0.0, -kb2, kb4],
                [-ka, 0.0, 0.0, ka, 0.0, 0.0],
                [0.0, -kb1, -kb2, 0.0, kb1, -kb2],
                [0.0, kb2, kb4, 0.0, -kb2, kb3],
            ],
            dtype=np.float64,
        )
        return k_local

    def transformation_matrix(self) -> np.ndarray:
        """Compute the 6x6 coordinate transformation matrix T.

        Transforms global nodal displacements u_e into local displacements u_e':
            u_e' = T @ u_e
            K_e  = T.T @ K_e' @ T
        """
        c = self.cos
        s = self.sin

        # 3x3 block transformation for a single node: [u_x, u_y, theta] -> [u', v', theta']
        # u' =  c*ux + s*uy
        # v' = -s*ux + c*uy
        # theta' = theta
        lam = np.array(
            [
                [c, s, 0.0],
                [-s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        T = np.zeros((6, 6), dtype=np.float64)
        T[0:3, 0:3] = lam
        T[3:6, 3:6] = lam
        return T

    def global_stiffness_matrix(self) -> np.ndarray:
        """Compute the 6x6 global element stiffness matrix K_e = T.T @ K_e' @ T."""
        T = self.transformation_matrix()
        K_local = self.local_stiffness_matrix()
        return T.T @ K_local @ T

    def compute_internal_forces(self, u_global_element: np.ndarray) -> MemberInternalForces:
        """Compute internal forces and energy breakdown given the 6 global element displacements.

        Args:
            u_global_element: Array of 6 global displacements
                              [ux1, uy1, th1, ux2, uy2, th2].

        Returns:
            MemberInternalForces dataclass instance.
        """
        u_elem = np.asarray(u_global_element, dtype=np.float64).reshape(6)
        T = self.transformation_matrix()
        u_local = T @ u_elem

        u1_p, v1_p, th1_p, u2_p, v2_p, th2_p = u_local

        L = self.length
        E = self.material.E
        A = self.section.A
        I = self.section.I

        # Axial normal force (tension positive)
        # N = (EA / L) * (u2' - u1')
        axial_strain = (u2_p - u1_p) / L
        axial_force = float(E * A * axial_strain)

        # Local flexural forces from local stiffness:
        # V1' = (12EI/L^3)*(v1' - v2') + (6EI/L^2)*(th1' + th2')
        # M1' = (6EI/L^2)*(v1' - v2') + (4EI/L)*th1' + (2EI/L)*th2'
        # M2' = (6EI/L^2)*(v1' - v2') + (2EI/L)*th1' + (4EI/L)*th2'
        kb1 = 12.0 * E * I / (L**3)
        kb2 = 6.0 * E * I / (L**2)
        kb3 = 4.0 * E * I / L
        kb4 = 2.0 * E * I / L

        dv = v1_p - v2_p
        shear_force = float(kb1 * dv + kb2 * (th1_p + th2_p))
        m1 = float(kb2 * dv + kb3 * th1_p + kb4 * th2_p)
        m2 = float(kb2 * dv + kb4 * th1_p + kb3 * th2_p)

        # Strain energy decomposition:
        # Axial strain energy: U_axial = 0.5 * (EA / L) * (u2' - u1')^2
        u_axial = 0.5 * (E * A / L) * ((u2_p - u1_p) ** 2)

        # Bending strain energy: U_bending = 0.5 * u_b.T @ K_b @ u_b
        # where u_b = [v1', th1', v2', th2']
        u_b = np.array([v1_p, th1_p, v2_p, th2_p], dtype=np.float64)
        k_b = np.array(
            [
                [kb1, kb2, -kb1, kb2],
                [kb2, kb3, -kb2, kb4],
                [-kb1, -kb2, kb1, -kb2],
                [kb2, kb4, -kb2, kb3],
            ],
            dtype=np.float64,
        )
        u_bending = 0.5 * float(u_b @ k_b @ u_b)

        # Numerical safety: strain energy is quadratic positive-semidefinite, clamp tiny negatives
        u_axial = max(0.0, float(u_axial))
        u_bending = max(0.0, float(u_bending))
        u_total = u_axial + u_bending

        return MemberInternalForces(
            axial_force=axial_force,
            bending_moment_1=m1,
            bending_moment_2=m2,
            shear_force=shear_force,
            strain_energy_axial=u_axial,
            strain_energy_bending=u_bending,
            strain_energy_total=u_total,
        )

