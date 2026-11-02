"""Hand-checkable validation test: 3-member symmetric triangular frame with closed-form results."""

import pytest
import numpy as np

from src.fem.material import Material, RectangularSection
from src.fem.mesh import FrameMesh2D
from src.fem.solver import FrameSolver2D


class TestHandCheckableTriangularFrame:
    """Rigorous analytical comparison for a 3-member triangular frame."""

    def test_triangular_frame_analytical_solution(self):
        """Hand-calculable 3-member symmetric triangular frame.

        Structure:
            Node 1: (0, 0), pinned (ux=0, uy=0)
            Node 2: (L, 0), roller_x (uy=0)
            Node 3: (L/2, H), apex node with vertical point load P downward (Fy = -P)

        Analytical derivations by Method of Joints:
            Let alpha be the diagonal inclination angle with the horizontal:
                tan(alpha) = H / (L/2) = 2*H / L
                sin(alpha) = H / sqrt((L/2)^2 + H^2)
                cos(alpha) = (L/2) / sqrt((L/2)^2 + H^2)

            Apex vertical equilibrium:
                2 * N_diag * sin(alpha) = P
                => N_diag = P / (2 * sin(alpha))  (compression: -N_diag)

            Node 1 horizontal equilibrium:
                N_bottom - N_diag * cos(alpha) = 0
                => N_bottom = N_diag * cos(alpha) = P / (2 * tan(alpha)) = P * L / (4 * H) (tension)

            Reactions:
                R_y1 = P / 2,  R_y2 = P / 2,  R_x1 = 0
        """
        L = 2.0  # m (base span)
        H = 1.0  # m (apex height)
        P = 1000.0  # N
        E = 2.1e11  # Pa
        t = 0.01  # m (slender: L/t = 200)
        b = 1.0  # m

        mat = Material(E=E, density=7850.0)
        sec = RectangularSection(t=t, b=b)

        # Analytical expectations
        L_diag = float(np.hypot(L / 2.0, H))
        sin_a = H / L_diag
        cos_a = (L / 2.0) / L_diag
        tan_a = H / (L / 2.0)

        expected_ry1 = P / 2.0
        expected_ry2 = P / 2.0
        expected_rx1 = 0.0

        expected_n_diag = P / (2.0 * sin_a)   # Compression magnitude (~707.11 N)
        expected_n_bottom = P / (2.0 * tan_a)  # Tension magnitude (500.0 N)

        # Assemble FE model
        mesh = FrameMesh2D()
        mesh.add_node(1, 0.0, 0.0)
        mesh.add_node(2, L, 0.0)
        mesh.add_node(3, L / 2.0, H)

        mesh.add_element(1, 1, 2, mat, sec)  # Bottom chord
        mesh.add_element(2, 1, 3, mat, sec)  # Left diagonal
        mesh.add_element(3, 2, 3, mat, sec)  # Right diagonal

        mesh.add_pinned_support(1)
        mesh.add_roller_x_support(2)
        mesh.add_load(3, fx=0.0, fy=-P, m=0.0)

        solver = FrameSolver2D(mesh)
        res = solver.solve()

        # 1. Reactions check (exact down to machine precision)
        rx1, ry1, rm1 = res.get_nodal_reaction(1)
        rx2, ry2, rm2 = res.get_nodal_reaction(2)
        assert np.isclose(rx1, expected_rx1, atol=1e-12)
        assert np.isclose(ry1, expected_ry1, rtol=1e-6)
        assert np.isclose(ry2, expected_ry2, rtol=1e-6)

        # 2. Member forces check
        n_bottom_computed = res.internal_forces[1].axial_force
        n_diag_left_computed = res.internal_forces[2].axial_force
        n_diag_right_computed = res.internal_forces[3].axial_force

        # Diagonals are in compression (negative)
        assert np.isclose(n_diag_left_computed, -expected_n_diag, rtol=1e-3)
        assert np.isclose(n_diag_right_computed, -expected_n_diag, rtol=1e-3)

        # Bottom chord is in tension (positive)
        assert np.isclose(n_bottom_computed, expected_n_bottom, rtol=1e-3)

        # 3. Truss behavior: axial strain energy dominates (>99.9%)
        assert res.axial_energy_fraction > 0.999
        assert res.bending_energy_fraction < 0.001

        # 4. Symmetry of displacements
        ux1, uy1, _ = res.get_nodal_displacement(1)
        ux2, uy2, _ = res.get_nodal_displacement(2)
        ux3, uy3, _ = res.get_nodal_displacement(3)

        assert np.isclose(ux1, 0.0, atol=1e-12)
        assert np.isclose(uy1, 0.0, atol=1e-12)
        assert np.isclose(uy2, 0.0, atol=1e-12)
        # Apex horizontal displacement is exactly half of the roller elongation
        assert np.isclose(ux3, ux2 / 2.0, rtol=1e-6)

