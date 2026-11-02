"""Rigorous validation test suite for 2D Euler-Bernoulli frame solver.

All 7 required validation cases are implemented with analytical comparisons:
1. Axial bar: tip displacement = PL / (EA)
2. Cantilever with tip load: tip deflection = PL^3 / (3EI)
3. Simply supported beam with central load: deflection = PL^3 / (48EI)
4. Unconstrained element rigid-body zero-energy modes (rank deficiency = 3)
5. Symmetric structure under symmetric load gives symmetric displacements
6. Mesh refinement invariance: splitting a beam into more elements preserves deflection
7. Global and member equilibrium: reactions balance applied loads exactly

Tolerance: 1e-6 relative unless justified (many pass at machine precision ~1e-14).
"""

import pytest
import numpy as np

from src.fem.material import Material, RectangularSection
from src.fem.elements import FrameElement2D
from src.fem.mesh import FrameMesh2D
from src.fem.solver import FrameSolver2D, SingularMatrixError


class TestFEValidationSuite:
    """Analytical validation tests matching Phase 1 specifications."""

    def test_1_axial_bar(self):
        """Validation Test 1: Tip displacement of an axially loaded bar.

        Analytical formula:
            delta_tip = P * L / (E * A)
        """
        L = 2.5
        E = 2.1e11
        t = 0.02
        b = 1.0
        P = 15000.0  # N

        mat = Material(E=E, density=7850.0)
        sec = RectangularSection(t=t, b=b)  # A = b * t = 0.02 m^2

        mesh = FrameMesh2D()
        mesh.add_node(1, 0.0, 0.0)
        mesh.add_node(2, L, 0.0)
        mesh.add_element(1, 1, 2, mat, sec)

        # Boundary condition: Node 1 clamped
        mesh.add_fixed_support(1)
        # Load: axial tension at tip
        mesh.add_load(2, fx=P, fy=0.0, m=0.0)

        solver = FrameSolver2D(mesh)
        res = solver.solve()

        ux_tip, uy_tip, th_tip = res.get_nodal_displacement(2)
        expected_disp = (P * L) / (E * sec.A)

        # Verification: relative tolerance 1e-6
        assert np.isclose(ux_tip, expected_disp, rtol=1e-6)
        assert np.isclose(uy_tip, 0.0, atol=1e-12)
        assert np.isclose(th_tip, 0.0, atol=1e-12)

        # Reaction at root must equal -P
        rx, ry, rm = res.get_nodal_reaction(1)
        assert np.isclose(rx, -P, rtol=1e-6)
        assert np.isclose(ry, 0.0, atol=1e-12)
        assert np.isclose(rm, 0.0, atol=1e-12)

        # Strain energy: pure axial, zero bending
        assert np.isclose(res.axial_energy_fraction, 1.0, atol=1e-12)
        assert np.isclose(res.bending_energy_fraction, 0.0, atol=1e-12)

    def test_2_cantilever_tip_load(self):
        """Validation Test 2: Tip deflection of a cantilever beam with transverse end load.

        Analytical Euler-Bernoulli formulas:
            deflection v_tip = P * L^3 / (3 * E * I)
            rotation   theta = P * L^2 / (2 * E * I)
        """
        L = 3.0
        E = 2.1e11
        t = 0.025
        b = 1.0
        P = -800.0  # Downward point load

        mat = Material(E=E, density=7850.0)
        sec = RectangularSection(t=t, b=b)  # I = b * t^3 / 12

        mesh = FrameMesh2D()
        mesh.add_node(1, 0.0, 0.0)
        mesh.add_node(2, L, 0.0)
        mesh.add_element(1, 1, 2, mat, sec)

        mesh.add_fixed_support(1)
        mesh.add_load(2, fx=0.0, fy=P, m=0.0)

        solver = FrameSolver2D(mesh)
        res = solver.solve()

        ux_tip, uy_tip, th_tip = res.get_nodal_displacement(2)
        expected_deflection = (P * (L**3)) / (3.0 * E * sec.I)
        expected_rotation = (P * (L**2)) / (2.0 * E * sec.I)

        # Relative tolerance 1e-6
        assert np.isclose(uy_tip, expected_deflection, rtol=1e-6)
        assert np.isclose(th_tip, expected_rotation, rtol=1e-6)
        assert np.isclose(ux_tip, 0.0, atol=1e-12)

        # Reaction at root: Ry = -P, Rm = -P * L
        rx, ry, rm = res.get_nodal_reaction(1)
        assert np.isclose(rx, 0.0, atol=1e-12)
        assert np.isclose(ry, -P, rtol=1e-6)
        assert np.isclose(rm, -P * L, rtol=1e-6)

        # Strain energy: pure bending, zero axial
        assert np.isclose(res.bending_energy_fraction, 1.0, atol=1e-12)
        assert np.isclose(res.axial_energy_fraction, 0.0, atol=1e-12)

    def test_3_simply_supported_beam_center_load(self):
        """Validation Test 3: Central deflection of a simply supported beam with midpoint load.

        Analytical Euler-Bernoulli formulas:
            center deflection v_mid = P * L^3 / (48 * E * I)
            end rotations     theta = +/- P * L^2 / (16 * E * I)
        """
        L = 4.0
        E = 2.0e11
        t = 0.04
        b = 1.0
        P = -2000.0  # Downward point load at midpoint

        mat = Material(E=E, density=7800.0)
        sec = RectangularSection(t=t, b=b)

        mesh = FrameMesh2D()
        mesh.add_node(1, 0.0, 0.0)
        mesh.add_node(2, L / 2.0, 0.0)
        mesh.add_node(3, L, 0.0)

        mesh.add_element(1, 1, 2, mat, sec)
        mesh.add_element(2, 2, 3, mat, sec)

        # Boundary conditions: Pinned at left (node 1), roller at right (node 3)
        mesh.add_pinned_support(1)
        mesh.add_roller_x_support(3)

        # Applied load at center node
        mesh.add_load(2, fx=0.0, fy=P, m=0.0)

        solver = FrameSolver2D(mesh)
        res = solver.solve()

        ux_mid, uy_mid, th_mid = res.get_nodal_displacement(2)
        expected_mid_deflection = (P * (L**3)) / (48.0 * E * sec.I)
        expected_end_slope = (P * (L**2)) / (16.0 * E * sec.I)

        # Check midpoint
        assert np.isclose(uy_mid, expected_mid_deflection, rtol=1e-6)
        assert np.isclose(th_mid, 0.0, atol=1e-12)  # Midpoint slope must be zero by symmetry

        # Check end rotations
        _, _, th_left = res.get_nodal_displacement(1)
        _, _, th_right = res.get_nodal_displacement(3)
        assert np.isclose(th_left, expected_end_slope, rtol=1e-6)
        assert np.isclose(th_right, -expected_end_slope, rtol=1e-6)

        # Reactions at supports: each bears half the vertical load
        rx1, ry1, rm1 = res.get_nodal_reaction(1)
        rx3, ry3, rm3 = res.get_nodal_reaction(3)
        assert np.isclose(ry1, -P / 2.0, rtol=1e-6)
        assert np.isclose(ry3, -P / 2.0, rtol=1e-6)
        assert np.isclose(rx1, 0.0, atol=1e-12)
        assert np.isclose(rm1, 0.0, atol=1e-12)
        assert np.isclose(rm3, 0.0, atol=1e-12)

    def test_4_rigid_body_zero_energy_modes(self):
        """Validation Test 4: Unconstrained element has exactly 3 zero-energy rigid-body modes.

        In 2D planar space, a free element has 3 rigid-body degrees of freedom
        (2 translations, 1 rotation). Therefore, the 6x6 element stiffness matrix
        must possess exactly 3 zero eigenvalues and 3 positive eigenvalues.
        """
        # Test for various member orientations
        for angle_deg in [0.0, 30.0, 45.0, 90.0, 135.0]:
            angle_rad = np.radians(angle_deg)
            L = 3.5
            mat = Material(E=2.1e11)
            sec = RectangularSection(t=0.03)

            x1, y1 = 1.0, 2.0
            x2 = x1 + L * np.cos(angle_rad)
            y2 = y1 + L * np.sin(angle_rad)

            elem = FrameElement2D(
                element_id=1,
                node_i=1,
                node_j=2,
                coords_i=(x1, y1),
                coords_j=(x2, y2),
                material=mat,
                section=sec,
            )

            K_elem = elem.global_stiffness_matrix()
            eigenvalues = np.linalg.eigvalsh(K_elem)

            max_eig = eigenvalues[-1]
            zero_eigs = eigenvalues[:3]
            pos_eigs = eigenvalues[3:]

            # The 3 smallest eigenvalues must be zero relative to max_eig
            for z in zero_eigs:
                assert abs(z) / max_eig < 1e-12, f"Eigenvalue {z} not zero at angle {angle_deg}"

            # The 3 largest eigenvalues must be strictly positive
            for p in pos_eigs:
                assert p / max_eig > 1e-6, f"Eigenvalue {p} not positive at angle {angle_deg}"

            # Rigid body translation vectors must give K @ u = 0
            u_tx = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
            u_ty = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
            assert np.allclose(K_elem @ u_tx, np.zeros(6), atol=1e-8)
            assert np.allclose(K_elem @ u_ty, np.zeros(6), atol=1e-8)

            # Rigid body rotation about node 1:
            dth = 0.001
            u_rot = np.array(
                [0.0, 0.0, dth, -(y2 - y1) * dth, (x2 - x1) * dth, dth]
            )
            # Energy of rigid rotation: 0.5 * u^T @ K @ u = 0
            u_rot_energy = 0.5 * float(u_rot @ K_elem @ u_rot)
            assert abs(u_rot_energy) < 1e-12

    def test_5_symmetric_structure_symmetric_response(self):
        """Validation Test 5: Symmetric structure under symmetric load gives symmetric displacements.

        Setup: A symmetric portal frame:
            - Left column: (0, 0) to (0, H)
            - Right column: (W, 0) to (W, H)
            - Beam: (0, H) to (W/2, H) to (W, H)
            - Base nodes fixed.
            - Downward load applied at roof midpoint (W/2, H).
        Symmetry checks:
            - Horizontal displacement at midpoint = 0
            - Rotation at midpoint = 0
            - ux(left knee) = -ux(right knee)
            - uy(left knee) = uy(right knee)
            - theta(left knee) = -theta(right knee)
        """
        W = 4.0
        H = 3.0
        P = -5000.0  # Vertical load at apex

        mat = Material(E=2.1e11, density=7850.0)
        sec = RectangularSection(t=0.03)

        mesh = FrameMesh2D()
        mesh.add_node(1, 0.0, 0.0)        # Left base
        mesh.add_node(2, 0.0, H)          # Left knee
        mesh.add_node(3, W / 2.0, H)      # Roof midpoint
        mesh.add_node(4, W, H)            # Right knee
        mesh.add_node(5, W, 0.0)          # Right base

        mesh.add_element(1, 1, 2, mat, sec)  # Left column
        mesh.add_element(2, 2, 3, mat, sec)  # Left roof half
        mesh.add_element(3, 3, 4, mat, sec)  # Right roof half
        mesh.add_element(4, 4, 5, mat, sec)  # Right column

        mesh.add_fixed_support(1)
        mesh.add_fixed_support(5)

        mesh.add_load(3, fx=0.0, fy=P, m=0.0)

        solver = FrameSolver2D(mesh)
        res = solver.solve()

        ux_mid, uy_mid, th_mid = res.get_nodal_displacement(3)
        ux_left, uy_left, th_left = res.get_nodal_displacement(2)
        ux_right, uy_right, th_right = res.get_nodal_displacement(4)

        # Midpoint symmetry
        assert np.isclose(ux_mid, 0.0, atol=1e-12)
        assert np.isclose(th_mid, 0.0, atol=1e-12)

        # Knee symmetry (tolerance 1e-6 relative, verified to < 1e-12)
        assert np.isclose(ux_left, -ux_right, rtol=1e-6)
        assert np.isclose(uy_left, uy_right, rtol=1e-6)
        assert np.isclose(th_left, -th_right, rtol=1e-6)

        # Reaction symmetry
        rx1, ry1, rm1 = res.get_nodal_reaction(1)
        rx5, ry5, rm5 = res.get_nodal_reaction(5)
        assert np.isclose(rx1, -rx5, rtol=1e-6)
        assert np.isclose(ry1, ry5, rtol=1e-6)
        assert np.isclose(rm1, -rm5, rtol=1e-6)
        assert np.isclose(ry1 + ry5, -P, rtol=1e-6)

    def test_6_mesh_refinement_invariance(self):
        """Validation Test 6: Mesh refinement preserves exact tip deflection.

        Because Euler-Bernoulli cubic Hermite shape functions exactly represent
        the analytical solution for nodal point loads, refining the mesh into
        1, 2, 4, 8, 16, or 32 elements must yield identical nodal tip deflections
        within floating-point precision (< 1e-12 relative difference).
        """
        L = 3.0
        E = 2.1e11
        t = 0.02
        P = -600.0

        mat = Material(E=E, density=7850.0)
        sec = RectangularSection(t=t)
        expected_deflection = (P * (L**3)) / (3.0 * E * sec.I)
        expected_rotation = (P * (L**2)) / (2.0 * E * sec.I)

        refinement_levels = [1, 2, 4, 8, 16, 32]
        tip_deflections = []
        tip_rotations = []

        for n_elem in refinement_levels:
            mesh = FrameMesh2D()
            x_nodes = np.linspace(0.0, L, n_elem + 1)
            for idx, x in enumerate(x_nodes):
                mesh.add_node(idx + 1, x, 0.0)

            for i in range(n_elem):
                mesh.add_element(i + 1, i + 1, i + 2, mat, sec)

            mesh.add_fixed_support(1)
            mesh.add_load(n_elem + 1, fx=0.0, fy=P, m=0.0)

            solver = FrameSolver2D(mesh)
            res = solver.solve()

            _, uy_tip, th_tip = res.get_nodal_displacement(n_elem + 1)
            tip_deflections.append(uy_tip)
            tip_rotations.append(th_tip)

            # Check convergence to exact analytical solution at each mesh level (tolerance 1e-6)
            assert np.isclose(uy_tip, expected_deflection, rtol=1e-6)
            assert np.isclose(th_tip, expected_rotation, rtol=1e-6)

        # Check invariance across all mesh levels
        for v in tip_deflections:
            assert np.isclose(v, tip_deflections[0], rtol=1e-12)
        for th in tip_rotations:
            assert np.isclose(th, tip_rotations[0], rtol=1e-12)

    def test_7_equilibrium_reactions_sum_to_applied_loads(self):
        """Validation Test 7: Global reactions balance applied loads exactly.

        Verifies:
            sum(F_x,applied) + sum(R_x) = 0
            sum(F_y,applied) + sum(R_y) = 0
            sum(M_0,applied) + sum(M_0,R) = 0
        across an inclined multi-member frame structure with mixed boundary conditions.
        Tolerance: 1e-6 relative to applied load magnitude.
        """
        mat = Material(E=2.1e11, density=7850.0)
        sec = RectangularSection(t=0.03)

        mesh = FrameMesh2D()
        mesh.add_node(1, 0.0, 0.0)   # Clamped support
        mesh.add_node(2, 2.0, 1.5)   # Inclined node with loads
        mesh.add_node(3, 4.0, 0.0)   # Pinned support
        mesh.add_node(4, 5.0, 2.5)   # Free node with moment
        mesh.add_node(5, 6.0, 0.0)   # Roller support

        mesh.add_element(1, 1, 2, mat, sec)
        mesh.add_element(2, 2, 3, mat, sec)
        mesh.add_element(3, 2, 4, mat, sec)
        mesh.add_element(4, 4, 5, mat, sec)

        mesh.add_fixed_support(1)
        mesh.add_pinned_support(3)
        mesh.add_roller_x_support(5)

        # Mixed loads
        mesh.add_load(2, fx=1500.0, fy=-2500.0, m=300.0)
        mesh.add_load(4, fx=-800.0, fy=-1200.0, m=-600.0)

        solver = FrameSolver2D(mesh)
        res = solver.solve()

        eq = res.check_equilibrium()

        # Equilibrium must be satisfied within tolerance 1e-6 relative to applied loads
        assert eq["rel_err_fx"] < 1e-6, f"Relative Fx error exceeds 1e-6: {eq['rel_err_fx']}"
        assert eq["rel_err_fy"] < 1e-6, f"Relative Fy error exceeds 1e-6: {eq['rel_err_fy']}"
        assert eq["rel_err_m0"] < 1e-6, f"Relative M0 error exceeds 1e-6: {eq['rel_err_m0']}"


class TestSingularMatrixAndMechanisms:
    """Test mechanism detection and singular matrix error handling across edge cases."""

    def test_case_a_floating_unsupported_node_raises(self):
        """Case (a): A floating unsupported node in the mesh must be flagged."""
        mat = Material(E=2.1e11)
        sec = RectangularSection(t=0.02)

        mesh = FrameMesh2D()
        mesh.add_node(1, 0.0, 0.0)
        mesh.add_node(2, 2.0, 0.0)
        mesh.add_node(3, 1.0, 1.0)  # Floating isolated node
        mesh.add_element(1, 1, 2, mat, sec)

        mesh.add_fixed_support(1)
        mesh.add_load(2, fy=-100.0)

        solver = FrameSolver2D(mesh)
        with pytest.raises(SingularMatrixError, match="Floating unsupported node"):
            solver.solve()

    def test_case_b_too_few_supports_raises(self):
        """Case (b): A structure with too few supports (< 3 rigid-body constraints) must be flagged."""
        mat = Material(E=2.1e11)
        sec = RectangularSection(t=0.02)

        # Subcase b1: Only 1 pinned support (only 2 DOFs constrained, free rotation in plane)
        mesh1 = FrameMesh2D()
        mesh1.add_node(1, 0.0, 0.0)
        mesh1.add_node(2, 2.0, 0.0)
        mesh1.add_element(1, 1, 2, mat, sec)
        mesh1.add_pinned_support(1)
        mesh1.add_load(2, fy=-100.0)

        with pytest.raises(SingularMatrixError, match="Insufficient boundary conditions"):
            FrameSolver2D(mesh1).solve()

        # Subcase b2: Parallel roller supports that allow unrestrained horizontal translation
        mesh2 = FrameMesh2D()
        mesh2.add_node(1, 0.0, 0.0)
        mesh2.add_node(2, 2.0, 0.0)
        mesh2.add_node(3, 4.0, 0.0)
        mesh2.add_element(1, 1, 2, mat, sec)
        mesh2.add_element(2, 2, 3, mat, sec)
        mesh2.add_roller_x_support(1)  # only uy fixed
        mesh2.add_roller_x_support(2)  # only uy fixed
        mesh2.add_roller_x_support(3)  # only uy fixed
        mesh2.add_load(2, fx=100.0)

        with pytest.raises(SingularMatrixError, match="Defective boundary conditions"):
            FrameSolver2D(mesh2).solve()

    def test_case_c_slender_13x5_grid_frame_not_flagged(self):
        """Case (c): A legitimate slender 13x5 grid frame (hundreds of members) must NOT be flagged."""
        nx, ny = 13, 5
        Lx, Ly = 12.0, 4.0
        mat = Material(E=2.1e11, density=7850.0)
        # Highly slender struts: t = 5 mm, bay length = 1.0 m (slenderness L/t = 200)
        sec = RectangularSection(t=0.005, b=1.0)

        mesh = FrameMesh2D()
        node_coords = {}
        node_id = 1
        for j in range(ny):
            for i in range(nx):
                x = i * (Lx / (nx - 1))
                y = j * (Ly / (ny - 1))
                mesh.add_node(node_id, x, y)
                node_coords[(i, j)] = node_id
                node_id += 1

        elem_id = 1
        # Horizontal members
        for j in range(ny):
            for i in range(nx - 1):
                mesh.add_element(elem_id, node_coords[(i, j)], node_coords[(i + 1, j)], mat, sec)
                elem_id += 1
        # Vertical members
        for j in range(ny - 1):
            for i in range(nx):
                mesh.add_element(elem_id, node_coords[(i, j)], node_coords[(i, j + 1)], mat, sec)
                elem_id += 1

        # Fix all base nodes at y = 0
        for i in range(nx):
            mesh.add_fixed_support(node_coords[(i, 0)])

        # Apply lateral and vertical shear loads at top right corner
        mesh.add_load(node_coords[(nx - 1, ny - 1)], fx=150.0, fy=-300.0)

        # 65 nodes, 112 elements -> must solve without false positive
        assert mesh.num_nodes == 65
        assert mesh.num_elements == 112

        solver = FrameSolver2D(mesh)
        res = solver.solve()

        # Solution must be valid, finite, and strictly in equilibrium
        assert res.compliance > 0.0
        assert np.isfinite(res.displacements).all()
        eq = res.check_equilibrium()
        assert eq["rel_err_fx"] < 1e-6
        assert eq["rel_err_fy"] < 1e-6

    def test_case_d_genuine_mechanism_must_be_flagged(self):
        """Case (d): A genuine mechanism must MUST be flagged with SingularMatrixError.

        A multi-component structure where one component is well-supported (clamped),
        but an adjacent component has a rotational mechanism.
        """
        mat = Material(E=2.1e11)
        sec = RectangularSection(t=0.02)

        mesh = FrameMesh2D()
        # Stable component
        mesh.add_node(1, 0.0, 0.0)
        mesh.add_node(2, 2.0, 0.0)
        mesh.add_element(1, 1, 2, mat, sec)
        mesh.add_fixed_support(1)
        mesh.add_fixed_support(2)

        # Mechanism component: Bar with 1 pinned node, free rotation
        mesh.add_node(3, 0.0, 2.0)
        mesh.add_node(4, 2.0, 2.0)
        mesh.add_element(2, 3, 4, mat, sec)
        mesh.add_pinned_support(3)  # Node 3 pinned, node 4 free -> rotational mechanism

        mesh.add_load(4, fy=-50.0)

        solver = FrameSolver2D(mesh)
        with pytest.raises(SingularMatrixError):
            solver.solve()
