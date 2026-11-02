# Lattice-Bio: Bio-Inspired Pruning vs. Genetic Algorithms for 2D Lattice Design

> **Research Project:** *"Bone Remodeling vs Evolution: A Controlled Comparison of Bio-Inspired Pruning and Genetic Algorithms for Lightweight 2D Lattice Design."*

This repository implements a controlled, research-grade comparison between two distinct structural optimization paradigms for lightweight 2D lattice networks under strictly equal material budgets:
1. **Method A (Bone Remodeling-Inspired Pruning):** Iterative removal of under-loaded structural members, simulating cellular-level bone resorption and remodeling in trabecular bone architectures.
2. **Method B (Genetic Algorithm):** Evolutionary search exploring topological member presence and discrete cross-sectional thickness profiles.
3. **Baselines:** Benchmark periodic honeycombs and stochastic Voronoi lattices evaluated at matched volume fractions.

---

## Phase 1: 2D Frame Finite Element (FE) Solver

Phase 1 establishes and rigorously validates the 2D Euler-Bernoulli frame element solver. A frame solver (3 degrees of freedom per node: $u_x$, $u_y$, $\theta$) is mandatory because honeycomb lattices and cellular materials exhibit bending-dominated deformation modes that cannot be modeled by pin-jointed truss elements.

### 1. Element Formulation & Cross-Section Properties

Each member connects node $i$ at $(x_i, y_i)$ to node $j$ at $(x_j, y_j)$ with length $L = \sqrt{(x_j - x_i)^2 + (y_j - y_i)^2}$, orientation angle $\theta = \operatorname{atan2}(y_j - y_i, x_j - x_i)$, and direction cosines $c = \cos\theta, s = \sin\theta$.

#### Rectangular Section Derivation:
For an in-plane member thickness (width) $t$ and unit out-of-plane depth $b = 1.0$:
- **Cross-sectional area:**
  $$A = b \cdot t = 1.0 \cdot t = t \quad [\text{m}^2]$$
- **Second moment of area (in-plane flexure):**
  $$I = \frac{b \cdot t^3}{12} = \frac{1.0 \cdot t^3}{12} = \frac{t^3}{12} \quad [\text{m}^4]$$

#### Coordinate Transformation:
Local degrees of freedom $\mathbf{u}_e' = [u_1', v_1', \theta_1', u_2', v_2', \theta_2']^T$ relate to global degrees of freedom $\mathbf{u}_e = [u_{x1}, u_{y1}, \theta_1, u_{x2}, u_{y2}, \theta_2]^T$ via:
$$\mathbf{u}_e' = \mathbf{T} \mathbf{u}_e, \quad \mathbf{T} = \begin{bmatrix} \mathbf{\Lambda} & \mathbf{0}_{3\times 3} \\ \mathbf{0}_{3\times 3} & \mathbf{\Lambda} \end{bmatrix}, \quad \mathbf{\Lambda} = \begin{bmatrix} c & s & 0 \\ -s & c & 0 \\ 0 & 0 & 1 \end{bmatrix}$$
The global stiffness matrix is $\mathbf{K}_e = \mathbf{T}^T \mathbf{K}_e' \mathbf{T}$, assembled into a sparse CSR structural matrix.

#### Energy Partition & Compliance:
- **Axial strain energy:**
  $$U_{\text{axial}} = \frac{1}{2} \frac{EA}{L} (u_2' - u_1')^2 = \frac{1}{2} \frac{N^2 L}{EA}$$
- **Bending strain energy:**
  $$U_{\text{bending}} = \frac{1}{2} \mathbf{u}_{b}'^T \mathbf{K}_b' \mathbf{u}_{b}', \quad \mathbf{u}_{b}' = [v_1', \theta_1', v_2', \theta_2']^T$$
- **Compliance:** $C = \mathbf{F}^T \mathbf{u} = 2(U_{\text{axial}} + U_{\text{bending}})$.

---

## Verification & Validation Suite

All 7 required analytical validation benchmarks and mechanism failure detection tests are automated and reproducible.

### Summary of Validation Results

| Test ID | Description | Metric | Expected | Computed | Relative Error | Tolerance | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Test 1** | Axial bar tip displacement: $PL / (EA)$ | Tip $u_x$ (m) | $5.952381 \times 10^{-6}$ | $5.952381 \times 10^{-6}$ | $0.0$ | $1.0 \times 10^{-6}$ | **PASS** |
| **Test 2** | Cantilever tip deflection: $PL^3 / (3EI)$ | Tip $u_y$ (m) | $-0.032143$ | $-0.032143$ | $2.16 \times 10^{-16}$ | $1.0 \times 10^{-6}$ | **PASS** |
| **Test 3** | Simply supported beam central load: $PL^3 / (48EI)$ | Mid $u_y$ (m) | $-0.011429$ | $-0.011429$ | $0.0$ | $1.0 \times 10^{-6}$ | **PASS** |
| **Test 4** | Unconstrained element rigid-body modes | Zero modes count | $3$ | $3$ | $9.76 \times 10^{-17}$ | $1.0 \times 10^{-10}$ | **PASS** |
| **Test 5** | Symmetric portal frame under symmetric load | Knee asymmetry | $0.0$ | $4.85 \times 10^{-12}$ | $4.85 \times 10^{-12}$ | $1.0 \times 10^{-6}$ | **PASS** |
| **Test 6** | Mesh refinement invariance ($1 \to 32$ elements) | Max refinement error | $-0.032143$ | $-0.032143$ | $3.10 \times 10^{-11}$ | $1.0 \times 10^{-6}$ | **PASS** |
| **Test 7** | Global equilibrium: $\sum \mathbf{R} + \sum \mathbf{F} = \mathbf{0}$ | Max relative error | $0.0$ | $4.04 \times 10^{-14}$ | $4.04 \times 10^{-14}$ | $1.0 \times 10^{-6}$ | **PASS** |
| **Test 8** | Mechanism / singularity detection | `SingularMatrixError` raised | $1.0$ | $1.0$ | $0.0$ | $0.0$ | **PASS** |

---

## Quickstart & Reproducibility

### 1. Run Complete Pytest Suite
```bash
python -m pytest
```

### 2. Execute Config-Driven Validation Script
```bash
python run_validation.py --config configs/phase1_validation.yaml
```

### 3. Run Interactive Demonstration
```bash
python demo/demo_cantilever.py
```

