# Lattice-Bio: Bone Remodeling vs Evolution for Lightweight 2D Lattice Design

[![Tests](https://img.shields.io/badge/Tests-76%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-100%25%20Deterministic-orange.svg)](reproduce_all.py)
[![3D-Printable](https://img.shields.io/badge/CAD-3D%20Printable%20STL-purple.svg)](stl/)

> **Research Paper:** *"Bone Remodeling vs Evolution: A Controlled Comparison of Bio-Inspired Pruning and Genetic Algorithms for Lightweight 2D Lattice Design."*

---

## 1. Overview & Research Scope

This repository provides an open, publication-grade research framework comparing two biological design paradigms for lightweight 2D lattice networks under strictly equal material budgets and standardized boundary conditions:
- **Method A (Remodeling-Inspired Pruning):** Iterative cellular resorption of under-utilized structural members based on strain energy density and Wolff's law of bone transformation.
- **Method B (Genetic Algorithm - GA):** Population-based evolutionary search exploring discrete strut cross-sections with an active volume budget repair operator.
- **Baselines:** Benchmark regular hexagonal honeycombs, stochastic Voronoi lattices (with Lloyd relaxation), and uniformly thinned ground structures evaluated under identical volume fraction budgets ($v_f \in \{0.20, 0.30, 0.40\}$).

All lattices are solved using a validated 2D Euler-Bernoulli frame element solver (3 degrees of freedom per node: $u_x, u_y, \theta$) to rigorously model both axial stretching and flexural bending deformation modes. Standardized boundary and load pads are incorporated into all designs, subtracting pad volume equally to guarantee fair material allocation.

---

## 2. Quickstart & Master Reproduction

The repository is 100% config-driven and deterministic. A single master CLI script (`reproduce_all.py`) manages all test executions, data regenerations, and paper figures.

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/g0vindoh/lattice-bio.git
cd lattice-bio
pip install -e .
# Alternatively:
pip install numpy scipy pandas matplotlib pyyaml pytest
```

### 1. Run Complete Pytest Suite (65/65 Passing)
```bash
python reproduce_all.py --test
# Or directly via pytest:
python -m pytest -v
```

### 2. Single-Command Paper Figure & Table Generation (Fast, ~8 s)
Regenerate all 8 publication figures (Fig 1 to Fig 8) and summary markdown tables directly from the verified experimental CSV results:
```bash
python reproduce_all.py --mode figures
```

### 3. Launch the Interactive Web Demo Locally
Launch the zero-dependency interactive lattice playback application:
```bash
python reproduce_all.py --serve-demo --port 8000
```
Then navigate to: **`http://localhost:8000/index.html`** in your browser.

### 4. Export 3D-Printable STL Solid Models (Additive Manufacturing)
Export watertight 3D printable STL files of optimal lattices with thickened struts and cylindrical node hubs:
```bash
python reproduce_all.py --mode stl
```
Pre-generated binary STL files are available in `stl/`:
- `stl/lattice_pruned_LC1_cantilever.stl` (Cantilever truss)
- `stl/lattice_pruned_LC2_bending_arch.stl` (3-Point bending arch)
- `stl/lattice_multiload_robust.stl` (Multi-load robust lattice)

### 5. Compile-Ready LaTeX & Overleaf Manuscript
The full journal preprint is ready for submission in `paper/`:
- `paper/manuscript.tex`: IEEE/Elsevier format with all real data, tables, and equations.
- `paper/references.bib`: Seminal biomechanics, cellular solids, and optimization citations.

### 6. Full Experimental Campaign Reproduction (E1 to E7 from scratch)
```bash
python reproduce_all.py --mode full
```

---

## 3. Core Empirical Findings

### Finding 1: Remodeling Pruning Achieves Superior Quality with 45x Fewer Solves
At $v_f = 0.20$ across Cantilever (LC1), 3-Point Bending (LC2), and Compression (LC3), Method A converges to optimal topologies in **19 to 20 FE evaluations (< 0.5 s runtime)**, whereas Method B requires **900 FE evaluations (~130–150 s runtime)** to reach comparable stiffness.

| Load Case | Method | Median Compliance (J) | Median Stiffness (N/m) | Specific Stiffness | FE Solves | Runtime (s) | Active Struts |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LC1 (Cantilever)** | **Method A (Pruning)** | **0.003777** | **264.77** | **0.05657** | **19** | **0.46** | **81** |
| LC1 | Method B (GA) | 0.004454 | 224.53 | 0.04823 | 900 | 130.13 | 162 |
| LC1 | Uniform Ground Struct. | 0.006447 | 155.12 | 0.03293 | 1 | 0.05 | 208 |
| LC1 | Voronoi Lattice | 0.049272 | 20.30 | 0.00431 | 1 | 0.05 | 144 |
| LC1 | Regular Honeycomb | 0.829996 | 1.20 | 0.00026 | 1 | 0.05 | 33 |
| **LC2 (3-Point Bend)**| **Method A (Pruning)** | **0.000491** | **2038.16** | **0.43454** | **20** | **0.56** | **81** |
| LC2 | Method B (GA) | 0.000491 | 2037.96 | 0.43337 | 900 | 151.37 | 165 |
| **LC3 (Compression)** | **Method A (Pruning)** | **0.000013** | **79000.00** | **16.7728** | **20** | **0.52** | **81** |
| LC3 | Method B (GA) | 0.000023 | 42775.10 | 9.14006 | 900 | 138.93 | 158 |

### Finding 2: Stretch-Dominated vs. Bending-Dominated Kinematics
Partitioning strain energy into axial stretching ($U_{\text{axial}}$) and flexural bending ($U_{\text{bending}}$) explains why honeycombs exhibit up to $220\times$ lower stiffness:
- **Optimized Lattices (Pruning & GA):** Strongly stretch-dominated (**$89.3\% - 99.8\%$ axial strain energy**).
- **Hexagonal Honeycombs:** Overwhelmingly bending-dominated (**$99.6\% - 99.9\%$ bending strain energy**).
- **Voronoi Foams:** Intermediate, flexure-dominated (**$80\% - 92\%$ bending energy**).

### Finding 3: The Robustness Specialization Gap
Optimizing specifically for one loading regime induces severe vulnerability to off-axis loading:
- An optimal cantilever truss (LC1) degrades by **$32,837\times$** when subjected to central 3-point bending (LC2).
- Conversely, a 3-point bending arch (LC2) generalizes relatively well to cantilever loading ($4.94\times$ degradation).

### Finding 4: Wolff's Law Area Redistribution Boosts Stiffness by +138%
Reallocating 50% of the resorbed strut volume into the top 15% most heavily loaded active members increases structural stiffness from $264.8$ N/m to **$630.3$ N/m ($2.38\times$)** under the exact same material budget ($v_f = 0.20$).

### Finding 5: Multi-Load Remodeling & Cowin Fabric Tensor Anisotropy
Simulating stochastic daily loading spectra ($\mathcal{S} = \{ (\mathbf{F}^{(k)}, w_k) \}$) with Multi-Load Bone Remodeling:
- Completely resolves the $32,837\times$ specialization cliff, producing a lattice that simultaneously supports Cantilever shear ($K = 313.9$ N/m) and 3-Point bending ($K = 3,144.9$ N/m).
- Stereological Cowin 2nd-order fabric tensor $\mathbf{H}$ decomposition confirms that the principal structural orientation directly aligns with the global principal stress field, validating Wolff's trajectorial hypothesis in quantitative tensor form.

---

## 4. Repository Structure

```
lattice-bio/
├── README.md                      # Comprehensive project documentation
├── LICENSE                        # MIT License
├── pyproject.toml                 # Package definition & pytest configuration
├── reproduce_all.py               # Master single-command reproduction CLI
├── run_validation.py              # Phase 1 analytical FE validation runner
├── run_phase2_baselines.py        # Phase 2 baseline benchmark runner
├── run_phase3_optimizers.py       # Phase 3 single-case optimizer benchmark
├── run_phase4_experiments.py      # Phase 4 full experimental campaign (E1 to E7)
├── export_demo_data.py            # Precomputed replay dataset exporter for demo
├── configs/
│   ├── phase1_validation.yaml     # Analytical FE validation benchmark cases
│   ├── phase2_baselines.yaml      # Baseline geometry & evaluation parameters
│   ├── phase3_optimizers.yaml     # Single-case optimizer parameters
│   └── phase4_experiments.yaml    # Complete experimental campaign parameters
├── src/
│   ├── fem/                       # 2D Euler-Bernoulli frame FE solver
│   │   ├── elements.py            # Local stiffness, transformation, energy partition
│   │   ├── material.py            # Material & RectangularSection derivations
│   │   ├── mesh.py                # FrameMesh2D container & DOF mapping
│   │   ├── solver.py              # Sparse CSR assembly & multi-tier singularity check
│   │   ├── postprocess.py         # Internal forces, moments, energy decomposition
│   │   └── load_cases.py          # YAML load case parser (LC1, LC2, LC3)
│   ├── geometry/                  # Lattice topology generators & boundary pads
│   │   ├── domain.py              # 2D rectangular domain & volume fraction helper
│   │   ├── ground_structure.py    # Over-connected grid with collinear overlap filter
│   │   ├── honeycomb.py           # Hexagonal tiling with Liang-Barsky domain clipping
│   │   ├── voronoi.py             # Random Voronoi with Lloyd relaxation & edge merge
│   │   └── pads.py                # Standardized boundary load & support pads
│   ├── optimizers/                # Optimization algorithms
│   │   ├── base.py                # Abstract BaseLatticeOptimizer & snapshot logging
│   │   ├── pruning.py             # Remodeling-inspired pruning with Wolff's law
│   │   └── ga.py                  # Genetic algorithm with volume budget repair
│   ├── metrics/                   # Mechanics metrics & non-parametric statistics
│   │   ├── metrics.py             # Compliance, stiffness, specific stiffness, energy
│   │   ├── feasibility.py         # BFS connectivity, load path, slenderness checks
│   │   └── statistics.py          # Mann-Whitney U, Holm-Bonferroni, Vargha-Delaney A12
│   ├── experiments/               # Experiment campaign execution engine
│   │   ├── baseline_runner.py     # Single-design evaluation harness
│   │   └── experiment_runner.py   # Multi-threaded E1-E7 campaign executor
│   └── export/                    # Visualization & web demo export
│       ├── plotting.py            # Matplotlib static lattice geometry & deformed shapes
│       ├── plot_paper_figures.py  # Publication figures (Fig 1 to Fig 8)
│       └── json_exporter.py       # JSON serialization for static web demo
├── tests/                         # Pytest test suite (65 tests)
│   ├── test_fem_element.py        # Element stiffness, rotation, energy tests
│   ├── test_fem_mesh_and_solver.py# Mesh assembly & boundary conditions
│   ├── test_fem_validation.py     # 7 analytical benchmarks & mechanism checks
│   ├── test_geometry_baselines.py # Honeycomb & Voronoi volume fraction matching
│   ├── test_geometry_ground_structure.py # Collinear filtering & grid generation
│   ├── test_hand_checkable_frame.py # Triangular frame analytical comparison
│   ├── test_feasibility_and_metrics.py   # Slenderness & connectivity checks
│   ├── test_optimizers_pruning.py # Remodeling pruning convergence & Wolff's law
│   ├── test_optimizers_ga.py      # GA repair operator & elitism preservation
│   ├── test_statistics.py         # Mann-Whitney U & Holm-Bonferroni unit tests
│   ├── test_phase4_experiments.py # Fast integration tests for E1-E7 runners
│   ├── test_demo_export.py        # Web demo JSON schema verification
│   └── test_reproduce.py          # Master reproduction CLI verification
├── results/                       # CSV experimental outputs & markdown summaries
├── figures/                       # Publication figures (paper_fig1 to paper_fig8)
├── demo/                          # Standalone static web application
│   ├── index.html                 # Synchronized dual viewport & scoreboard
│   ├── style.css                  # Modern publication theme styling
│   ├── app.js                     # Canvas 2D engine with animated deformation
│   └── data/lattice_data.json     # Precomputed multi-scenario snapshots
└── paper/
    └── manuscript.md              # Complete publication-grade research manuscript
```

---

## 5. Mathematical Formulations

### Rectangular Strut Cross-Section Derivation
For an in-plane strut thickness $t$ and unit out-of-plane depth $b = 1.0\text{ m}$:
- **Cross-Sectional Area:**
  $$A = b \cdot t = 1.0 \cdot t = t \quad [\text{m}^2]$$
- **Second Moment of Area (In-Plane Flexure):**
  $$I = \frac{b \cdot t^3}{12} = \frac{1.0 \cdot t^3}{12} = \frac{t^3}{12} \quad [\text{m}^4]$$

### Local and Global Frame Element Stiffness
Local degrees of freedom $\mathbf{u}_e' = [u_1', v_1', \theta_1', u_2', v_2', \theta_2']^T$ relate to global degrees of freedom $\mathbf{u}_e = [u_{x1}, u_{y1}, \theta_1, u_{x2}, u_{y2}, \theta_2]^T$ via coordinate transformation:
$$\mathbf{u}_e' = \mathbf{T} \mathbf{u}_e, \quad \mathbf{T} = \begin{bmatrix} \mathbf{\Lambda} & \mathbf{0}_{3\times 3} \\ \mathbf{0}_{3\times 3} & \mathbf{\Lambda} \end{bmatrix}, \quad \mathbf{\Lambda} = \begin{bmatrix} \cos\theta & \sin\theta & 0 \\ -\sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$$
The element global stiffness matrix is $\mathbf{K}_e = \mathbf{T}^T \mathbf{K}_e' \mathbf{T}$, assembled into a sparse CSR structural matrix.

### Energy Decomposition & Compliance
- **Axial Strain Energy:** $U_{\text{axial}} = \frac{1}{2} \frac{EA}{L}(u_2' - u_1')^2 = \frac{1}{2} \frac{N^2 L}{EA}$
- **Bending Strain Energy:** $U_{\text{bending}} = \frac{1}{2} \mathbf{u}_b'^T \mathbf{K}_b' \mathbf{u}_b'$, where $\mathbf{u}_b' = [v_1', \theta_1', v_2', \theta_2']^T$
- **Compliance:** $C = \mathbf{F}^T \mathbf{u} = 2 (U_{\text{axial}} + U_{\text{bending}})$

---

## 6. Analytical Validation Suite (Appendix)

Every module of the FE frame solver was validated against independent closed-form analytical solutions before optimizer development.

| Test ID | Benchmark Description | Analytical Formula | Expected | Computed | Relative Error | Tolerance | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Test 1** | Axial bar tip elongation | $u_x = \frac{P L}{E A}$ | $5.952381 \times 10^{-6}\text{ m}$ | $5.952381 \times 10^{-6}\text{ m}$ | $0.0$ | $1.0 \times 10^{-6}$ | **PASS** |
| **Test 2** | Cantilever tip deflection | $u_y = -\frac{P L^3}{3 E I}$ | $-0.03214286\text{ m}$ | $-0.03214286\text{ m}$ | $2.16 \times 10^{-16}$ | $1.0 \times 10^{-6}$ | **PASS** |
| **Test 3** | Simply supported center deflection| $u_y = -\frac{P L^3}{48 E I}$ | $-0.01142857\text{ m}$ | $-0.01142857\text{ m}$ | $0.0$ | $1.0 \times 10^{-6}$ | **PASS** |
| **Test 4** | Unconstrained rigid-body modes | $\operatorname{dim} \operatorname{ker}(\mathbf{K}_e) = 3$ | $3$ zero eigenvalues | $3$ zero eigenvalues | $9.76 \times 10^{-17}$ | $1.0 \times 10^{-10}$ | **PASS** |
| **Test 5** | Portal frame structural symmetry | $\|u_{\text{left}}\| - \|u_{\text{right}}\| = 0$ | $0.0\text{ m}$ | $4.85 \times 10^{-12}\text{ m}$ | $4.85 \times 10^{-12}$ | $1.0 \times 10^{-6}$ | **PASS** |
| **Test 6** | Mesh refinement invariance ($1 \to 32$) | $u_y(N=1) = u_y(N=32)$ | $-0.03214286\text{ m}$ | $-0.03214286\text{ m}$ | $3.10 \times 10^{-11}$ | $1.0 \times 10^{-6}$ | **PASS** |
| **Test 7** | Global equilibrium reactions | $\sum \mathbf{R} + \sum \mathbf{F} = \mathbf{0}$ | $\mathbf{0}$ | $4.04 \times 10^{-14}\text{ N}$ | $4.04 \times 10^{-14}$ | $1.0 \times 10^{-6}$ | **PASS** |
| **Test 8** | Mechanism / singularity detection | `SingularMatrixError` raised | $1.0$ | $1.0$ | $0.0$ | Exact | **PASS** |

---

## 7. Known Modeling Assumptions & Limitations

To ensure complete scientific integrity, all limitations are explicitly stated:
1. **2D Planar Modeling:** The simulation is restricted to in-plane 2D frame elements. Out-of-plane buckling, torsional warping, and 3D architectural effects are not captured.
2. **Linear Elasticity & Small Strains:** Euler-Bernoulli beam theory assumes linear elasticity and small deflections. Geometric non-linearities ($P\text{-}\Delta$ effects) and material plasticity are omitted.
3. **No Local Strut Buckling Constraint:** Optimization maximizes linear stiffness without an explicit Euler buckling constraint ($P_{\text{cr}} = \pi^2 E I / L_{\text{eff}}^2$). Very slender struts under high compression may buckle prior to yield in physical prototypes.
4. **Grid-Restricted Topologies:** Strut connectivity is restricted to nodes located on the discrete Cartesian grid ($13 \times 5$). Continuous nodal coordinate optimization (shape optimization) is not performed.

---

## 8. Interactive Web Demo

The repository includes a standalone static web application located in `demo/`. The application requires **zero backend dependencies** and replays precomputed optimization trajectories directly from `demo/data/lattice_data.json`:
- **Interactive Scenarios:** Select between LC1, LC2, LC3 at volume fractions $v_f \in \{0.20, 0.30, 0.40\}$.
- **Synchronized Viewports:** Dual Canvas 2D renderers comparing Remodeling Pruning side-by-side with Genetic Algorithm search.
- **Animated Load Test:** Toggle displacement magnification with members colored dynamically by strain energy, axial stress, or thickness.
- **Benchmark Scoreboard:** Real-time metrics comparing compliance, stiffness, mass, and energy partition directly against Honeycomb and Voronoi baselines.

Launch locally with:
```bash
python reproduce_all.py --serve-demo
```
Or open `demo/index.html` directly in any modern browser.

---

## 9. Citation

If you use this benchmark codebase or research findings in your work, please cite:

```bibtex
@article{lattice_bio_2026,
  title   = {Bone Remodeling vs Evolution: A Controlled Comparison of Bio-Inspired Pruning and Genetic Algorithms for Lightweight 2D Lattice Design},
  author  = {Lattice-Bio Research Contributors},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://github.com/g0vindoh/lattice-bio}
}
```

---

## 10. License

This project is licensed under the permissive **MIT License** — see the [LICENSE](LICENSE) file for details.
