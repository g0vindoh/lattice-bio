# Bone Remodeling vs Evolution: A Controlled Comparison of Bio-Inspired Pruning and Genetic Algorithms for Lightweight 2D Lattice Design

**Authors:** Lattice-Bio Research Contributors  
**Date:** November 2026  
**Repository:** [https://github.com/lattice-bio/lattice-bio](https://github.com/lattice-bio/lattice-bio)  
**Git Commit Hash:** `29d59fb`  
**Artifact Archive:** Fully reproducible from seeds and configuration files via `reproduce_all.py`

---

## Abstract

Bio-inspired structural optimization frequently invokes biological archetypes—such as Darwinian natural selection or trabecular bone functional adaptation—to synthesize lightweight cellular architectures. However, comparative literature often suffers from confounding variables, including inconsistent boundary condition modeling, disparate finite element (FE) formulations, unequal volume fraction budgets, and unadjusted statistical testing. In this study, we present a controlled, reproducible benchmark comparing two bio-inspired design paradigms under strictly equal material budgets and standardized boundary conditions:
1. **Method A (Bone Remodeling-Inspired Pruning):** Iterative cellular resorption of under-utilized struts based on strain energy density and Wolff's law of functional adaptation.
2. **Method B (Genetic Algorithm - GA):** Population-based evolutionary search operating over discrete strut cross-sections with an active volume repair operator.
3. **Baselines:** Benchmark regular hexagonal honeycombs and stochastic Voronoi (trabecular-like) cellular lattices evaluated at identically matched volume fractions ($v_f \in \{0.20, 0.30, 0.40\}$).

All lattices are analyzed using a validated 2D Euler-Bernoulli frame element solver (3 DOFs/node: $u_x, u_y, \theta$) to rigorously capture flexural deformation modes. Standardized boundary and load pads are incorporated into all designs, ensuring that support compliance and volume expenditure are normalized across all topologies.

Across three canonical load cases (Cantilever Tip Shear LC1, Simply Supported 3-Point Bending LC2, and Compression Block LC3), **Method A achieves equivalent or superior structural stiffness ($K = 264.8$ N/m on LC1, $K = 79,000$ N/m on LC3 at $v_f = 0.20$) requiring only 19 to 20 FE evaluations (< 0.5 s runtime)**, whereas **Method B requires 900 FE evaluations (~130–150 s runtime) to reach comparable stiffness ($K = 224.5$ N/m on LC1, $K = 42,775$ N/m on LC3)**. Kinematic strain energy partitioning reveals that both optimized lattices converge to strongly **stretch-dominated deformation regimes ($89.3\% - 99.8\%$ axial strain energy)**, whereas hexagonal honeycombs remain overwhelmingly **bending-dominated ($99.6\% - 99.9\%$ flexural strain energy)**, explaining why honeycombs exhibit two to three orders of magnitude lower stiffness at equal mass. Cross-load evaluation uncovers an extreme **robustness gap**: topologies optimized for cantilever shear experience a **$32,837\times$ compliance degradation** when subjected to 3-point bending. Finally, biologically incorporating Wolff's law area redistribution (reallocating resorbed material to highly utilized struts) yields an additional **$+138\%$ stiffness enhancement ($264.8 \to 630.3$ N/m)** under identical material budgets. The entire computational pipeline is deterministic, config-driven, validated by 65 passing automated tests, and accompanied by an interactive open-source web demonstration.

---

## 1. Introduction

Lightweight cellular lattices are widely employed in aerospace, automotive, biomedical, and additive manufacturing applications where high specific stiffness (stiffness-to-weight ratio) and energy absorption are critical performance metrics (Gibson & Ashby, 1997; Schaedler & Carter, 2016). In recent decades, structural engineers and computer scientists have turned to biological systems for inspiration, drawing upon two distinct evolutionary and physiological phenomena:
1. **Phylogenetic Adaptation (Evolution):** Simulated via Genetic Algorithms (GAs) and evolutionary computation, where prospective topologies undergo mutation, recombination, and survival-of-the-fittest selection across generations (Holland, 1992; Goldberg, 1989).
2. **Ontogenetic Adaptation (Bone Remodeling):** Trabecular bone dynamically adapts its internal architecture throughout an organism's lifetime. In accordance with Wolff's law of bone transformation (Wolff, 1892) and modern mechanobiological theories (Cowin, 1986; Huiskes et al., 2000; Alsheghri et al., 2021), osteoclasts resorb bone matrix from mechanically quiescent trabeculae, while osteoblasts deposit bone matrix along principal stress trajectories, yielding high stiffness-to-weight cellular networks.

Despite hundreds of publications proposing bio-inspired optimization algorithms, the literature exhibits substantial methodological weaknesses:
- **Lack of Controlled Baselines:** Novel heuristics are frequently benchmarked only against naive random search or arbitrary initial states rather than standard cellular solid archetypes (such as regular honeycombs or relaxed Voronoi foams).
- **Unequal Evaluation Budgets & Volume Accounting:** Algorithmic comparisons often fail to enforce equal material mass budgets or neglect the mass of connection pads and load application fixtures.
- **Inadequate Kinematic Modeling:** Pin-jointed truss solvers are frequently misapplied to cellular solids; truss formulations fail to represent bending moments, treating bending-dominated lattices (such as hexagonal honeycombs) as singular kinematic mechanisms.
- **Selective Reporting:** Non-parametric statistical tests, multiple-comparison corrections, and failure cases (such as cross-load sensitivity) are routinely omitted.

### Contributions
This study resolves these gaps through an open, reproducible computational benchmark:
1. **Controlled Quantitative Comparison:** We benchmark bone remodeling-inspired pruning (Method A) directly against a Genetic Algorithm (Method B) and two classical cellular baselines across identical design domains, volume fractions ($0.20, 0.30, 0.40$), and loading conditions.
2. **Kinematic Strain Energy Partitioning:** Using a validated Euler-Bernoulli frame FE formulation, we quantify the exact split between axial strain energy ($U_{\text{axial}}$) and flexural bending strain energy ($U_{\text{bending}}$), demonstrating why remodeling and evolutionary search converge to stretch-dominated behavior while periodic cellular networks remain trapped in bending.
3. **The Robustness Specialization Gap:** We establish a quantitative cross-load transfer matrix demonstrating that extreme topological specialization carries severe vulnerability to off-axis loading.
4. **Wolff's Law Biological Area Redistribution:** We demonstrate that dynamic material redistribution during strut resorption achieves a $2.38\times$ stiffness improvement at zero additional material expenditure.

---

## 2. Mathematical Formulation & Structural Mechanics

### 2.1 2D Euler-Bernoulli Frame Element
Consider a structural network discretized into 2D frame elements in the Cartesian coordinate system $(x, y)$. Each node $i$ possesses three degrees of freedom (DOFs):
$$\mathbf{u}_i = [u_{xi}, u_{yi}, \theta_i]^T$$
where $u_{xi}$ is horizontal displacement, $u_{yi}$ is vertical displacement, and $\theta_i$ is in-plane rotation.

For an element $e$ connecting node 1 at $(x_1, y_1)$ to node 2 at $(x_2, y_2)$:
- **Element Length:** $L = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$
- **Orientation Angle:** $\theta = \operatorname{atan2}(y_2 - y_1, x_2 - x_1)$
- **Direction Cosines:** $c = \cos\theta, \quad s = \sin\theta$

#### Rectangular Cross-Section
For an in-plane strut thickness $t$ and unit out-of-plane depth $b = 1.0\text{ m}$:
- **Cross-Sectional Area:**
  $$A = b \cdot t = t \quad [\text{m}^2]$$
- **Second Moment of Area (In-Plane Flexure):**
  $$I = \frac{b \cdot t^3}{12} = \frac{t^3}{12} \quad [\text{m}^4]$$

#### Local and Global Stiffness Matrices
In local coordinates $(x', y')$, the local element stiffness matrix $\mathbf{K}_e'$ is given by:
$$\mathbf{K}_e' = \begin{bmatrix}
\frac{EA}{L} & 0 & 0 & -\frac{EA}{L} & 0 & 0 \\
0 & \frac{12EI}{L^3} & \frac{6EI}{L^2} & 0 & -\frac{12EI}{L^3} & \frac{6EI}{L^2} \\
0 & \frac{6EI}{L^2} & \frac{4EI}{L} & 0 & -\frac{6EI}{L^2} & \frac{2EI}{L} \\
-\frac{EA}{L} & 0 & 0 & \frac{EA}{L} & 0 & 0 \\
0 & -\frac{12EI}{L^3} & -\frac{6EI}{L^2} & 0 & \frac{12EI}{L^3} & -\frac{6EI}{L^2} \\
0 & \frac{6EI}{L^2} & \frac{2EI}{L} & 0 & -\frac{6EI}{L^2} & \frac{4EI}{L}
\end{bmatrix}$$

The coordinate transformation matrix $\mathbf{T}$ relating local displacements $\mathbf{u}_e'$ to global displacements $\mathbf{u}_e$ is:
$$\mathbf{u}_e' = \mathbf{T} \mathbf{u}_e, \quad \mathbf{T} = \begin{bmatrix} c & s & 0 & 0 & 0 & 0 \\ -s & c & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & c & s & 0 \\ 0 & 0 & 0 & -s & c & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \end{bmatrix}$$

The element stiffness in global coordinates is:
$$\mathbf{K}_e = \mathbf{T}^T \mathbf{K}_e' \mathbf{T}$$

Global assembly into the structural stiffness matrix $\mathbf{K}$ is performed using sparse Compressed Sparse Row (CSR) storage:
$$\mathbf{K} = \sum_{e=1}^{N_e} \mathbf{L}_e^T \mathbf{K}_e \mathbf{L}_e$$
where $\mathbf{L}_e$ is the topological Boolean assembly mapping matrix.

### 2.2 Boundary Conditions & Singularity Safeguards
Boundary constraints (fixed, pinned, roller) are applied via row-column elimination on the reduced system:
$$\mathbf{K}_{ff} \mathbf{u}_f = \mathbf{F}_f$$
where subscript $f$ denotes unconstrained (free) degrees of freedom.

To prevent artificial numerical convergence on kinematically unstable structures (mechanisms or disconnected subgraphs), we apply a multi-tier singularity detection protocol:
1. **Topological Graph Connectivity:** Breadth-First Search (BFS) confirms that all active elements belong to a single connected component that intersects both support nodes and load nodes.
2. **Rigid-Body Mode Projection Rank:** Unconstrained kinematic motions are identified by computing the nullspace dimension of $\mathbf{K}_{ff}$.
3. **Equilibrated Pivoting Check:** Symmetrical Jacobi equilibration verifies that the minimum pivot ratio $\min |D_{ii}| / \max |D_{ii}|$ exceeds $10^{-12}$. Any structural instability raises an explicit `SingularMatrixError` and triggers optimizer rollback.

### 2.3 Energy Decomposition and Metrics
Once the nodal displacement vector $\mathbf{u}$ is solved, member internal forces are evaluated:
- **Axial Force:** $N = \frac{EA}{L}(u_2' - u_1')$
- **Bending Moments:**
  $$M_1 = \frac{6EI}{L^2} v_1' + \frac{4EI}{L} \theta_1' - \frac{6EI}{L^2} v_2' + \frac{2EI}{L} \theta_2'$$
  $$M_2 = \frac{6EI}{L^2} v_1' + \frac{2EI}{L} \theta_1' - \frac{6EI}{L^2} v_2' + \frac{4EI}{L} \theta_2'$$

#### Strain Energy Partition
For each element, the strain energy splits cleanly into uncoupled axial and flexural components:
$$U_{\text{axial}}^{(e)} = \frac{1}{2} \frac{N^2 L}{EA} = \frac{1}{2} \frac{EA}{L} (u_2' - u_1')^2$$
$$U_{\text{bending}}^{(e)} = \frac{1}{2} \mathbf{u}_{be}'^T \mathbf{K}_{b}' \mathbf{u}_{be}'$$
where $\mathbf{u}_{be}' = [v_1', \theta_1', v_2', \theta_2']^T$.

Total internal strain energy is $U_{\text{total}} = \sum_e (U_{\text{axial}}^{(e)} + U_{\text{bending}}^{(e)})$.
The global compliance is:
$$C = \mathbf{F}^T \mathbf{u} = 2 U_{\text{total}}$$
Structural stiffness under unit reference load is $K = 1 / C$, and specific stiffness is:
$$K_{\text{specific}} = \frac{1 / C}{M_{\text{total}}} = \frac{K}{\rho V_{\text{total}}}$$

---

## 3. Design Domain, Boundary Pads, and Baselines

### 3.1 Design Domain & Standardized Load Cases
The design domain is a 2D rectangular region of width $W = 3.0\text{ m}$ and height $H = 1.0\text{ m}$ (aspect ratio 3:1), with unit depth $b = 1.0\text{ m}$ and total bounding volume $V_{\text{domain}} = 3.0\text{ m}^3$. Structural material is structural steel ($E = 210\text{ GPa}, \nu = 0.30, \rho = 7850\text{ kg/m}^3$).

Three standardized load cases (LC1, LC2, LC3) are defined in external YAML configurations:
- **LC1 (Cantilever Tip Shear):** Fully fixed support on the left edge ($x = 0$); downward transverse point load $F_y = -1000\text{ N}$ at the right tip mid-height $(3.0, 0.5)$.
- **LC2 (Simply Supported 3-Point Bending):** Pinned support at bottom-left $(0, 0)$, roller support at bottom-right $(3.0, 0)$, and central vertical load $F_y = -1000\text{ N}$ applied at top-center $(1.5, 1.0)$.
- **LC3 (Compression Block):** Fixed boundary conditions along the entire bottom boundary ($y = 0$); distributed downward compressive nodal loads $\sum F_y = -1000\text{ N}$ applied across the top surface ($y = 1.0$).

### 3.2 The Boundary Pad Fairness Rule
A critical flaw in prior structural optimization benchmarks is applying concentrated point loads or supports directly to delicate interior struts, creating local stress singularities and inconsistent boundary compliance. To ensure strict fairness across all designs, we implement standardized **load/support pads**:
- Fixed, thin boundary members ($t_{\text{pad}} = 5.0\text{ mm}$) are placed at all load application points and support locations across every candidate lattice.
- The volume occupied by these pads ($V_{\text{pad}}$) is computed analytically and subtracted equally from the material budget of every design:
  $$V_{\text{lattice}}^{\text{budget}} = v_f \cdot V_{\text{domain}} - V_{\text{pad}}$$
- Optimization and baseline thickness scaling operate solely on the remaining internal volume budget.

### 3.3 Ground Structure Candidate Network
The design space for Method A and Method B is initialized as an over-connected ground structure on a $13 \times 5$ Cartesian nodal grid (65 nodes). Candidate members are generated between all node pairs separated by distance $d \le R_{\max} = 0.60\text{ m}$. To prevent non-physical overlapping elements, a strict collinear intersection filter eliminates any candidate member $AB$ if a third grid node $C$ lies collinearly along $AB$. The standard ground structure contains **208 unique, non-overlapping candidate members**.

### 3.4 Baseline Architectures
1. **Regular Hexagonal Honeycomb:** Tiled regular hexagonal cells with cell edge length $l = 0.35\text{ m}$. Cells are clipped to the domain boundary $[0, W] \times [0, H]$ using the Liang-Barsky line clipping algorithm. Strut thickness $t_{\text{hc}}$ is solved analytically via $t_{\text{hc}} = V_{\text{lattice}}^{\text{budget}} / L_{\text{hc}}^{\text{total}}$ to match the target volume fraction to machine precision ($< 10^{-12}$).
2. **Trabecular-Like Voronoi Lattice:** Stochastic Voronoi seeds ($N = 45$) are mirrored across domain boundaries to guarantee perpendicular closure. Three iterations of Lloyd centroidal relaxation are applied to eliminate pathological narrow angles and homogenize cell distribution, mimicking trabecular bone morphogenesis. Edges shorter than $5\%$ of average length are merged, clipped to the domain boundary, and thickness scaled to match $v_f$ exactly.
3. **Uniformly Thinned Ground Structure:** The complete 208-member ground structure with all strut cross-sections uniformly set to $t_{\text{uniform}} = V_{\text{lattice}}^{\text{budget}} / L_{\text{gs}}^{\text{total}}$.

---

## 4. Optimization Algorithms

```
========================================================================================
ALGORITHM 1: Remodeling-Inspired Pruning with Wolff's Law Area Redistribution (Method A)
========================================================================================
Input: Ground structure G = (V, E), Target volume fraction v_f, Removal rate eta = 0.05
Output: Optimized lattice topology and cross-section assignment

1: Initialize all active members: a_e = 1, t_e = t_max for all e in E
2: Attach fixed load/support boundary pads
3: while current volume fraction v(a, t) > v_f do
4:     Solve 2D Euler-Bernoulli FE system: K(a, t) u = F
5:     Compute internal forces and member strain energy density: rho_e = U_e / V_e
6:     Rank active members in ascending order of utilization: rho_(1) <= rho_(2) <= ...
7:     Identify candidate subset of lowest rho_e: S_prune = { (1), ..., (k) } where k = ceil(eta * |E_active|)
8:     Tentatively remove S_prune: a_e' = 0 for e in S_prune
9:     Check feasibility (graph connectivity, load path, mechanism check)
10:    if feasible then
11:        Commit removal: a_e = a_e'
12:        if Wolff's redistribution enabled then
13:            Reallocate 50% of removed volume to top 15% highest-utilized struts:
14:            t_top = sqrt(t_top^2 + Delta A / b)
15:        end if
16:    else
17:        Rollback S_prune, reduce removal rate eta = eta / 2, or prune single member
18:    end if
19: end while
20: Scale final active member thicknesses to match target v_f exactly
21: return Final verified design
========================================================================================
```

```
========================================================================================
ALGORITHM 2: Genetic Algorithm with Active Volume Budget Repair Operator (Method B)
========================================================================================
Input: Ground structure G = (V, E), Population size P = 30, Generations G_max = 30,
       Crossover rate p_c = 0.85, Mutation rate p_m = 0.05
Output: Best elite lattice chromosome

1: Chromosome representation: vector x in {0, 1, ..., S_levels}^|E|
2: Initialize random population of P individuals
3: for gen = 1 to G_max do
4:     for each individual i in 1 ... P do
5:         Apply Volume Budget Repair Operator:
6:             Compute volume V(x_i)
7:             if V(x_i) > V_budget: iteratively decrement highest/random non-zero genes
8:             if V(x_i) < V_budget: iteratively increment random genes until V(x_i) ~= V_budget
9:         Assemble FE mesh and verify feasibility (connectivity, load path, singularity)
10:        if feasible then
11:            Solve K u = F; Fitness(x_i) = 1.0 / Compliance(u)
12:        else
13:            Fitness(x_i) = 1e-12 (Heavy death penalty)
14:        end if
15:    end for
16:    Preserve top 2 elite chromosomes (Elitism)
17:    Perform tournament selection (k_tournament = 3) to construct mating pool
18:    Apply uniform crossover with probability p_c
19:    Apply point mutation with probability p_m
20: end for
21: return Best elite individual
========================================================================================
```

---

## 5. Experimental Results & Statistical Benchmarks

All experiments were executed on an AMD processor architecture under Windows 64-bit with Python 3.14 / NumPy 2.2 / SciPy 1.15. Random number seeds were controlled and logged.

### 5.1 Experiment E1: Final Structural Performance Benchmark
Table 1 compiles the final structural compliance ($C$), structural stiffness ($K = 1/C$), and specific stiffness at $v_f = 0.20$ across all methods and baselines.

**Table 1: Benchmark structural performance at volume fraction $v_f = 0.20$ (Experiment E1).**
*Compliance values in Joules; stiffness in N/m; runtime in seconds.*

| Load Case | Optimization Method | Median Compliance (J) | IQR Compliance (J) | Median Stiffness (N/m) | Median Specific Stiffness | FE Solves | Runtime (s) | Active Struts |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LC1 (Cantilever)** | **Method A (Pruning)** | **0.003777** | **0.000000** | **264.77** | **0.05657** | **19** | **0.46** | **81** |
| LC1 | Method B (GA) | 0.004454 | 0.000379 | 224.53 | 0.04823 | 900 | 130.13 | 162 |
| LC1 | Uniform Ground Struct. | 0.006447 | 0.000000 | 155.12 | 0.03293 | 1 | 0.05 | 208 |
| LC1 | Voronoi Lattice | 0.049272 | 0.004758 | 20.30 | 0.00431 | 1 | 0.05 | 144 |
| LC1 | Regular Honeycomb | 0.829996 | 0.000000 | 1.20 | 0.00026 | 1 | 0.05 | 33 |
| **LC2 (3-Point Bend)**| **Method A (Pruning)** | **0.000491** | **0.000000** | **2038.16** | **0.43454** | **20** | **0.56** | **81** |
| LC2 | Method B (GA) | 0.000491 | 0.000010 | 2037.96 | 0.43337 | 900 | 151.37 | 165 |
| LC2 | Uniform Ground Struct. | 0.000552 | 0.000000 | 1812.08 | 0.38473 | 1 | 0.05 | 208 |
| LC2 | Voronoi Lattice | 0.046830 | 0.029012 | 21.35 | 0.00453 | 1 | 0.05 | 144 |
| LC2 | Regular Honeycomb | 0.918492 | 0.000000 | 1.09 | 0.00023 | 1 | 0.05 | 33 |
| **LC3 (Compression)** | **Method A (Pruning)** | **0.000013** | **0.000000** | **79000.00** | **16.7728** | **20** | **0.52** | **81** |
| LC3 | Method B (GA) | 0.000023 | 0.000002 | 42775.10 | 9.14006 | 900 | 138.93 | 158 |
| LC3 | Uniform Ground Struct. | 0.000026 | 0.000000 | 37907.60 | 8.04832 | 1 | 0.05 | 208 |
| LC3 | Voronoi Lattice | 0.000103 | 0.000031 | 9717.52 | 2.06317 | 1 | 0.05 | 144 |
| LC3 | Regular Honeycomb | 0.047059 | 0.000000 | 21.25 | 0.00451 | 1 | 0.05 | 33 |

#### Non-Parametric Hypothesis Testing
To verify whether evolutionary optimization significantly outperforms the stochastic trabecular baseline, two-sided Mann-Whitney U tests were performed comparing GA and Voronoi samples across 9 conditions ($3\text{ LCs} \times 3\text{ VFs}$). Across all conditions, GA demonstrated complete rank superiority over Voronoi ($U = 0, p_{\text{raw}} = 0.007936$, Rank-Biserial correlation $r = 1.0$, Vargha-Delaney effect size $A_{12} = 0.0$, indicating that every GA design outperformed every Voronoi design). Applying strict Holm-Bonferroni family-wise error rate control over the 9 simultaneous comparisons yields adjusted $p = 0.0714$.

---

### 5.2 Experiment E2: Convergence Dynamics
Figure 3 captures the optimization trajectories:
- **Remodeling Pruning (Method A):** Monotonically removes material along the steepest strain-energy gradient. Because it prunes $5\%$ of low-strain members per step, it converges to the volume fraction budget in exactly **19 to 20 iterations**, utilizing fewer than 20 FE solves in $0.46\text{ s}$.
- **Genetic Algorithm (Method B):** Begins with randomly distributed gene levels. Although the volume repair operator guarantees budget satisfaction, evolutionary recombination requires approximately **15 generations (450 FE solves)** before discovering efficient load-bearing diagonals, plateauing at 30 generations (900 FE solves).
- **Computational Speedup:** Method A delivers equivalent or superior structural quality with a **$45\times$ reduction in FE solver evaluations** and a **$280\times$ wall-clock speedup** relative to Method B.

---

### 5.3 Experiment E4: The Robustness Gap (Cross-Load Evaluation)
To examine topological specialization versus generalization, designs optimized for one load case were evaluated on the other two load cases. Table 2 presents the cross-load degradation ratio:
$$\text{Degradation Ratio} = \frac{C(\text{Topology } i \text{ under Load } j)}{C(\text{Topology } j \text{ under Load } j)}$$

**Table 2: Cross-load compliance degradation matrix (Experiment E4 at $v_f = 0.20$).**
*Values denote fold-increase in compliance relative to the natively optimized structure (1.0 = native).*

| Optimized For | Evaluated on LC1 (Cantilever) | Evaluated on LC2 (3-Point Bend) | Evaluated on LC3 (Compression) |
| :--- | :--- | :--- | :--- |
| **LC1 (Cantilever Truss)** | **1.00** | **32,836.98** | **1,470.21** |
| **LC2 (3-Point Arch)** | **4.94** | **1.00** | **2,256.49** |
| **LC3 (Compression Columns)**| **5,010.09** | **18.13** | **1.00** |

#### Crucial Structural Insight
The cantilever-optimized topology (LC1) experiences a catastrophic **$32,837\times$ degradation** when tested under central 3-point bending (LC2). This occurs because the LC1 design prunes all struts near the bottom center $(x = 1.5, y = 0)$ where cantilever bending moment is minimal, leaving no vertical load path when a point load is subsequently applied at $(1.5, 1.0)$. Conversely, the 3-point bending arch (LC2) generalizes relatively well to the cantilever load ($4.94\times$ degradation) because its curved truss arch spans the full length of the domain.

---

### 5.4 Experiment E5: Kinematics (Axial vs. Bending Strain Energy Share)
Table 3 reports the kinematic deformation breakdown between axial stretching and flexural bending.

**Table 3: Kinematic strain energy partition at volume fraction $v_f = 0.20$ (Experiment E5).**

| Method / Baseline | Load Case | Axial Strain Energy ($U_{\text{axial}}$) | Bending Strain Energy ($U_{\text{bending}}$) | Kinematic Classification |
| :--- | :--- | :--- | :--- | :--- |
| **Method A (Pruning)** | LC1 | **89.31%** | **10.69%** | **Stretch-Dominated** |
| **Method B (GA)** | LC1 | **99.71%** | **0.29%** | **Stretch-Dominated** |
| **Honeycomb Baseline** | LC1 | **0.40%** | **99.60%** | **Bending-Dominated** |
| **Voronoi Baseline** | LC1 | **13.99%** | **86.01%** | **Bending-Dominated** |
| **Method A (Pruning)** | LC2 | **95.39%** | **4.61%** | **Stretch-Dominated** |
| **Method B (GA)** | LC2 | **99.65%** | **0.35%** | **Stretch-Dominated** |
| **Honeycomb Baseline** | LC2 | **0.14%** | **99.86%** | **Bending-Dominated** |
| **Voronoi Baseline** | LC2 | **8.40%** | **91.60%** | **Bending-Dominated** |
| **Method A (Pruning)** | LC3 | **99.66%** | **0.34%** | **Stretch-Dominated** |
| **Method B (GA)** | LC3 | **99.82%** | **0.18%** | **Stretch-Dominated** |
| **Honeycomb Baseline** | LC3 | **0.15%** | **99.85%** | **Bending-Dominated** |
| **Voronoi Baseline** | LC3 | **20.43%** | **79.57%** | **Bending-Dominated** |

#### Why Honeycombs Underperform
Hexagonal honeycombs deform predominantly via the flexure of their horizontal and oblique cell walls ($> 99.6\%$ bending energy). In classical cellular solid theory (Gibson & Ashby, 1997), the effective modulus of bending-dominated cellular solids scales quadratically with relative density ($E^* / E_s \propto (v_f)^2$ or $(v_f)^3$), whereas stretch-dominated triangulated trusses scale linearly ($E^* / E_s \propto v_f$). At $v_f = 0.20$, this theoretical scaling disparity manifests as a **$220\times$ lower stiffness in honeycombs ($K = 1.20$ N/m) compared to optimized trusses ($K = 264.77$ N/m)**.

---

### 5.5 Experiment E6: Ablation Studies & Wolff's Law
We performed systematic parameter ablations on Method A under LC1 ($v_f = 0.20$):
1. **Pruning Removal Rate ($\eta$):** $\eta = 0.02$ requires 42 FE solves ($K = 256.0$ N/m); $\eta = 0.05$ requires 19 FE solves ($K = 264.8$ N/m); $\eta = 0.10$ requires only 11 FE solves but suffers premature topological disconnection ($K = 151.4$ N/m). Thus, $\eta = 0.05$ represents the optimal Pareto trade-off between speed and structural quality.
2. **Utilization Metric:** Axial stress utilization ($\sigma_{\text{axial}}$) yields slightly higher stiffness ($K = 303.99$ N/m) than strain energy density ($K = 264.77$ N/m), while peak surface stress including flexure ($\sigma_{\text{peak}}$) yields lower stiffness ($K = 212.13$ N/m) due to over-penalizing slight bending moments.
3. **Wolff's Law Area Redistribution:** Reallocating $50\%$ of resorbed strut volume into the top $15\%$ most heavily loaded active struts increases structural stiffness from $264.8$ N/m to **$630.3$ N/m ($2.38\times$ / $+138\%$ stiffness boost)** with 60 highly reinforced primary struts under the exact same volume fraction budget.

---

### 5.6 Experiment E7: Computational Scaling
Table 4 records computational scaling as the grid size expands from $9 \times 3$ (102 members) to $21 \times 9$ (948 members).

**Table 4: Computational scaling across grid resolutions (Experiment E7).**

| Grid Size | Nodes | Candidate Struts | Grid Gen. Time (s) | Single FE Solve (ms) | Total FE Solves | Total Optimization Time (s) | Final Stiffness (N/m) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **9 x 3** | 27 | 102 | 0.005 | 13.82 | 18 | **0.19** | 317.16 |
| **13 x 5** | 65 | 208 | 0.023 | 30.55 | 19 | **0.41** | 264.77 |
| **17 x 7** | 119 | 566 | 0.109 | 68.01 | 18 | **1.00** | 537.72 |
| **21 x 9** | 189 | 948 | 0.271 | 103.66 | 35 | **2.43** | 334.37 |

Single sparse FE solves scale near-linearly ($O(N^{1.2})$), allowing Method A to optimize a dense 948-member network in **2.43 seconds**.

---

## 6. Discussion & Biological Interpretation

### 6.1 Ontogenetic vs. Phylogenetic Adaptation
Our controlled benchmark highlights fundamental differences between ontogenetic remodeling and evolutionary search:
- **Bone Remodeling (Pruning)** operates via localized, physics-informed physiological feedback: osteoclasts resorb bone where local strain energy falls below a biological setpoint. Because physical stress directly indicates which struts carry negligible load, pruning moves deterministically downhill along the compliance gradient.
- **Evolutionary Search (GA)** lacks intrinsic knowledge of structural mechanics. Mutations and crossovers randomly perturb chromosome bitstrings without awareness of force equilibrium, relying on survival-of-the-fittest selection to filter out pathological designs. Consequently, the GA expends hundreds of FE evaluations recreating basic structural load paths that remodeling discovers in fewer than 10 iterations.

### 6.2 Threats to Validity & Mitigation
- **Confounding Support Volume:** Mitigated by the boundary pad fairness protocol.
- **Mechanism Singularities:** Mitigated by the multi-tier graph and rank check.
- **Multiple Comparisons:** Mitigated by Holm-Bonferroni correction.

---

## 7. Known Modeling Assumptions & Limitations

To ensure scientific integrity, the following assumptions and limitations are explicitly declared:
1. **2D Planar Frame Elements:** The current framework is restricted to two-dimensional planar kinematics. Three-dimensional torsional and out-of-plane buckling modes are not modeled.
2. **Linear Elasticity & Small Displacements:** Euler-Bernoulli beam theory assumes linear elasticity ($E = \text{const}$) and infinitesimal deformations. Plastic yielding, geometric non-linearity ($P\text{-}\Delta$ effects), and finite rotations are omitted.
3. **Absence of Local Strut Buckling:** The optimizer maximizes stiffness without an explicit Euler buckling constraint ($P_{\text{cr}} = \pi^2 E I / L_{\text{eff}}^2$). In slender compression members ($L/t > 50$), buckling would precede tensile yielding.
4. **Discrete Ground Structure Boundary:** Topology is restricted to node positions defined on the Cartesian grid. Nodal position optimization (shape optimization) is not performed.

---

## 8. Conclusion

We conducted a controlled comparison of bone remodeling-inspired pruning and genetic algorithms for lightweight 2D lattice design under strictly equal material budgets. Our findings demonstrate:
1. Remodeling-inspired pruning converges to optimal stretch-dominated topologies in **19 to 20 FE solves (< 0.5 s)**, achieving equal or superior stiffness to a 900-solve GA ($280\times$ faster).
2. Both bio-inspired optimization methods produce stretch-dominated lattices ($89\% - 99\%$ axial energy), outperforming regular honeycombs ($> 99\%$ bending energy) by up to $220\times$ in stiffness.
3. Extreme topological specialization creates severe cross-load vulnerability ($32,837\times$ degradation).
4. Biologically incorporating Wolff's law area redistribution provides a $+138\%$ stiffness enhancement at zero additional material cost.

---

## Appendix: FE Solver Analytical Validation Suite

All 7 required analytical benchmarks passed with machine-precision accuracy ($< 10^{-6}$ relative error).

| Test Benchmark | Analytical Closed-Form Formula | Expected Value | Computed Value | Relative Error | Tolerance | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Axial Bar** | $u_x = \frac{P L}{E A}$ | $5.952381 \times 10^{-6}\text{ m}$ | $5.952381 \times 10^{-6}\text{ m}$ | $0.0$ | $1.0 \times 10^{-6}$ | **PASS** |
| **2. Cantilever Beam** | $u_y = -\frac{P L^3}{3 E I}$ | $-0.03214286\text{ m}$ | $-0.03214286\text{ m}$ | $2.16 \times 10^{-16}$ | $1.0 \times 10^{-6}$ | **PASS** |
| **3. Simply Supported Beam** | $u_y = -\frac{P L^3}{48 E I}$ | $-0.01142857\text{ m}$ | $-0.01142857\text{ m}$ | $0.0$ | $1.0 \times 10^{-6}$ | **PASS** |
| **4. Rigid-Body Modes** | $\operatorname{dim} \operatorname{ker}(\mathbf{K}_{\text{unconstrained}}) = 3$ | $3$ zero eigenvalues | $3$ zero eigenvalues | $9.76 \times 10^{-17}$ | $1.0 \times 10^{-10}$ | **PASS** |
| **5. Portal Frame Symmetry** | $\|u_{\text{left}}\| - \|u_{\text{right}}\| = 0$ | $0.0\text{ m}$ | $4.85 \times 10^{-12}\text{ m}$ | $4.85 \times 10^{-12}$ | $1.0 \times 10^{-6}$ | **PASS** |
| **6. Mesh Refinement Invariance** | $N_{\text{elements}} = 1 \to 32$ | $-0.03214286\text{ m}$ | $-0.03214286\text{ m}$ | $3.10 \times 10^{-11}$ | $1.0 \times 10^{-6}$ | **PASS** |
| **7. Equilibrium Reactions** | $\sum \mathbf{R} + \sum \mathbf{F} = \mathbf{0}$ | $\mathbf{0}$ | $4.04 \times 10^{-14}\text{ N}$ | $4.04 \times 10^{-14}$ | $1.0 \times 10^{-6}$ | **PASS** |
