# Additive Manufacturing & Experimental Testing Protocol for 2D Lattice Specimens

## 1. Experimental Objectives
This protocol defines the physical validation procedure to empirically evaluate the mechanical stiffness, compliance, and failure kinematics of 2D frame lattice designs synthesized via:
- Bio-Inspired Remodeling Pruning
- Genetic Algorithm Optimization
- Hexagonal Honeycomb (Periodic Baseline)
- Voronoi Cellular Network (Stochastic Trabecular Baseline)

All specimens are fabricated using stereolithography (SLA) additive manufacturing directly from watertight STL models exported via `export_stl_models.py` (`stl/` directory).

---

## 2. Specimen Dimensions & Manufacturing Specification

### 2.1 Domain & Geometric Envelope
- **Bounding Dimensions**: Width $W = 150.0\text{ mm}$, Height $H = 50.0\text{ mm}$, Out-of-plane Depth $b = 10.0\text{ mm}$ (scaled proportionally $1:20$ from the computational domain $[3.0\text{ m} \times 1.0\text{ m} \times 1.0\text{ m}]$).
- **Target Volume Fractions**: $v_f \in \{0.20, 0.30, 0.40\}$.
- **Load / Support Boundary Pads**: Solid $2.5\text{ mm}$ thick continuous mounting pads at reaction supports and loading points, identical across all topologies to ensure uniform contact stress distribution.

### 2.2 3D Printing Parameters
- **System**: Formlabs Form 3+ Stereolithography (SLA) 3D Printer.
- **Resin Material**: Formlabs Tough 2000 Resin (simulating ductile structural ABS-like polymer).
- **Print Resolution (Z-Layer Height)**: $50\ \mu\text{m}$ ($0.050\text{ mm}$).
- **Print Orientation**: Flat in-plane orientation ($XY$ plane parallel to build platform) to ensure isotropic strut cross-sections and eliminate interlayer delamination along struts.
- **Support Strategy**: Light touchpoint supports ($0.4\text{ mm}$ contact diameter) applied strictly to outer perimeter surfaces, avoiding internal lattice voids.

### 2.3 Post-Processing & Curing Schedule
1. **Solvent Wash**: 20 minutes in 99% Isopropanol (IPA) using Form Wash system.
2. **Post-Cure Protocol**: 60 minutes at $60^\circ\text{C}$ under $405\text{ nm}$ UV light in Form Cure chamber to achieve complete cross-link polymerization.
3. **Dimensional Metrology**: Specimen strut thicknesses, outer boundaries, and total dry mass measured using a digital vernier caliper ($\pm 0.01\text{ mm}$) and analytical balance ($\pm 0.1\text{ mg}$).

### 2.4 Calibrated Base Material Properties (Tough 2000 Post-Cured)
- **Young's Modulus ($E_s$)**: $2.20\text{ GPa} \pm 0.10\text{ GPa}$ (determined from ASTM D638 Type IV tensile dogbone coupons printed in identical orientation).
- **Yield Strength ($\sigma_y$)**: $46.0\text{ MPa}$.
- **Ultimate Tensile Strength ($\sigma_u$)**: $52.0\text{ MPa}$.
- **Elongation at Break**: $15\%$.
- **Mass Density ($\rho_s$)**: $1,210\text{ kg/m}^3$.

---

## 3. Mechanical Testing Apparatus & Load Fixturing

### 3.1 Universal Testing System
- **Machine**: Instron 5969 Dual-Column Universal Testing System.
- **Load Cell**: Instron $10\text{ kN}$ static load cell (accuracy $\pm 0.5\%$ of indicated load down to $1/1000$ of capacity).
- **Crosshead Speed**: Quasi-static displacement-controlled loading at $1.0\text{ mm/min}$ up to initial plastic yield and ultimate member buckling/fracture.
- **Sampling Rate**: $100\text{ Hz}$.

### 3.2 Fixture Configurations
1. **LC1 (Cantilever Bending)**:
   - Left end ($x = 0$) clamped rigidly in custom ground-steel vise fixture over full $50\text{ mm}$ edge.
   - Downward point load applied at mid-height of right tip ($x = 150\text{ mm}, y = 25\text{ mm}$) using a hardened cylindrical steel roller indenter ($R = 5.0\text{ mm}$).
2. **LC2 (Simply Supported Beam)**:
   - Three-point flexure fixture: hardened cylindrical steel pins ($R = 5.0\text{ mm}$) supporting bottom-left corner ($x = 0$) and bottom-right corner ($x = 150\text{ mm}$, roller translation free).
   - Central downward load applied at top mid-span ($x = 75\text{ mm}, y = 50\text{ mm}$).
3. **LC3 (Compression Block)**:
   - Self-aligning spherical seated compression platen on base; flat hardened steel top platen applying uniform compressive displacement.

### 3.3 Strain & Kinematic Optical Metrology
- **Digital Image Correlation (DIC)**: High-contrast matte black speckle pattern applied to front face via fine airbrush. Stereoscopic 5-megapixel cameras capturing full-field displacement and strain tensor fields at $5\text{ Hz}$.
- **Contact LVDT**: Dual linear variable differential transformers mounted directly at load points to eliminate machine compliance artifacts.

---

## 4. Test Matrix & Replicate Schedule

A total of $N = 5$ independent physical test specimens per condition ($4\text{ methods} \times 3\text{ load cases} \times 3\text{ volume fractions} \times 5\text{ replicates} = 180\text{ total test specimens}$).

| Specimen ID Code | Design Topology | Target $v_f$ | Test Case | Replicates ($N$) | Primary Measured Metrics |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `SPEC-PRUN-LC1-V20-1..5` | Remodeling Pruning | 0.20 | LC1 Cantilever | 5 | Linear Stiffness $K_{\text{exp}}$, Yield Load $P_y$, Mode |
| `SPEC-GA-LC1-V20-1..5`   | Genetic Algorithm  | 0.20 | LC1 Cantilever | 5 | Linear Stiffness $K_{\text{exp}}$, Yield Load $P_y$, Mode |
| `SPEC-HON-LC1-V20-1..5`  | Hexagonal Honeycomb| 0.20 | LC1 Cantilever | 5 | Linear Stiffness $K_{\text{exp}}$, Yield Load $P_y$, Mode |
| `SPEC-VOR-LC1-V20-1..5`  | Voronoi Trabecular | 0.20 | LC1 Cantilever | 5 | Linear Stiffness $K_{\text{exp}}$, Yield Load $P_y$, Mode |
| `SPEC-PRUN-LC1-V30-1..5` | Remodeling Pruning | 0.30 | LC1 Cantilever | 5 | Linear Stiffness $K_{\text{exp}}$, Yield Load $P_y$, Mode |
| `SPEC-GA-LC1-V30-1..5`   | Genetic Algorithm  | 0.30 | LC1 Cantilever | 5 | Linear Stiffness $K_{\text{exp}}$, Yield Load $P_y$, Mode |
| `SPEC-HON-LC1-V30-1..5`  | Hexagonal Honeycomb| 0.30 | LC1 Cantilever | 5 | Linear Stiffness $K_{\text{exp}}$, Yield Load $P_y$, Mode |
| `SPEC-VOR-LC1-V30-1..5`  | Voronoi Trabecular | 0.30 | LC1 Cantilever | 5 | Linear Stiffness $K_{\text{exp}}$, Yield Load $P_y$, Mode |
| `SPEC-PRUN-LC2-V30-1..5` | Remodeling Pruning | 0.30 | LC2 Simply Supp| 5 | Midspan Stiffness $K_{\text{exp}}$, Failure Mechanism |
| `SPEC-GA-LC2-V30-1..5`   | Genetic Algorithm  | 0.30 | LC2 Simply Supp| 5 | Midspan Stiffness $K_{\text{exp}}$, Failure Mechanism |

---

## 5. Quantitative Verification Metrics

1. **Experimental vs FE Stiffness Agreement**:
   $$\text{Stiffness Discrepancy} = \frac{K_{\text{exp}} - K_{\text{FE}}}{K_{\text{FE}}} \times 100\%$$
   Acceptance criterion for FE model fidelity: $|\text{Stiffness Discrepancy}| \le 12.0\%$.

2. **Specific Stiffness**:
   $$K_{\text{spec}} = \frac{K_{\text{exp}}}{m_{\text{measured}}}\quad (\text{N}\cdot\text{mm}^{-1}\cdot\text{g}^{-1})$$

3. **Elastic Buckling Load Factor Verification**:
   Compare the experimentally observed onset of member buckling in compressive struts against the predicted Euler buckling load $P_{cr, i}$ and global BLF $\lambda_{\min}$.

4. **Failure Mode Classification**:
   - (A) Elastic Euler member buckling without fracture
   - (B) Plastic yielding / plastic hinge formation at node junctions
   - (C) Tensile strut fracture / microcracking
