# Phase 1 FE Solver Validation Summary

Configuration: `configs/phase1_validation.yaml`

| Test ID   | Description                                                     | Metric                                         |     Expected |     Computed |   Relative Error |   Tolerance | Status   |
|:----------|:----------------------------------------------------------------|:-----------------------------------------------|-------------:|-------------:|-----------------:|------------:|:---------|
| Test 1    | Axial bar tip displacement: PL / (EA)                           | Tip displacement ux (m)                        |  5.95238e-06 |  5.95238e-06 |      0           |       1e-06 | PASS     |
| Test 2    | Cantilever tip deflection: PL^3 / (3EI)                         | Tip deflection uy (m)                          | -0.032143    | -0.032143    |      4.31753e-16 |       1e-06 | PASS     |
| Test 3    | Simply supported beam central deflection: PL^3 / (48EI)         | Midpoint deflection uy (m)                     | -0.011429    | -0.011429    |      6.07153e-16 |       1e-06 | PASS     |
| Test 4    | Unconstrained element rigid-body modes (multiplicity 3)         | Zero-energy modes count (ratio < 1e-12)        |  3           |  3           |      9.75951e-17 |       1e-10 | PASS     |
| Test 5    | Symmetric portal frame under symmetric load: knee ux asymmetry  | Relative knee asymmetry |ux_L + ux_R| / |ux_L| |  0           |  6.311e-12   |      6.311e-12   |       1e-06 | PASS     |
| Test 6    | Mesh refinement invariance (1 to 32 elements): tip deflection   | Max relative error across 6 refinement levels  | -0.032143    | -0.032143    |      4.22857e-11 |       1e-06 | PASS     |
| Test 7    | Global equilibrium: reactions sum to applied loads (Fx, Fy, M0) | Max relative equilibrium error                 |  0           |  1.0758e-13  |      1.0758e-13  |       1e-06 | PASS     |
| Test 8    | Singular matrix / mechanism detection (rotational mechanism)    | SingularMatrixError raised (1=True, 0=False)   |  1           |  1           |      0           |       0     | PASS     |
