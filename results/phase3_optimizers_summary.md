# Phase 3 Optimization Results: Pruning vs Genetic Algorithm

Target Volume Fraction: 0.20

| Method                       | Load Case   |   Target VF |   Achieved VF |   Compliance (J) |   Stiffness (N/m) |   FE Solves |   Runtime (s) |   Active Members | Axial Share (%)   | Bending Share (%)   |
|:-----------------------------|:------------|------------:|--------------:|-----------------:|------------------:|------------:|--------------:|-----------------:|:------------------|:--------------------|
| Method A: Remodeling Pruning | LC1         |         0.2 |      0.19875  |      0.00377686  |           264.77  |          19 |      0.569216 |               81 | 89.3%             | 10.7%               |
| Method B: Genetic Algorithm  | LC1         |         0.2 |      0.199761 |      0.00426245  |           234.607 |         900 |     25.3115   |              169 | 99.7%             | 0.3%                |
| Method A: Remodeling Pruning | LC2         |         0.2 |      0.199167 |      0.000490638 |          2038.16  |          20 |      0.57919  |               76 | 95.4%             | 4.6%                |
| Method B: Genetic Algorithm  | LC2         |         0.2 |      0.1998   |      0.000480643 |          2080.55  |         900 |     30.7028   |              157 | 99.7%             | 0.3%                |
| Method A: Remodeling Pruning | LC3         |         0.2 |      0.2      |      1.26582e-05 |         79000     |          20 |      0.556951 |               76 | 99.7%             | 0.3%                |
| Method B: Genetic Algorithm  | LC3         |         0.2 |      0.19912  |      2.14916e-05 |         46529.9   |         900 |     32.5813   |              151 | 99.8%             | 0.2%                |
