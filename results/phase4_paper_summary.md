# Phase 4 Experimental Campaign: Final Paper Benchmark Summary

Generated at: 2026-11-02 22:30:08 | Total Runtime: 1628.0 s

## 1. Experiment E1: Statistical Performance Summary (vf = 0.20)

| load_case   | method                 |   median_compliance |   iqr_compliance |   median_stiffness |   median_specific_stiffness |   evaluations |    runtime |
|:------------|:-----------------------|--------------------:|-----------------:|-------------------:|----------------------------:|--------------:|-----------:|
| LC1         | GeneticAlgorithm       |         0.0044538   |      0.000378646 |          224.527   |                 0.0482251   |           900 | 130.126    |
| LC1         | Honeycomb              |         0.829996    |      0           |            1.20483 |                 0.000255802 |             1 |   0.05     |
| LC1         | RemodelingPruning      |         0.00377686  |      0           |          264.77    |                 0.056568    |            19 |   0.457827 |
| LC1         | UniformGroundStructure |         0.00644658  |      0           |          155.121   |                 0.0329344   |             1 |   0.05     |
| LC1         | Voronoi                |         0.0492722   |      0.00475757  |           20.2954  |                 0.00430901  |             1 |   0.05     |
| LC2         | GeneticAlgorithm       |         0.000490688 |      1.02595e-05 |         2037.96    |                 0.433374    |           900 | 151.372    |
| LC2         | Honeycomb              |         0.918492    |      0           |            1.08874 |                 0.000231155 |             1 |   0.05     |
| LC2         | RemodelingPruning      |         0.000490638 |      0           |         2038.16    |                 0.434542    |            20 |   0.563113 |
| LC2         | UniformGroundStructure |         0.000551853 |      0           |         1812.08    |                 0.38473     |             1 |   0.05     |
| LC2         | Voronoi                |         0.0468298   |      0.0290123   |           21.3539  |                 0.00453374  |             1 |   0.05     |
| LC3         | GeneticAlgorithm       |         2.33781e-05 |      1.64732e-06 |        42775.1     |                 9.14006     |           900 | 138.933    |
| LC3         | Honeycomb              |         0.0470587   |      0           |           21.2501  |                 0.00451169  |             1 |   0.05     |
| LC3         | RemodelingPruning      |         1.26582e-05 |      0           |        79000       |                16.7728      |            20 |   0.51516  |
| LC3         | UniformGroundStructure |         2.63799e-05 |      0           |        37907.6     |                 8.04832     |             1 |   0.05     |
| LC3         | Voronoi                |         0.000102907 |      3.07669e-05 |         9717.52    |                 2.06317     |             1 |   0.05     |

## 2. Hypothesis Testing: Mann-Whitney U Tests with Holm-Bonferroni Correction

| load_case   |   target_vf | comparison    |   u_statistic |   p_value_raw |   p_value_holm_adjusted |   rank_biserial_r |   vargha_delaney_a12 | is_significant_corrected   |
|:------------|------------:|:--------------|--------------:|--------------:|------------------------:|------------------:|---------------------:|:---------------------------|
| LC1         |         0.2 | GA_vs_Voronoi |             0 |    0.00793651 |               0.0714286 |                 1 |                    0 | False                      |
| LC1         |         0.3 | GA_vs_Voronoi |             0 |    0.00793651 |               0.0714286 |                 1 |                    0 | False                      |
| LC1         |         0.4 | GA_vs_Voronoi |             0 |    0.00793651 |               0.0714286 |                 1 |                    0 | False                      |
| LC2         |         0.2 | GA_vs_Voronoi |             0 |    0.00793651 |               0.0714286 |                 1 |                    0 | False                      |
| LC2         |         0.3 | GA_vs_Voronoi |             0 |    0.00793651 |               0.0714286 |                 1 |                    0 | False                      |
| LC2         |         0.4 | GA_vs_Voronoi |             0 |    0.00793651 |               0.0714286 |                 1 |                    0 | False                      |
| LC3         |         0.2 | GA_vs_Voronoi |             0 |    0.00793651 |               0.0714286 |                 1 |                    0 | False                      |
| LC3         |         0.3 | GA_vs_Voronoi |             0 |    0.00793651 |               0.0714286 |                 1 |                    0 | False                      |
| LC3         |         0.4 | GA_vs_Voronoi |             0 |    0.00793651 |               0.0714286 |                 1 |                    0 | False                      |

## 3. Experiment E4: Cross-Load Generalization Degradation Matrix

| optimized_for   |        LC1 |        LC2 |     LC3 |
|:----------------|-----------:|-----------:|--------:|
| LC1             |    1       | 32837      | 1470.21 |
| LC2             |    4.93826 |     1      | 2256.49 |
| LC3             | 5010.09    |    18.1327 |    1    |

## 4. Experiment E5: Kinematics (Axial vs Bending Energy Shares at vf = 0.20)

| experiment   | method            | load_case   |   axial_energy_share |   bending_energy_share | kinematic_class      |
|:-------------|:------------------|:------------|---------------------:|-----------------------:|:---------------------|
| E5           | RemodelingPruning | LC1         |           0.89306    |             0.10694    | Stretch-dominated    |
| E5           | GeneticAlgorithm  | LC1         |           0.997114   |             0.00288576 | Stretch-dominated    |
| E5           | Honeycomb         | LC1         |           0.00396075 |             0.996039   | Bending-dominated    |
| E5           | Voronoi           | LC1         |           0.139875   |             0.860125   | Intermediate/Bending |
| E5           | RemodelingPruning | LC2         |           0.953876   |             0.0461237  | Stretch-dominated    |
| E5           | GeneticAlgorithm  | LC2         |           0.996537   |             0.00346342 | Stretch-dominated    |
| E5           | Honeycomb         | LC2         |           0.00135143 |             0.998649   | Bending-dominated    |
| E5           | Voronoi           | LC2         |           0.0840449  |             0.915955   | Intermediate/Bending |
| E5           | RemodelingPruning | LC3         |           0.996643   |             0.00335679 | Stretch-dominated    |
| E5           | GeneticAlgorithm  | LC3         |           0.998196   |             0.00180404 | Stretch-dominated    |
| E5           | Honeycomb         | LC3         |           0.00146278 |             0.998537   | Bending-dominated    |
| E5           | Voronoi           | LC3         |           0.204277   |             0.795723   | Intermediate/Bending |

## 5. Experiment E6: Ablation Studies (LC1, vf = 0.20)

| experiment   | ablation_group                | parameter_name     | parameter_value       |   compliance_J |   stiffness_N_m |   fe_solves |   runtime_s |   active_members |
|:-------------|:------------------------------|:-------------------|:----------------------|---------------:|----------------:|------------:|------------:|-----------------:|
| E6           | Pruning_Removal_Rate          | removal_rate       | 0.02                  |     0.00390622 |         256.002 |          42 |    1.03163  |               81 |
| E6           | Pruning_Removal_Rate          | removal_rate       | 0.05                  |     0.00377686 |         264.77  |          19 |    0.370331 |               81 |
| E6           | Pruning_Removal_Rate          | removal_rate       | 0.1                   |     0.00660633 |         151.37  |          11 |    0.225188 |               78 |
| E6           | Pruning_Utilization_Metric    | utilization_metric | strain_energy_density |     0.00377686 |         264.77  |          19 |    0.421802 |               81 |
| E6           | Pruning_Utilization_Metric    | utilization_metric | axial_stress          |     0.00328962 |         303.986 |          19 |    0.523738 |               81 |
| E6           | Pruning_Utilization_Metric    | utilization_metric | peak_stress           |     0.00471419 |         212.126 |          19 |    0.497144 |               81 |
| E6           | Pruning_Wolffs_Redistribution | redistribute_area  | False                 |     0.00377686 |         264.77  |          19 |    0.506091 |               81 |
| E6           | Pruning_Wolffs_Redistribution | redistribute_area  | True                  |     0.00158663 |         630.267 |          25 |    0.579793 |               60 |

## 6. Experiment E7: Computational Scaling Benchmark

| experiment   |   nx |   ny |   num_nodes |   num_members |   grid_generation_time_s |   single_fe_solve_time_s |   total_fe_solves |   total_optimization_time_s |   final_compliance_J |   final_stiffness_N_m |
|:-------------|-----:|-----:|------------:|--------------:|-------------------------:|-------------------------:|------------------:|----------------------------:|---------------------:|----------------------:|
| E7           |    9 |    3 |          27 |           102 |               0.00459456 |                0.0138214 |                18 |                    0.194885 |           0.00315296 |               317.162 |
| E7           |   13 |    5 |          65 |           208 |               0.0229182  |                0.0305529 |                19 |                    0.409009 |           0.00377686 |               264.77  |
| E7           |   17 |    7 |         119 |           566 |               0.109439   |                0.0680079 |                18 |                    1.00306  |           0.0018597  |               537.721 |
| E7           |   21 |    9 |         189 |           948 |               0.270845   |                0.10366   |                35 |                    2.42513  |           0.00299067 |               334.374 |
