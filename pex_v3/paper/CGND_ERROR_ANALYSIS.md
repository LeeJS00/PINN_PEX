# C_gnd Error Origin Analysis

_Mesh-curriculum 5-seed ensemble on cross-design test (95,594 nets)_

## Baseline

- gnd MAPE: median **19.896%**, mean 23.664%
- cpl MAPE: median **15.153%**, mean 19.850%
- total MAPE: median 7.888%

## 1. gnd error vs cpl error correlation

- Spearman ρ = 0.3342 (p=0.00e+00)
- Pearson r  = 0.2412 (p=0.00e+00)

**Interpretation**: Strong positive correlation suggests gnd/cpl errors share root cause (geometry, model capacity), NOT trading off.

## 2. Top features by Spearman |ρ| vs gnd error

| feature | \|Spearman ρ\| | Spearman ρ | Pearson r |
|---|---:|---:|---:|
| `compact_gnd_estimate_fF` | 0.1540 | -0.1540 | -0.1132 |
| `bbox_xy_um2` | 0.1486 | -0.1486 | -0.0391 |
| `vss_n_cuboids` | 0.1483 | -0.1483 | -0.0589 |
| `total_metal_area_um2` | 0.1439 | -0.1439 | -0.1126 |
| `compact_cpl_estimate_total_fF` | 0.1431 | -0.1431 | -0.0915 |
| `fanout` | 0.1406 | -0.1406 | -0.0802 |
| `broadside_overlap_total_um2` | 0.1359 | -0.1359 | -0.0853 |
| `n_edges_lt_1um` | 0.1348 | -0.1348 | -0.1162 |
| `aspect_ratio` | 0.1338 | -0.1338 | 0.0061 |
| `total_wire_length_um` | 0.1317 | -0.1317 | -0.1210 |
| `spacing_p25_um` | 0.1316 | 0.1316 | 0.1072 |
| `lateral_overlap_total_um2` | 0.1315 | -0.1315 | -0.0765 |
| `spacing_p50_um` | 0.1292 | 0.1292 | 0.1114 |
| `spacing_p95_um` | 0.1242 | 0.1242 | 0.1131 |
| `broadside_overlap_p95_um2` | 0.1232 | -0.1232 | -0.0888 |

## 3. Quartile breakdown (median feature value per gnd-error quartile)

| feature | Q1 (low err) | Q2 | Q3 | Q4 (high err) |
|---|---:|---:|---:|---:|
| `compact_gnd_estimate_fF` | 0.067 | 0.055 | 0.042 | 0.040 |
| `bbox_xy_um2` | 1.638 | 1.173 | 0.743 | 0.647 |
| `vss_n_cuboids` | 1767.000 | 1679.000 | 1574.500 | 1567.000 |
| `total_metal_area_um2` | 0.340 | 0.279 | 0.213 | 0.206 |
| `compact_cpl_estimate_total_fF` | 0.571 | 0.459 | 0.346 | 0.374 |
| `fanout` | 20.000 | 15.000 | 1.000 | 1.000 |
| `gnd_rel_err` | 0.047 | 0.146 | 0.258 | 0.426 |

## 4. Per-design breakdown

| design | n_nets | gnd median | gnd mean | cpl median | total median |
|---|---:|---:|---:|---:|---:|
| intel22_nova_f3 | 92,425 | 19.971% | 23.725% | 15.190% | 7.881% |
| intel22_tv80s_f3 | 3,169 | 17.704% | 21.874% | 14.369% | 8.233% |

## 5. Layer-mix stratification (gnd MAPE by dominant layer)

| dominant layer | n_nets | gnd median | gnd mean | cpl median |
|---|---:|---:|---:|---:|
| `M2` | 73,640 | 19.430% | 23.245% | 15.264% |
| `M3` | 19,651 | 21.727% | 25.207% | 14.896% |
| `M4` | 1,708 | 20.839% | 25.297% | 13.686% |
| `M5` | 595 | 17.958% | 19.886% | 13.362% |

## 6. Outlier characterization (top 50 highest gnd error)

| feature | top-50 median | overall median | ratio |
|---|---:|---:|---:|
| `compact_gnd_estimate_fF` | 0.416 | 0.048 | 8.61× |
| `bbox_xy_um2` | 56.086 | 0.929 | 60.35× |
| `vss_n_cuboids` | 1407.000 | 1626.000 | 0.87× |
| `total_metal_area_um2` | 2.797 | 0.247 | 11.34× |
| `compact_cpl_estimate_total_fF` | 1.268 | 0.427 | 2.97× |
| `fanout` | 92.000 | 13.000 | 7.08× |

## 7. Signed error (under vs over)

- median signed error: **-9.244%**
- mean signed error:   -4.200%
- % nets where pred < golden: **63.8%**
- % nets where pred > golden: **36.2%**

