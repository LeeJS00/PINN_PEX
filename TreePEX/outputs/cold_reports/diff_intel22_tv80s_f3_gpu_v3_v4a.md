# Feature comparison: `intel22_tv80s_f3`

- baseline label: `baseline` (n_selected=120, wall=121.4s)
- patched  label: `gpu_v3_v4a` (n_selected=120, wall=77.1s)
- common nets: V3 120 / V4 118

## Per-block runtime
| Block | n | baseline sum (s) | patched sum (s) | speedup | base mean (s) | patched mean (s) | base p95 (s) | patched p95 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V3 (41-D) | 120 | 45.827 | 1.647 | 27.82× | 381.89 ms | 13.73 ms | 2360.1 ms | 39.5 ms |
| V4 H3 (26-D) | 118 | 72.792 | 72.854 | 1.00× | 616.88 ms | 617.40 ms | 3176.5 ms | 3065.8 ms |

## V3 per-feature value drift (sorted by MAE%)
| Feature | n | baseline mean | patched mean | abs max diff | MAE | MAE% | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| `spacing_p25_um` | 120 | 0.9073 | 0.907 | 0.137 | 0.001968 | 2.959% | 0.99945 |
| `n_edges_3_to_4um` | 120 | 175.7 | 175.8 | 23 | 0.525 | 2.357% | 0.99871 |
| `spacing_p50_um` | 120 | 1.837 | 1.832 | 0.396 | 0.008006 | 1.837% | 0.99629 |
| `broadside_overlap_total_um2` | 120 | 0.7045 | 0.7053 | 0.8206 | 0.02018 | 1.106% | 0.98622 |
| `n_edges_1_to_3um` | 120 | 320.8 | 320.1 | 79 | 1.917 | 0.977% | 0.97603 |
| `lateral_overlap_total_um2` | 120 | 7.056 | 6.927 | 4.471 | 0.1295 | 0.668% | 0.99390 |
| `compact_cpl_estimate_total_fF` | 120 | 2.019 | 1.998 | 1.146 | 0.03857 | 0.580% | 0.99413 |
| `broadside_overlap_p95_um2` | 120 | 0.004817 | 0.004746 | 0.004739 | 7.083e-05 | 0.568% | 0.99368 |
_(34 more features omitted; sorted by MAE%)_

## V4 H3 per-feature value drift (sorted by MAE%)
| Feature | n | baseline mean | patched mean | abs max diff | MAE | MAE% | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| `agg_count_above_target_z` | 118 | 3.146e+04 | 3.146e+04 | 0 | 0 | 0.000% | 1.00000 |
| `agg_count_below_target_z` | 118 | 6.206e+04 | 6.206e+04 | 0 | 0 | 0.000% | 1.00000 |
| `agg_count_within_1um_xyz` | 118 | 1909 | 1909 | 0 | 0 | 0.000% | 1.00000 |
| `agg_count_within_3um_xyz` | 118 | 1.476e+04 | 1.476e+04 | 0 | 0 | 0.000% | 1.00000 |
| `agg_count_within_5um_xyz` | 118 | 3.891e+04 | 3.891e+04 | 0 | 0 | 0.000% | 1.00000 |
| `agg_n_distinct` | 118 | 592.2 | 592.2 | 0 | 0 | 0.000% | 1.00000 |
| `target_n_cuboids_check` | 118 | 107.9 | 107.9 | 0 | 0 | 0.000% | 1.00000 |
| `top1_agg_size_um2` | 118 | 264.4 | 264.4 | 0 | 0 | 0.000% | 1.00000 |
_(18 more features omitted; sorted by MAE%)_
