# Feature comparison: `intel22_tv80s_f3`

- baseline label: `baseline` (n_selected=120, wall=121.4s)
- patched  label: `patched_v4a` (n_selected=120, wall=107.1s)
- common nets: V3 120 / V4 118

## Per-block runtime
| Block | n | baseline sum (s) | patched sum (s) | speedup | base mean (s) | patched mean (s) | base p95 (s) | patched p95 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V3 (41-D) | 120 | 45.827 | 32.675 | 1.40× | 381.89 ms | 272.29 ms | 2360.1 ms | 1527.6 ms |
| V4 H3 (26-D) | 118 | 72.792 | 71.412 | 1.02× | 616.88 ms | 605.19 ms | 3176.5 ms | 3040.1 ms |

## V3 per-feature value drift (sorted by MAE%)
| Feature | n | baseline mean | patched mean | abs max diff | MAE | MAE% | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| `spacing_p25_um` | 120 | 0.9073 | 0.9086 | 0.218 | 0.002848 | 6.796% | 0.99873 |
| `broadside_overlap_total_um2` | 120 | 0.7045 | 0.6856 | 1.513 | 0.02891 | 1.248% | 0.96316 |
| `spacing_p50_um` | 120 | 1.837 | 1.836 | 0.249 | 0.005513 | 1.238% | 0.99815 |
| `n_edges_3_to_4um` | 120 | 175.7 | 175.9 | 7 | 0.2333 | 0.988% | 0.99984 |
| `compact_cpl_estimate_total_fF` | 120 | 2.019 | 1.964 | 3.09 | 0.05936 | 0.802% | 0.98081 |
| `lateral_overlap_total_um2` | 120 | 7.056 | 6.892 | 4.537 | 0.1646 | 0.729% | 0.99097 |
| `n_edges_1_to_3um` | 120 | 320.8 | 320.4 | 43 | 1.267 | 0.686% | 0.98986 |
| `broadside_overlap_p95_um2` | 120 | 0.004817 | 0.004728 | 0.004433 | 8.864e-05 | 0.672% | 0.99321 |
| `lateral_overlap_p95_um2` | 120 | 0.02361 | 0.02339 | 0.009396 | 0.0002216 | 0.372% | 0.99527 |
| `n_edges_lt_1um` | 120 | 236.4 | 236.6 | 43 | 1.283 | 0.250% | 0.99804 |
_(32 more features omitted; sorted by MAE%)_

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
| `top1_layer_diff_flag` | 118 | 0.0339 | 0.0339 | 0 | 0 | 0.000% | 1.00000 |
| `top1_mean_dz_um` | 118 | 0.2427 | 0.2427 | 0 | 0 | 0.000% | 1.00000 |
_(16 more features omitted; sorted by MAE%)_
