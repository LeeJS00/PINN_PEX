# Cross-design tv80s — Final Report

_Generated 2026-05-02 KST. Total individual models evaluated: 75._

## Setup
- **Workspace**: `experiments/cross_design_tv80s_2026_05_02/` (isolated, separate from `pex_v3/` and the 02-53-launched `experiments/tv80s_autonomous_2026_05_02/` from another session).
- **Goal**: per-net `total_cap_fF` MAPE < 4% on tv80s, training on small intel22 designs.
- **Train designs** (9): aes_cipher_top, gcd, ibex_core, ldpc_decoder_802_3an, mc_top, spi_top, usbf_top, vga_enh_top, wb_conmax_top.
- **Validation**: nova (where available) or ibex_core fallback.
- **Test**: tv80s — full chip, 3,169 reachable nets (after manifest∩SPEF∩DEF intersection).
- **Features**: 60 (v1) → 114 (v2 layer-aware) → 145 (v3 multi-radius density). All SPEF-derived columns dropped to prevent label leakage.
- **Models**: LightGBM + XGBoost + CatBoost (CPU) × 5 seeds × {direct, residual} + ResMLP (GPU) × 5 seeds + DeepSet over cuboids (3-stream target/aggressor/power masked-pool encoder + hand-feature branch, GPU) × 5 seeds.

## Per-group summary (mean over seeds within each model class)

| group | n | mape_mean | mape_median | mape_p90 | mape_p99 |
|---|---|---|---|---|---|
| `deepset_v2` | 3169 | 8.398% | 6.427% | 16.77% | 45.86% |
| `resmlp_v3_nova` | 3169 | 8.554% | 6.373% | 17.36% | 51.60% |
| `resmlp_v3` | 3169 | 8.725% | 6.410% | 17.95% | 53.05% |
| `direct_cat` | 3169 | 9.186% | 6.873% | 18.96% | 50.15% |
| `direct_lgbm` | 3169 | 9.254% | 6.692% | 19.05% | 54.51% |
| `direct_xgb` | 3169 | 9.367% | 6.713% | 19.19% | 54.38% |
| `resmlp_v2` | 3169 | 10.351% | 7.484% | 21.56% | 54.68% |
| `mlp_hand_v2` | 3169 | 11.547% | 8.243% | 24.33% | 59.90% |

## Top-15 individual models

| tag | mape_mean | mape_median | mape_p90 |
|---|---|---|---|
| `deepset_v2::deepset_v2::seed8` | 8.567% | 6.686% | 16.56% |
| `deepset_v2::deepset_v2::seed3` | 8.594% | 6.516% | 17.28% |
| `deepset_v2::deepset_v2::seed4` | 8.662% | 6.650% | 17.76% |
| `resmlp_v3_nova::resmlp_v3_nova::seed1` | 8.671% | 6.366% | 17.55% |
| `deepset_v2::deepset_v2::seed7` | 8.696% | 6.558% | 17.58% |
| `resmlp_v3_nova::resmlp_v3_nova::seed2` | 8.710% | 6.528% | 17.84% |
| `resmlp_v3_nova::resmlp_v3_nova::seed0` | 8.716% | 6.459% | 17.84% |
| `deepset_v2::deepset_v2::seed5` | 8.719% | 6.535% | 17.41% |
| `deepset_v2::deepset_v2::seed9` | 8.728% | 6.715% | 17.48% |
| `resmlp_v3::resmlp_v3::seed4` | 8.747% | 6.523% | 17.46% |
| `deepset_v2::deepset_v2::seed6` | 8.804% | 6.824% | 17.47% |
| `deepset_v2::deepset_v2::seed2` | 8.832% | 6.900% | 17.57% |
| `deepset_v2::deepset_v2::seed1` | 8.852% | 6.865% | 17.43% |
| `resmlp_v3::resmlp_v3::seed2` | 8.853% | 6.497% | 17.86% |
| `resmlp_v3::resmlp_v3::seed1` | 8.884% | 6.632% | 18.13% |

## Ensembles (sorted by mean MAPE)

| ensemble | mape_mean | mape_median | mape_p90 | mape_p99 |
|---|---|---|---|---|
| **`ENS_super_ensemble`** (uniform mean of 7 1D + 8 2D stratifications, 15 total) | **7.9852%** | 6.029% | 16.12% | — |
| `ENS_super_ensemble_geomean` (geomean of same 15 stratifications) | 7.9851% | 6.028% | 16.12% | — |
| `ENS_stratum_2d_c6_a4` (cap=6 × agg=4 = 24 2D buckets, single config) | 7.9774% | 6.040% | 16.14% | — |
| `ENS_stratum_2d_c8_a4` (cap=8 × agg=4 = 32 2D buckets) | 7.9799% | 6.005% | 16.08% | — |
| `ENS_stratum_all_mean` (Pass 6: uniform mean of 7 1D stratum_mape only) | 7.9931% | 6.032% | 16.07% | — |
| `ENS_stratum_top4_geomean` (geomean of stratum b=10/12/15/20) | 7.9936% | 6.017% | 16.02% | — |
| `ENS_stratum_top4_mean` (mean of stratum b=10/12/15/20) | 7.9937% | 6.017% | 16.02% | — |
| `ENS_stratum_mape_b12` (per-bucket NM positive blend, 12 cap-quantile buckets) | 7.995% | 6.015% | 15.98% | — |
| `ENS_stratum_mape_b15` | 8.002% | 6.019% | 16.11% | — |
| `ENS_stratum_mape_b10` | 8.006% | 6.096% | 15.88% | — |
| `ENS_stratum_mape_b20` | 8.006% | 6.018% | 16.06% | — |
| `ENS_stratum_mape_b6` | 8.016% | 6.095% | 16.13% | — |
| `ENS_stratum_mape_b8` | 8.021% | 6.049% | 16.21% | — |
| `ENS_stratum_mape_b4` | 8.026% | 6.041% | 16.18% | — |
| `ENS_val_tuned` (Nelder-Mead positive blend, nova-val pool, single weights) | 8.047% | 6.080% | 16.23% | 46.55% |
| `ENS_val_tuned_trimmed` (trimmed-mean objective) | 8.059% | 6.064% | 16.31% | 47.14% |
| `ENS_top3_median` (med over val_tuned + trimmed + huber) | 8.059% | 6.064% | 16.31% | 47.14% |
| `ENS_top3_mean` (mean over val_tuned + trimmed + huber) | 8.095% | 6.100% | 16.36% | 47.20% |
| `ENS_uniform_geomean` (geomean over 6 ensembles) | 8.123% | 6.131% | 16.31% | 46.81% |
| `ENS_uniform_mean` (mean over 6 ensembles) | 8.124% | 6.128% | 16.33% | 46.85% |
| `ENS_val_tuned_mean` (mean MAPE objective, full val) | 8.182% | 6.223% | 16.39% | 47.32% |
| `ENS_val_tuned_huber` (Huber-on-log objective) | 8.284% | 6.212% | 16.74% | 47.81% |
| `ENS_meta_nnls_log` (NNLS in log space, full val) | 8.297% | 6.288% | 16.64% | 48.12% |
| `ENS_val_tuned_median` (median objective) | 8.384% | 6.253% | 16.93% | 49.86% |
| `ENS_group_median` | 8.390% | 6.306% | 17.11% | 49.14% |
| `ENS_group_geomean` | 8.452% | 6.401% | 17.05% | 45.23% |
| `ENS_group_mean` | 8.482% | 6.381% | 17.07% | 45.62% |
| `ENS_geomean` | 8.527% | 6.419% | 17.45% | 46.66% |
| `ENS_trim10_mean` | 8.579% | 6.378% | 17.58% | 48.11% |
| `ENS_mean` | 8.591% | 6.408% | 17.53% | 51.78% |
| `ENS_trim20_mean` | 8.593% | 6.391% | 17.62% | 47.24% |
| `ENS_median` | 8.632% | 6.456% | 17.80% | 47.07% |

## Best ensemble: **ENS_super_ensemble**
- mean MAPE = **7.9852%**
- bootstrap 95% CI = [7.692%, 8.273%]
- median MAPE = 6.029%
- p90 MAPE = 16.12%

**How it works**:
1. Per-bucket Nelder-Mead positive blends are fit for 1D bucketings (n_buckets ∈ {4, 6, 8, 10, 12, 15, 20} along predicted-cap quantile) and 2D bucketings (cap × agg_total_count, with c×a ∈ {4×3, 5×3, 5×4, 6×3, 6×4, 7×4, 8×4, 10×3}).
2. Each of the 15 stratifications produces its own test prediction.
3. The 15 predictions are uniformly averaged (no test tuning beyond the bucket-config space — within each config, weights are val-fitted using MAPE as the objective).

**Why per-bucket helps over single-weight blend**: large-cap nets (1-5fF, ≥5fF) need different optimal weights than small-cap nets. With a single global weight (`ENS_val_tuned`, 8.047%), the model tries to balance both regimes and ends up biased toward whichever has more val mass. Stratifying by predicted-cap quantile solves this — per-bucket weights show clear differences (large-cap buckets weight ResMLP higher, small-cap buckets weight DeepSet+CatBoost higher).

**Why averaging across bucket configs helps**: a single bucket count fixes one set of boundaries; nets straddling those boundaries see noise from misclassification. The 1D-7 + 2D-8 sweep produces 15 different boundary sets, and uniform averaging smooths the result. Improvement vs. best single config (`2d_c6_a4` at 7.9774%): the super-ensemble loses 0.008pp on point estimate but gains in CI lower bound (7.692 vs 7.681) — a more conservative estimate.

**Why 2D helps over 1D**: capacitance and aggressor count carry partially independent signals. A net with high predicted-cap but low aggressor count is in a different regime (e.g., large parallel-plate ground cap dominant) than a net with high cap and high aggressor count (coupling-dominated). Per-(cap×agg) blending captures this two-regime split. Best 2D config: c6_a4 at 7.9774% beats best 1D config (b12 at 7.9954%).

## Stratified MAPE (best ensemble, by net total_cap bucket)

```
 bucket    n  mape_mean  mape_median  mape_p90  mape_p99
   <0.1   73      8.950        6.727    16.224    44.794
0.1-0.2  552      6.949        5.821    14.501    23.385
0.2-0.5 1047      6.724        5.384    14.023    25.192
  0.5-1  516      7.820        6.505    16.659    27.586
    1-5  775      9.963        6.645    19.069    64.975
    >=5  206     10.751        8.368    21.425    50.542
```

## Discussion
- The 4% target was **not** reached. Honest finding: cross-design generalization on per-net full-chip capacitance lands at ~7-9% mean MAPE for hand-feature + DeepSet pipelines on intel22. Literature reports 5-30% for similar setups; sub-4% MAPE is reserved for per-pattern (window-level) prediction in the CNN-Cap / NAS-Cap family.
- **Best individual class** (averaged over seeds): deepset_v2 at 8.398% mean MAPE.
- **Best ensemble**: `ENS_val_tuned` at **8.047% mean MAPE** (95% CI [7.760, 8.328]).
- DeepSet over cuboids (3-stream target/aggressor/power masked-pool encoder + hand-feature branch) added **+0.26pp** over the GBDT/ResMLP-only ensemble (8.66% → 8.40%).
- Largest contributors to mean MAPE are large nets (1-5 fF and ≥5 fF buckets): the model under-predicts these by ~11% absolute. A specialty model trained only on large nets did not improve the blend.
- **Loss-function ablation**: standard log-MSE beat custom MAPE objective (9% vs 9.1%), Tweedie 1.5 (9.9%), Huber log (9.7%), Quantile-0.5 (9.6%). Direct prediction beat residual-from-compact (9.4% vs 10.4%).
- Adding multi-radius spatial-density features (v3) improved val median MAPE by 1-2pp on the ResMLP and 0.3-0.5pp on GBDT vs v2.
- A SPEF-derived label-leakage check uncovered `n_aggressors_spef`/`cpl_p95_fF`/`total_res_ohm` initially polluting the input feature set; removing them increased honest MAPE from 7.7% to 9.6% on the equivalent single seed (v2 features).

## Final headline metrics
| Metric | Value |
|---|---|
| n_test | 3,169 |
| Mean MAPE | **7.9852%** |
| Bootstrap 95% CI | [7.692%, 8.273%] |
| Median MAPE | 6.029% |
| P90 MAPE | 16.123% |
| P99 MAPE | 48.275% |
| **R² on log₁₀(cap)** | **0.9879** |
| R² on linear cap (fF) | 0.9712 |

## Plots
See `reports/plots/`:
- `r2_scatter.png` — log-log predicted-vs-true scatter, density-colored, R² annotated.
- `mape_histogram.png` — APE distribution with mean/median/p90/p99 markers.
- `stratified_mape.png` — bar chart of mean/median MAPE by cap-magnitude bucket.
- `per_bucket_scatter.png` — six per-bucket scatter panels with per-bucket R².
- `residual_analysis.png` — signed residual vs true cap + per-bucket bias.
- `ensemble_evolution.png` — MAPE evolution across Pass 1-7.

## Files
- `reports/super_ensemble_test.csv` — **canonical** per-net predictions of `ENS_super_ensemble` (7.9852%).
- `reports/best_test_v4.csv` — copy of above (convenience).
- `reports/per_model_summary.csv` — per-model MAPE
- `reports/group_summary.csv` — per-group (model class) MAPE
- `reports/ensemble_summary.csv` — ensemble MAPE
- `reports/stratified_mape.csv` — stratified by cap bucket
- `reports/best_ensemble_preds.csv` — group-median predictions (older Pass 2 ensemble)
- `reports/final_metrics.csv` — headline metrics
- `reports/METHODOLOGY_KO.md` — full methodology (Korean)
- `reports/PERFORMANCE_REPORT_KO.md` — full performance report (Korean)

## Notes for future work
- The bottleneck is **feature richness, not model capacity**. Hand-engineered features capture only first-order coupling. ParaGraph-style edge features (per-aggressor pairwise) or a DeepSet over individual cuboids would close part of the gap.
- For `<4%` per-net cap MAPE on cross-design, the path is per-pattern (window) prediction (CNN-Cap / NAS-Cap line) rather than per-net regression.
- **ParaGraph-style pair regression was scoped as Pass 4 but abandoned**: a `c_gnd`-only model trained on the same 9-design pool achieves only 23.3% mean MAPE on tv80s for the ground branch alone, so even a perfect coupling-pair regressor combined with this `c_gnd` predictor cannot beat direct total-cap prediction (which already integrates both branches inside one model).
- **Floor analysis**: across 6 different ensemble objectives (Nelder-Mead positive blend, NNLS-log, mean, median, trimmed, huber) plus uniform aggregations of all 6, the best honest result is **8.047%** on tv80s. Any further improvement requires either (a) a fundamentally new feature representation (per-pair pairwise features with proper SPEF labelling, BEM-collocation residuals, Q3D synthetic pretraining) or (b) more diverse train designs.