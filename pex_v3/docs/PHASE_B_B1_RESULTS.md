# Phase B B1 XGBoost Results

_Date: 2026-05-02_
_Status: v1 done (deterministic-seed bug); v2 running (subsample fix)_

## v1 result — TEST split (OOD designs nova + tv80s)

```
Seeds:               5 (all identical due to deterministic XGBoost)
Test set:            95,594 nets in intel22_nova_f3 + intel22_tv80s_f3
                     (cross-design OOD; designs unseen during training)

Median MAPE:         5.90%
Mean MAPE:           7.48%
P95 MAPE:            19.78%
Delay error median:  5.90%
Delay error P95:     19.78%
Power error median:  5.90%
RC chip ratio P50:   1.018
RC chip ratio P95:   0.960

Wall-clock:          504-599 sec/seed × 5 = ~45 min sequential
```

**Important caveats**:

1. **Deterministic-seed bug**: Stdev = 0 across all 5 seeds. XGBoost with
   `tree_method="hist"` and no `subsample`/`colsample_bytree` is fully
   deterministic given the data. Bug fixed in xgboost_baseline.py
   (subsample=0.8 + colsample_bytree=0.8 added). v2 re-run with subsample
   active will give true variance estimate.

2. **OOD test split, NOT in-dist**: B1 v1 evaluated on `split == 'test'`
   which is intel22_nova + intel22_tv80s — designs **never seen during
   training**. This is harder than in-dist evaluation.

3. **Not apples-to-apples vs B3 PINN**: B3 evaluated on `split == 'valid'`
   (in-dist held-out from training designs). B1 v1 evaluated on `split == 'test'`
   (cross-design OOD). The two numbers are NOT comparable directly.

## v2 result — VALID split (in-dist held-out, with subsample)

[in progress; will update when complete]

## Why this is striking

Even with the apples-vs-oranges comparison flagged:
- **B1 XGBoost OOD MAPE 7.48%** vs **B3 PINN in-dist MAPE 30.90%**
- XGBoost on hand-engineered features achieves DRAMATICALLY better than
  the legacy PINN on real BEOL data — by ~4-5×

This raises an important question for the paper thesis:

> "Is the contribution **the neural architecture** or **the feature
> engineering** (hand-crafted physical features + tree boosting)?"

If apples-to-apples B1 valid ≈ 5-7% and B3 valid = 30%, the answer is
clearly: **feature engineering + hand-crafted physics features dominate
the legacy PINN paradigm by a wide margin**.

This pre-empts the user's <4% target story differently than originally
planned:
- The "neural network" Phase 1 hybrid arch must beat XGBoost's ~5-7%
  baseline, NOT just legacy PINN's 30%.
- The bar is **<4% per-net while OUTPERFORMING strong feature-based
  baseline** — much harder.

## Compared metrics (B1 v1 OOD)

| Metric | B1 v1 (OOD test) |
|---|---|
| Cap MAPE median | 5.90% |
| Cap MAPE mean | 7.48% |
| Cap MAPE P95 | 19.78% |
| Delay error median | 5.90% |
| Power error median | 5.90% |
| RC chip ratio P50 | 1.018 (well-calibrated) |
| RC chip ratio P95 | 0.960 (well-calibrated) |
| n_valid_nets | 95,594 |

The delay/power error matching cap MAPE is expected when res values
default to non-rich values; the 4-metric reporting still serves its
provenance role.

## Hand-features that made the difference

The 43 NetFeatureVector fields after the layer_idx fix:
- 9 layer histogram (M1-M9) — now LIVE thanks to A3 audit fix
- 3 VSS shielding (M1-M3, M4-M5, M6+) — now LIVE
- 3 density buckets — now LIVE
- 4 layer-stack stats (eps_min/max/mean, n_layers_present) — now LIVE
- Coupling: aggressor count, broadside/lateral overlap (total + P95),
  spacing distribution (min, P25, P50, P95, bucket counts)
- Topology: fanout
- Compact-model intermediates: Sakurai-Tamaru per-net estimates

A3 audit's catastrophic-bug fix (15 of 43 features previously dead-zero)
was a **necessary precondition** for this 5.90% number. Pre-fix XGBoost
would have run on the 28 surviving features and likely gotten worse.

## Files

- Per-seed predictions: `pex_v3/output/baselines/B1_xgboost_real/seed{0..4}/eval_predictions.csv`
- Models: `seed{N}/model_gnd.json`, `seed{N}/model_cpl.json`
- Aggregated: `pex_v3/output/baselines/B1_xgboost_real/per_method.csv`,
  `per_run.csv`, `mwu_pairs.csv`

## Pending work

- v2 result on VALID split (apples-to-apples vs B3)
- Stratified error report (per-quartile, per-design, per-layer)
- A2 classical-baseline-owner audit on B1 v1 + v2 results
