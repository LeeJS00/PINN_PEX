# C1 — CTS Mode B Isotonic Post-Correction (DESIGN)

_Auto-optimize sweep variant — capacity-zero output post-process_

## Goal

Reduce the giant-CTS Mode B `gnd` MAPE outliers (Top-50 median ~270%) without
modifying the trained baseline (`HybridPexV3Mesh` 5-seed) and without touching
any non-CTS quartile (Q1..Q3 must stay flat or improve).

## Anti-leak strategy (non-negotiable)

| Phase | Data | Operation | Use of test data |
|---|---|---|---|
| Fit | `eval_logger_valid.parquet` (12,594 nets) | Fit `IsotonicRegression(pred → gold)` per seed | **Never** |
| Apply | `eval_logger_test.parquet` (95,594 nets) | Predict corrected `gnd_pred` per seed | Read-only |
| Aggregate | per-seed corrected predictions | D2 across-seed stats | Standard |

Per-seed isolation: each baseline seed has its own held-out val + test predictions.
We fit a separate isotonic per seed and apply per seed; results are then
aggregated via `aggregate_ablation_summary.py`. **No cross-seed leakage**, no
test data ever touches the fit.

## Mode B identification (data-driven choice)

Before designing the corrector we re-checked which axis identifies the
giant-CTS Mode B:

- **Original CGND_ERROR_ANALYSIS.md** Top-50 (5-seed ensemble, by `gnd_rel_err`):
  median compact_gnd 1.41 fF, median bbox 56 µm², median fanout 92,
  median pred/gold ratio **3.7×** (model OVER-predicts).
- **Top-1pct compact_gnd_estimate_fF on test** (n=956): median `gnd_rel_err`
  10.7%, ratio 0.95 — these are NOT the Mode B failures. They're well-predicted
  giant nets.
- **Top-50 by `gnd_rel_err`** caught by top-1pct compact_gnd: **0 / 50**.
- **Top-50 by `gnd_rel_err`** caught by top-1pct gnd_pred: **0 / 50**.
- **Top-50 by `gnd_rel_err`** caught by top-5pct compact_gnd: **0 / 50**.

Conclusion: there is **no single 1D feature axis** that pre-identifies the test
Top-50 outliers. We therefore drop the "Mode B-only" idea (C1a) — it cannot
have any effect because the corrector never sees the failing nets. We
substitute with a **full-distribution isotonic** (C1b) which corrects the
systematic bias visible across all deciles, then check the kill criterion
on the gnd-quartile axis.

## Two variants evaluated

### C1a — Mode B-only isotonic (original spec; KEPT for completeness)
- Fit on val nets where `compact_gnd_estimate_fF` ≥ val 99th percentile
  (~126 nets per seed).
- Apply only to test nets where `compact_gnd_estimate_fF` ≥ same
  threshold (~683-956 nets per seed).
- Expected outcome from prototype: identity for ~99% of test, near-zero
  Top-50 effect (Top-50 outliers are not in this set).

### C1b — Full-distribution log-space isotonic (operative variant)
- Fit `IsotonicRegression(out_of_bounds='clip')` on
  `(log gnd_pred, log gnd_gold)` over the **entire** val set (12,594 nets).
- Apply via `corrected = exp(IR.predict(log gnd_pred))` to the **entire**
  test set.
- Log-space chosen because the bias is multiplicative (val multipliers
  ~1.05-1.34× across deciles, see prototype log).
- Mass-conservation: total cap is preserved on val (regression is least-squares
  monotone; bias-correcting increases each decile mean by the val
  multiplier).

Both variants emit identical schema; aggregator + stratifier handle either.

## Feature space justification (1D)

Codex constraint: 1D for transparency and to avoid overfitting on a 12,594-net
val set. Choices considered:

| Axis | Justification | Verdict |
|---|---|---|
| `gnd_pred` | Direct calibration target; model output already encodes geometry | **CHOSEN** for C1b |
| `compact_gnd_estimate_fF` | Strong analytic prior; available pre-inference | **CHOSEN** for C1a |
| `fanout` | Discrete; high-fanout val outliers exist | Rejected — coarse |
| `bbox_xy_um2` | Continuous; correlated with cap | Rejected — redundant with compact_gnd |

C1b uses `gnd_pred` because:
1. It is a continuous, dense axis (no zero-bin issue).
2. Isotonic on `(pred → gold)` is the textbook recalibration target.
3. Log-space handles the 4-orders-of-magnitude range without bin sparsity.

C1a uses `compact_gnd_estimate_fF` because Mode B was originally defined
on this analytic axis.

## Decision gate (Codex revised criterion)

C1 PASSES if **all** of:
1. **Mode B (Top-50 gnd_rel_err on test) reduced by ≥ 25% relative**
   (e.g., 269% → ≤ 201%).
2. **Q1/Q2/Q3 gnd MAPE NOT regressed by > 0.3pp absolute**.
3. **Test total MAPE not regressed by > 0.2pp absolute**.

If C1 PASSES → eligible for stacking on InputSubset survivors.
If C1 FAILS → report negative finding, document why, no further C1 action.

## Per-seed protocol

For seed s ∈ {0, 1, 2, 3, 4}:
1. Load `pex_v3/output/phase1_mesh_5seed/seed{s}/eval_logger_valid.parquet`.
2. Load `pex_v3/output/phase1_mesh_5seed/seed{s}/eval_logger_test.parquet`.
3. Fit `IsotonicRegression(out_of_bounds='clip')` on val (variant-specific
   data slice + transform).
4. Apply to test → `gnd_pred_corrected`.
5. Recompute `total_pred = gnd_pred_corrected + cpl_pred` and write
   `outputs/c1_cts_isotonic/{variant}/seed{s}/eval_logger_test.parquet` with
   the **same schema** as baseline (so `aggregate_ablation_summary.py` and
   `stratify_eval.py` work unmodified).
6. Also copy through `eval_logger_valid.parquet` (val unchanged) and write
   a `summary.json` mimicking baseline's schema (so D2 picks up
   final_test/final_valid).

## Output layout

```
pex_v3/experiments/auto_optimize_2026_05_03/
  variants/c1_cts_isotonic/
    DESIGN.md                       — this file
    c1_isotonic.py                  — implementation
    smoke_seed0.json                — single-seed smoke result
  outputs/c1_cts_isotonic/
    c1a_modeB/                      — Mode B-only variant
      seed{0..4}/eval_logger_*.parquet, summary.json
      stratified/                   — per-seed not used; ensemble-level only
    c1b_full/                       — full-distribution variant
      seed{0..4}/eval_logger_*.parquet, summary.json
      stratified/
  reports/
    c1_cts_isotonic_summary.json     — D2 aggregator (5-seed)
    c1_cts_isotonic_stratified.json  — D3 stratified MAPE
```

## Stacking caveat (PLAN.md Codex Round 2)

If C1 PASSES on baseline output, the isotonic is fit on baseline val. If
later stacked on InputSubset, the C1 isotonic MUST be REFIT on InputSubset's
own val output — never transfer baseline-fit isotonic to a different model.
This is enforced by per-seed re-fit at apply time.

## Risk acknowledgments

1. **Top-50 distribution shift** (val high-fanout vs test low-fanout
   over-predict): isotonic in log-space cannot fix this if the over-prediction
   regime is not present on val. Documented in prototype output.
2. **Multiplicative bias** dominates (~1.20× expand globally), so a global
   isotonic improves bulk gnd median by ~1.5-2pp at risk of harming Top-50.
   Decision gate explicitly checks the Top-50 outlier metric to catch this.
3. **Per-seed fit yields per-seed correctors**, so 5-seed std should remain
   comparable to baseline std.
