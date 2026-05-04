# Phase 1 Tier 2 — First Real-BEOL Result + Diagnosis

_Date: 2026-05-02_
_Status: single-seed result, β-FAIL but architecture validated_

## TL;DR

Phase 1 hybrid_v3 with bounded residual fine-tuned on real v3 valid:
**best epoch 19 reached total MAPE 7.19%** — **4.3× better than legacy PINN
(30.90%) but still 1.5× worse than XGBoost (4.66%)**. β-strategy gate
(gnd<8%, cpl<8%, total<4%) NOT met. Root cause: `compact_gnd_estimate_fF`
analytic prior is poorly calibrated (median ratio 0.35 — 3× under-estimate),
forcing the residual multiplier into an unstable wide range.

## Quantitative

```
Day-1 (zero-init residual = analytic only):
  gnd 65.80%  cpl 81.09%  total 38.31%

clamp=log(1.5) (multiplier ∈ [0.67, 1.5]) — original A5 spec:
  After 8 epochs (early stop): total 26.62%  ← saturated at clamp limit

clamp=log(20) (multiplier ∈ [0.05, 20]) — looser:
  Best epoch 19:        total 7.19%  ← β-FAIL but ↓4.3× vs legacy PINN
  Final epoch 59:       total 10.77%
  Test (OOD nova+tv80s): total 11.79%

Comparison anchors:
  B3 PINN (legacy):     31.08%
  Hybrid_v3 (best):     7.19%   (4.3× better than B3)
  B1 XGBoost:            4.66%   (1.5× better than hybrid)
```

## Why hybrid is better than B3 PINN but worse than XGBoost

| | B3 PINN | Hybrid_v3 | B1 XGBoost |
|---|---|---|---|
| Architecture | DeepPEX (legacy 1M params, frozen encoder) | hybrid analytic + residual (11K params) | tree boosting (~100K leaves) |
| Per-channel separation | implicit (KCL closure) | explicit (β-strategy, 2 heads) | implicit (one regressor per channel) |
| Inductive bias | physics router on per-cuboid | analytic Sakurai prior + bounded multiplier | none (purely structural) |
| Training data | tile-level (raw cuboids) | per-net (43 hand features) | per-net (43 hand features) |
| Training time | 4.5 h/seed | ~10 min/seed | ~10 min/seed |
| **Total median MAPE** | **30.90%** | **7.19% (best)** | **4.66%** |

**Hybrid wins over B3** because:
- Hand features compress information well (XGBoost shows this)
- Per-channel separation prevents cancellation (β strategy)
- Analytic prior provides scaffolding even if poorly calibrated

**Hybrid loses to XGBoost** because:
- Smaller capacity (11K vs 100K effective params)
- Bounded multiplier limits expressivity when analytic is far off
- 2-layer MLP < deep tree ensemble for non-linear feature interactions

## Diagnostic: analytic prior calibration

```
Analytic / golden ratio on v3 valid (12,594 nets):

compact_gnd_estimate_fF / c_gnd_fF:
  median:  0.347   (analytic UNDER-estimates by ~3×)
  p5/p95:  0.18 / 0.94   (5× spread)
  range:   0.10 - 7.31    (some nets analytic is 7× off)

compact_cpl_estimate_total_fF / c_cpl_total_fF:
  median:  1.810   (analytic OVER-estimates by ~2×)
  p5/p95:  0.77 / 4.38

→ multiplier needed: [0.10, 6×] for gnd; [0.23, 1.30] for cpl
→ clamp=log(1.5)=±50% way too tight
→ clamp=log(20)=±2000% works but creates training instability
```

The `compact_gnd_estimate_fF` was built as a Sakurai-Tamaru placeholder
in `feature_dataset.py:_compact_gnd_estimate_fF` — it uses
`d = abs(layer - 0) * 0.1` as a coarse approximation. Real BEOL stack has
varying per-layer thicknesses + etch-stop layers, so this is at best
an order-of-magnitude estimate.

## Path forward — 4 options ranked

### Option E (recommended): NNLS-calibrate the analytic prior

Re-fit `compact_gnd / compact_cpl` magnitudes via NNLS on v3 train data
so median ratio ≈ 1.0. This makes the bounded residual paradigm (small
multiplier ≤ ±50%) work as designed.

- Effort: ~1 day (port logic from `src/data/_archive/calibration_extractor.py`)
- Expected outcome: hybrid_v3 with clamp=log(1.5) reaches stable <8% per-channel
- Paper narrative: "data-driven calibrated analytic prior + bounded
  residual" — clean physics story

### Option F: Direct MLP regression (no analytic prior)

Skip the bounded multiplier. Train MLP on log(c_gnd) directly. Architecture
becomes "neural XGBoost" — likely matches XGBoost (~5%) but loses the
physics-informed pitch.

- Effort: ~2 days (refactor hybrid_v3)
- Expected outcome: ~5% (same as XGBoost)
- Paper narrative: weak (no clear paradigm contribution)

### Option G: XGBoost as analytic prior

Use XGBoost prediction as the "analytic baseline", residual learns small
corrections.

- Effort: ~1 day (wrap XGBoost as differentiable evaluator at training
  time, or precompute predictions)
- Expected outcome: marginal improvement over XGBoost (~4-4.5%)
- Paper narrative: "boosted hybrid" — compelling if it actually beats
  XGBoost

### Option H: Bigger MLP + AdamW + LR scheduler

Scale residual head to 256→256→256 (~150K params), use cosine LR
schedule, train 200 epochs.

- Effort: ~half day
- Expected outcome: marginal improvement; analytic prior still pollutes
- Paper narrative: same story but bigger model — doesn't help

## Recommendation

**Pursue Option E** as primary next step. Restore the bounded-residual
paradigm by fixing the calibration so multiplier ≈ 1.0 in median.
Expected outcome puts Phase 1 hybrid at ~8% per-channel — closing the
gap to XGBoost while preserving the physics-informed narrative.

Concrete next sub-tasks:
1. Port `calibration_extractor.py` logic to `pex_v3/src/baselines/calibration_v3.py`
2. NNLS-fit per-layer ρ multipliers using v3 train split (no leakage)
3. Recompute `compact_gnd_estimate_fF` with calibrated ρ
4. Re-run Phase 1 Tier 2 fine-tune (target: clamp=log(1.5), stable, <8% per-channel)
5. 5-seed multi-GPU on calibrated path

## Files

- `pex_v3/src/trainers/finetune_hybrid_v3.py`
- `pex_v3/scripts/10_finetune_hybrid_smoke.py`
- `pex_v3/output/phase1_finetune_smoke/{model.pt, summary.json, history.json}`
- This doc: `pex_v3/docs/PHASE1_TIER2_FIRST_RESULT.md`
- Memory: `project_phase1_tier2_first_result.md` (next)
