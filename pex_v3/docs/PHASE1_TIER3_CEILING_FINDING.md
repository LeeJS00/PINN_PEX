# Phase 1 Tier 3 — Hand-Feature Ceiling Identified

_Date: 2026-05-02_
_Status: paper-grade finding; Phase 1 plan revised (4th time this session)_

## TL;DR

**Hand-engineered NetFeatureVector (42 dims) + any flexible ML model
(tree boosting OR deep MLP) hit an identical ~4.66% total MAPE ceiling**
on real v3 BEOL data. Per-channel: ~21% gnd, ~12.6% cpl — also identical
across architectures. The 4.66% headline benefits from gnd/cpl
cancellation that BOTH XGBoost and unbounded MLP independently learn.

The hybrid bounded-residual paradigm UNDERPERFORMS this ceiling (~7-10%)
because the analytic prior (Sakurai-Tamaru placeholder) is poorly
calibrated and the multiplier bound conflicts with the necessary
correction range. **Phase 1 paradigm to break the ceiling requires
per-cuboid mesh features**, not better residual design on hand features.

## Three architectures, same data, same eval

```
Eval set: v3 valid 12,594 in-dist nets, single seed (n=1).

Method                         params   gnd_med   cpl_med   total_med
B1 XGBoost                      ~100K   20.6%     12.4%     4.66%
Option F deep MLP (this turn)   286K    21.0%     12.6%     4.66%   ← matches!
Hybrid_v3 bounded clamp=2.5     11K     21.4%     13.8%     9.54%   ← UNDERPERFORMS
Hybrid_v3 bounded clamp=20      11K     20.9%     14.7%     7.19% best
B3 PINN legacy                  ~1M     —         —         30.90%  ← far worse
```

## Why this is a feature ceiling, not architecture ceiling

Two completely different model classes hit identical numbers:
- **XGBoost**: tree-additive, ~100K leaves, deterministic given data
- **Deep MLP (Option F)**: 256×3 hidden, 286K params, AdamW + cosine LR

Both reached **4.66% total / 21% gnd / 12.6% cpl** in <60 seconds of
training. The convergence to identical numbers from independent
architectures says the bottleneck is the **information content of the
42-dim NetFeatureVector**, not how a model uses it.

## Per-channel cancellation is intrinsic to hand features

Both unbounded models exhibit the same pattern:
- gnd over-predicts at scale ~21%
- cpl under-predicts at scale ~12%
- Total error ~5% via partial cancellation

This is the model's optimal strategy given the features. **The
cancellation isn't architecture-specific** — it emerges anywhere a
flexible regressor is given hand features that confuse gnd vs cpl
allocation.

## Hybrid_v3 underperforms because of analytic-prior×clamp interaction

```
Median(c_gnd / compact_gnd) = 0.35      → analytic 3× under-estimate
P5/P95 ratio gnd = 0.18 / 0.94          → 5× spread

Required multiplier range for fit: [0.10, 6×]
clamp=log(1.5)=±50%      → too tight, saturates at 26.62%
clamp=log(20)=±2000%     → unstable training, best 7.19%
clamp=log(2.5) + calibrated → 7.72% best (this Tier 3)
```

The bounded-residual paradigm was designed assuming analytic ≈ truth
(small correction needed). With Sakurai-Tamaru placeholder, analytic is
3-5× off, requiring corrections beyond the clamp range. Calibrating the
median fixes the bias but spread remains, so the model still loses to
Option F's free MLP.

## Implications for paper

### Phase 1 paradigm contribution must shift

**Was** (per Phase 1 spec): "Hybrid analytic + bounded neural residual
beats XGBoost via physics-informed prior."

**After data**: That story doesn't hold. Bounded paradigm with current
analytic prior **underperforms** XGBoost. Per-channel ceiling is
intrinsic to features, not architecture.

**Revised Phase 1 contribution narrative**:

> "We identify a per-net hand-feature ceiling (4.66% total / 21% gnd /
> 12.6% cpl) shared by XGBoost and deep MLP. Per-channel β-strategy
> exposes that the headline 4.66% benefits from gnd/cpl cancellation.
> To break this ceiling AND deliver per-channel honesty, Phase 1
> paradigm uses per-cuboid mesh features (`mesh_v3`) with per-pair
> attention; the hand-feature ceiling becomes the empirical anchor
> against which mesh contribution is measured."

### Two-paper plan strengthened

**Paper #1A (methodology)** — paper-grade content already in hand:
- 32.89pp data-fix gain (H1+H3 alone)
- B1 XGBoost stratified analysis showing cancellation
- **NEW: hand-feature ceiling demonstration** (XGBoost = MLP = 4.66%)
- Ablations showing what data fixes contribute

**Paper #1B (paradigm)** — Phase 1.5 work:
- Mesh_v3 + per-cuboid hybrid arch
- Demonstrate breaking the 4.66% ceiling AND per-channel <8%
- Compare to all hand-feature baselines as anchor

## Updated risk register

R1 was "Phase 1 hybrid can't beat XGBoost per-channel" — **VALIDATED
HIGH-RISK**: bounded paradigm on hand features cannot beat XGBoost. Mesh
paradigm (Phase 1.5) is required.

R-NEW: Mesh_v3 implementation cost (~6 days per A6 spec) — but the
paper #1B story now requires it. Without mesh, paper #1B has no clear
contribution above paper #1A.

## Phase 1 plan — 4th revision (this session)

```
Tier 0: analytic_base + residual + hybrid                          ✅ DONE
Tier 1: synthetic pretrain + K3 canary                             ✅ K3 fired → DROPPED
Tier 2: real-BEOL fine-tune (bounded)                              ✅ DONE (β-FAIL, 7.7% best)
Tier 3: NNLS calibration                                           ✅ DONE (no breakthrough; ~7.7%)
Tier 4 (NEW): Option F deep MLP on hand features                   ✅ DONE this turn (4.66% — matches XGBoost = ceiling)
Tier 5 (NEW): Mesh_v3 per-cuboid features + paradigm-shift hybrid  ⏳ next critical sprint
```

## Concrete next steps

### Option A — Ship paper #1A first, defer paper #1B

1. Run 5-seed Option F (big MLP) for proper variance estimate
2. Run 5-seed B4 Compact+GAM (Sakurai+GAM physics anchor)
3. Run 5-seed B2 ParaGraph capped reproduction
4. Build paper #1A around: data fix + cancellation + ceiling
5. Submit to ICCAD/DAC
6. Paper #1B (mesh paradigm) is follow-up work

Estimated time-to-paper #1A: ~2 weeks. Strong methodology contribution.

### Option B — Pivot to mesh_v3 for paper #1B

1. Implement mesh_v3 per A6 spec (~6 days; 24h MVP feasible)
2. Per-cuboid hybrid arch: encoder + per-pair attention + bounded residual
3. Train on real v3 (not synthetic — K3 already showed synthetic is no-op)
4. Target: break <4% total AND <8% per-channel
5. Risk: per-cuboid model might also hit a different ceiling

Estimated time-to-paper #1B: ~6-8 weeks. High-risk high-reward.

### Recommended: A then B

Paper #1A is paper-grade in 2 weeks; paper #1B is contingent. Reduces
risk of empty-handed outcome. A2 (classical-baseline-owner) and A1
(benchmarking-statistician) would both endorse this sequencing.

## Files

- `pex_v3/output/phase1_tier3_calibrated/` — Tier 3 calibrated bounded run
- This doc: `pex_v3/docs/PHASE1_TIER3_CEILING_FINDING.md`
- Memory: `project_phase1_tier3_ceiling_finding.md` (next)
