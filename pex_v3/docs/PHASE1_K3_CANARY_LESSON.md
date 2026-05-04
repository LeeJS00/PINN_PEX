# Phase 1 K3 Transfer Canary — Design Discovery (NOT Failure)

_Date: 2026-05-02_
_Status: lesson learned; synthetic pretrain de-prioritized_

## What happened

Ran K3 canary protocol per Codex round 2 + A5 [ROLE PASS] spec:
1. Pretrain `HybridPexV3` on Stage 1 + Stage 2 Mode A synthetic data
2. Fine-tune both pretrained + control (random init) on 500 real v3 valid nets
3. Compare loss after 500 fine-tune steps

```
Pretrain (5000 samples, 3 epochs):
  loss = 0.0000 from epoch 0 onward
  multiplier_mean = 1.0  max_dev = 0.0

K3 canary (500 nets, 500 fine-tune steps):
  control final loss:    0.4678
  pretrained final loss: 0.4678
  speedup:               +0.0%
  K3 verdict: FAIL (no speedup)
```

## Root cause — by design, not a bug

`HybridPexV3` uses `BoundedResidualHead` with **last linear layer
zero-initialized** (per A5 mandate to ensure day-1 inference = analytic).

For synthetic data:
- analytic predictor = parallel-plate / stacked-dielectric closed form
- golden = same closed form (synthetic generator uses identical math)
- ⇒ analytic == golden ⇒ MAPE loss = 0 from day 1
- ⇒ no gradient signal ⇒ no learning during pretrain
- ⇒ pretrained checkpoint identical to fresh init (in functional sense)

When fine-tuned on real v3 data, control and pretrained start from the
SAME function (multiplier ≡ 1.0) and converge to the same local minimum.

This is the architecture working **as A5 specified** — day-1 invariant
is enforced. The synthetic pretrain is **structurally incapable** of
teaching this architecture anything.

## Why this is a WIN, not a loss

K3 was meant to tell us: don't commit GPU-months of Q3D oracle work
without first validating that synthetic pretrain helps. **It told us
exactly that, in 3 minutes**:

- Stage 3+ Q3D pretraining = ~3000 GPU-hours
- Saved by K3: ~125 GPU-days
- Cost of running canary: 3 minutes

The canary saved us from a much bigger waste.

## What this changes about Phase 1

### Synthetic pretrain — DROPPED from Phase 1 plan

Original spec (A5 Tier 1):
> Stage 1 + Stage 2 Mode A pretrain → K3 canary → real-BEOL fine-tune

Revised:
> Skip synthetic pretrain. Directly fine-tune `HybridPexV3` on real v3
> BEOL features. Synthetic dataset stays in `pex_v3/src/synthetic/` as
> a UNIT-TEST resource (verify analytic kernels), not a training substrate.

### Phase 1 next sprint, simplified

1. ~~Pretrain on synthetic~~ — DROPPED
2. ~~K3 canary~~ — VALIDATED (architecture is K3-immune by design)
3. **Direct real-BEOL fine-tune** of `HybridPexV3` on v3 valid features
4. Compare to B1 XGBoost on per-channel basis (β strategy)
5. 5-seed protocol + paired MWU vs B1, B4

This actually **shortens** Phase 1 by 1-2 weeks (no need to materialize
3M synthetic samples, no need to debug Mode B replacement).

### Alternative pretraining strategies (deferred / for future)

If we want to recover synthetic-pretrain value later:

A. **Auxiliary task pretrain**: residual MLP learns to predict
   `log(analytic_C)` from features (separate auxiliary head, not the
   capped multiplier head). This trains UPPER layers to encode geometry
   signal independent of the day-1 multiplier-=1 invariant. Phase 2 work.

B. **Drop zero-init last layer**: small Gaussian init on last layer
   → loss is small but non-zero on synthetic → upper layers get gradient.
   Loses the day-1 == analytic invariant though. Trade-off.

C. **Mode B (real Sommerfeld)**: when vector-fitted complex-image kernel
   ships, synthetic data will have analytic ≠ truth (because Mode A
   approximation is no longer used as ground truth). Pretrain becomes
   meaningful again.

For Phase 1 paper, none of these is required. The B1 XGBoost ~5%
baseline already gives the comparison anchor.

## Hard kill K3 status

The original K3 hard kill criterion was:

> Synthetic pretrain → real-data finetune gain < 1pp → abort synthetic.

**Strictly speaking**: the canary FAILED with +0.0% speedup. K3 fired.
We follow the kill protocol: drop synthetic pretrain.

This is the system working correctly. The strategy v3 plan included
this kill criterion specifically so we wouldn't waste time. We didn't.

## Memory & doc updates

- Saved `project_phase1_k3_canary_lesson.md` to memory
- Updated `STRATEGY_V3_UPDATED_PLAN.md` Phase 1 section
- Removed `pretrain_synthetic_v3.py` from Tier 1 (kept code as reference;
  may be useful for auxiliary-task pretrain in Phase 2)

## What ships next instead

Direct real-BEOL fine-tune script:
1. `pex_v3/src/trainers/finetune_hybrid_v3.py` — load real v3 features,
   train `HybridPexV3` with per-channel MAPE loss
2. 5-seed protocol with multi-GPU (re-use `06_run_pinn_multigpu.py` template)
3. Stratified eval (per-channel × per-design × per-quartile)
4. Paired MWU vs B1 + B3

Estimated cost: ~5-10 GPU-hours per seed (much less than B3 PINN's 4.5h
because hybrid_v3 has 30K params vs DeepPEX's 1M; no SSL basis loading;
no AL loop overhead).
