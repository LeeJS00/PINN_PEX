# InputSubset + ClampNorm — Combined-Stack Variant

_Auto-Optimize Sweep 2026-05-03. Phase 1 candidate "best model"._

## 0. One-line

Stack two architecturally orthogonal levers on top of `HybridPexV3Mesh`:
**InputSubset** (per-channel input zero-masking with a SHARED encoder) at
the encoder INPUT, and **ClampNorm** (joint-norm projection clamp) at the
residual OUTPUT. Same parameter count as the locked baseline (44,738),
same curriculum schedule, day-1 invariant preserved.

## 1. Why this combination

The two single-seed smokes both passed against `HybridPexV3Mesh`:

| Variant       | test gnd | test cpl | test total | last_valid total | Δ vs baseline                |
|---------------|----------|----------|------------|------------------|------------------------------|
| Baseline (5s) | 20.49%   | 15.53%   | 8.27%      | 8.27%            | —                            |
| InputSubset   | 19.05%   | 15.02%   | 7.22%      | n/a              | gnd −1.44 / total −1.05 pp   |
| ClampNorm     | 20.89%   | 15.22%   | 7.36%      | 6.66%            | cpl −0.31 / total −0.91 pp   |

Each lever attacks a different failure mode in the baseline:

- **InputSubset attacks input information:** the gnd encoder no longer
  sees `(semantic_type, is_target, net_type)` (channels 6, 7, 9), which
  for a "self-only" physical quantity (`c_gnd_fF`) are noise. The
  encoder still pools over aggressor cuboids but cannot tell them apart,
  acting as a geometric blur that biases the embedding toward
  geometry-only features. Best evidence: gnd MAPE drops 1.44 pp; the
  largest single-lever lift on the worst per-channel error.
- **ClampNorm attacks output gradient flow:** the joint
  `δ × min(1, C/||(δ_gnd, δ_cpl)||₂)` projection replaces the
  element-wise clamp's 0/1 gradient cliff with a smooth rank-1
  Jacobian. This eliminates the curriculum-transition spikes that
  inflate the "last-step" valid total. Best evidence: last_valid total
  6.66% (vs 8.27% baseline last; 6.18% baseline best) — the
  best-vs-last gap nearly closes, which means the 5-seed lock can ship
  the last checkpoint without losing the cherry-picked best.

The mechanisms compose because they touch **non-overlapping regions of
the forward graph**:

```
INPUT     →  cuboid_in           ┐
                                 │  InputSubset:  gnd path zeros ch{6,7,9}
                                 │                cpl path identity (full)
ENCODER   →  shared CuboidSetEnc ┘
                                 │
HEAD      →  residual MLPs (gnd, cpl)
                                 │  ClampNorm:   joint-norm projection
OUTPUT    →  multiplier (exp δ_eff)             on (δ_gnd, δ_cpl) per net
```

InputSubset only changes the encoder's INPUT tensor; ClampNorm only
changes the SHAPE of the residual saturation. There is no shared
parameter modified by both; they cannot trivially conflict.

## 2. What does NOT compose (the A1-trap to avoid)

The combination "InputSubset + per-channel SEPARATE encoders" is
**explicitly forbidden**. That re-creates A1's failure mode (per-channel
encoder duplicates the cuboid encoder, +9.0K params, test gnd 21.60%,
+1.11pp regression). InputSubset's value statement is "different INPUT,
SHARED weights" — coupling the encoder weights is what makes capacity
stay flat. Combining InputSubset with a per-channel encoder swap would
add capacity and re-trigger Phase-2 overfit.

The combination "ClampNorm + per-pair clamp head" (B1) is also
out-of-scope here: B1 changes the cpl analytic prior, which would change
the residual scale and require a re-tuned cap. Either lever is
investigated independently before composition.

## 3. Day-1 invariant

Both InputSubset and ClampNorm preserve day-1 individually. The combined
forward path:

```
δ_gnd, δ_cpl  =  residual_mlp(encoder(masked_input)).squeeze(-1)
              =  0  (zero-init last linear)        ← unchanged from each single
n             =  sqrt(0² + 0² + ε²)  =  ε
s             =  min(1, C / ε)        =  1         ← clamp-norm identity at zero
δ_eff         =  s · δ                =  0
mul           =  exp(0)               =  1
pred          =  analytic · 1         =  analytic  ✓
```

Verified by `test_day1_analytic` (atol = 1e-5).

## 4. Param budget

| Submodule          | Baseline / IS / CN / Combined | Notes                    |
|--------------------|-------------------------------|--------------------------|
| `cuboid_encoder`   | 9,024                         | shared, identical        |
| `gnd_residual`     | 17,601                        | unchanged                |
| `cpl_residual`     | 18,113                        | unchanged                |
| **Total trainable**| **44,738**                    | exact match required     |
| Buffers (extra)    | 2 × (1,1,10) float (20 floats)| `gnd_channel_mask`,
                                                    `cpl_channel_mask`           |

The two channel-mask buffers are **non-trainable** (registered via
`register_buffer`). The clamp cap is reused from the existing
`gnd_residual._clamp_bound` buffer — no new buffer added by ClampNorm.

Verified by `test_param_count` (`pc["total"] == 44_738` exact equality).

## 5. Curriculum schedule (UNCHANGED)

```
epoch  0–49   :  C = log(1.5) ≈ 0.405   (Phase 0)
epoch 50–149  :  C = log(2.5) ≈ 0.916   (Phase 1)
epoch 150–199 :  C = log(4.0) ≈ 1.386   (Phase 2)
```

The combined model exposes the same `set_clamp_bounds(value)` hook as
both singles, which updates the `_clamp_bound` buffer on both heads.
ClampNorm reads that buffer at forward time (per-net joint norm).

## 6. Risks + mitigation

### Risk A — ClampNorm cap interacts with masked-input δ_gnd magnitude

Hypothesis: with InputSubset, the gnd encoder loses interaction-channel
features and may produce a residual logit `δ_gnd` of a different
magnitude than baseline. The joint norm `n = √(δ_gnd² + δ_cpl²)` then
saturates the clamp differently — for example, if `δ_gnd` is
systematically smaller (less information → smaller deviation from 1.0
multiplier), then `n ≈ |δ_cpl|` more often, and the projection
effectively becomes per-channel clamp on `δ_cpl` alone (with
`δ_eff_gnd = (C/|δ_cpl|) · δ_gnd ≈ small fraction`).

Direction of the effect:
- In the InputSubset single-seed result, gnd MAPE actually IMPROVED
  (19.05% vs 20.49% baseline), so the masked-input gnd path is NOT
  starved — it produces meaningful corrections, just toward a tighter
  geometry-only target.
- The best-case for ClampNorm in this regime: when `δ_gnd` is
  well-controlled (smaller), the joint cap leaves more "budget" for
  `δ_cpl`, which the cpl path can use (cpl MAPE was 15.02% under
  InputSubset, already better than ClampNorm-alone 15.22%).
- Worst case: `δ_gnd` becomes degenerate (always near zero) and the
  joint norm is dominated by cpl, making ClampNorm degenerate to
  per-channel cpl clamp. This would lose ClampNorm's coupling benefit
  but is still no worse than baseline element-wise clamp on cpl alone.

Mitigation: the smoke run logs Phase 2 |Δvalid| (max + mean) to detect
oscillation re-emergence. If ClampNorm's stability gain disappears in
the combined model, that diagnoses the cap-interaction effect.

### Risk B — Encoder dead-input pathology compounds

InputSubset's known caveat: encoder first-linear columns
`W[:, 6], W[:, 7], W[:, 9]` only fire from cpl-path gradients. With
ClampNorm dampening cpl gradients above the cap (`Jacobian norm = C/n`),
those input columns may receive even weaker signal than under
InputSubset alone. This is most acute in Phase 0 (`C = 0.405`, harshest
projection) when the heads are still warming up.

Mitigation: zero-init residual heads make `δ ≈ 0` in early epochs, so
the projection is effectively identity (n ≪ C → s = 1). The dead-input
issue should be no worse than InputSubset alone for the first ~20 epochs
of Phase 0; if it manifests, it would show as a slower drop in
train_loss during Phase 0 vs InputSubset alone.

### Risk C — Mode-B giant-net under-correction

The same "Mode B" worry from ClampNorm's standalone DESIGN.md applies:
CTS / clock nets needing large multipliers may be over-clamped when the
joint norm saturates. InputSubset does not change the analytic prior,
so this risk is identical to ClampNorm-alone. The C1 isotonic
post-correction step (next phase, after this combined smoke) is intended
to address the long-net tail and is orthogonal to this combined model.

## 7. Hypothesis

If both mechanisms compose additively:

| Metric           | Baseline | InputSubset | ClampNorm | Combined predicted (additive) | Combined gate threshold        |
|------------------|----------|-------------|-----------|-------------------------------|---------------------------------|
| test gnd         | 20.49%   | 19.05%      | 20.89%    | ~19.0% (IS-dominated)         | ≤ 18.5% to PASS strictly        |
| test cpl         | 15.53%   | 15.02%      | 15.22%    | ~14.7% (intersection)         | ≤ 14.7% to PASS strictly        |
| test total       | 8.27%    | 7.22%       | 7.36%     | ~6.5% (gnd-cpl decorrelation) | ≤ 6.8% to PASS strictly         |
| last_valid total | 8.27%    | n/a         | 6.66%     | ~6.5% (CN-dominated stability)| ≤ 6.5% to PASS strictly         |

Sub-additive is acceptable — the gates are conservative (each ≥ 0.1pp
tighter than the better single).

## 8. Decision gate (inherited from task brief)

PASS if at least ONE of:
- test gnd ≤ **18.5%** (better than InputSubset 19.05%)
- test cpl ≤ **14.7%** (better than InputSubset 15.02%, ClampNorm 15.22%)
- test total ≤ **6.8%** (better than InputSubset 7.22%, ClampNorm 7.36%)
- last_valid total ≤ **6.5%** (better than ClampNorm 6.66%)

AND no metric regresses by > 0.5pp vs the better of the two singles
(`min(IS, CN)` per metric).

If PASSES → recommend 5-seed lock for `HybridPexV3MeshInputSubsetClampNorm`.
If FAILS → diagnose which lever dominates / interferes; archive variant.

## 9. Implementation contract

```python
class HybridPexV3MeshInputSubsetClampNorm(nn.Module):
    """Composes InputSubset (input zero-mask, shared encoder) with
    ClampNorm (joint-norm residual projection)."""

    def __init__(self, ..., gnd_interaction_channels=(6, 7, 9)):
        # ONE shared cuboid_encoder
        # gnd_residual + cpl_residual unchanged from baseline
        # gnd_channel_mask buffer (1, 1, 10) with zeros at {6, 7, 9}
        # cpl_channel_mask buffer (1, 1, 10) all-ones (identity, kept for symmetry)

    def _predict_joint(analytic_gnd, analytic_cpl,
                       self_features, pair_features,
                       cuboids, padding_mask):
        # 1. InputSubset masking
        gnd_input = cuboids * self.gnd_channel_mask        # zeros ch{6,7,9}
        cpl_input = cuboids                                 # identity
        # 2. Two encoder forwards (same shared weights)
        gnd_emb = self.cuboid_encoder(gnd_input, padding_mask)
        cpl_emb = self.cuboid_encoder(cpl_input, padding_mask)
        # 3. Residual logits via direct .mlp access (bypass head's clamp)
        feats_gnd = torch.cat([self_features, gnd_emb], dim=-1)
        feats_cpl = torch.cat([pair_features, cpl_emb], dim=-1)
        logit_gnd = self.gnd_residual.mlp(feats_gnd).squeeze(-1)
        logit_cpl = self.cpl_residual.mlp(feats_cpl).squeeze(-1)
        # 4. Joint-norm projection clamp
        cap = self.gnd_residual._clamp_bound
        n = sqrt(logit_gnd² + logit_cpl² + EPS²)            # softened sqrt
        s = clamp(cap / n, max=1.0)
        logit_gnd_eff, logit_cpl_eff = s * logit_gnd, s * logit_cpl
        # 5. Multiplicative residual
        return analytic_gnd * exp(logit_gnd_eff), analytic_cpl * exp(logit_cpl_eff)
```

The standalone `predict_gnd` and `predict_cpl` API methods retain the
ClampNorm fallback (assume the missing logit is 0). The trainer + smoke
script exclusively use `_predict_joint` so this fallback is never on the
training path.

**Forward cost**: 2× encoder forward (because gnd/cpl see different
inputs) + 2× residual MLP — same as InputSubset (which also runs
encoder twice). ClampNorm alone runs encoder once + 2× residual MLP, so
the combined model has ~10% more wall-clock per step due to the
second encoder forward; in practice the encoder is small (9K params,
~10% of total) so the slowdown is bounded.

## 10. Files

| Path                                                                                                              | Role                            |
|-------------------------------------------------------------------------------------------------------------------|---------------------------------|
| `pex_v3/src/models/hybrid_v3_mesh_input_subset_clamp_norm.py`                                                     | Combined model class            |
| `pex_v3/tests/test_hybrid_v3_mesh_input_subset_clamp_norm.py`                                                     | Unit tests (8)                  |
| `pex_v3/scripts/38_finetune_mesh_input_subset_clamp_norm_smoke.py`                                                | Single-seed smoke               |
| `pex_v3/configs/ablation_manifest.yaml`                                                                           | Variant registry                |
| `pex_v3/experiments/auto_optimize_2026_05_03/outputs/input_subset_clamp_norm/seed42/`                             | Smoke artifacts                 |
| `pex_v3/experiments/auto_optimize_2026_05_03/variants/input_subset_clamp_norm/DESIGN.md`                          | This file                       |

## 11. Open questions for `pex-physics-architect`

1. **Cap-budget reallocation under InputSubset:** if the gnd path
   produces systematically smaller `|δ_gnd|` (because less input
   information), the joint norm leaves more budget for `δ_cpl`. Is this
   physically sensible (cpl truly needs more correction headroom) or
   does it artificially over-fit cpl?
2. **Gradient asymmetry through the shared encoder:** InputSubset
   already imposes that the encoder's `W[:, 6:8], W[:, 9]` columns only
   receive cpl-path gradient. ClampNorm dampens cpl gradient above the
   cap (`Jacobian norm = C/n`). Is the implied "extra dampening of
   interaction-channel learning" a concern for cpl quality?
3. **Phase 2 stability under combined forward:** ClampNorm-alone showed
   `Phase 2 max |Δvalid| ≪ baseline 3.6pp`. Does the InputSubset masking
   noise re-introduce oscillation? The smoke logs `phase2_max_abs_delta`
   for direct comparison.
