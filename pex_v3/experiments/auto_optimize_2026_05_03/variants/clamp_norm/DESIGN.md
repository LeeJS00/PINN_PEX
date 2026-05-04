# ClampNorm — Clamp-on-Residual-Norm Variant

_Single-seed smoke design doc. Auto-Optimize Sweep 2026-05-03._

## 1. Motivation

Baseline `HybridPexV3Mesh` (5-seed locked, 44.7K params) shows **curriculum-transition
instability** in Phase 2 (clamp 1.386, epochs 150-200):

- Per-epoch valid total MAPE swings up to **3.6 pp** (seed 0: 6.18% → 9.80% across
  consecutive epochs in Phase 2).
- `best_valid_total = 6.18%` vs `last_step_valid_total = 8.69%` (gap **2.5 pp**).

A1 agent observation (file: A1 audit notes): the element-wise `torch.clamp(δ, -C, +C)`
applied to the per-net residual logit produces a **gradient cliff** — `∂clamp(δ)/∂δ`
is `1` for `|δ|<C` and `0` for `|δ|>C`. When training pushes `|δ|` near the cap, the
encoder weights see a stochastically-discontinuous gradient signal across batches and
oscillate.

In Phase 2 the cap is loosest (`C = log 4 ≈ 1.386`), so more residual logits are
**near or beyond** the cap (vs Phase 0's tight `C = log 1.5 ≈ 0.405`). This explains
why instability is worst in Phase 2, not Phase 0.

## 2. Hypothesis

Replace the **element-wise hard clamp on each residual scalar** with a
**vector-norm projection clamp on the joint (gnd, cpl) residual logit**:

> When the L2 norm of the per-net residual VECTOR `δ = (δ_gnd, δ_cpl)` exceeds
> the curriculum cap `C`, scale BOTH components down by the same factor so that
> `||δ_eff||_2 = C`. Below the cap, identity (no scaling).

This preserves the day-1 invariant, preserves the curriculum schedule,
and removes the per-component gradient cliff: when `||δ|| > C`, both
components contribute smoothly to the projection scale, so the gradient
of `δ_eff` w.r.t. `δ` is a smooth `(2 × 2)` Jacobian (rank-1 update plus
identity-times-scalar) rather than a `0/1` indicator.

## 3. Math

### 3.1 Per-net joint residual vector

For each net `b`, the model computes two residual logits (scalars) via
the existing `BoundedResidualHead.mlp`:

```
δ_gnd[b] = MLP_gnd( self_features[b]  ⊕ embed[b] )   ∈ R
δ_cpl[b] = MLP_cpl( pair_features[b] ⊕ embed[b] )    ∈ R
```

Stack into a **per-net joint residual vector**:

```
δ[b] = (δ_gnd[b], δ_cpl[b])  ∈ R^2
```

### 3.2 Norm-projection clamp

Let `C` be the current curriculum cap (scalar buffer, set per-epoch by
`set_clamp_bounds`). Define the L2 norm per net:

```
n[b] = ||δ[b]||_2 = sqrt(δ_gnd[b]^2 + δ_cpl[b]^2 + 0)
```

Define the projection scale (vectorized; `eps = 1e-12` for numerical
safety against division by zero at day-1):

```
s[b] = min( 1, C / max(n[b], eps) )
```

Apply the scale uniformly to both components:

```
δ_eff[b] = s[b] · δ[b]      i.e.   δ_eff_gnd[b] = s[b] · δ_gnd[b]
                                   δ_eff_cpl[b] = s[b] · δ_cpl[b]
```

Multiplier and prediction are unchanged:

```
mul_gnd[b] = exp(δ_eff_gnd[b]),    pred_gnd[b] = analytic_gnd[b] · mul_gnd[b]
mul_cpl[b] = exp(δ_eff_cpl[b]),    pred_cpl[b] = analytic_cpl[b] · mul_cpl[b]
```

**Norm choice — per-NET, joint over (gnd, cpl).** L2 is computed over the
2-vector `(δ_gnd, δ_cpl)` per net, NOT over the batch dimension and NOT
over per-cuboid logits (the residual heads already operate on per-net
pooled features, so there is no per-cuboid axis at this layer). Per-net
joint norm is the only interpretation that:

1. Preserves day-1 invariant exactly.
2. Differs meaningfully from element-wise clamp (a per-net per-channel
   norm reduces to `|·|`, which equals element-wise clamp on a scalar).
3. Couples gnd and cpl regularization naturally — the model's joint
   "deviation budget" per net is bounded.

### 3.3 Day-1 invariant proof

At initialization, `BoundedResidualHead` has the last linear zero-initialized
(`nn.init.zeros_(final.weight); nn.init.zeros_(final.bias)`). Therefore at day-1:

```
δ_gnd = 0,   δ_cpl = 0
n = sqrt(0 + 0) = 0
s = min(1, C / max(0, eps)) = min(1, C / eps)
```

`C / eps = 0.405 / 1e-12 ≈ 4e11`, so `s = min(1, 4e11) = 1`. Then
`δ_eff = 1 · 0 = 0`, so `mul = exp(0) = 1`, so `pred = analytic`. ✓

(Note: `eps = 1e-12` is chosen so that `C/eps ≫ 1` for any realistic `C`,
guaranteeing `s = 1` whenever `||δ|| < C` regardless of `C` magnitude.)

### 3.4 Curriculum schedule (UNCHANGED)

```
epoch  0–49   :  C = log(1.5) ≈ 0.405   (Phase 0)
epoch 50–149  :  C = log(2.5) ≈ 0.916   (Phase 1)
epoch 150–200 :  C = log(4.0) ≈ 1.386   (Phase 2)
```

`set_clamp_bounds(value)` updates `_clamp_bound` buffer on **both**
residual heads, identical to baseline. The norm-projection layer reads
the buffer at forward time, so curriculum is a drop-in.

### 3.5 Gradient analysis

Let `n = ||δ||`, `s = min(1, C/n)`, `δ_eff = s · δ`.

**Below the cap** (`n < C`): `s = 1`, so `δ_eff = δ`, Jacobian
`∂δ_eff/∂δ = I`. Identity in-region. Same as element-wise clamp.

**Above the cap** (`n > C`): `s = C/n`, so

```
δ_eff = (C/n) · δ
∂δ_eff_i/∂δ_j  =  (C/n) · I_ij  -  (C/n^3) · δ_i · δ_j
```

This is a smooth **rank-1 perturbation of a scaled identity** — it
preserves direction information and degrades smoothly with `n`, in
contrast to the element-wise clamp's `0/1` indicator. The norm of the
Jacobian is `C/n < 1`, so above-cap gradients are dampened but never
truncated to zero.

**At the cap** (`n = C`): both regions agree (`s = 1` is reached from
both sides), Jacobian is continuous up to a kink in the second
derivative — first-order smooth. No discontinuity in gradient sign or
magnitude across the boundary.

This matches Codex Round 2 verdict: physics OK, gradient continuous.

## 4. Risk + Mitigation

### Risk A: Joint scaling may damp the wrong channel.

If a net's true residual is purely `(large_gnd, 0)`, the joint norm
clamp at `C = 1.386` would scale BOTH down to `(C, 0)` — but cpl was
already 0, so this is harmless. The risk is the symmetric case: model
emits `(spurious_large_gnd, useful_cpl)` and the joint norm compresses
the useful cpl. Mitigation:

- **Monitor cpl test MAPE** specifically against baseline. Kill criterion
  enforces no metric regresses by > 0.5 pp (cpl included).
- The element-wise clamp would in this case truncate `δ_gnd` to `±C`
  and pass `δ_cpl` through unchanged, so this is the one regime where
  ClampNorm is theoretically WORSE. Phase-2 instability evidence
  suggests the actual failure mode in baseline is `|δ_gnd|, |δ_cpl|`
  BOTH near the cap (encoder oscillating to satisfy two competing
  errors), in which case joint projection helps.

### Risk B: Per-net norm clamping dampens "Mode B" giant-net residuals.

CTS / clock nets need large multiplicative correction (the Strike #6
finding). Joint norm clamping with `C = 1.386` caps both gnd and cpl
multipliers per net, so a giant CTS net needing `mul_gnd = 5.0`
(`δ_gnd ≈ 1.61`) cannot get there even if `δ_cpl ≈ 0`. With
element-wise clamp, the same net would saturate at `mul_gnd = exp(C) = 4.0`.
With ClampNorm, if `δ_cpl = 0`, then `n = |δ_gnd|`, `s = C/|δ_gnd|`, so
the gnd output is `exp(C · sign(δ_gnd)) = exp(±C) = exp(±1.386) ∈ {0.25, 4.0}`.
**Identical to element-wise clamp in the single-channel-saturation case.**
The two clamps differ ONLY when both channels are simultaneously near
the boundary.

Conclusion: Risk B is mitigated by construction — single-channel
saturation behaves identically to baseline. Risk only materializes
in the joint-saturation regime, which is also exactly the regime
the hypothesis predicts is causing the instability.

### Risk C: `eps = 1e-12` interaction with `float32`.

`1e-12` is below `float32` ulp at unit scale (`~1.19e-7`), but the
critical operation is `max(n, eps)` in the denominator. At day-1
`n = 0.0` exactly (zero-init), so `max(0, 1e-12) = 1e-12`, and
`C / 1e-12` overflows `float32` only if `C > 3.4e38` (it's `1.386`,
so OK). The result `s = min(1, 4e11) = 1` is well-defined.
**No `float64` upcasting needed.**

## 5. Comparison with baseline

| Aspect              | Baseline (HybridPexV3Mesh) | ClampNorm                            |
|---------------------|----------------------------|--------------------------------------|
| Param count         | 44,738                     | 44,738 (identical)                   |
| Residual head MLPs  | unchanged                  | unchanged                            |
| Cuboid encoder      | unchanged                  | unchanged                            |
| Curriculum schedule | 0.405 → 0.916 → 1.386      | 0.405 → 0.916 → 1.386 (unchanged)    |
| Clamp formula       | `clamp(δ, -C, +C)` per chan | `δ × min(1, C/||(δ_gnd, δ_cpl)||_2)` |
| Day-1 invariant     | yes (mul=1)                | yes (mul=1, see §3.3 proof)           |
| Gradient at cap     | `0/1` indicator (cliff)    | rank-1 + scaled identity (smooth)    |
| Joint reg coupling  | none                       | yes (gnd + cpl share a budget)       |

## 6. API parity (drop-in)

`HybridPexV3MeshClampNorm` exposes the same API as `HybridPexV3Mesh`:

- `predict_gnd(analytic_C_fF, self_features, cuboids, padding_mask)` → `(B,)`
- `predict_cpl(analytic_C_fF, pair_features, cuboids, padding_mask)` → `(B,)`
- `set_clamp_bounds(value: float)` → updates BOTH residual heads' `_clamp_bound`
- `parameter_count() → dict`

**Implementation note:** because the norm-projection requires BOTH
`δ_gnd` and `δ_cpl` to compute `n`, the model must run BOTH residual
MLPs whenever EITHER `predict_gnd` or `predict_cpl` is called for a
given net. This is a **forward-cost increase ≈ 2×** for the residual
heads (negligible vs encoder cost), but does NOT change parameter count.
To preserve `predict_gnd` / `predict_cpl` API parity (single call returns
single output), the model internally calls `_predict_joint(...)` which
returns both, and the public `predict_*` methods discard the unused
half. For training, the trainer calls both back-to-back, so the second
call's compute is essentially free if the same minibatch is reused with
cached embeddings (current trainer already calls both per minibatch).

## 7. Decision gate (revised criterion)

ClampNorm passes smoke if at least ONE of:

- test gnd ≤ **19.5%** (-1 pp)
- test cpl ≤ **14.5%** (-1 pp)
- test total ≤ **7.27%** (-1 pp)

AND no metric regresses by > 0.5 pp absolute.
AND curriculum-transition gain preserved (transition gain > 0).

If passes → recommend 5-seed lock via `run_ablation_5seed.py`.
If fails → drop variant + post-mortem document failure mode.

## 8. Files

| Path                                                                                                | Role                |
|-----------------------------------------------------------------------------------------------------|---------------------|
| `pex_v3/src/models/hybrid_v3_mesh_clamp_norm.py`                                                    | Model class         |
| `pex_v3/tests/test_hybrid_v3_mesh_clamp_norm.py`                                                    | Unit tests (5)      |
| `pex_v3/scripts/37_finetune_mesh_clamp_norm_smoke.py`                                               | Single-seed smoke   |
| `pex_v3/experiments/auto_optimize_2026_05_03/outputs/clamp_norm/seed42/`                            | Smoke artifacts     |
| `pex_v3/experiments/auto_optimize_2026_05_03/variants/clamp_norm/DESIGN.md`                         | This file           |

## 9. Baseline numbers (for direct comparison)

From `pex_v3/output/phase1_mesh_5seed/seed0/{summary.json, history.json}`:

- Day-1 valid total: 20.69%
- Best valid total: 6.18% @ epoch 170
- Last (epoch 199) valid total: 8.69%
- Final test: gnd 20.80%, cpl 15.53%, total 8.76%
- Curriculum transition jumps:
  - epoch 49 → 50: +1.69 pp (9.86 → 11.55%) — Phase 0→1 onset (worse before better)
  - epoch 50 → 52: -2.37 pp (11.55 → 7.49%)
  - epoch 149 → 150: +1.76 pp (7.83 → 9.58%) — Phase 1→2 onset
  - epoch 150 → 153: -2.98 pp (9.58 → 6.61%)
- Phase 2 (epochs 150-200): max |Δvalid| = 3.6 pp, mean |Δvalid| = 1.1 pp
