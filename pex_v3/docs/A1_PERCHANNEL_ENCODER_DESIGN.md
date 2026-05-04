# A1 — Per-channel Separate Cuboid Encoders (Design)

_Author: neural-operator-architect (PINN-PEX architecture lead)_
_Date: 2026-05-03_
_Status: DRAFT — pre-implementation (Step 1 of 4)_

## 1. Motivation

Current PINN headline `HybridPexV3Mesh` (5-seed locked, 44K params) hits a per-channel ceiling on cross-design test:

| metric | best valid | last valid | last test |
|---|---|---|---|
| gnd MAPE | 19.18% | 19.19% | **20.49%** |
| cpl MAPE | 12.54% | 12.69% | **15.53%** |
| total MAPE | 6.18% | 8.69% | **8.27%** |

Strike #8 systematic diagnostic concluded:
- Adding scalar features (cell-OBS, Liberty pin caps, z-scored variants) consistently HURTS test (+2 to +5pp).
- The `CuboidSetEncoder` already saturates the cell-complexity proxy that those scalars encoded.
- gnd and cpl errors are positively correlated (ρ=0.33) — they share a common cause AND are pushed against each other when forced to share representation.

The single shared `CuboidSetEncoder` is the only pre-residual-head feature pathway. By Codex's ranking it is the highest-leverage architectural intervention untested in the original paper sweep:

> "If gnd Q4 errors come from small-fanout/small-bbox nets (a cell-internal substrate-area question) and cpl errors come from broadside-overlap nets (a coupling-geometry question), the same set embedding is being asked to encode two very different statistics. Splitting the encoder lets each head specialize."

## 2. Architecture

### 2.1 Diagram

```
                    +-----------------+
   raw cuboids ---> | gnd_encoder     | --emb_gnd--> [self_features ⊕ emb_gnd] ---> gnd_residual --x--> pred_gnd
   padding_mask     | (CuboidSetEnc)  |                                                                  |
                    +-----------------+                                              analytic_gnd -------+
                                                                                                              
                    +-----------------+
   raw cuboids ---> | cpl_encoder     | --emb_cpl--> [pair_features ⊕ emb_cpl] ---> cpl_residual --x--> pred_cpl
   padding_mask     | (CuboidSetEnc)  |                                                                  |
                    +-----------------+                                              analytic_cpl -------+
```

Two **independent** `CuboidSetEncoder` instances (separate weights, identical architecture) feed two independent `BoundedResidualHead`s. Both encoders see the same raw cuboid tensor; gradient paths are fully disjoint.

### 2.2 Drop-in contract

`HybridPexV3MeshPerChannel` MUST expose the exact same forward API as `HybridPexV3Mesh` so existing trainer code (`19_finetune_hybrid_mesh_smoke.py` and the 5-seed launcher) work unchanged after a single class-name swap:

```python
predict_gnd(analytic_C_fF, self_features, cuboids, padding_mask) -> Tensor
predict_cpl(analytic_C_fF, pair_features, cuboids, padding_mask) -> Tensor
set_clamp_bounds(clamp_bound: float) -> None
parameter_count() -> dict
```

### 2.3 What is NOT changed

To preserve attribution and isolate the per-channel-encoder effect:

| Component | Status | Reason |
|---|---|---|
| Analytic priors (`compact_gnd_estimate_fF`, `compact_cpl_estimate_total_fF`) | UNCHANGED | shared physics; per-channel encoder change must not be confounded with prior change |
| `fit_per_layer_calibration` (NNLS) | UNCHANGED | already per-channel; reuse exactly |
| Bounded multiplicative residual `exp(clamp(δ))` | UNCHANGED | curriculum killer feature, -1.89pp at Phase 0→1 transition |
| RES_CLAMP curriculum schedule (0.405 → 0.916 → 1.386) | UNCHANGED | one-change-per-cycle rule |
| β-strategy per-channel MAPE loss | UNCHANGED | same loss, separately backprops to gnd vs cpl encoders |
| Optimizer (Adam, lr=1e-3, wd=1e-5) | UNCHANGED | -- |
| Eval pipeline (validate_calibration → evaluate_full_split) | UNCHANGED | identical schema → ensemble + 5-seed code already works |
| `CuboidSetEncoder` internal architecture (3-pool mean+max+sum, 2-layer MLP, embed_dim=64) | UNCHANGED | one-change-per-cycle rule |
| `BoundedResidualHead` zero-init last layer | UNCHANGED | day-1 invariant |

## 3. Parameter budget

`CuboidSetEncoder` with defaults (in_dim=10, hidden=64, embed_dim=64, n_layers=2):

```
cuboid_mlp[0]: 10×64 + 64 = 704
cuboid_mlp[2]: 64×64 + 64 = 4160
cuboid_mlp[4]: 64×64 + 64 = 4160
total per encoder: 9024
```

Residual heads (BoundedResidualHead, hidden=64, n_hidden=2):

```
gnd_residual: in_dim = 16 + 192 = 208
  Linear(208,64) = 208×64+64 = 13,376
  Linear(64,64)  = 4160
  Linear(64,1)   = 65
  total ≈ 17,601 (matches baseline summary.json)

cpl_residual: in_dim = 24 + 192 = 216
  Linear(216,64) = 216×64+64 = 13,888
  Linear(64,64)  = 4160
  Linear(64,1)   = 65
  total ≈ 18,113 (matches baseline summary.json)
```

| Variant | cuboid encoder(s) | residual heads | total |
|---|---|---|---|
| Baseline `HybridPexV3Mesh` (shared) | 9,024 | 35,714 | **44,738** |
| A1 `HybridPexV3MeshPerChannel` (split) | 18,048 (×2) | 35,714 | **53,762** |

Budget cap: ≤ 100K. **A1 = 53,762 = 1.20× baseline = comfortably within cap.**

### Compute & memory

- 2× encoder forward → ~2× FLOPs in encoder section. Encoder is small relative to data loading; expect ~5-10% wall-clock slowdown.
- Activation memory: 2 forward graphs of `(B, N_max, embed_dim)` ≈ +(256 × 512 × 64 × 4 bytes) = +33 MB per batch. Negligible on A6000 (48 GB).
- Backward pass: 2 disjoint graphs; no extra synchronization required.

## 4. Inductive-bias rationale (why this is not a parameter-doubling trick)

1. **Separate encoders allow specialization.** The gnd encoder can learn that small-bbox & small-fanout nets need a substrate-area-proxy embedding, while the cpl encoder can learn that broadside-overlap & spacing-min-um signals need a coupling-geometry embedding. With shared weights, gradient signal is averaged across these two semantically different objectives.
2. **Disjoint gradient paths reduce destructive interference.** With ρ=0.33 in error, the shared encoder receives push-pull updates whenever an outlier exists in only one channel. Disjoint paths let each head optimize without cross-channel interference.
3. **Day-1 invariant preserved.** Encoders are NOT in the day-1 critical path because the residual head's last linear layer is zero-init → multiplier = 1.0 regardless of encoder output. So separating encoders is a pure capacity/specialization change with no shift in initial behavior.

## 5. Risks & mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| 2× encoder ↔ 2× train compute | Low | Encoder is ~20% of total batch time; expect <15% slowdown. 200-epoch run still fits in <1.5h on A6000. |
| Per-channel overfit (more capacity, less inductive sharing) | Medium | Test set (95K nets, OOD) is the kill-criterion. If valid improves but test regresses, kill criterion fires. |
| Activation memory blowup with longer cuboid sequences | Low | `max_cuboids_per_net=512` cap unchanged. Memory delta is ~33 MB per batch (negligible). |
| Day-1 invariant breakage (e.g., if encoder somehow leaks to output) | Low | Explicit unit test (`test_day1_analytic`) asserts pred = analytic at init. |
| Gradient leak between encoders (e.g., shared buffer, accidental tied params) | Low | Explicit unit test (`test_gradient_isolation`) asserts ∂loss_gnd/∂cpl_encoder.weight == 0 and vice-versa. |

## 6. Decision gate (Codex revised kill criterion)

Single-seed smoke (full 200 epochs preferred, 30-epoch sanity acceptable if time-limited) must show on cross-design test:

| Metric | Baseline (test, last) | A1 must achieve | Verdict if missed |
|---|---|---|---|
| gnd MAPE | 20.49% | ≤ **19.5%** (≥ 1.0pp improvement) | A1 dropped, A3 deprioritized |
| total MAPE | 8.27% | ≤ **8.27%** (no regression) | A1 dropped |

If BOTH miss → REPORT FAILURE — do not recommend full 5-seed.
If only one misses but the other improves significantly → human-in-loop call.
If both meet → recommend 5-seed lock.

For 30-epoch (Phase 0 only) sanity smoke:
- day-1 (epoch 0) MAPE must match calibrated analytic prior baseline (~21% gnd, ~13% cpl, ~20% total per `summary.json:day1_valid`).
- epoch 30 valid gnd should be reasonable (< 25% — anything beyond means divergence).
- Sanity-only verdict; user decides full run.

## 7. Files

| Path | Purpose |
|---|---|
| `pex_v3/docs/A1_PERCHANNEL_ENCODER_DESIGN.md` | THIS doc |
| `pex_v3/src/models/hybrid_v3_mesh_perchannel.py` | new model class |
| `pex_v3/tests/test_hybrid_v3_mesh_perchannel.py` | 4 unit tests |
| `pex_v3/scripts/35_finetune_mesh_perchannel_smoke.py` | single-seed smoke |
| `pex_v3/output/phase1_mesh_perchannel_smoke/` | smoke outputs (summary.json, history.json, model.pt) |

## 8. Open physics-side questions for `pex-physics-architect`

These are NOT blockers for A1 implementation but should be reviewed before any 5-seed lock claim:

1. **Should the analytic prior also be split by channel?** Currently `compact_gnd_estimate_fF` and `compact_cpl_estimate_total_fF` are already separate; A1 does not change that. But if A1 succeeds, a follow-up question is whether a per-channel version of `analytic_base_v3` (e.g., separate calibrators for gnd-vs-cpl per layer) yields additional benefit. Out of scope for A1.

2. **Does the per-channel encoder break any conservation check?** The bounded multiplicative residual preserves sign and (for gnd) typically ratio bounds. With separate encoders, gnd_pred and cpl_pred are computed on independent feature paths but both still go through `analytic × exp(clamp(δ))`. No new conservation violation.

3. **Calibration retains independence assumption.** `fit_per_layer_calibration` already treats gnd and cpl independently. A1 reinforces this by giving each its own learned representation — physically defensible since gnd is a self-cap (substrate ↔ conductor) and cpl is mutual cap (conductor ↔ conductor).
