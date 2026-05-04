# InputSubset — per-channel input zero-masking

## One-line

Same shared `CuboidSetEncoder` weights as the locked baseline, but the
gnd-head call sees an input tensor whose interaction-info columns are
zeroed, while the cpl-head call sees the full tensor.

## Why this exists

A1 (per-channel separate encoders, +9.0K params) was killed: test gnd
21.60% (+1.11pp regression). The Strike #7 / Strike #8 / A1 trifecta
proves capacity-add at the encoder is a dead lever — it Phase-2-overfits.

A1's analysis suggested the root cause might not be "more capacity" but
"different information needed per channel":
- `c_gnd_fF` (net to substrate) physically depends on the target net's
  own conductor geometry + dielectric stack only. Aggressor presence is
  noise.
- `c_cpl_total_fF` (net to all neighbors) is by definition an
  interaction quantity — it MUST see both target and aggressor geometry.

InputSubset attacks the same hypothesis without adding parameters: same
encoder weights, but pass two different masked inputs.

## Cuboid feature layout (10 channels)

Verified by inspecting `/data/PINNPEX/data/processed_v3/intel22/per_net_cuboids/intel22_*.npz`:

| idx | name           | dtype  | range / unique values                        | meaning |
|-----|----------------|--------|----------------------------------------------|---------|
| 0   | x_rel          | f32    | continuous, ~ [-7, +7]                       | tile-relative xy |
| 1   | y_rel          | f32    | continuous, ~ [-7, +7]                       | tile-relative xy |
| 2   | z_abs          | f32    | quantized per metal (M1=0.58, ..., M9=1.19)  | absolute z layer |
| 3   | w              | f32    | [0.004, 4.95]                                | cuboid width  |
| 4   | h              | f32    | [0.0,  3.96]                                 | cuboid height |
| 5   | d              | f32    | per-layer thickness (0.066/0.071/0.087)      | cuboid depth (z extent) |
| 6   | semantic_type  | f32    | {0.5, 1.0}  (0.5 = pin, 1.0 = wire)          | conductor role |
| 7   | is_target      | f32    | {0.0, 1.0}  (0 = aggressor, 1 = target)      | INTERACTION flag |
| 8   | eps            | f32    | per-layer (intel22 sample = 2.8 here)        | permittivity |
| 9   | net_type       | f32    | currently always 0.0 (VSS-aggressor unused)  | reserved |

Per-net store contents (after `18_extract_per_net_cuboids.py`): about
**76% of rows are aggressors** (ch7 == 0) and 24% are target (ch7 == 1).
Despite the script's filter intent, the per-net `.npz` retains both
target and aggressor cuboids — the filter only enforces "the net has at
least one cuboid in this tile." This is the data the encoder pools over.

## Channel partition

| Group           | Indices       | Rationale |
|-----------------|---------------|-----------|
| GEO_CORE        | 0,1,2,3,4,5   | spatial position + extents — purely geometric |
| MATERIAL        | 8             | dielectric permittivity — material physics |
| INTERACTION     | 6,7,9         | semantic_type, is_target, net_type — describe role / pair structure |

- `gnd input mask`  → keep `GEO_CORE ∪ MATERIAL` = `[0,1,2,3,4,5,_,_,8,_]`,
  zero out indices 6, 7, 9.
- `cpl input mask`  → identity (keep all 10 columns).

## Implementation contract

```
encoder = CuboidSetEncoder(in_dim=10, ...)            # ONE instance, shared
gnd_input = cuboids * gnd_channel_mask                # mask is (1, 1, 10) buffer
gnd_emb = encoder(gnd_input, padding_mask)
gnd_pred = analytic_gnd * gnd_residual([self_features, gnd_emb])

cpl_input = cuboids                                    # full tensor
cpl_emb = encoder(cpl_input, padding_mask)
cpl_pred = analytic_cpl * cpl_residual([pair_features, cpl_emb])
```

The mask is a fixed `(1, 1, 10)` float32 buffer registered on the model
(no parameters, no gradient). Multiplication is element-wise.

## Why pure column-zero (not row-zero of aggressors)

The brief says: "gnd encoder call: input = (x, y, z, w, h, d, eps) —
geometry + permittivity only." Literal reading is column-only zeroing.
Row-zeroing aggressor cuboids would be a stronger ablation that drops
~76% of the per-net signal entirely; it's a separate variant for future
ablation but not the first smoke. Column-only also keeps the test
`test_input_mask_correctness` trivially checkable on intermediate
tensors.

Trade-off: with column-only, the gnd encoder still sees aggressor
cuboids in its pool — but it can no longer tell them apart from target
cuboids (ch7 zeroed). The hypothesis is that this geometric "blur" is
preferable to the current setup where the same shared embedding has to
serve both gnd's "what's MY geometry?" and cpl's "what's my geometry vs.
neighbors?" objectives simultaneously.

## Param budget

Baseline `HybridPexV3Mesh` total = **44,738** params:
- `cuboid_encoder` 9,024
- `gnd_residual`   17,601
- `cpl_residual`   18,113

InputSubset adds:
- 1 buffer (10 floats, NOT trainable)
- 0 modules

So **InputSubset total ≈ 44,738 params** — within ≤ 50K budget,
zero new trainable params.

## Day-1 invariant

- Residual heads zero-init last linear → multiplier = 1.0 → forward
  output exactly = analytic prior. Same as baseline.
- The mask only affects intermediate encoder activations; since those
  feed a residual head whose last-layer is zero, day-1 forward is
  unchanged.

## Smoke kill criterion (Codex revised)

Single-seed (42), 200 epoch, GPU 0. PASS if at least ONE of:
- test gnd ≤ **19.5%** (-1pp from 20.49%)
- test cpl ≤ **14.5%** (-1pp from 15.53%)
- test total ≤ **7.27%** (-1pp from 8.27%)

AND no metric regresses by > 0.5pp absolute.

If PASSES → recommend 5-seed lock via `run_ablation_5seed.py
--variant HybridPexV3MeshInputSubset --seeds 0 1 2 3 4`.

If FAILS → drop variant, document failure mode (Phase 2 overfit?
curriculum break? day-1 sanity violated?).

## Open questions for neural-operator-architect review

1. **Aggressor row-zeroing as Variant 1.5**: should we pre-bake an
   aggressor-only-row-zero variant for direct comparison? Or wait until
   InputSubset smoke result first?
2. **Encoder dead-input pathology**: if gnd input columns 6,7,9 are
   always zero in training, the encoder's first-linear-layer columns
   (W[:, 6:8], W[:, 9]) only ever fire from cpl-path gradients. Does
   this implicit gradient asymmetry cause dead neuron issues? Smoke
   should reveal this in train_loss curve shape.
3. **Mask signaling via constant vs. learnable**: keeping the mask as a
   non-trainable buffer keeps capacity at exactly 0 added. A future
   variant could learn a per-channel dropout-like temperature
   (parameter count +10) — explicitly out of scope here.
