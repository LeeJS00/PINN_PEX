# Phase 1 Hybrid Architecture Specification

_Status: design — pre-Codex round 3 review draft_
_Date: 2026-05-01_
_Owners: pex-physics-architect + neural-operator-architect_

This spec defines the Phase 1 model architecture for **per-pattern**
capacitance prediction (paper #1) under Strategy γ + δ. Phase 2 will
generalize to full-net via aggregator on top.

The architecture is **hybrid**: an *analytic* layered Green's function
provides the physics base; a *bounded* neural residual learns the
correction StarRC's pattern-matching engine encodes that the analytic
formula misses. ResCap (ASPDAC 2025) demonstrated this paradigm produces
SOTA-class accuracy with high data efficiency; Strategy v3 takes the
same paradigm and pushes it deeper (analytic Green's function instead
of compact closed-form formulas).

---

## 1. Scope and goals

**Phase 1 scope (paper #1 candidate)**:
- Predict capacitance for an isolated **canonical pattern** (a small
  geometric window: 1-10 conductors over 1-5 ILD layers, ≤10×10×20 μm)
- Target metric: **per-pattern MAPE < 4%** (CNN-Cap / NAS-Cap territory)
- Comparable to: CNN-Cap (TODAES 2022) Total <1.3%, NAS-Cap (2024) Total
  0.74%, BUT with **novel hybrid analytic+neural architecture**, not
  just a deeper ResNet on grid voxelization

**What the model takes as input**:
- Conductor surface mesh (Phase 1 representation; replaces Phase 0 cuboid tile)
- Per-conductor metadata (target/aggressor flag, layer index)
- Layer stack ε(z) profile (vector)

**What the model outputs**:
- Per-conductor self-capacitance to ground (C_self)
- Per-pair coupling capacitance (C_cpl[i, j])

**Out of scope for Phase 1**:
- Full-net aggregation (Phase 2 paper)
- Cross-PDK transfer (Phase 3)
- Diffusion-based / generative modeling (Codex round 1 verdict: not for
  this paradigm)

---

## 2. Input representation: conductor surface mesh

Replaces the legacy `(N, 10)` cuboid-tile representation. Per-conductor
patches on actual surfaces, no voxelization aliasing.

### 2.1 Patch format

Each conductor is decomposed into rectangular surface patches (Manhattan
routing → axis-aligned rectangles). Per patch:

| Field | Shape | Meaning |
|---|---|---|
| `xy_center_um` | (2,) | patch xy centroid |
| `z_um` | (1,) | patch z position |
| `extent_xy_um` | (2,) | patch xy half-widths |
| `area_um2` | (1,) | patch area |
| `normal` | (3,) | unit outward normal (top/bottom/side face) |
| `layer_idx` | (1,) | metal layer index (1..M9 for intel22) |
| `is_target` | (1,) | 1.0 target conductor, 0.0 aggressor |
| `conductor_id` | (1,) | 0..N-1 within the pattern |

**Why patches, not cuboids**: BEM collocation operates on conductor
surfaces. Volume cuboids carry irrelevant interior points; surface
patches carry the relevant boundary points + normals.

### 2.2 Pattern-level wrapper

A pattern is a set of conductors plus an ε-stack:

```
Pattern = {
    patches: list[Patch],         # all surface patches across all conductors
    n_conductors: int,
    layer_stack: {                # the ILD stack relevant to this pattern
        z_boundaries_um: list[float],
        eps_per_layer: list[float],
    },
    target_conductor_id: int,     # which conductor is "the target net"
    aggressor_conductor_ids: list[int],
}
```

### 2.3 Patch generation (Phase 1 utility)

A separate utility in `pex_v3/src/preprocessing/mesh_v3.py` (to be
written) takes a pattern descriptor (geometry + ε-stack) and produces
patches via:

1. Each conductor is a 3D box → 6 face rectangles
2. Each face rectangle is sub-meshed at characteristic length
   `h = min(layer_pitch, conductor_min_extent) × 0.5`
3. Each sub-rectangle becomes one patch

For BEOL Manhattan routing, all faces are axis-aligned, so meshing is
trivial. Curved/bend regions use larger patch counts but standard
rectangular sub-mesh still works.

---

## 3. Analytic baseline: layered Green's function

The base predictor uses a **closed-form layered Green's function** with
the following modes (matching `pex_v3/src/synthetic/`):

### 3.1 Mode A — stacked-dielectric series

For a target patch sitting between top and bottom ground patches, with
ILD stack between, the analytic capacitance is:

```
C_A(target, ground) = ε₀ · A_overlap / Σ_i (d_i / ε_i)
```

This handles broadside coupling correctly for parallel-plate-like
geometry common to BEOL signal-over-power layouts.

### 3.2 Mode B — single-interface image-charge correction

For a target patch above an aggressor at lateral offset, with ε
discontinuity at one z-interface (top metal vs ILD; ILD vs metal cap),
the image method gives:

```
C_B(target, aggressor) = ε₀ · A_eff / d_eff  ·  (1 + α · k · d/√A)
```

where `k = (ε_above - ε_below) / (ε_above + ε_below)` is the reflection
coefficient, captures asymmetry the Phase 0 ε channel was losing.

### 3.3 Mode C (deferred to optimization phase)

Full Sommerfeld layered Green's function via Vector Fitting / complex
image rational approximation. Direct quadrature is O(10⁻³ s/eval),
prohibitive at 10⁷ training samples. Deferred — Mode A + Mode B cover
the physics phenomenology; Mode C only matters at sub-percent floor.

### 3.4 Implementation notes

- `pex_v3/src/synthetic/ground_truth.py` already has Mode A + Mode B
  closed-forms (validated by 30 tests, all green).
- The Phase 1 model wraps these as a differentiable analytic predictor
  `phi_analytic(pattern: Pattern) -> dict[(i, j) -> C_fF]`.
- Differentiability matters because the residual network sees gradients
  through the analytic value during training (allows residual head to
  learn relative correction).

---

## 4. Neural residual: bounded correction

The residual is **multiplicative** with a bounded scale, following ResCap
and Codex round 2 mandate ("bound `||R||/||φ||` to prevent neural taking
over").

### 4.1 Residual head architecture

```
For each (target, aggressor) pair:

    pair_features = [
        analytic_log_C,                    # log of analytic prediction
        layer_pair_one_hot,                # M_i × M_j onehot
        eps_above_target,                  # local ε
        eps_below_target,
        eps_above_aggressor,
        eps_below_aggressor,
        log_d_um,                          # surface-to-surface distance
        log_A_overlap_um2,                 # broadside overlap area
        log_A_lateral_um2,                 # lateral overlap area
        log_perimeter_um,                  # contour length
        normal_dot_product,                # cos angle between conductor normals
        n_conductors_in_pattern,           # density proxy
        target_pattern_density,            # local routing density
    ]                                       # ≈ 24 dims after expansion

    residual_logit = MLP_residual(pair_features)     # MLP: 24 → 64 → 64 → 1
    residual_logit = clamp(residual_logit, -RES_CLAMP, +RES_CLAMP)
    multiplier = exp(residual_logit)                 # bounded in [exp(-R), exp(R)]
    
    C_pred = C_analytic × multiplier
```

### 4.2 Bound parameter

`RES_CLAMP = log(2.0)` initially → multiplier ∈ [0.5, 2.0]. The model
cannot deviate more than 2× from analytic; if it tries, the analytic is
the regularizer. As training progresses, bound can be loosened (curriculum):

| Epoch | RES_CLAMP | Multiplier range |
|---|---|---|
| 0-50 | log(1.5) ≈ 0.405 | [0.67, 1.5] |
| 50-150 | log(2.0) ≈ 0.693 | [0.5, 2.0] |
| 150+ | log(3.0) ≈ 1.099 | [0.33, 3.0] |

This prevents day-1 noise from blowing up. Bounded residuals are the
key reason ResCap claims "data efficient" — the model can't overfit by
moving far from physics.

### 4.3 Initialization

Last layer of `MLP_residual` weight + bias **zero-initialized**, so day-1
output is exactly zero → multiplier = 1.0 → C_pred = C_analytic. Allows
clean attribution of "what gain came from the residual" by comparing
day-1 to converged.

### 4.4 Self-capacitance head

C_self predictions use a separate head with the same residual structure
but operating on per-conductor (not pair) features:

```
self_features = [
    analytic_log_C_self,
    layer_idx_onehot,
    eps_above, eps_below,
    log_perimeter, log_total_area, ...
]                                                        # ≈ 16 dims

self_logit = MLP_self(self_features)
self_logit = clamp(self_logit, -RES_CLAMP, +RES_CLAMP)
C_self_pred = C_self_analytic × exp(self_logit)
```

---

## 5. Loss function

Aligned with `feedback_loss_design_principles.md` Rules 1-5.

### 5.1 Primary signal: cap MAPE

```
loss_mape = mean over valid pairs/selfs of:
    |C_pred - C_golden| / C_golden.clamp(min=eps)
```

Direct MAPE (not SymMAPE) so gradient stays alive in the high-error
regime (Rule 1: "align loss with eval metric").

### 5.2 Heteroscedastic weighting

```
cap_weight = clamp(C_golden / median_C, min=0.3, max=20.0)
loss_mape_weighted = mean over pairs of  cap_weight · |C_pred - C_golden| / C_golden
```

20× max to prevent CTS-class large nets from dominating; 0.3 min so tiny
nets aren't entirely ignored (Rule 2).

### 5.3 Zero-target supervision

```
loss_zero = 0.1 × smooth_l1(
    C_pred[C_golden < eps],
    C_golden[C_golden < eps],
    beta=0.05,
)
```

Specifically NOT log-space, since `log1p(C_pred)` zero-pen vanishes
exactly where we want gradient (Rule 3).

### 5.4 KCL closure (internal consistency)

For per-net aggregation (Phase 2 path; not in Phase 1's per-pattern loss):

```
loss_kcl = smooth_l1(
    C_self_total + Σ_aggressors C_cpl,
    C_total_pred.detach()                # detach! — closure, not extra teacher
)
```

(Rule 4 — KCL pulls heads to consistency, not to a redundant teacher.)

### 5.5 Total loss

```
loss = (
    1.0 * loss_mape +
    0.5 * loss_mape_weighted +           # secondary
    0.1 * loss_zero +                    # zero-target
    0.05 * loss_smoothness +             # neighbor-pair smoothness (TBD)
)
```

No bundled changes per Rule 5 — adjust ONE coefficient per validation cycle.

---

## 6. Synthetic pretraining hookup

Pretrain the residual MLP on synthetic Stage 1 + Stage 2 data before
finetuning on real BEOL.

### 6.1 Pretrain protocol

```
Stage 1 (parallel plate, 1M samples):
    - Trivial: residual should learn multiplier ≈ 1.0 (analytic is exact)
    - Sanity: if residual moves more than 1% from 1.0, the model is
      broken — abort
    - This is the "is the wiring correct?" gate

Stage 2 Mode A (stacked dielectric, 2M samples):
    - Same: analytic is exact, residual should stay at 1.0
    - Confirms the layer-stack-aware analytic is wired correctly

Stage 2 Mode B (single interface, 2M samples):
    - Gentle correction: residual learns small (0.95-1.05) deviation
      from leading-order image-method approximation

Transfer canary (after Stage 1+2 pretrain):
    - Finetune on 500-1000 net subset of real intel22 v3 data
    - 1000 steps
    - Compare to no-pretrain control
    - K3 GATE: pretrained init must drop loss ≥50% faster than control
```

### 6.2 Stage 3+ (deferred)

Stage 3 (3D box pairs via Q3D) and Stage 4/4.5 (multi-conductor + real
density) cost ~3000 GPU-hours on commercial 3D solver. Only commit
after Stage 1-2 transfer canary passes.

---

## 7. Evaluation protocol

Phase 1 evaluation uses CNN-Cap / NAS-Cap-style benchmarks adapted to
our PDK:

### 7.1 Pattern-level eval set

- Sample 5,000 canonical patterns from intel22 v3 dataset (real BEOL
  geometry, real StarRC labels)
- Patterns drawn from layouts NOT in any training pattern
- Stratified across 5 cap-magnitude quartiles + 4 layer buckets

### 7.2 Metrics reported

Per `pex_v3/src/evaluation/metrics.py` four-column convention:
1. **Cap MAPE** (primary, target <4%)
2. **Delay error** (RC-equivalent, ResCap convention)
3. **Power error** (downstream, ParaFormer convention)
4. **RC chip-ratio percentile** (chip-level distribution match)

### 7.3 5-seed protocol

- 5 seeds × 1 model = 5 runs per ablation cell
- Mann-Whitney U + Cohen's d + bootstrap 95% CI on median
- All four metrics reported per seed
- `per_method.csv`, `mwu_pairs.csv`, `bootstrap_ci.csv` written by
  `seed_aggregator.py`

### 7.4 Strong baseline comparison (Phase 0.5 outputs)

The Phase 1 model must beat:
- B1: XGBoost on hand features
- B2: ParaGraph reproduction
- B3: legacy DeepPEX_Model on rebuilt v3 data
- B4: Compact + GAM

with statistically supported margin (p<0.05 + Cohen's d ≥ 0.5).

---

## 8. Reviewer-defensibility

Anticipated reviewer objections and our preempt:

### "Why not just use FastCap (BEM) directly?"

FastCap is a 3D field solver; runs at ~1 sec / pattern. Our paradigm
runs at <1 ms / pattern by amortizing the BEM into a learned residual
on top of analytic Green's function. We get FastCap accuracy at ParaGraph
speed.

### "Why is StarRC used as oracle when StarRC is itself a tool?"

Acknowledged in paper as a limitation. We report:
1. Synthetic pretraining stage agreement vs **closed-form analytic**
   (not StarRC)
2. Phase 3 cross-PDK validation that demonstrates the paradigm transfers
3. (Optional) Cross-validation against Q3D / FastCap for a 1000-pattern
   subset, when license available

### "Layered Green's function isn't novel"

Correct — Sommerfeld 1909, image method, FastCap 1991. Our novelty is:
1. **Hybrid**: analytic base + bounded learned residual
2. **BEOL-specific**: tuned for IC layout (Manhattan, layer-stacked)
3. **Synthetic pretraining**: closed-form curriculum that transfers
4. **Stratified error reporting**: top-metal / long-parallel buckets
   improve, not just aggregate MAPE

This stack of contributions, not any single one, is the paper.

### "Per-pattern <4% in literature already (CNN-Cap, NAS-Cap)"

Correct. Our differentiation is:
- **Architecture is hybrid + bounded + analytic-aware**, vs CNN-Cap
  pure ResNet on grid
- **Data efficiency** demonstrated via Stage 1-2 analytic pretraining
- **Compositional** — Phase 2 paper shows pattern → full-net aggregation
  achieving full-net <4% (where CNN-Cap doesn't operate)

---

## 9. Open questions for Codex round 3

Before any Phase 1 code lands, Codex deliberation on:

1. **Patch generation granularity**: target characteristic length
   `h = layer_pitch × 0.5` — is this correct vs say `pitch × 0.25` for BEM
   convergence? What's the patch count budget that keeps inference cheap?

2. **Residual bound RES_CLAMP**: is `log(2.0) = 0.693` right? Too tight =
   model can't fix big analytic errors; too loose = neural takes over.
   Should this be per-pair-type adaptive (e.g., looser for cross-layer)?

3. **Self-capacitance vs pair-coupling separation**: Phase 1 has two heads
   (`MLP_self`, `MLP_residual` for pairs). Should they share an encoder?
   If so, what's the shared representation?

4. **Loss weighting stability**: 1.0/0.5/0.1/0.05 — Codex round 1 flagged
   "don't bundle correlated changes." What's the safest single-coefficient
   sweep to validate the design?

5. **Failure-mode preemption**: BEM conditioning issues (Codex round 2 P1)
   — for our hybrid, where does the conditioning enter? The analytic part
   has no matrix solve; the residual is pointwise. So we sidestep it,
   right? Or is there a hidden conditioning issue we're missing?

6. **K3 transfer canary spec tightening**: 50% loss drop in 1000 steps is
   the proposed bar. Is this empirically correct for layered-media physics
   pretraining, or should it be 30% / 70%?

7. **Phase 1→2 interface**: Phase 1 outputs per-pair C_cpl, per-conductor
   C_self. Phase 2 needs these aggregated to per-net. What's the right
   contract — "per-pair tensors out + aggregator on top," or "richer
   intermediate (charge density σ on patches)"?

---

## 10. Acceptance criteria

Phase 1 implementation is "done" when:

- [ ] `pex_v3/src/preprocessing/mesh_v3.py` produces conductor surface meshes
- [ ] `pex_v3/src/models/hybrid_v3.py` implements the hybrid architecture
- [ ] Synthetic Stage 1 pretrain → residual stays at multiplier ≈ 1.0 (verifies wiring)
- [ ] Stage 2 pretrain → residual deviates within bounds
- [ ] Transfer canary passes K3 (≥50% loss drop in 1000 steps)
- [ ] Phase 0.5 baselines beaten on real intel22 v3 data, 5-seed MWU + Cohen's d ≥ 0.5
- [ ] Per-pattern MAPE < 4% on stratified eval set
- [ ] Stratified error report (per-quartile, per-layer, per-design)
  shows improvements distributed (not just average), specifically
  top-metal and long-parallel buckets

When all green → write paper #1 draft.

---

## 11. Implementation order (post-Codex round 3)

1. `mesh_v3.py` — conductor surface mesh utility
2. `analytic_base_v3.py` — differentiable Mode A + Mode B Green's function
3. `residual_head_v3.py` — bounded MLP residual head
4. `hybrid_v3.py` — composes analytic + residual + heads
5. `pretrain_synthetic.py` — Stage 1+2 training loop
6. `transfer_canary.py` — K3 gating script
7. `train_pattern_v3.py` — finetune on real v3 BEOL
8. `eval_pattern_v3.py` — stratified eval entrypoint
