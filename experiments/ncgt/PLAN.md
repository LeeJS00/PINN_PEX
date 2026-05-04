# NCGT (Net-Centric Graph Transformer) — Paradigm A+ Plan

_Created: 2026-05-01_
_Status: v4 — post Phase 0 audit (11 designs × 300 nets validated; aggressor cap calibrated, supervision gate passed)_
_Track: structural paradigm shift, replaces tile-cuboid + 1-hop GNN baseline_

## 0. Goal

- **Accuracy**: net MAPE ≤ **4%** (5-seed mean), R² ≥ **0.97**, p95 error ≤ **8%** on intel22 in-dist + OOD.
- **Runtime**: ≤ **5 min/small chip** (intel22-class), ≤ **3 h/10M-instance chip** on RTX A6000.
- **Scope**: NCGT backbone (Option A) + physics-guided residual head (Option B-light). Full BEM kernel deferred.

## 1. Why this paradigm + why it can hit 4%

Current baseline plateaus at 55-65% MAPE. PROJECT_REPORT.md §8 documents 4 floor evidence (slope=0.5, plateau, CPL chip ratio, outliers).

Literature precedent for ≤4% pure-ML PEX:

| Paper | Reported error | Key strategy |
|---|---|---|
| CNN-Cap (TODAES 2022) | Total <1.3%, Coupling <10% @ 99.5% | Per-window pattern, ResNet, grid representation |
| NAS-Cap (2024) | Total **0.74%**, Coupling **1.7%** | NAS + 8× geometric augmentation |
| SRAM 2-stage (GLSVLSI 2024) | 19× error reduction vs ParaGraph | 5-bin classifier + per-bin MLP, focal loss |
| ParaFormer (ASPDAC 2025) | Cap R² **0.9630**, Power MAPE **1.45%** | Heterogeneous graph + transformer + GradNorm |
| ResCap (ASPDAC 2025) | Delay **0.06%**, Power **0.16%**, no outliers | **Physics-guided linear base + ML residual** |
| DeepRWCap (arXiv 2511.06831) | 1.24% ± 0.53% | Neural-guided FRW |

**4% is achievable.** SOTA combines: physics base + residual learning + multi-bin specialization + heterogeneous types + augmentation + gradient normalization. Plan v2 incorporates all six.

## 2. Architecture

### 2.1 Primitive: conductor segment + virtual subsegments

Conductor segment between **explicit topology breaks** (via, pin, jog, branch, layer change). Long segments (>L_subdiv=4μm) split into virtual subsegments to align with SPEF *RES segmentation.

Per-segment feature (12D — same as v1):
```
[x_mid, y_mid, z, dx, dy, w, h, layer_idx, semantic_type, role,
 net_class, is_subdivision]
```

### 2.2 Heterogeneous typed embeddings (ParaFormer)

**Critical change vs v1**: `role` + `net_class` are NOT just feature fields. They route to **type-specific embedding tables**:

```
node_type ∈ {target, signal_aggr, power_VDD, power_VSS, pin, branch_node}
```

(Phase 0 finding: same-layer/cross-layer split classifier is unreliable — collapsed to `signal_aggr` until classifier bug resolved in Phase 1. Vias not enumerated as aggressors per audit.)

Each type has its own (12D → 64D) embedding MLP. Concatenated with shared geometry embedding (64D) → final 128D z_seg. Type-specific embeddings prevent representation collision between physically distinct primitives (DS-PINN macro hijack mode reappeared because all types fed single MLP — this fix breaks that path).

Layer-stack embedding (separate, joined inside encoder): `(ε_above, ε_below, etch_stop, t_metal, t_dielectric)`.

### 2.3 Per-net graph (Phase 0 calibrated)

- Targets `T_n` (incl. virtual subsegments). Cap **1K** (audit: P95=288, max=607).
- Aggressors `A_n` within `R_aggr=12μm` (3D L∞). **Vias excluded** (audit: vias = 65.6% of aggressors at R=20μm with no cap contribution).
- Aggressor cap: **4K signal + 2K power = 6K total**. Closest-distance pruning at boundary.
- Edges: `E_local` (≤4μm, all pairs), `E_mid` (4-8μm, kNN k=8 per target), `E_long` (8-12μm, top-1 by parallel-overlap per (target, aggr_net)).
- See PHASE0_AUDIT.md for cap derivation.

### 2.4 Backbone: sparse 3D attention transformer

`L=4` blocks, `d=128`, `heads=8`. Same as v1:
- Local sparse attention (R_attn=4μm, k=64).
- Long-range carried by **explicit edges**, not global token.
- Global token: **readout-only** (Phase 2 ablation gate).
- ALiBi-3D relative-position encoding.

**No batch normalization** — NAS-Cap reported BN detrimental for cap regression.

### 2.5 Physics-guided residual heads (ResCap — KEY CHANGE vs v1)

Instead of NN predicting cap directly, predict **residual** over physics base:

```
C_predicted = C_physics_base × (1 + softplus_residual_correction)
```

Where `softplus_residual_correction ∈ [-0.5, +1.0]` (capped, clamp(-0.5, 1.0)) — model can adjust 50% down or 100% up.

**GND physics base** (per segment):
```python
# Parallel-plate area + Sakurai-Tamaru fringe (well-established BEOL formulas)
C_gnd_base(s) = ε₀ · ε_eff · (
    A_top(s) / d_top(s) +              # plate to layer above
    A_bot(s) / d_bot(s) +              # plate to layer below
    P(s) · log1p(t/d) · (2/π)          # Sakurai-Tamaru fringe
)
```

**CPL physics base** (per edge):
```python
# Sakurai-Tamaru lateral or broadside formula based on layer pair
if same_layer:
    C_cpl_base(e) = ε₀ · ε_pair · L_overlap(e) · log1p(t_metal / d_lateral(e)) · (2/π)
else:
    C_cpl_base(e) = ε₀ · ε_pair · A_overlap(e) / d_vertical(e)
```

Both physics bases are differentiable (no scipy). NN learns multiplicative correction from `[z_target, z_aggr, geom, rel_pose, layer_pair_emb, z_global]`.

**Why this changes accuracy ceiling**:
- ResCap reports 0.16% MAPE on power because physics base captures 90%+ of cap, NN refines 10%.
- Data-efficient: intel22's 220K nets gives ~22M edges of supervision, which is ample for residual learning even though small for from-scratch learning.

### 2.6 Multi-bin specialized heads (SRAM 2-stage)

Single GND/CPL head can't handle 60× dynamic range (CTS 30+ fF vs avg 0.5 fF). Bin-specialized heads:

**GND bin classifier** (per segment): 5 bins by C_physics_base magnitude:
- B0: (0, 0.01) fF
- B1: [0.01, 0.1) fF
- B2: [0.1, 1) fF
- B3: [1, 10) fF
- B4: [10, ∞) fF

Each bin has its own residual MLP head: `MLP_bin_k`. At inference, bin is selected by `argmax(classifier_logits)`; at training, soft-routing with classifier as teacher (focal loss for class imbalance, α_k = 1/freq_k).

**CPL bin classifier** (per edge): 5 bins by `C_cpl_base` magnitude. Same structure.

**Loss**:
- Classification loss: focal cross-entropy on bin assignment.
- Regression loss: per-bin MAPE on the predicted bin.

Boundary smoothing: nets/edges within 10% of bin boundary get loss in both adjacent bins (avoids hard boundary error spikes).

### 2.7 Frozen head input contract (Option B compatibility)

Same as v1:

```python
@dataclass
class CPLHeadInput:
    z_target: Tensor       # (E, d)
    z_aggr: Tensor         # (E, d)
    geom_target: Tensor    # (E, 6)  [x,y,z,dx,dy, layer_idx]
    geom_aggr: Tensor      # (E, 6)
    area_target: Tensor    # (E, 1)  side-area exposed
    area_aggr: Tensor      # (E, 1)
    rel_pose: Tensor       # (E, 4)  [|Δr|, parallel_overlap, broadside_flag, layer_pair_idx]
    layer_pair_emb: Tensor # (E, 16)
    z_global: Tensor       # (E, d)
    physics_base: Tensor   # (E, 1)  C_cpl_base from §2.5
```

Future Option B (full BEM kernel) replaces `softplus(MLP(...))` with `ε₀ · A_t · A_a · G_θ(rel_pose, layer_pair)` — no upstream change.

### 2.8 Net-level aggregation + KCL closure

```
C_gnd_net = Σ_{s ∈ T} C_gnd_seg(s)
C_cpl_edge[e] = head_cpl(e)
C_cpl_net[a] = Σ_{e: aggr_net(e)=a} C_cpl_edge[e]
C_total_net = C_gnd_net + Σ_a C_cpl_net[a]

KCL: smooth_l1(C_gnd_net + Σ C_cpl_net, C_total_net.detach())
```

## 3. Data pipeline

### 3.1 New files (`experiments/ncgt/src/data/`)

- `segment_extractor.py`: DEF → segments + topology breaks + virtual subsegments + heterogeneous type assignment.
- `aggressor_index.py`: 3D KD-tree, R_aggr query.
- `edge_builder.py`: E_local / E_mid / E_long enumeration.
- `physics_base.py`: differentiable Sakurai-Tamaru / parallel-plate base computations.
- `spef_to_targets.py`: SPEF *N → segment mapping, *CAP → edge target aggregation.
- `geometric_aug.py`: 8× cap-invariant transformations (NAS-Cap).
- `ncgt_dataset.py`: PyTorch dataset with augmentation in collate.

### 3.2 Geometric data augmentation (NAS-Cap — initial 6×, expand only after Phase 0 isotropy verification)

Capacitance is invariant under symmetries that preserve the layer stack and BEOL pitch direction. **xy-isotropy is NOT assumed without verification** (Codex r2 P1 D): real BEOL has preferred routing directions per layer (M_odd horizontal, M_even vertical), and fill density / etch may break 90°-rotation invariance.

Phase 1 default (verified-safe subset, 6×):
1. Identity
2. xy-rotation 180°
3. x-reflection
4. y-reflection
5. xy-diagonal reflection
6. xy-anti-diagonal reflection

Phase 0 isotropy verification: pick 100 nets across 5 designs, run physics-base under each transform; if max(C) - min(C) > 1% relative, reject 90°/270° rotations permanently. If passes, expand to 8× (NAS-Cap full set).

Apply randomly per-batch in collate. Storage: 1×; effective training data: 6× (8× post-verification). Implementation: rotation/reflection of `(x_mid, y_mid)` and `(dx, dy)` in segment features; edges' `rel_pose` recomputed.

### 3.3 SPEF supervision mapping (line-on-wire + WIRE-preferred tie-break)

**Phase 0 audit result**: strict containment yields only 15.7% — but **WIRE-preferred tie-break recovers to 85.6%** usable per-edge supervision (well above 30% gate).

- **Line-on-wire containment**: SPEF *N node → our-segment via parametric projection onto wire centerline. Project (x, y) onto segment's p_start→p_end line; require (a) projection within [0, L] of segment, (b) perpendicular distance ≤ w/2 + 5nm tolerance.
- **Tie-break for ambiguous mapping** (multiple segments contain node): prefer WIRE > VIA, then minimum perpendicular distance.
- *CAP entries `(n1, n2, c)` → aggregate per (target_seg, aggr_seg) bin when BOTH endpoints map (containment OR tie-broken).
- `is_supervised[e] = 1` for edges with both endpoints mapped.
- Edges still failing: supervised only via net-total aggregation.

Hybrid loss: net-total always + per-edge MAPE on `is_supervised[e]==1` (~86% of edges).

OOD design `nova_f3` has 27.8% unmapped (worst, vs ~10% typical). Watch in Phase 1; if Phase 1 OOD MAPE > Phase 1 in-dist by > 5pp, may need to halve `L_subdiv` to 2μm.

### 3.4 Stratified net-level split (Codex-Q10 from v1)

Hash by `(design_name, net_name)`, stratified by 12 buckets `(net_size_bucket × cpl_dominance_bucket)`. 90/10 train/valid per bucket.

### 3.5 Dataset build

`experiments/ncgt/scripts/build_ncgt_dataset.py`:
- Output: `/data/PINNPEX/data/ncgt/<design_name>/<net_name>.pkl.gz`.
- Manifest: `dataset_manifest_ncgt.csv` with `(design_name, net_name, split, n_target_segs, n_aggr_segs, n_edges, gnd, cpl_total, edge_supervised_frac, gnd_bin, cpl_bin)`.

Estimated cost: 4-8 GPU-hours one-time, 32 workers CPU-bound.

## 4. Loss design (multi-task with GradNorm — composite tasks only)

Codex r2 P1 B: GradNorm with 12+ tasks unstable when bin-specific losses are intermittent. Solution: **composite tasks at the GradNorm interface, internal weighting per bin hand-tuned**.

```python
# Per-bin regression losses (5 GND + 5 CPL) — internal, NOT exposed to GradNorm
loss_gnd_bin_k = MAPE(pred_gnd_residual, gt_gnd_residual)[bin==k]
loss_cpl_bin_k = MAPE(pred_cpl_residual, gt_cpl_residual)[bin==k]

# Hand-aggregated composite (frequency-balanced, internal)
loss_gnd_composite = Σ_k (1 / sqrt(freq_gnd_k)) · loss_gnd_bin_k
loss_cpl_composite = Σ_k (1 / sqrt(freq_cpl_k)) · loss_cpl_bin_k

# Bin classification (focal, α_k = 1/freq_k)
loss_gnd_cls = focal_ce(gnd_bin_logits, gnd_bin_target, alpha=alpha_gnd)
loss_cpl_cls = focal_ce(cpl_bin_logits, cpl_bin_target, alpha=alpha_cpl)

# Net-level (regularization)
loss_net_gnd = MAPE(C_gnd_net_pred, C_gnd_net_gt)
loss_net_cpl = MAPE(C_cpl_net_pred, C_cpl_net_gt)

# KCL closure (Rule 4)
loss_kcl = smooth_l1(C_gnd + Σ C_cpl_net, C_total.detach())

# Zero-target supervision (Rule 3)
loss_zero = smooth_l1(pred[gt<eps], gt[gt<eps], beta=0.05)

# 6 composite tasks fed to GradNorm (ParaFormer's reported task count)
loss = GradNorm(L=[
    loss_gnd_composite,
    loss_cpl_composite,
    loss_gnd_cls + loss_cpl_cls,    # joint classification task
    loss_net_gnd + loss_net_cpl,     # joint net-total regularization
    loss_kcl,
    loss_zero,
])
```

**GradNorm**: balances gradient norms of these 6 composite tasks across shared backbone parameters. α=1.5 (ParaFormer default). Updates task weights every step. Initial 100 steps: hand-tuned weights `[3, 3, 1, 0.5, 0.1, 0.1]`, then switch to GradNorm.

## 5. Training plan

### Phase 0 — Pre-flight audit (2-3 days)

Six audits (Codex r2 P1+P2 incorporated):

1. **Geometric distributions**: segments/net, aggressors/net, edges/net, target subsegments/net, P50/P95/P99.
2. **Bin distributions per design + per layer**: gnd/cpl bin frequencies; if any bin has <100 samples in a design, merge with neighbor.
3. **Heterogeneous type counts** (Codex r2 P2 C): per-design counts for {target, signal_aggr_same/cross_layer, power_VDD/VSS, via, pin, branch_node}. Sparse types (<100 samples per design) → merge with closest type.
4. **SPEF mapping ambiguity rate** (Codex r2 P1 E): fraction of SPEF *N nodes uniquely contained in our segments. If <30% on average, increase `L_subdiv` resolution before Phase 1.
5. **Augmentation invariance numerical check** (Codex r2 P1 D): pick 100 nets across 5 designs, run physics-base under each transform; if max(C)-min(C) > 1% relative for 90°/270° rotations, restrict augmentation to 6× (4 reflections + 180°).
6. **Worst-tail memory profiling**: LDPC, CTS, PWR mesh — actual memory cost on A6000 with proposed caps.

VDD/VSS dominance per net (Codex r1 P3 Q1) folded into audit 3.

### Phase 1 — Smoke + physics-base sanity (2-3 days)

- 1-net forward + grad check.
- 10-net overfit test on gcd_f3.
- **Physics-base-only inference** (`residual_correction=0`): measure baseline MAPE. ResCap-style framing: how much does pure physics get? If 10-20% MAPE, residual learning has ample headroom.

### Phase 2 — Single-design supervised baseline (5-7 days, gated rollout)

Codex r2 P1 H: no published precedent for stacking 5 SOTA strategies. Strict gate ladder, one feature per step:

- **2.0 baseline**: gcd_f3, single GND head + single CPL head, no augmentation, no GradNorm, hand-tuned loss weights, physics-base + residual. Target: MAPE ≤ 10%, R² ≥ 0.85.
- **2.1 +residual ablation** (Codex r2 P2 A): residual-only vs residual+bins. If bins don't improve over residual-only by >2pp, simplify to no-bins.
- **2.2 +augmentation**: turn on 6× verified-safe augmentation. Expected: 1.5-2× error reduction (NAS-Cap).
- **2.3 +bins** (if 2.1 confirms benefit): 5-bin classifier + per-bin residual heads. Expected heteroscedastic improvement.
- **2.4 +GradNorm**: switch from hand-tuned to 6-composite GradNorm. Expected: training stability, <1pp MAPE change.
- **Final Phase 2 target**: MAPE ≤ 5% in-dist, R² ≥ 0.95.
- **Synthetic far-coupling ablation**: 2-segment far-coupling tests (Codex r1 Q2).
- **Gate**: if any sub-step regresses by >2pp, halt and root-cause before continuing.

### Phase 3 — Multi-design supervised (5-7 days)

- All TRAIN_DEFS, full feature set (bins + augmentation + GradNorm + physics base).
- Stratified-split val.
- Target: ≤ 8% net MAPE.

### Phase 4 — SSL pretrain (5-7 days, only if Phase 3 passes)

Physics-aligned pretexts (Codex Q7):
- Masked-edge coupling rank prediction.
- Coarse coupling load (per-segment scalar regression).
- Contrastive layer-pair matching.

Auxiliary supervised on subset with golden SPEF.

### Phase 5 — AL finetune + 5-seed (1 week)

- AL loop: net-native.
- **Selection signal**: dropout-MC variance / pred (uncertainty density), tie-break by edge_supervised_frac.
- 5-seed protocol from start.
- Target: **4% net MAPE 5-seed mean, R² ≥ 0.97**.

### Phase 6 — OOD eval + runtime profile (3 days)

- TEST_DEFS, 5-seed.
- End-to-end runtime profile.
- Pass/fail vs §0 targets.

## 6. Pattern fast-path (post-Phase 6 optional)

CNN-Cap reports <1.3% per-window. For short single-tile nets (77% of all nets per intel22), a CNN-Cap-style per-window head runs at ~0.06ms (NAS-Cap), while NCGT runs at ~5ms. Hybrid:

```python
if n_tiles(net) <= 1 and net not power:
    use CNN-Cap-style fast path
else:
    use NCGT slow path
```

Adds 1.5-3× total runtime speedup. Defer to post-target hit; if 4% achieved without fast path, fast path is pure runtime optimization.

## 7. Reused from current baseline

Same as v1: `def_parser`, `lef_parser`, `cell_parser`, `layer_parser`, `spef_writer`, `oracle`, `profiler`. Read-only imports.

## 8. What's intentionally NOT in this plan

- Full BEM kernel (replaces residual MLP with `ε₀·A·G_θ`): post-Phase 6, head-swap via frozen `CPLHeadInput`.
- Surface-patch primitive: post-target.
- Multi-PDK transfer: post-target.
- Distillation, calibration JSON, γ head, DSPINN, GINO: archived, do not reintroduce.

## 9. Risk register (post-Codex round 2)

| Risk | Severity | Mitigation |
|---|---|---|
| GradNorm instability with bin-sparse losses | **P1** (Codex r2 B) | 6 composite tasks only at GradNorm interface; per-bin weighting handled internally with `1/sqrt(freq)` |
| Augmentation isotropy assumption wrong | **P1** (Codex r2 D) | Phase 0 numerical verification; default 6× (no 90°/270° rotations) until verified |
| SPEF mapping noise pollutes per-edge gradient | **P1** (Codex r2 E) | Strict containment only, no fallback; ambiguous edges supervised at net-total only |
| Stacking 5 SOTA strategies has no precedent | **P1** (Codex r2 H) | Phase 2 gated rollout 2.0→2.4, halt-and-revise on >2pp regression |
| Physics base wrong formula | **P1** | Phase 1 physics-only baseline; if base MAPE >50%, formula revision |
| Bin classifier mis-routes by base bias | **P2** (Codex r2 A) | Phase 1 ablation residual-only vs residual+bins; boundary smoothing in loss |
| Type starvation on small designs | **P2** (Codex r2 C) | Phase 0 type count audit; sparse types (<100/design) merged before Phase 1 |
| Heterogeneous embeddings overfit | **P2** | Type-specific MLPs share final projection; dropout per type-MLP |
| Memory blow on dense nets | **P2** | Phase 0 audit; per-net caps; chunked edge eval |
| Per-net split unstable on small designs | **P2** | 12-bucket stratification |
| Global token hijack | **P2** | Readout-only; Phase 2 ablation gate |

## 10. Deliverables

- [x] PLAN.md v2 (this).
- [ ] PHASE0_AUDIT.md.
- [ ] segment_extractor.py + tests.
- [ ] aggressor_index.py + tests.
- [ ] edge_builder.py + tests.
- [ ] physics_base.py + tests (Sakurai-Tamaru, parallel-plate; differentiable).
- [ ] spef_to_targets.py + tests.
- [ ] geometric_aug.py + tests (8× invariance verified numerically).
- [ ] ncgt_dataset.py.
- [ ] ncgt_model.py (encoder, backbone, bin-specialized residual heads).
- [ ] gradnorm.py.
- [ ] train_ncgt.py.
- [ ] build_ncgt_dataset.py.
- [ ] smoke_test_ncgt.py.
- [ ] RESULTS.md (Phase 0-6 logs).

## 11. Out-of-scope guardrails

- Do not touch `src/`, `configs/config.py`, `run_active_learning.py`, `scripts/build_dataset*.py` — other sessions' surface.
- Reuse via read-only imports.
- Outputs land in `output_intel22/ncgt/`.

## 12. Literature anchors (Plan v2 SOTA basis)

- **CNN-Cap** (TODAES 2022, arXiv 2107.06511): Total <1.3%, Coupling <10% — per-window field-solver replacement.
- **NAS-Cap** (arXiv 2408.13195): Total 0.74%, Coupling 1.7% — 8× geometric augmentation, no BN.
- **SRAM 2-stage** (GLSVLSI 2024, arXiv 2507.06549): 19× error reduction — 5-bin classifier + per-bin MLP, focal loss.
- **ParaFormer** (ASPDAC 2025): R² 0.9630 cap, MAPE 1.45% power — heterogeneous graph + transformer + GradNorm.
- **ResCap** (ASPDAC 2025): Delay 0.06%, Power 0.16%, no outliers — physics base + ML residual, 215× speedup.
- **DeepRWCap** (arXiv 2511.06831): 1.24% ± 0.53% — neural-guided FRW.
- **GNN-Cap** (RG 375573477): chip-scale GNN.

Plan v2 incorporates **all six core SOTA strategies** (residual learning, multi-bin specialization, geometric augmentation, heterogeneous types, GradNorm, no-BN).
