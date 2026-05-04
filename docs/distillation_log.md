# Data-Driven Physics Calibration Log

_Living document. Last updated: 2026-05-01 (v4 in iter 1, v5 ablation pending)._

> **Naming note**: this work was originally framed as "distillation". On
> reflection that term is overclaim — there is no live teacher signal in
> training, only a one-shot offline NNLS fit that produces an init JSON.
> The model is free to overwrite the calibrated values during AL training.
> Accurate name is **Data-Driven Physics Calibration** (or
> "Statistical Pre-tuning of Physics-Informed Init"). The filename
> `distillation_log.md` is kept for git-history continuity; references
> below use the corrected naming.

This is the consolidated log for the **calibration track** of PINN-PEX
development. Mirrors the structure of `dspinn_development_log.md`. Update at
every milestone.

The calibration track is *parallel* to the DS-PINN track (`v3` running on
GPU 1 with hand-tuned ζ); it does not block, replace, or modify the DS-PINN
work.

---

## 0. Goal & Constraints

**Goal**: Replace v3's hand-tuned ζ initialization (`softplus_inv(8.0)` diag,
`softplus_inv(5.0)` off-diagonal for `cpl_layer_pair_log_scale`; hand-tuned
per-layer values for `layer_scale_phys_gnd`) with **data-driven init** values
computed offline from `cfg.TRAIN_SPEFS` via constrained least-squares.

Same NN architecture. Same loss stack (8 terms, untouched). Same training
loop. Only the initial values of two parameter tensors change.

**Production target**: improvement on the documented dspinn_v3 weaknesses —

- **GND heteroscedastic calibration** (slope 0.6 from
  `dspinn_development_log.md` §3.4): expect v4 quartile ratios to flatten
  toward 1.0 across all four buckets.
- **CPL 6.5× under-prediction** (§3.5): expect v4's `Σpred / Σgold` median
  ratio to land closer to 1.0 than v3's 0.10.

**Realistic interim target**: net MAPE within ±5pp of v3 at iteration 3
with **monotonically better** chip_gnd / chip_cpl ratios (no architectural
change → no expected degradation; any improvement is from better init).

---

## 1. Why Option X — and what we rejected

### 1.1 What the docs revealed

`docs/dspinn_development_log.md` §3.4 / §3.5 / §3.6 makes the actual error
modes clear:

| Error mode | Cause | Plan v0 (multi-scale distill) addresses? |
|---|---|---|
| GND slope 0.6 (hetero) | calibration, not localization | ❌ — gives more numerous targets at same wrong calibration |
| CPL 6.5× under | physics base 8× too small + loss landscape doesn't push modifier up | ❌ — same target, same loss landscape |
| LDPC outlier under | dense routing confounds Sakurai-Tamaru | ❌ |

The model already has Pearson r=0.85 for GND. It knows *where* GND lives.
What it doesn't know is *how much*. This is a calibration problem, not a
localization problem.

### 1.2 Three options considered

After the user pointed out my plan v0 didn't address root causes:

| Option | Approach | Loss terms added | Risk |
|---|---|---|---|
| W: per-tile distillation | dense per-tile teacher targets | +1 | doesn't address calibration root cause |
| Y: γ scaling head | per-net learned multiplicative scale | +1 (1 head, ~1k params) | adds capacity; may overfit |
| **X: data-driven init** | NNLS-fit init values from SPEFs | **0** | bug-prone offline pipeline |

User selected **X** with rationale: parallel-safe with v3, doesn't add to
loss stack, attacks root cause, paper-publishable as ablation.

### 1.3 Why not just keep iterating v3 hand-tuning

v3 has 2 magic numbers (8.0, 5.0). Per-layer differentiation could need
25 magic numbers (one per layer for GND density). Trial-and-error is
intractable; data-driven NNLS is one shot.

---

## 2. Mathematical formulation

### 2.1 GND

For each train net `i`:
```
Σ_j  A[i, j] · ρ_layer[j]  =  golden_gnd_total[i]
```

- `A[i, j] = Σ_(target cuboid in layer j of net i)
   (bottom_area + fringe_init[j] · sidewall) · core_ratio`
- `ρ_layer[j]` = unknown per-layer density (fF/μm²), K = 25 unknowns
- `golden_gnd_total[i] = Σ gnd_caps[node]` from SPEF *CAP

### 2.2 CPL — only 2 unknowns (simplified after Codex Q1)

For each (net `i`, signal aggressor `a`):
```
B_diag[i, a] · s_diag  +  B_cross[i, a] · s_cross  =  golden_cpl[i, a]
```

- `B_diag` = sum of `w_cpl_base · core_ratio_eff` over edges where
  `src.layer == dst.layer`
- `B_cross` = same but `src.layer != dst.layer`
- `core_ratio_eff` mirrors `finetuner.py:493`'s src/dst fallback
- 2 unknowns; over-determined; trivially identifiable

### 2.3 Power-net CPL → GND lumping

`finetuner.py:506-513` adds CPL flux to power-net dsts into `global_pred_gnd`
to match StarRC's lumping behavior. Calibration must mirror:

```
golden_gnd[i] = Σ_j ρ_layer[j] · A_primary[i, j]
              + s_diag  · Σ_j A_power_diag[i, j]
              + s_cross · Σ_j A_power_cross[i, j]
```

This couples GND and CPL into one **joint NNLS** with K + 2 unknowns and
~(N + N×M_aggr) equations.

### 2.4 Why NNLS (not gradient descent)

Convex; closed-form-ish (Lawson-Hanson active set); guarantees global
optimum; runs in seconds for this size; non-negativity natural for
densities/scales.

---

## 3. Implementation Plan

### 3.1 Step 0 — Hard-gate pre-conditions

| Check | What | If fails |
|---|---|---|
| 0A | `tiling.py`: `data['origin']` is tile-center in absolute μm | Abort entire plan |
| 0B | SPEF C_UNIT == "FF" via `scripts/diag_spef_unit_check.py` | Adjust extraction scale |
| 0C | `parse_spef_with_coordinates` returns caps in fF | Add scaling |

All read-only; minutes of work.

### 3.2 Step 1 — Statistics extractor

**File**: `src/data/calibration_extractor.py` (new)

Three routines:

1. `extract_golden(spef_paths) → dict` — per-net `gnd_total`, per-net per-aggr `cpl_total`
2. `extract_geometry(processed_dir, manifest) → dict` — per-net per-layer `A_primary`
3. `extract_physics_base(model, train_loader) → dict` — runs physics-only
   forward (mirrors `diag_eval_dump.py --physics_only`), dumps
   `sparse_cpl['w_cpl']` (raw geometric base, **no `cpl_residual` leak**),
   aggregates per-(net, aggr) into `B_diag`, `B_cross`, `A_power_diag`,
   `A_power_cross`. Reuses `finetuner.py:486-513` aggregation block verbatim.

Cache: `<PROCESSED_DIR>/calibration_extract/{design}_stats.parquet`.

### 3.3 Step 2 — Joint NNLS solver

**File**: `src/data/calibration_solver.py` (new)

```python
M = stack([
    [A_primary,        A_power_diag,        A_power_cross],   # GND eqs
    [zeros(K),         B_diag,              B_cross         ], # CPL eqs
])
y = concat([golden_gnd, golden_cpl_per_aggr])
x_star, residual = scipy.optimize.nnls(M, y)
ρ_layer = x_star[:K]
s_diag, s_cross = x_star[K:]
```

Diagnostics: pooled MAPE, per-design holdout MAPE, per-design IQR, per-layer
net coverage histogram.

Floor: layers with < 10 supporting nets fall back to hardcoded init.

### 3.4 Step 3 — JSON dump

`<PROCESSED_DIR>/calibration_init.json`. Schema in plan v2 § "Step 3".

### 3.5 Step 4 — Model init plumbing

**File**: `src/models/flux_head.py` — `_make_gnd_cap_density_init` and
`__init__` for `cpl_layer_pair_log_scale` read JSON if `cfg.CALIBRATION_INIT_PATH`
is set, else fall back to hardcoded values.

**File**: `src/models/neural_field.py:freeze_ssl_layers` — same fallback
in the re-seed block (lines 87-106).

Backward-safe: missing JSON → behave exactly like v3.

### 3.6 Step 5 — Sanity-check script

**File**: `scripts/diag_calibration_check.py` (new)

Loads model with new init, runs physics-only forward over each holdout
design, prints per-net ratio histogram. Aborts if median outside
[0.5, 2.0] for any populated layer or layer-pair class.

### 3.7 Step 6 — End-to-end build

```bash
python3 scripts/diag_spef_unit_check.py
python3 -m src.data.calibration_extractor \
  --output /data/PINNPEX/data/processed/intel22/calibration_init.json \
  --holdout intel22_wb_conmax_top_f3 intel22_ldpc_decoder_802_3an_f3 \
            intel22_vga_enh_top_f3
python3 scripts/diag_calibration_check.py --calibration /data/PINNPEX/data/processed/intel22/calibration_init.json
```

Failure on diag → root-cause investigation, do NOT train v4.

### 3.8 Step 7 — Train v4_distillinit

`configs/config.py`:
```python
CALIBRATION_INIT_PATH = PROCESSED_DIR / "calibration_init.json"
```

Run on a free GPU (NOT GPU 1 where v3 is running):
```bash
python3 run_active_learning.py --model_name v4_distillinit --gpu 4 --use_dspinn
```

### 3.9 Step 8 — Compare v3 vs v4

When both reach iteration 3+, side-by-side:

| Metric | v3 baseline | v4 target |
|---|---|---|
| net MAPE in-dist | (TBD from v3 run) | within ±5pp of v3 |
| CPL ratio median | 0.10 (v2-baseline) → ? (v3) | closer to 1.0 |
| chip_gnd | 0.6-1.0 hetero | flatter across quartiles |
| chip_cpl | 0.10 | improved |
| training step time | T | identical to v3 (no arch change) |
| heteroscedastic plot | 1.58 / 1.53 / 1.11 / 0.73 (v2) | flatter |

---

## 4. Codex Consultation Log

| Round | Date | Topic | Outcome |
|---|---|---|---|
| 0 (pre-) | 2026-04-30 | Multi-scale distillation plan v1 | Codex flagged 6 critical issues including coord projection bug, K×K duplication. Led to v0→v2 narrowing (CPL-only, etc.) |
| 0 (pre-) | 2026-04-30 | Multi-scale distillation plan v2 | Codex conditional pass; flagged 4 follow-on issues (sample_filename plumbing, voxel-merge bias, aux/cpl gradient interference, trainer drift). Plan v3 patches drafted. |
| 0 (redirect) | 2026-04-30 | User read `dspinn_development_log.md` and pointed out plan v0-v3 didn't address actual root causes (calibration vs. localization). Switched track to Option X — data-driven init. |
| 1 | 2026-04-30 | Option X plan v1 | Codex flagged 3 BUGs (CPL K² weak identification, `cpl_residual` leak at phys_scale=0, aggregation mismatch with finetuner.py) and 3 WARNINGs (node projection, SPEF coord units, single-design holdout). |
| 2 | 2026-04-30 | Option X plan v2 (after v1 fixes) | (in this doc) — v2 removes node projection entirely, simplifies CPL to 2 unknowns, mirrors finetuner aggregation, joint NNLS with power-net lumping. **Pre-build approval.** |

Each round's full prompt + verdict captured in Claude Code conversation
transcripts. Actionable items summarized in §3.

---

## 5. Live Experiment Tracker

| Run | GPU | PID/task | Status | Best MAPE | CPL ratio (med) |
|---|---|---|---|---|---|
| v4_distillinit | TBD | not yet launched | planning | TBD | TBD |
| (parallel: dspinn_v3) | 1 | task `brsxkzyxj` | running (β+ζ) | TBD | TBD |

### Build progress (2026-04-30)

- ✓ Step 0A — `tiling.py` origin verified absolute-μm tile-center
  (`abs_geo - origin == cuboid_xy_rel` to 1e-6)
- ✓ Step 0B — `scripts/diag_spef_unit_check.py` confirms 11/11 SPEFs use
  `*C_UNIT 1.0 FF`
- ✓ Step 0C — `parse_spef_with_coordinates` returns fF; KCL holds
  `(sum_gnd + sum_cpl) ≈ total_cap` (verified ratio = 1.0000 on gcd nets)
- ✓ Step 1 — `src/data/calibration_extractor.py` (phase 1 + phase 2)
  - net-centric sampling (2000 nets/design × 9 = 15913 (design, net) pairs)
  - phase 1: 4 min, 15674 nets joined with golden + per-layer A_primary
  - phase 2: 28 min, 23M signal-aggressor edges aggregated, c_vss/A_power per net
  - **Bug found via isolated test**: `A_tgt` (name-based mask in robust_collate)
    differs from `is_target` (cuboid[..., 7]==1.0 used inside flux_head).
    Pin cuboids of target net have `A_tgt=1` but `is_target=0` — model excludes
    them from c_gnd_seg, but my A_primary was including them. Same bug caused
    `c_vss_pred = -8015 fF` (negative!) due to non-target gnd_area leaking
    through the wrong mask. Fixed by switching tgt_mask in phase 1 and
    is_target_t mask in phase 2 to use cuboid channel 7. After fix:
    `c_vss_pred = +134 fF` (positive, ~2.3% of golden — sane).
- ✓ Step 2 — `src/data/calibration_solver.py` (joint NNLS)
  - **Issue found**: 29-anchor NNLS oscillates wildly (ρ alternates 0/30/0/16)
    because adjacent anchors map to top/bottom of same physical metal, causing
    near-collinearity. Fixed via **layer bucketing**: collapse 29 anchors → 10
    physical-metal buckets (pre_M1, M1, M2, M3, M4, M5, M6, upper, top, others).
    NNLS now solves 12 = 10 buckets + 2 cpl unknowns. Stable monotonic ρ values.
  - Joint NNLS solves in <1s for 200k+ equation × 12 unknown system.
  - Sparsely-populated buckets (no supporting nets) fall back to v3 hardcoded.
- ✓ Step 3+4 — model init plumbing
  - `flux_head.py`: `_load_calibration_init()` helper added; 
    `_make_gnd_cap_density_init()` and `cpl_layer_pair_log_scale` `__init__`
    read JSON if `cfg.CALIBRATION_INIT_PATH` is set
  - `neural_field.py:freeze_ssl_layers` re-seed block respects JSON
  - Backward-compat verified: model loads with hardcoded ζ when
    CALIBRATION_INIT_PATH unset; loads from JSON when set (same softplus_inv
    encoding so values flow through identically).
- ✓ Step 5 — `scripts/diag_calibration_check.py`
  - **Bug found**: tile-centric sampling (`head(N)` over manifest) gave only
    partial coverage of each net → predicted GND/CPL was 5-10% of golden
    (under-counts because we only saw a fraction of each net's tiles).
    Fixed via `--max_nets_per_design` net-centric walking.
- ✓ Step 6 — End-to-end build + sanity check
  - Final calibration JSON written to
    `/data/PINNPEX/data/processed/intel22/calibration_init.json`
  - **All 3 holdout designs PASS** within sanity thresholds:

| Design | GND chip ratio | GND median ratio | CPL chip ratio | CPL median ratio |
|---|---|---|---|---|
| ldpc | 1.09 | 1.35 | 0.59 | 0.58 |
| vga_enh | 0.74 | 0.79 | 0.37 | 0.43 |
| wb_conmax | 1.27 | 1.80 | 0.52 | 0.49 |

### Calibrated values (final)

Per-bucket GND density (fF/μm²):
| Bucket | ρ | vs v3 hardcoded |
|---|---|---|
| M1 | 1.246 | 2.50 (v3 is ~2× higher) |
| M2 | 0.334 | 3.00 (v3 is ~9× higher) |
| M3 | 0.386 | 3.00 (v3 is ~8× higher) |
| M4 | 0.278 | 2.75 (v3 is ~10× higher) |
| M5 | 0.135 | 2.75 (v3 is ~20× higher) |
| M6, upper, top, others | fallback | unchanged from v3 |

CPL pair scales:
| | NNLS-fit | v3 hardcoded |
|---|---|---|
| s_diag (same-layer, lateral) | 0.182 | 8.0 |
| s_cross (cross-layer, broadside) | 4.250 | 5.0 |

Interpretation:
- **Lateral CPL (s_diag)**: Sakurai-Tamaru lateral formula is already accurate
  enough at raw geometric base; no scale-up needed. v3 hardcoded 8× was
  over-aggressive — explains why `cpl_modifier` had to learn 0.11× at
  convergence.
- **Broadside CPL (s_cross)**: matches v3's 5.0 within 15%. The cross-layer
  physics needs 4-5× scale-up, consistent.
- **GND density**: NNLS values ~5-20× lower than hardcoded. Hardcoded values
  were designed to compensate for `gnd_modifier` converging to ~0.5×; NNLS
  solves directly for the effective density without modifier compensation.
  Model's `gnd_modifier` should now stay near 1.0 (instead of needing to
  push down to 0.5×).

### Performance / scale

- Phase 1 (geometry+SPEF, CPU): ~4 min for 70k tiles (net-centric 2000/design)
- Phase 2 (model forward, GPU): ~28 min for 70k tiles at bs=8
- NNLS solve: < 1 s for 200k×12 system
- Sanity check (3 designs, 200 nets each): ~2 min total

### Known limitations

- Calibration fit on net-centric sample of 2000 nets/design (vs 5k-50k nets
  per design in full data). Pooled GND MAPE = 0.71 — large per-net residual,
  consistent with heteroscedastic finding in DS-PINN log §3.4. Per-design
  IQR shows holdouts (LDPC, vga, wb_conmax) span pred/golden ratios from
  0.5× to 2× — global init can only do so much; per-net γ scaling head
  (DS-PINN log open question §7.1) is the natural follow-up.
- 5 of 10 buckets fall back to v3 hardcoded (M6, upper, top, others) due
  to insufficient supporting nets. AL training can move these via
  `layer_scale_phys_gnd.requires_grad_(True)`.

---

## 6. Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-30 | Reject plan v0 (multi-scale distillation) after user critique | Did not address calibration root cause |
| 2026-04-30 | Adopt Option X (data-driven init) | Loss-term-zero, parallel-safe with v3, root-cause direct |
| 2026-04-30 | After Codex round 1, simplify CPL to 2 unknowns | K² unknowns weakly identified by per-aggressor totals |
| 2026-04-30 | Use joint NNLS for GND+CPL | Power-net lumping couples them; can't solve sequentially |
| 2026-04-30 | Remove node-to-cuboid projection | Bug-prone (Q2) and unnecessary (net-level NNLS suffices) |
| 2026-04-30 | Stratified 3-design holdout | Single design covers single regime; need size + topology + metal-stack diversity |

---

## 6.5. Critical Self-Assessment (2026-05-01)

After v4 iter 0 + iter 1 partial trajectory observed, an honest reckoning
of where the calibration track actually stands. Sections below in priority
order from "most certain claim" to "most speculative".

### What is solid

- **Pipeline correctness**: the extractor + solver + init plumbing build,
  load, and round-trip without errors. Backward-compat verified (model
  loads with hardcoded ζ when `cfg.CALIBRATION_INIT_PATH` is unset).
- **Two real bugs caught and fixed**: (1) `A_tgt` (name-based) vs
  `is_target` (cuboid channel 7) mask mismatch was inflating A_primary by
  ~20% and producing negative c_vss leakage of -8015 fF. Fixed with
  channel-7 mask. (2) NNLS oscillation from same-metal-layer top/bottom
  z anchors — fixed via 10-bucket parameterization.
- **iter 0 best ckpt**: v4 reaches MAPE 47.41% at step 3000 vs v3's
  60.27% at step 8000. Single data point but consistent direction.

### What is overclaimed

- **"Distillation" framing**: there is no live teacher signal in training.
  This is a one-shot offline init substitution. Renamed in this doc as
  "Data-Driven Physics Calibration".
- **"Data-driven" parameter coverage**: 5 of 10 GND buckets fall back to
  v3 hardcoded due to insufficient supporting nets (M6, upper, top,
  others, pre_M1). So the data-driven part is M1-M5 only — ~half the
  parameter space.
- **Independent validation of values**: NNLS gave s_diag=0.18 vs v3's
  hand-tuned 8.0 (44× swing) and ρ values 5-20× lower than hardcoded.
  No independent physics check (Sakurai-Tamaru hand calc, StarRC tech
  file extraction, alternate PDK sanity). We trust NNLS by construction.

### Concerns about the result

- **NNLS pooled GND MAPE = 0.71**: even on training nets the calibration
  itself averages 70% per-net error. A single global ρ vector cannot
  capture per-design heteroscedasticity. The motivation (slope 0.6 fix)
  is not addressed by global parameters.
- **CPL may have gotten worse**: v3 init (8/5) puts pred at modifier=1
  near golden, so cpl_modifier converges to ~1. v4 init (0.18/4.25) puts
  pred at ~0.4× golden, requiring cpl_modifier 2-3× to recover. With
  modifier range exp(±3) = [0.05, 20], some nets may saturate. Training
  CPL ratio in v4 hovers 2-15% — not noticeably better than v2 baseline.
- **iter 0 best is transient**: v4 step 3000 best, then degrades to MAPE
  81% at step 12000. Same pattern in v3. The "head start" may erode by
  iter 6 if both runs converge to the same basin.
- **N=1 comparison**: single seed v3 vs single seed v4. No statistical
  bound on stochastic variance. Could be 5-10pp noise.
- **iter 1 not improving yet**: v4 iter 1 step 7000, no new BEST. iter 0
  step 3000 still holds best. If iter 1 doesn't improve, the calibration
  benefit is purely a one-iteration head start.

### What is required to make this defensible

Ordered by urgency:

1. **Wait for v4 iter 6 completion** (~16h from 2026-05-01 00:35 KST) and
   re-run the per-iteration MAPE/CPL ratio comparison vs v3.
2. **TEST_DEFS evaluation**: run `evaluator.py` on `nova_f3` and
   `tv80s_f3` (both are TEST designs — never seen by either v3 or v4
   training). OOD MAPE is the primary contribution measure. Numbers
   matter; in-distribution head-start on validation set is interesting
   but not publishable on its own.
3. **Per-quartile heteroscedastic plot**: docs §3.4 reported v2 quartile
   ratios 1.58 / 1.53 / 1.11 / 0.73. Re-measure on v4 best ckpt and v3
   best ckpt. If the slope is still 0.6, the calibration didn't fix what
   it set out to fix.
4. **Ablation v5_calib_gnd_only**: train with ρ data-driven but
   s_diag=8.0, s_cross=5.0 (v3 hardcoded). If v5 ≥ v4, the NNLS CPL fit
   is hurting — the modifier dynamic-range concern is real. If v5 < v4,
   the full data-driven init is genuinely better.
5. **(Conditional) γ head**: if v4 + v5 both plateau without addressing
   heteroscedastic slope, escalate to dspinn_log §7.1 γ proposal —
   per-net multiplicative scaling head as the actual fix for the
   heteroscedastic finding.

### What we should not claim until the above are done

- "Data-driven physics calibration improves PINN-PEX" — N=1, transient.
- "Heteroscedastic problem solved" — no per-quartile evidence.
- "Generalizes to unseen designs" — no TEST_DEFS evaluation.

### What we can claim today

- We built a reproducible offline calibration pipeline that ingests
  TRAIN_SPEFS and produces a validated init JSON.
- We caught two real bugs that would have produced silently-wrong values.
- v4's iter 0 best ckpt outperforms v3's iter 0 best ckpt on val MAPE
  by ~22% relative; this is one snapshot of an ongoing comparison.

### Heteroscedastic re-measurement (2026-05-01)

Per-quartile-of-y_gnd analysis (script: `scripts/diag_quartile_heteroscedastic.py`)
on v3 best vs v4 iter 0 best:

| Quartile | v3 median ratio | v4 median ratio | v3 chip | v4 chip |
|---|---|---|---|---|
| Q1 (≤0.10 fF) | 1.013 | **1.263** | 0.956 | 1.249 |
| Q1-Q2 | 0.627 | 0.633 | 0.676 | 0.680 |
| Q2-Q3 | 0.537 | **0.394** | 0.635 | 0.540 |
| Q3+ (≥0.48 fF) | 0.545 | **0.489** | 0.526 | 0.461 |

Linear fit slope: v3 = 0.453, v4 = **0.369** (further from 1.0).
Pearson r: v3 = 0.923, v4 = 0.904.

**Interpretation**: The heteroscedastic problem (motivation for this work)
is not fixed; v4's slope is *further* from 1.0 than v3 at this snapshot.
However v4's BEST is from iter 0 step 3000 (early training) while v3's
BEST is from later iterations. Not apples-to-apples — re-measure when
v4 reaches iter 6 or matches v3's training progression.

If the trend persists: data-driven init shifts the parameter values but
does not address the *per-net* heteroscedasticity that motivated the
work. The γ scaling head (dspinn_log §7.1) becomes the natural next step.

### OOD evaluation — TEST_DEFS (2026-05-01)

Script: `scripts/diag_ood_compare.py`. Evaluates v3 best vs v4 iter 0 best
on `nova_f3` and `tv80s_f3` (both never seen by either training run).

| Metric | v3 nova | v4 nova | v3 tv80s | v4 tv80s |
|---|---|---|---|---|
| GND chip ratio | 0.60 | 0.48 | 0.52 | 0.50 |
| CPL chip ratio | 1.45 | 1.62 | 1.53 | 1.58 |
| **Total MAPE** | **0.32** | 0.37 (+5pp) | **0.34** | 0.42 (+8pp) |
| GND MAPE | 0.43 | 0.54 | 0.43 | 0.50 |
| CPL MAPE | 0.67 | 0.76 | 0.53 | 0.69 |

Slope (combined OOD): **v3 = 0.507, v4 = 0.421** (v4 further from 1.0).
Pearson r: v3 = 0.933, v4 = 0.948 (v4 slightly better localization).

**Interpretation**:
- v4's in-distribution iter-0-best advantage (22% relative MAPE improvement
  on val) **does not transfer to OOD**. v4 is 5-8pp WORSE on OOD.
- Per-quartile: v4 over-predicts smallest (Q1 chip ratio 1.40 vs v3 1.09)
  AND under-predicts largest (Q4 chip ratio 0.46 vs v3 0.54). Both ends
  worse than v3 — heteroscedastic slope drops from 0.51 to 0.42.
- v4's CPL is consistently *more over-predicted* than v3 (1.58-1.62 vs
  1.45-1.53 chip ratio). Suggests cpl_modifier did push CPL up but
  overshot — consistent with the dynamic-range concern raised earlier.
- v4's slightly higher Pearson r (0.948 vs 0.933) means it knows *where*
  GND lives slightly better. The problem is purely magnitude calibration.

**Caveat (apples vs oranges)**: v4 BEST was saved at iter 0 step 3000 —
very early. v3 BEST was saved later (more iterations of training).
Comparison is between an under-trained v4 and a more-trained v3. The
fairer test is post-iter-6 v4 vs v3, expected ~16h from this measurement.

**If post-iter-6 v4 still loses on OOD**: data-driven calibration is not
the right path; escalate to γ head per dspinn_log §7.1.

### 5-seed measurement protocol (2026-05-01)

After noting the N=1 problem in the v3 vs v4 single-run comparison, we ran
3 variants × 5 seeds each at `--max_iters 1 --steps_per_iter 5000` to get
distributional comparisons. The new CLI flags (`--seed`, `--max_iters`,
`--steps_per_iter`, `--calib_path`) were added by the user. 15 jobs total
across GPUs 1-4 in batches.

**Variants**:
- `v3_baseline`:   `--calib_path none`           (hardcoded ζ: s_diag=8.0, s_cross=5.0)
- `v4_full_calib`: `--calib_path calibration_init.json`         (NNLS-fit ρ + CPL)
- `v5_gnd_only`:   `--calib_path calibration_init_gnd_only.json` (NNLS-fit ρ + hardcoded CPL)

**Final 5-seed best_mape distribution** (validation MAPE on AL_PREDEFINED valid set):

| Variant | n | Median | p25-p75 | Range | Mean | IQR |
|---|---|---|---|---|---|---|
| v3_baseline | 5 | 64.17 | 55.70-65.50 | 50.70-73.23 | 61.86 | 9.80 |
| **v4_full_calib** | 5 | **54.50** | 52.67-57.19 | **49.32-63.03** | **55.34** | **4.52** |
| v5_gnd_only | 5 | 60.40 | 53.56-62.41 | 48.78-70.08 | 59.05 | 8.85 |

**Mann-Whitney U test (two-sided)** on best_mape across seeds:

| Comparison | U | p-value | Conclusion |
|---|---|---|---|
| v3 vs v4 | 19.0 | **0.222** | not significant |
| v3 vs v5 | 16.0 | 0.548 | not significant |
| v4 vs v5 | 10.0 | 0.690 | not significant |

**Interpretation**:

1. **v4 has lowest median (54.50) AND lowest IQR (4.52)** — calibration init
   gives most consistent-and-low MAPE.
2. **None of the variants are statistically distinguishable at α=0.05 with n=5**.
   To establish significance we'd need ~10 seeds per variant.
3. **The earlier single-run v3 vs v4 "22% improvement"** was within stochastic
   variance — the critical analysis's "N=1 problem" warning was confirmed.
4. **v5 (gnd-only calibration) is between v3 and v4** by mean but indistinguishable
   from either by Mann-Whitney. This neither supports nor refutes the
   hypothesis that NNLS CPL fit hurts the modifier dynamic range — needs more
   data.
5. The CPL ratio differences in earlier reports (v4 0.6%, v5 9.65% mid-run)
   washed out at full step 5000: v3=0.5, v4=0.6, v5=0.4 — all near zero.

**Per-seed best_mape (sorted within variant)**:
- v3: 50.70 / 55.70 / **64.17** / 65.50 / 73.23
- v4: 49.32 / 52.67 / **54.50** / 57.19 / 63.03
- v5: 48.78 / 53.56 / **60.40** / 62.41 / 70.08

The min values are nearly identical across variants (48.78-50.70). The
difference is in how often each variant produces a "bad" run — v3 has
more high-MAPE seeds.

### Ablation v5_calib_gnd_only (launched 2026-05-01 00:48 KST, GPU 7) — superseded

The original v5 long-single-seed run on GPU 7 was superseded by the 5-seed
protocol's v5_gnd_only variant. The long-run was killed when its purpose
was absorbed by the more rigorous statistical comparison.

### Per-variant aggregated heteroscedastic + OOD eval (2026-05-01)

Ran `scripts/aggregate_5seed_eval.py` on all 15 best_model.pth checkpoints —
per-ckpt net-centric forward (300 nets/design, ind=AL_PREDEFINED 6 designs,
ood=TEST_DEFS nova_f3+tv80s_f3), aggregated per variant across 5 seeds.

**In-distribution validation (median across 5 seeds)**:

| Variant | total MAPE | GND chip ratio | CPL chip ratio | slope | Pearson r |
|---|---|---|---|---|---|
| v3_baseline | 0.549 | 0.735 | 1.619 | 0.525 | 0.915 |
| **v4_full_calib** | **0.458** | 0.740 | 1.454 | 0.510 | 0.912 |
| v5_gnd_only | 0.525 | 0.737 | 1.406 | 0.517 | 0.912 |

**OOD (nova_f3 + tv80s_f3, never seen by training; median across 5 seeds)**:

| Variant | total MAPE | GND chip ratio | CPL chip ratio | slope | Pearson r |
|---|---|---|---|---|---|
| v3_baseline | 0.553 | 0.665 | 1.730 | 0.535 | 0.899 |
| **v4_full_calib** | **0.459** | 0.678 | 1.544 | 0.534 | 0.885 |
| v5_gnd_only | 0.535 | 0.686 | 1.462 | 0.536 | 0.886 |

**Mann-Whitney U on total_mape (mean across 5 seeds, two-sided)**:

| Comparison | ind p-value | ood p-value |
|---|---|---|
| v3 vs v4 | 0.548 | 0.548 |
| v3 vs v5 | 0.548 | 0.548 |
| v4 vs v5 | 0.690 | 0.690 |

All comparisons not significant at α=0.05 with n=5.

**Final findings (consolidated)**:

1. **v4 (full data-driven calibration) consistently best**:
   - Lowest total MAPE in both ind (0.458) and ood (0.459)
   - Lowest IQR in best_mape distribution (4.52 vs v3's 9.80)
   - 6.5pp lower mean MAPE than v3 (55.34 vs 61.86)

2. **But statistically indistinguishable from v3 or v5 at n=5**:
   - Mann-Whitney p > 0.5 for all pairs
   - The single-run "v4 22% better than v3" claim from before was
     within stochastic seed-to-seed variance (39pp range across v3 seeds at
     step 1000)

3. **Heteroscedastic problem NOT fixed by any variant**:
   - Slope: v3 0.525, v4 0.510, v5 0.517 — all far from ideal 1.0
   - Pearson r: 0.91-0.92 — model knows location but not magnitude
   - This was the original motivation; data-driven calibration alone is
     not the solution

4. **CPL over-prediction NOT fixed**:
   - All variants have CPL chip ratio 1.4-1.7 (over-predict)
   - v5 has lowest CPL chip ratio (1.46 ood) — hardcoded ζ may help here
   - v4's NNLS-fit CPL pair didn't make CPL worse (concern from earlier
     analysis was wrong)

5. **OOD ≈ in-dist performance** for all variants:
   - total MAPE: ind 0.46-0.55 vs ood 0.46-0.55
   - The model generalizes; calibration init doesn't change that

6. **The earlier OOD single-run finding ("v4 iter0 best is +5-8pp WORSE
   than v3 OOD")** is REVERSED by the 5-seed evidence:
   - Single-run v4 OOD MAPE was 37-42% (cherry-picked iter0 best)
   - 5-seed v4 OOD median is 0.459 (= 45.9%)
   - 5-seed v3 OOD median is 0.553 (= 55.3%)
   - With proper averaging, v4 is BETTER on OOD by ~10pp
   - The earlier finding was an artifact of comparing under-trained v4
     vs more-trained v3

### Final verdict

The data-driven calibration init has a small but consistent positive effect
on PINN-PEX MAPE (median 6-9pp improvement, narrower distribution) that
falls below statistical significance at n=5. The original "distillation"
claim was overclaim — this is parameter pre-tuning that gives a modest,
non-significant gain.

**Heteroscedastic problem (motivation) is NOT solved**: slope ≈ 0.5 across
all variants. The γ scaling head (dspinn_log §7.1) remains the recommended
next step. The data-driven init may be combined with γ as orthogonal fixes.

For paper purposes: the contribution is too thin on its own. Either:
- Combine with γ head and report joint improvement
- Frame as "ablation showing that calibration init + γ together work"
- Or acknowledge negative-but-careful result: "we tested data-driven
  calibration with proper statistical methodology; it does not significantly
  improve MAPE"

### Cleanup status

- 5 v3_baseline + 5 v4_full_calib + 5 v5_gnd_only checkpoints under
  `output_intel22/active_learning/m5_*/best_model.pth` (preserve for any
  follow-up analysis).
- Per-seed + per-variant CSVs under `output_intel22/active_learning/m5_summary/`
- Raw eval data: `eval_raw_ind.csv`, `eval_raw_ood.csv`
- Original long single-run v5_calib_gnd_only on GPU 7 was killed when the
  5-seed protocol absorbed its purpose.

Tests whether the NNLS-fit CPL pair scales (s_diag=0.18, s_cross=4.25)
hurt training. v5 uses NNLS ρ but reverts CPL pair to v3 hardcoded
(8.0, 5.0). If v5 ≥ v4: NNLS CPL fit is the problem, ρ alone is fine.
If v5 ≤ v4: full data-driven init is genuinely better. If both ≤ v3 final:
data-driven init isn't the right path, escalate to γ head.

Run config:
- `--model_name v5_calib_gnd_only --gpu 7 --use_dspinn`
- `--calib_path /data/PINNPEX/data/processed/intel22/calibration_init_gnd_only.json`
- Same SSL basis (bem_ssl_ep181), same predefined dataset cache, same 6 designs.

---

## 7. Open Questions / Future Work

1. **Per-layer-pair CPL scale** — v2 uses 2 unknowns (s_diag, s_cross). If
   the holdout MAPE shows residual structure correlated with specific layer
   pairs, escalate to per-layer-pair-distance-class (K+1 unknowns) or full
   K×K with regularization.

2. **γ head as follow-up** — even after data-driven init, the heteroscedastic
   slope-0.6 problem may persist (it's a *per-net* effect, not just per-layer).
   `dspinn_development_log.md §7.1` proposes γ (per-net multiplicative scaling
   head). Estimated 12-20% MAPE if combined with v4's calibrated init.

3. **Outlier-aware extraction** — top-100 worst nets (LDPC decoder, dense
   CTS) carry ~10pts of MAPE. Should NNLS use robust regression (L1) to
   not over-fit to outliers? Or trim?

4. **Per-design calibration** — if pooled IQR > 2, per-design init JSONs
   may fit better. Engineering cost is moderate.

5. **Cross-PDK transfer** — if asap7 and intel22 require different
   calibrations, the JSON should be PDK-keyed. Currently a single global
   value per anchor.

6. **Joint with γ** — γ scaling head can potentially absorb some of
   the per-net residual the static init can't. Co-design possibility.

7. **Calibration drift over training** — does the model's `gnd_modifier`
   stay near 1.0 throughout training, or drift? If it drifts, the init
   advantage may erode. Monitor with the existing probe.

---

## 8. References

- Codebase root: `/home/jslee/projects/PINNPEX/`
- Pipeline overview: `/home/jslee/projects/PINNPEX/CLAUDE.md`
- DS-PINN log: `docs/dspinn_development_log.md` (parallel track, root-cause source)
- Plan files (working drafts, archived):
  - `/tmp/distillinit_plan_v1.md` (Codex round 1 input)
  - `/tmp/distillinit_plan_v2.md` (current spec, post-fix)
- Calibration JSON output target:
  `/data/PINNPEX/data/processed/intel22/calibration_init.json`

---

_To update this log: edit in place, bump the date at the top, append to the
relevant section. Keep §5 (live tracker) current with each milestone._
