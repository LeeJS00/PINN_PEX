# exp_013: Per-pair coupling residual regression

_Owner: pex-cpl-allocator-owner. 2026-05-04. Smoke-pass timebox 90 min._

## Diagnosis (v10 = α=0.2 XGB-Mesh blend) — measured on tv80s

| Slice | Value |
|---|---:|
| n predicted pairs | 55,845 |
| n golden pairs | 89,917 |
| common pairs (intersection) | 37,859 (42.1% of golden) |
| **per-pair MAPE mean** | **368.6%** |
| **per-pair MAPE median** | **76.9%** |
| **signed bias mean** | **+308%** (over-predicting) |
| pairs missing from prediction | 52,058 (57.9% of golden) |

Stratified by golden c_pair magnitude:
- `<0.001` fF: median 690% (n=1108)
- `0.005-0.05` fF (mid): median ~75%
- `>=0.1` fF: median 75% (still bad — uniform distribution bias)

**Root cause**: v10 distributes the v10 per-net `c_cpl_total` proportionally
to `(L_target × L_aggr) / d²` across top-20 aggressors per net. The per-pair
*magnitude* is therefore (XGB-Mesh-blended total) ÷ N_aggr × geometric-weight,
which has no per-pair physics — every aggressor for a given target gets the
same uniform share modulo the 1/d² weight.

## Strike #2 lesson encoded

Strike #2 used a UNIFORM analytic baseline (`compact_cpl_total / n_aggr`) and
learned a bounded multiplier; failed because lateral / vertical / via-cap
priors differ by 10-100×. The fix: **per-pair-SPECIFIC analytic priors**
based on actual (target, aggressor) geometry — Sakurai-Tamaru lateral for
same-layer, parallel-plate vertical for adjacent-layer.

## This experiment's plan

### Step 1: extraction infrastructure (training-set per-pair features)

Walk all topology pkls for designs in `cfg.TRAIN_DEFS` (9 designs). For each
target net build a global KD-tree of all its in-design aggressor segments,
then for each (target, aggressor) pair within `max_dist_um=5.0` extract
features:

| Feature | meaning |
|---|---|
| target_layer | dominant target layer (1..9) |
| agg_layer | dominant aggressor layer (1..9) |
| layer_gap | abs(target_layer - agg_layer) |
| same_layer | bool |
| min_dist_um | closest cuboid-to-cuboid lateral distance |
| mean_dist_um | mean lateral distance |
| sum_inv_d_um | Σ 1 / max(d, 0.05) |
| sum_inv_d2_um2 | Σ 1 / max(d², 0.05²) |
| L_overlap_lateral_um | Σ length where same-layer parallel-projection overlap |
| L_overlap_vertical_um2 | Σ overlap area for cross-layer (broadside) |
| target_total_metal_um2 | whole-target metal area |
| agg_total_metal_um2 | whole-aggressor metal area in this pair window |
| target_eps_mean | mean ε on target layer |
| agg_eps_mean | mean ε on aggressor layer |
| n_pairs_in_window | # cuboid-cuboid pairs satisfied |

Join with golden parquet (`/data/PINNPEX/.../per_pair_golden/<design>.parquet`)
to get target `c_pair_fF`.

Output: `train_pairs.parquet` (ALL train designs concatenated).

### Step 2: per-pair analytic prior

For each pair:

```
if same_layer:
    # Sakurai-Tamaru lateral
    h_metal = layer_thickness(target_layer)
    s_lat   = max(min_dist_um, 0.04)        # min spacing floor
    eps_lat = layer_eps(target_layer)
    C_pair  = EPS0_FF_UM * eps_lat * h_metal * L_overlap_lateral / s_lat * 1.10
else:
    # Vertical: parallel-plate over inter-metal dielectric
    d_inter = layer_z_diff(target_layer, agg_layer)
    eps_v   = inter_layer_eps(target, agg)
    C_pair  = EPS0_FF_UM * eps_v * L_overlap_vertical / max(d_inter, 0.04)
# fall-back when no overlap: 1/d² geometric
if L_overlap_*= 0:
    C_pair_fallback = (sum_inv_d2_um2 * h_metal * eps) * 1e-3
```

This is the per-pair-specific analytic baseline (NOT uniform).

### Step 3: residual regression (LightGBM, MAPE objective)

Target: `log(c_golden_pair_fF / c_analytic_pair_fF)` (log-residual, additive
in log space → natural for multiplicative correction).
Features: all listed above + `c_analytic_pair_fF`.
Train on `train_pairs.parquet` (TRAIN designs only).
Validate on a held-out within-train slice; evaluate on tv80s.

If LightGBM is unavailable, fallback to RandomForest or XGBoost MAPE.

### Step 4: smoke evaluation on tv80s

Apply the trained model:
1. Build per-net per-pair features for tv80s
2. Predict `c_pair_pred_fF = c_analytic × exp(log_residual_pred)`
3. Compute per-pair MAPE on common pairs (golden ∩ predicted)

Compare against:
- v10 baseline (368.6% mean, 76.9% median)
- analytic-only (no learned residual) — measures the analytic prior alone
- uniform-baseline reproduction (Strike #2-style) — sanity check

### Step 5: decision gate

- **GREEN** (per-pair median ≤ 35% AND coverage ≥ 60%): integrate into a
  SPEF post-process script `44_per_pair_calibrate_spef.py` that, for each
  D_NET, replaces the per-pair `c_pair` in `*CAP` lines with the regression
  output, RESCALED to preserve the v10 per-net `cpl_total`.
  Then 5-seed measure on tv80s + nova.
- **YELLOW** (per-pair median 35-50%): document and hand off; analytic +
  residual is partial improvement, not paper-grade.
- **RED** (per-pair median > 50% OR coverage < 50%): STOP, document failure
  mode, hand off to architect.

### Wall-clock budget

Per-pair extraction on tv80s alone is currently 1-2 min (sister codebase
benchmark). Target SPEF post-process adds ≤ 20 s on top of v10's 32 s →
total 52 s. Hard cap 60 s.

## Files to create

```
pex_v3/joint_pareto/
├── allocators/cpl/
│   └── per_pair_residual.py           # core API
├── experiments/exp_013_per_pair/
│   ├── PLAN.md                         (this file)
│   ├── diag_v10_per_pair.py           # already wrote
│   ├── 01_extract_train_features.py   # walk TRAIN designs, write train_pairs.parquet
│   ├── 02_train_residual.py           # LightGBM
│   ├── 03_smoke_tv80s.py              # apply + measure
│   └── results/
│       ├── train_pairs.parquet
│       ├── residual_model.lgbm
│       ├── tv80s_per_pair.csv
│       └── verdict.md
```

## Hard guardrails

- Stay in `pex_v3/joint_pareto/`. Do NOT modify `fast_spef_engine.py`
  directly; the new allocator runs as a SPEF post-process.
- Wall-clock cap 60 s on tv80s.
- TRAIN-only training set; nova / tv80s test never seen during fit.
- 5-seed only if smoke gate passes.
