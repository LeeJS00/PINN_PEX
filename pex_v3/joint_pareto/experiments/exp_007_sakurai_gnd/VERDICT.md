# exp_007 v8_sakurai — VERDICT: REJECT (regression on unmatched + total)

_Owner: pex-gnd-allocator-owner. Frozen 2026-05-03._

## Hypothesis

Replace v3's `length × width × ε × 0.22` per-segment c_gnd placeholder with
full Sakurai-Tamaru per-segment (top-plate + bottom-plate, layer-aware ε,
nearest-conductor distance, layer-aware fringe, NNLS per-layer multiplier).
Expected gain:

  - Lift unmatched-net (211 / 3,380) per-net c_gnd accuracy.
  - Provide physically-correct per-segment c_gnd distribution for STA-grade SPEF.
  - Possibly reduce gnd matched mean below 27 % via XGB-anchor-friendly spatial allocation.

## Pre-experiment diagnosis (Path-2 v3 / v7 frontier on tv80s, seed0)

Read `diagnose_summary.json` for the full numbers. Key facts:

  - **Per-layer gnd MAPE (matched)**: M3 worst at 33.4 % (1265 nets), M2 24.4 %
    (1220 nets), M4 27.1 %, M5 16.0 %. M1/M6/M7/M8 have zero matched nets in
    the topology cache (single-layer routing on this design is rare).
  - **Per-quartile (g_tot)**: Q1 (smallest 793 nets) 31.7 %, Q2 29.2 %,
    Q3 29.9 %, Q4 (largest) **18.7 %** — small nets dominate the error.
  - **Per-segment-count bucket**: 2-5 segs (1554 nets) at 31.7 %, 21-100
    (308 nets) at 15.5 %, 101+ (8 nets) at 6.5 %. Few-segment small nets
    are the long tail.

## What we built

  - `pex_v3/joint_pareto/allocators/gnd/sakurai_tamaru.py`
    + `LayerStackPlate(layer_info)` — precompute per-metal {ε_top, ε_bot,
       d_top, d_bot, fringe_α, NNLS_k, pre_factor} from `LayerInfoParser`.
    + `analytic_per_net_cap_estimate(segments, plate)` — drop-in
       replacement for `fast_spef_engine.analytic_per_net_cap_estimate` with
       Sakurai-Tamaru per-segment c_gnd.
    + `redistribute_node_caps_inplace(writer, plate)` — optional post-pass
       that overrides the legacy length-only `distribute_net_caps` per-node
       weighting with ST-weighted per-edge `(L × W × ε_layer × ST_factor)`.
  - `pex_v3/joint_pareto/experiments/exp_007_sakurai_gnd/engine.py`
    Parallel-pass-2 engine (copied from exp_006) with the ST analytic injected.
  - 5-seed driver: `run.sh`, `run_one_seed.py`, `aggregate_5seed.py`.

## 5-seed measurement on tv80s (16 workers / seed)

| Axis | v3 frontier (v7_parallel) | v8_sakurai | Δ |
|---|---:|---:|---:|
| Wall-clock (s) | 27.77 ± 0.77 | 28.40 ± 0.32 | +0.63 (within +10 % cap) |
| Total cap MAPE mean (%) | 7.035 ± 0.045 | **8.076 ± 0.045** | **+1.04** ❌ past ε (0.2) |
| Total cap MAPE median (%) | 5.441 ± 0.052 | 5.733 ± 0.064 | +0.29 |
| Total cap MAPE p95 (%) | 18.54 ± 0.35 | 24.35 ± 0.15 | +5.81 ❌ |
| **gnd matched mean (%)** | 27.20 ± 0.23 | 27.20 ± 0.23 | ±0.00 (XGB-pinned, identical) |
| **gnd unmatched mean (%)** | 21.50 (deterministic) | **35.61 (deterministic)** | **+14.11** ❌ |
| cpl matched mean (%) | 18.70 ± 0.07 | 18.70 ± 0.07 | ±0.00 (XGB-pinned, identical) |
| cpl unmatched mean (%) | 27.29 (deterministic) | **40.60 (deterministic)** | **+13.31** ❌ |
| R²(C) | 0.9934 ± 0.0002 | 0.9899 ± 0.0002 | -0.0035 |

`admit_to_frontier --dry-run` still reports "✅ ADMIT" because the candidate
isn't worse past ε on **every** axis (wall-clock is within +10 % cap), but
the variant doesn't strictly improve any axis and clearly regresses three
(total_mape_mean, total_mape_p95, unmatched gnd, unmatched cpl, R²(C)).

## Why the matched gnd is unchanged

The XGB anchor calibration (`scripts/16_xgb_calibrate_spef.py`) rescales
each *D_NET's *CAP block by `gnd_scale = xgb_pred_gnd / pinn_sum_gnd`. After
rescale, the matched-net `*CAP` block sums to exactly `xgb_pred_gnd` — so
matched gnd MAPE := `MAPE(xgb_pred_gnd, golden_gnd)` regardless of which
analytic produced the pre-rescale numbers. **The spatial allocator cannot
move matched gnd MAPE.** This was called out in the agent role doc and the
joint-pareto PROBLEM doc, and is now empirically confirmed.

The only things the gnd allocator can move are:
  1. Per-net c_gnd_total for the 211 unmatched nets (no XGB row).
  2. Per-cuboid c_gnd distribution that survives 1e-5 fF *CAP truncation —
     but the post-XGB rescale washes this back into a tight band per net.

## Why unmatched got worse

Sakurai-Tamaru gives unmatched-net c_gnd median 0.515 fF (vs 0.398 fF for
v3 placeholder; golden median is 0.477 fF). Median is closer to golden,
but the **per-net distribution is wider** because per-layer ε_top/d_top
differentiates m1-m6 (thin, dense) from m7-m8 (thick, sparse). Mean MAPE
is mean-of-relatives, so heavier tails dominate.

The c_cpl side is even worse: c_cpl_total = 1.3 × c_gnd_total inherits the
new wider distribution and golden-median offset, giving 40.60 % unmatched
cpl (vs 27.29 %).

## Bigger picture

This experiment confirms the agent-role doc's a-priori warning:

> the matched-net 27 % is the **XGBoost per-net prediction ceiling** … you
> break this ceiling by improving the underlying per-net or per-segment
> predictor — not by tuning the allocator alone.

Per-segment Sakurai-Tamaru physics is not enough. To break 27 % matched
gnd, we need either:
  - Per-segment XGB or Mesh PINN that emits **per-cuboid** c_gnd (not per-net),
    bypassing the per-net XGB anchor for matched nets.
  - A new per-net predictor that beats XGB's 19.93 % per-channel gnd ceiling
    (Phase 1 capacity sweep showed ~11-14 % is the per-net feature ceiling
    on hand features, so this likely needs `mesh_v3` per-cuboid features).

## Recommendation to pareto-architect

**REJECT v8_sakurai.** Do not admit to frontier. The Sakurai-Tamaru per-segment
analytic does not beat the v3 placeholder on this objective. Frontier stays
at v7_parallel.

## Concrete next-step proposal

  1. **Per-segment XGB head** that emits one c_gnd per *CAP line directly,
     trained on tile-level golden c_gnd (not aggregated to net). Bypasses
     the per-net XGB anchor for matched nets. Expected effort: 2-3 days.
  2. **Investigate the truncation hypothesis**: instrument `compare_spef`
     to count *CAP lines truncated per matched net. If many *CAP lines
     fall below 1e-5 fF for small nets (Q1/Q2 quartiles, where matched
     gnd MAPE is 31.7-29.2 %), the cause is line-truncation drift, and
     a smarter post-process (consolidate truncated-cap residual onto the
     surviving *CAP lines per net) would lift matched MAPE without any
     model change.
  3. **Do not invest in further allocator-only variants** for the c_gnd axis
     until either (1) or (2) is in.

## Files

  - Allocator module: `pex_v3/joint_pareto/allocators/gnd/sakurai_tamaru.py`
  - Engine: `pex_v3/joint_pareto/experiments/exp_007_sakurai_gnd/engine.py`
  - 5-seed runs: `runs/seed{0..4}_*`
  - Aggregate: `measurement.json`
  - Diagnosis: `diagnose_summary.json`, `per_net_layer_cache.json`
