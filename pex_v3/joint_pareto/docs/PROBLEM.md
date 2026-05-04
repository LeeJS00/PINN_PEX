# Joint-Pareto Problem Statement

_Owner: pex-pareto-architect. Updated as state changes._

## What is the joint problem?

Drive **wall-clock**, **c_gnd MAPE**, and **c_cpl MAPE** down at the same
time on full-chip SPEF generation for VLSI parasitic extraction, while
keeping R²(C) ≥ 0.99 paper-grade. Current Path-2 v3 baseline is
Pareto-dominant on per-net total cap MAPE but per-channel sits at the
XGBoost ceiling.

## Why a joint problem?

The four axes are not independent:

1. **Runtime ⇄ allocator complexity**: more accurate gnd allocator
   (Sakurai-Tamaru per-segment) costs CPU per net. 3D overlap c_cpl
   allocator costs more KD-tree queries.
2. **Per-channel ⇄ per-net total**: XGB rescales sum_gnd and sum_cpl
   exactly per net, so per-channel error for matched nets equals XGB's
   per-channel prediction error. Improving per-channel for matched nets
   requires breaking the XGB per-net prediction ceiling.
3. **Per-channel ⇄ R²(C)**: tighter per-channel improves total R²
   marginally; the dominant R² driver is per-net total accuracy.

## Why is per-channel stuck?

For matched nets (3,169 / 3,380 = 93.8 % of tv80s):

```
sum_gnd_after_xgb_rescale  := xgb_pred_gnd  per net
sum_cpl_after_xgb_rescale  := xgb_pred_cpl  per net
```

So per-channel MAPE for matched nets is exactly XGB's per-channel
prediction MAPE: gnd 19.93 %, cpl 16.13 % on tv80s test (5-seed).
Path-2 v3 sees gnd matched mean 27.37 % (worse than 19.93 % above) and
cpl 18.78 % (close to 16.13 %) — the gap on gnd is from `*CAP` line
truncation effects (per-node values < 1e-5 fF dropped) and possibly
unmatched-net contribution leaking into the matched bucket via the
hash-match logic.

## Three levers for per-channel improvement

### Lever 1: better placeholder for unmatched (gnd-allocator-owner)

Current placeholder is `length × width × ε × 0.22`. This lands on
golden median (0.477 fF unmatched) but per-net error variance is high.
Switch to true Sakurai-Tamaru per-segment with layer ε + fringe.

Expected gain: small on matched (XGB invariant), moderate on unmatched
(~5 pp gnd MAPE).

### Lever 2: better per-aggressor c_cpl distribution (cpl-allocator-owner)

Current: `(length_t × length_a) / dist²` over midpoints; same-layer +
±1 layer; top_k=20; max_dist=5 μm.
Better: 3D overlap area × ε / d_inter; layer-aware lateral vs vertical
physics; shielding from intervening conductors.

Expected gain: small on per-net matched cpl (XGB invariant), moderate
on per-aggressor distribution accuracy (downstream STA), and tail p95.

### Lever 3: replace XGB anchor with a Mesh PINN per-net anchor

Current: XGB ceiling 19.93 % gnd / 16.13 % cpl per-net mean MAPE on test.
Alternative: Mesh PINN per-net (44 K params, 5-seed best 6.26 % total),
which has its own per-channel ceiling (Mesh last-step gnd 20.49 % / cpl
15.53 %). Slight gain on cpl, tied on gnd.

Expected gain: 0–2 pp on per-channel, but per-net total may regress
from 4.66 % (XGB) to 6.26 % (Mesh).

## What is OUT of scope

- Adding hand-feature inputs (Strikes #7 and #8 verified failed)
- Re-attempting synthetic pretrain (K3 canary fired)
- Per-pair head with uniform analytic baseline (Strike #2 killed)
- Re-introducing legacy 1M DeepPEX inference (97 % runtime)

## What is IN scope

- Smarter per-cuboid analytic placeholders (Sakurai-Tamaru, layer ε)
- Smarter per-aggressor geometric distribution (3D overlap, shielding)
- Parallel pass-2 SPEF write (runtime-owner)
- Hybrid anchor: Mesh PINN where XGB CSV is missing (211 unmatched)
- Better post-process — XGB calibration with fallback to Mesh for unmatched

## Key references

- `pex_v3/paper/RESULTS_CONSOLIDATED.md` — paper-grade leaderboard
- `pex_v3/paper/METHOD.md` §8.3 — Path-1/Path-2 dual table
- `MEMORY.md` 🚀🚀 — Pareto-dominance summary
- `project_starrc_compat_cgnd_diagnosis.md` — c_gnd information ceiling
- `project_strike_2_perpair_negative.md` — per-pair head failure
- `project_strike_7_cell_features_negative.md`, `project_strike_8_pincap_negative.md`
