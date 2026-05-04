# Auto-Optimize Sweep — 2026-05-03

_Self-directed sweep to push PINN-PEX per-channel test MAPE below baseline._

## Baseline (lock)

`HybridPexV3Mesh` 5-seed (`pex_v3/output/phase1_mesh_5seed/`):
- valid total **6.26%** best / **8.27%** last (±0.108pp / ±0.342pp)
- test per-channel: **gnd 20.49% / cpl 15.53%**
- test total **8.27%**
- 44K params, 200 epoch curriculum (clamp 0.405 → 0.916 → 1.386)

## Targets

- gnd test ≤ **17%** (-3.5pp)
- cpl test ≤ **13%** (-2.5pp)
- total test ≤ **6.5%** last-step (-1.8pp), best-step ≤ **5%**

## Lever inventory after Week 1

### KILLED (capacity-add → Phase 2 overfit; 3 strikes)
- A1 per-channel separate encoders (single-seed test gnd 21.60%, total 8.82%)
- Strike #7 cell-OBS scalar features
- Strike #8 Liberty pin caps

### Surviving levers (this sweep)

| Variant | Type | Risk | GPU |
|---|---|---|---|
| **C1** CTS Mode B isotonic post-correction | output post-process, capacity-zero | low | 0 |
| **InputSubset** per-channel raw-cuboid input mask (shared encoder weights, channel-specific input) | input information re-routing, capacity-equal | medium | 1 |
| **ClampNorm** clamp-on-residual-norm vs clamp-on-logit | curriculum stabilizer, capacity-equal | medium | 2 |

### Decision gates

Each variant single-seed smoke must beat baseline on EITHER:
- test gnd ≤ 19.5% (-1pp), OR
- test cpl ≤ 14.5% (-1pp), OR
- test total ≤ 7.27% (-1pp)

If smoke passes → 5-seed lock via `pex_v3/scripts/run_ablation_5seed.py`.
If smoke fails → drop variant + report negative.

### Best-stack composition (post-funnel)

After all variants 5-seed locked:
1. Pick best-of-singles by paired Wilcoxon test vs baseline (D infrastructure auto-computes Cohen's d, MWU p-value, bootstrap CI).
2. Stack survivors (e.g., InputSubset model + C1 post-correction) → second 5-seed lock for the combined stack.
3. Hero result → `HERO.md` with anti-overclaim CI.

## Subdirectory layout

```
pex_v3/experiments/auto_optimize_2026_05_03/
  PLAN.md           — this file
  RESULTS.md        — running variant table
  HERO.md           — final best-stack with CI (created at end)
  variants/
    c1_cts_isotonic/      — C1 design + smoke artifacts
    input_subset/         — InputSubset design + smoke artifacts
    clamp_norm/           — ClampNorm design + smoke artifacts
  outputs/
    <variant>/seed{0..4}/ — 5-seed lock outputs (model.pt, summary.json, eval_logger.parquet)
    best_stack/seed{0..4}/ — final composed stack
  reports/
    <variant>_summary.json — D2 anti-overclaim aggregator output
    <variant>_stratified.json — D3 stratified MAPE
```

## Boundary

Per `pex_v3/CLAUDE.md`: only edit inside `pex_v3/`. All variant code, configs, and outputs live under this `experiments/auto_optimize_2026_05_03/` subtree or in `pex_v3/src/models/` (new model classes only — no edits to existing baseline files).

## Anti-overclaim

- 5-seed required before "improvement" claim
- Report last-step + best-step + ±std + Cohen's d + paired MWU p-value
- Stratified MAPE (per-design / per-quartile / per-fanout / per-layer) MUST also improve, not just total
- Best-stack number reported with bootstrap 95% CI

## Execution mode

Auto mode. Each variant runs single-seed smoke first; only winners go to 5-seed lock; best stack then 5-seed locked. No user confirmation between stages.

## Codex Round 2 verdicts (applied)

- **3-way parallel smoke**: GO (all non-capacity-add levers, A1 kill irrelevant). Single-seed must NOT be sole survivor decision; 5-seed required.
- **InputSubset constraint**: zero-masking ONLY (shared weights + input zeroing). Separate input projections = A1-in-disguise → FORBIDDEN.
- **ClampNorm**: SAFE day-1 invariant preserved (δ=0 → pred=analytic). Curriculum re-scaling sensitivity to monitor.
- **B1 (per-pair Sakurai)**: DEFERRED until 3-way smoke results.
- **Stacking C1 + InputSubset**: NO direct transfer of baseline-fit isotonic to InputSubset output. Must REFIT C1 isotonic on InputSubset's val output (held-out strict).

## GPU allocation

| Phase | Variant | GPUs |
|---|---|---|
| Smoke parallel | C1 | CPU (post-correction only) |
| Smoke parallel | InputSubset | 0 |
| Smoke parallel | ClampNorm | 7 |
| 5-seed lock | survivors | 0,1,2,3,4 (sequential per variant, ~20 min wall each) |
| Best-stack 5-seed | composed | 0,1,2,3,4 |

GPU 5/6 partially loaded — avoid.

