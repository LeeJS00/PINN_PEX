# pex_v3/joint_pareto — Runtime × Per-Channel MAPE Joint Optimization

_Created: 2026-05-03 late evening, post Path-2 v3 5-seed lock._

## Mission

Drive **runtime, c_gnd MAPE, c_cpl MAPE** down **simultaneously** without
trading any one axis off the others. Path-2 v3 (placeholder calibration)
already Pareto-dominates Path-1 Legacy on per-net total cap MAPE. The
remaining gap is **per-channel** (gnd 27 % · cpl 19 % matched-net mean), which
sits at the XGBoost ceiling and is unmoved by the placeholder fix.

## Joint objective

```
J = α · runtime_seconds + β · gnd_MAPE + γ · cpl_MAPE − δ · R²(C)

  default weights (Pareto target):
    α = 0.001  (penalty per second; 100s ≈ 0.1 unit)
    β = 1.0    (penalty per pp gnd MAPE)
    γ = 1.0    (penalty per pp cpl MAPE)
    δ = 100.0  (reward per 0.01 R²; near-1 needs heavy weighting)
```

For paper-grade reporting: track all four axes separately (no scalar J).
Use the joint J only for ranking variants.

## Current baseline (2026-05-03 late)

**Path-2 v3 (calibrated placeholder, 5-seed tv80s):**

| Axis | Value | Vs Path-1 Legacy |
|---|---:|---:|
| Wall-clock | 68.9 s | −12.5× (864 s) |
| C MAPE mean | 7.035 ± 0.045 pp | −3.93 pp |
| C MAPE median | 5.441 ± 0.052 pp | −0.33 pp |
| C MAPE p95 | 18.54 ± 0.35 pp | −25.76 pp |
| **gnd MAPE mean (matched)** | **27.37 %** | (Path-1 30.91 %) |
| **cpl MAPE mean (matched)** | **18.78 %** | (Path-1 22.15 %) |
| R²(C) | 0.993 | +0.010 |
| R MAPE | 2.21 % (det.) | unchanged |
| R²(R) | 0.9991 | match |

## Target (Pareto frontier next move)

| Axis | Target | Strategy |
|---|---:|---|
| Wall-clock | ≤ 75 s on tv80s | budget-cap: ±10 % from baseline |
| Per-net total mean | ≤ 6.5 % | from 7.04 % (push p95 outliers down) |
| **gnd matched-net mean** | **≤ 22 %** | break XGB ceiling via per-segment Sakurai-Tamaru |
| **cpl matched-net mean** | **≤ 13 %** | break XGB ceiling via 3D overlap-area + shielding |
| R²(C) | ≥ 0.995 | better p95 → tighter R² |

## Folder layout

| Path | Owner | Purpose |
|---|---|---|
| `docs/PROBLEM.md` | pareto-architect | Axis-by-axis state + blockers |
| `docs/BASELINE.md` | pareto-architect | Frozen current baseline numbers |
| `docs/EXPERIMENTS_LOG.md` | pareto-architect | Append-only log of every variant tried |
| `allocators/gnd/` | gnd-allocator-owner | c_gnd per-cuboid distribution variants |
| `allocators/cpl/` | cpl-allocator-owner | c_cpl per-aggressor distribution variants |
| `runtime/` | runtime-owner | profiling + benchmark harness |
| `experiments/exp_<NNN>_<tag>/` | runner | each variant: config + spef + compare |
| `results/leaderboard.json` | pareto-architect | Pareto frontier record |

## Specialist agents

Each axis has a dedicated specialist (role markdown at `.claude/agents/`):

- **`pex-runtime-owner`** — wall-clock budget, parallelization, profiling.
  Veto on changes that regress runtime > 10 % from baseline.
- **`pex-gnd-allocator-owner`** — per-cuboid c_gnd physics; analytic
  Sakurai-Tamaru parallel-plate + fringe; layer-stack ε. Owns gnd MAPE.
- **`pex-cpl-allocator-owner`** — per-aggressor c_cpl geometry; lateral
  + vertical coupling, 3D overlap area, shielding. Owns cpl MAPE.
- **`pex-pareto-architect`** — joint trade-offs; integrates the three
  specialists; gates entries to leaderboard. Reports Pareto frontier.

Invoke via the general-purpose agent wrapper:

```python
Agent(
    subagent_type="general-purpose",
    prompt="""You are the **<role>** specialist agent.
              Read /home/jslee/projects/PINNPEX/.claude/agents/<role>.md FIRST
              and operate strictly within that role.

              [task]"""
)
```

## Anti-patterns (lesson from Strikes #2/#7/#8)

1. Don't add scalar features hoping they help — Phase 2 over-fits, all metrics worse.
2. Don't re-attempt synthetic pretrain — K3 canary fired (analytic = truth).
3. Don't claim "X improves Y" without 5-seed paired MWU (anti-overclaim discipline).
4. Don't break the runtime cap — every variant must stay ≤ 75 s on tv80s.

## Hand-off references

- `pex_v3/paper/RESULTS_CONSOLIDATED.md` — paper-grade leaderboard
- `pex_v3/paper/METHOD.md` §8.3 — Path-1/Path-2 dual table
- `pex_v3/output/spef_e2e_fast_v3/` — current Path-2 v3 baseline artifacts
- `pex_v3/src/utils/fast_spef_engine.py` — current SPEF engine code
- `MEMORY.md` 🚀🚀 entry — Pareto-dominance summary
