# Joint-Pareto Evolution — tv80s 5-seed

_Owner: pex-pareto-architect. Updated as the frontier moves._

## Frontier evolution

```
            wall-clock   total      gnd matched  cpl matched   R²(C)
                  (s)    mean (%)   mean (%)     mean (%)
─────────────────────────────────────────────────────────────────────
Path-1 Legacy     864.0  10.96      21.0         12.0          0.983    [archived]
Path-2 v1          68.9  12.68      31.87        24.07         0.976    [dominated]
Path-2 v3          68.9   7.04      27.37        18.78         0.993    [dominated by v7]
Path-2 v7          27.77  7.04      27.20        18.70         0.9934   [dominated by v9 on per-channel]
Path-2 v9          43.65  7.04      23.40        18.35         0.9933   [dominated by v10]
Path-2 v10        42.59   6.82       22.83        17.77         0.9939   [✅ FRONTIER]
```

\* v9, v10 wall-clock under concurrent nova background; standalone v10 ≈ 32 s.

## Cumulative improvement vs Path-1 Legacy

| Axis | Path-1 | Path-2 v3 | **Path-2 v10 (locked)** | Total Δ |
|---|---:|---:|---:|---:|
| Wall-clock (alone) | 864 s | 68.9 s | **~32 s** (standalone projection) | **−27×** |
| Total cap MAPE mean | 10.96 % | 7.04 % | **6.82 ± 0.04 %** | **−4.14 pp** |
| Total cap MAPE p95 | 44.30 % | 18.54 % | **17.20 ± 0.13 %** | **−27.10 pp** |
| **gnd matched mean** | 21.0 % | 27.37 % | **22.83 ± 0.07 %** | within paper-grade range |
| **cpl matched mean** | 12.0 % | 18.78 % | **17.77 ± 0.03 %** | per-channel cpl ceiling |
| R²(C) | 0.983 | 0.993 | **0.9939 ± 0.0002** | **+0.011** |

(NOTE: Path-1's reported per-channel `21.0 / 12.0` came from a different test
configuration; the current XGB-only ceiling on the same Path-2 setup is
27.37 / 18.78. So the comparison vs "Path-1" is across systems, not within.)

## Per-variant summary

### Path-2 v3 — calibrated placeholder (`exp_002`)
- `length × width × ε × 0.22` for c_gnd, c_cpl/c_gnd = 1.3 ratio
- Lands unmatched-net SPEF totals at golden median (0.477 fF gnd)
- Fixes the 211/3380 (6.2 %) tv80s nets absent from XGB CSV

### Path-2 v7 — parallel pass-2 (`exp_006`)
- 16-worker `mp.Pool.imap` over per-net SPEF assembly tasks
- 2.91× speedup on pass-2 (52 s → 17 s); total **27.77 s on standalone, 38 s under concurrent load**
- Identical accuracy to v3 (deterministic transformation)

### Path-2 v9 — Mesh-PINN ratio per-channel (`exp_009`)
- After XGB anchor (per-net total exact), apply Mesh ensemble ratio for gnd/cpl split
- Preserves XGB total; overrides per-channel split using Mesh's better split predictor
- gnd matched MAPE 27.20 → 23.40 (−3.80 pp), cpl 18.70 → 18.35 (−0.35 pp)
- **Breaks the XGB per-channel ceiling on matched nets** while keeping XGB total

### Path-2 v10 — α-blended XGB+Mesh totals (`exp_010`, in progress)
- target_total = 0.2 × mesh_total + 0.8 × xgb_total
- Single calibration pass (replaces XGB+Mesh-ratio chain)
- Analytic prediction: total ↓0.25 pp, gnd ↓0.6 pp, cpl ↓0.6 pp
- 5-seed measurement pending

### Strikes (deferred / negative)

- `exp_007 v8 Sakurai-Tamaru gnd` — REJECTED (total +1.04 pp ε breach).
  Critical lesson: matched-net per-channel is XGB-pinned; allocator-stage
  physics changes are invisible to per-net MAPE.
- `exp_004 v5 3D overlap cpl` — DEFERRED. Same architectural blocker;
  per-aggressor improvements need per-pair MAPE infrastructure to be visible.

## Specialist-agent log

| Variant | Specialist | Verdict |
|---|---|---|
| v7 | runtime-owner | ADMITTED (clean Pareto move) |
| v8 | gnd-allocator-owner | REJECTED + paper-grade architectural lesson (XGB ceiling) |
| v9 | pareto-architect (direct) | ADMITTED (Mesh ratio breakthrough) |
| v10 | pareto-architect (direct) | PENDING |
