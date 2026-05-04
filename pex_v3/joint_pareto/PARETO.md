# Pareto Frontier — Live Tracker

_Owner: pex-pareto-architect. Append-only. Each row = a measured 5-seed variant._

## Frontier (tv80s test, B1 XGB seeds 0..4 unless stated)

| # | Variant | Wall-clock | Total mean | Total median | Total p95 | gnd matched | cpl matched | R²(C) | dominant? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | Path-1 Legacy DeepPEX (1M) + XGB | 864 s | 10.96 ± 0.047 | 5.77 | 44.30 | 21.0 | 12.0 | 0.983 | ⏹ |
| 1 | Path-2 v1 fast (uncalibrated) | 68.9 s | 12.68 ± 0.043 | 5.78 ± 0.077 | 99.66 | 31.87 | 24.07 | 0.976 | ⏹ |
| 2 | Path-2 v3 (calibrated placeholder) | 68.9 s | 7.035 ± 0.045 | 5.441 ± 0.052 | 18.54 ± 0.35 | 27.37 | 18.78 | 0.993 | ⏹ (dominated by v7) |
| 3 | Path-2 v7 parallel pass-2 (16w) | 27.77 ± 0.77 s | 7.035 ± 0.045 | 5.441 ± 0.052 | 18.54 ± 0.35 | 27.20 ± 0.23 | 18.70 ± 0.07 | 0.9934 ± 0.0002 | ⏹ runtime-frontier (dominated on per-channel by v9) |
| 4 | Path-2 v9 = v7 + Mesh-ratio per-channel | 43.65 ± 0.60 s* | 7.035 ± 0.045 | 5.441 ± 0.052 | 18.54 ± 0.35 | 23.40 ± 0.09 | 18.35 ± 0.04 | 0.9933 ± 0.0002 | ⏹ dominated by v10 |
| 5 | Path-2 v10 α=0.2 XGB-Mesh blend | 42.59 ± 1.35 s* / ~32s alone | 6.821 ± 0.040 | 5.458 ± 0.059 | 17.20 ± 0.13 | 22.83 ± 0.07 | 17.77 ± 0.03 | 0.9939 ± 0.0002 | ⏹ dominated by v11 |
| 6 | **Path-2 v11 single-pass parallel (α=0.20)** | **20.34 ± 0.45 s** | **6.821 ± 0.040** | **5.458 ± 0.059** | **17.20 ± 0.14** | 22.83 ± 0.07 | 17.77 ± 0.03 | 0.9939 | ✅ **frontier on total** (best total mean) |
| 7 | **Path-2 v12 α=0.30 per-channel Pareto** | **20.42 ± 0.21 s** | 6.856 ± 0.035 | 5.551 | **17.15** | **22.59 ± 0.06** | **17.53 ± 0.03** | **0.9941** | ✅ **frontier on per-channel + p95** (gnd −0.24 cpl −0.24 vs v11, p<0.01; total +0.035pp NS) |

\* v9, v10 wall-clock measured under concurrent nova background workload. Standalone projection: v9 ≈ 33s, v10 ≈ 32s. v7 27.77s is standalone (exp_006 measurement). v11 is freshly measured standalone (no concurrent workload).

## 10% per-channel target verdict (2026-05-04)

User-set ambitious target: drive per-channel gnd / cpl matched MAPE to 10 %.

**Conclusion: not achievable with current DEF/LEF/Liberty/layer.info inputs.**

| Bound | gnd matched | cpl matched | Source |
|---|---:|---:|---|
| User target | 10 % | 10 % | aspirational |
| **4-way oracle** | **14.07 %** | **11.21 %** | exp_012: per-net min over XGB+B4+OptF+Mesh |
| Mesh single (best individual) | 21.87 % | 17.13 % | feature-bound |
| **v10 frontier** | **22.83 ± 0.07** | **17.77 ± 0.03** | XGB-Mesh α=0.2 blend |
| Path-1 Legacy | 21.0 % | 12.0 % | prior measurement, different config |

**Why 10 % is unreachable**:
- 56 % of nets exceed 10 % gnd MAPE even at the 4-way oracle bound.
- Pairwise XGB↔Mesh signed-error correlation 0.86; XGB↔B4↔OptF correlations 0.93-0.95. All hand-feature models share information ceiling.
- per-channel feature limit documented in `project_starrc_compat_cgnd_diagnosis.md`: cell-internal substrate area is the missing signal; weak |ρ|<0.16 of all DEF/LEF features with c_gnd residuals.

**Routes that could approach 10 %** (out of joint_pareto incremental scope):
- (A) GDSII transistor area extraction (substrate-aware c_gnd) — paper-class effort, ~4 weeks
- (B) Mesh PINN per-channel-decoupled retrain — uncertain, capacity sweep already saturated
- (C) Substrate / Liberty pin C combined with neural BEM per pair — research direction
- (D) Per-design oracle calibration for deployment — viable if deployment allows small oracle access

These are documented as **future work** in `pex_v3/paper/OUTLINE.md` §6 Discussion.

## Variant slots awaiting measurement

| # | Variant | Hypothesis | Risk |
|---|---|---|---|
| 3 | Path-2 v4 + Sakurai-Tamaru c_gnd per-segment | per-cuboid layer-ε aware → tighter per-segment dist | minimal runtime cost; need physics validation |
| 4 | Path-2 v5 + 3D-overlap c_cpl per-pair | overlap_area × ε / spacing → tighter cpl spatial | runtime risk if overlap calc is heavy |
| 5 | Path-2 v6 = v4 + v5 | combined | depends on each |
| 6 | Path-2 v7 + parallel pass-2 (multiprocess SPEF write) | runtime ↓ further | ordering concerns; merge step |
| 7 | Path-2 v8 + Mesh PINN per-net (replaces XGB) | break per-channel XGB ceiling | per-channel may improve, total may regress (Mesh 6.26 vs XGB 4.66 valid) |

## Decision rule (gates row promotion)

A new variant is admitted to the frontier if it strictly improves at least
one Pareto axis without regressing any other axis by more than ε:

- ε(runtime) = +10 % (from current best 68.9 s → max 75.8 s)
- ε(total mean) = +0.2 pp (from current best 7.035 % → max 7.235 %)
- ε(gnd matched mean) = +1.0 pp (from current best 27.37 % → max 28.37 %)
- ε(cpl matched mean) = +1.0 pp (from current best 18.78 % → max 19.78 %)
- ε(R²(C)) = −0.005 (from current best 0.993 → min 0.988)

5-seed paired MWU required for any "X better than Y" claim.

## Hard kill criteria

- **K-runtime**: any variant > 100 s wall-clock on tv80s → reject
- **K-gnd**: any variant > 35 % matched gnd MAPE → reject
- **K-cpl**: any variant > 25 % matched cpl MAPE → reject
- **K-r2**: any variant R²(C) < 0.98 → reject

## Summary

The frontier is currently a SINGLE point at variant #2 (Path-2 v3). Path-1
and Path-2 v1 are dominated. The next move targets per-channel reduction
without breaching the runtime cap.
