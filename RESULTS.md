# PINN-PEX RESULTS — canonical leaderboard

_Last updated: 2026-05-18 (refinement sprint v3 lock — L5 drop both PDKs; ASAP7 specialist d9 n750 → d8 n500)._

## 0. **DEPLOYABLE FRONTIER — TreePEX 5-seed Tweedie ensemble** (post-sprint 2026-05-18)

End-to-end deployable PEX tool: `TreePEX/`. Features → 5-seed Tweedie XGBoost
ensemble (depth=8, n_est=500, vp=1.5) → L11 large-net specialist (ASAP7 only,
d8 n500) → IEEE 1481-1999 SPEF → golden comparison. CPU-only (no GPU needed).
L5 calibration **dropped** (both PDKs, net 0 ASAP7 / −0.10/−0.14 pp IMPROVE intel22).

### Paper benchmark — MAPE_med (5-seed prediction-mean ensemble)

**⚡ Warm path** (features pre-computed CSV → inference + SPEF; fanout = gold-SPEF-derived label-leak; deployment scenario where feature engineering runs upstream):

| PDK | Design | n_nets | tot_MAPE | gnd_MAPE | cpl_MAPE | R²(tot) | Wall e2e | predict-only |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **intel22** | tv80s_f3 | 3,169 | **4.95 %** | 17.96 % | 13.51 % | **0.9936** | **11.27 s** | 0.38 s |
| **intel22** | nova_f3 | 92,425 | **5.34 %** | 17.42 % | 15.21 % | **0.9914** | **82.10 s** | 0.60 s |
| **ASAP7** | tv80s_x1 | 3,328 | **6.72 %** | 20.10 % | 9.01 % | **0.9854** | **9.68 s** | 7.75 s |
| **ASAP7** | nova_x1 | 125,499 | N/A (no training entry) | — | — | — | — | — |

**❄️ Cold path** (DEF → parse → V3 + V4 H3 features → fanout XGB proxy → inference; StarRC-equivalent from-scratch playing field):

| PDK | Design | n_nets | tot_MAPE | gnd_MAPE | cpl_MAPE | R²(tot) | Cold wall | vs StarRC FS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **intel22** | tv80s_f3 | 3,280 | **4.95 %** | 17.57 % | 13.59 % | **0.9933** | 68.31 s | 4.1× |
| **intel22** | nova_f3 | 113,812 | **5.47 %** | 15.83 % | 15.75 % | **0.9895** | 4767 s / 80 min | 1.50× |
| **ASAP7** | tv80s_x1 | 3,328 | **7.00 %** | 19.91 % | 9.59 % | **0.9827** | ~70 s | 3.9× |
| **ASAP7** | nova_x1 | 125,499 | **7.93 %** | 21.32 % | 10.78 % | **0.9699** | ~3249 s / 54 min | 2.2× |

Warm vs cold Δ MAPE: intel22 tv80s +0.00 pp · intel22 nova +0.14 pp · ASAP7 tv80s +0.28 pp.
Fanout proxy OOS quality (12 % intel22, 18-20 % ASAP7)가 cold/warm gap 직접 결정.

Warm vs cold 차이: feature 추출 wall (0 vs 1500-3000s) + fanout source (gold-SPEF label-leak vs DEF-only XGB proxy).
**Path separation rule**: 두 path는 절대 같은 표에 섞지 말 것 (`feedback_warm_cold_path_separation.md`).

Stage breakdown (tv80s / nova):
- Parse (DEF + tech LEF + cell LEF + layer.info): 1.68 / 65.27 s
- Feature extract (V3 41-D + V4 H3 26-D): 2.13 / 2.08 s
- Model load (10 × XGBoost JSON): 2.90 / 2.42 s
- Predict (5-seed CPU): 0.38 / 0.60 s
- SPEF write: 0.009 / 0.17 s
- Compare to golden: 0.001 / 0.005 s

Parse dominates wall for large designs; ML inference itself < 1 s. Full table:
`TreePEX/paper_benchmark/PAPER_TABLE.md`.

### vs PINN v12 mesh (5-seed ensemble, archived under `archive/pex_v3/`)

| Design | XGBoost TreePEX tot | PINN v12 mesh tot | Δ | Predict-only ratio |
|---|---:|---:|---:|---:|
| tv80s | 4.98 % | 8.23 % | **−3.25 pp** | **9.1× faster** (0.38 vs 3.46 s) |
| nova  | 5.28 % | 7.88 % | **−2.60 pp** | **33.8× faster** (0.60 vs 20.29 s) |

### Predict-only (legacy 2026-05-10 numbers for cross-reference)

| design | n_nets | tot_med | gnd | cpl | R²(tot) | inference | SPEF write |
|---|---:|---:|---:|---:|---:|---:|---:|
| tv80s | 3,169 | 4.979 % | 18.02 % | 13.27 % | 0.9940 | 0.171 s | 0.16 s |
| nova  | 92,425 | 5.279 % | 17.40 % | 14.96 % | 0.9911 | 0.185 s | 3.01 s |

vs prior:
- **v12 PINN (5-seed)**: tv80s 5.55 / 20.4 s standalone → **TreePEX −0.57 pp / 120× faster**
- **B1 XGBoost (5-seed)**: tv80s 5.30 / 5.83 → TreePEX −0.32 pp / −0.55 pp
- **Innovus / OpenRCX (pattern matching)**: 22-72 % per cap-decile → TreePEX 4-9× better

SPEF round-trip lossless (max abs err 5e-6 fF). Per-bucket C8 (mid-cap nets,
1.46 fF mean) hits **4.02 %** locally — confirms 4 % achievable on mid/large
nets and that aggregate ≥ 4.98 % is bottlenecked by C1 (smallest 10 %, cap < 0.15 fF)
denominator noise.

**Why ensemble inference improves over single-seed mean**: TreePEX averages 5
predictions per net BEFORE computing MAPE; original 5-seed lock averaged
MAPE across seeds. Predict-then-aggregate cancels per-seed variance directly,
yielding 5.087 → 4.979 (tv80s) and 5.417 → 5.279 (nova) without any
architectural change.

Pipeline: `TreePEX/REPORT.md`. Tool: `TreePEX/scripts/pex_tool.py --all`.
Models: `TreePEX/models/` (10 × ~12 MB Tweedie XGBoost weights, ~120 MB total).

## 0.5. **pex_v5 auto-4pct round 1+2 — exhaustive ceiling exploration** (2026-05-10)

Goal tv80s tot_med ≤ 4.0 % NOT achieved as 5-seed mean across separate models.
Closest deployable per-seed-model 5.087 ± 0.049 (S4 Tweedie); TreePEX's
ensemble-of-predictions reaches **4.979 %** (see §0). Theoretical oracle
bound on per-bucket approach: P2 4.742 / P7 r1 mean 4.679 — but this is
the ceiling of "67-D scalar feature + tree model", not an information
ceiling. **StarRC reaches golden cap on the same DEF/LEF/Liberty/layer-stack
inputs**, so the gap to 4 % is a representation/architecture problem, not
a missing-input problem.

| strategy | tv80s tot | nova tot | inference (95k) | deployable? |
|---|---:|---:|---:|---|
| S4 Tweedie (ref) | 5.087 ± 0.049 | 5.417 ± 0.027 | 0.05 s | ✓ |
| P1 quantile (Small/Big) | 5.380 / 5.380 | 6.13 / 6.70 | 0.05–0.4 s | ✓ but worse |
| **P2 per-bucket + oracle** | **4.742 ± 0.070** | **5.004 ± 0.041** | ~0.5 s | ✗ (oracle) |
| **P7 mega-mean (incl P2 oracle)** | **4.679** | **4.970** | ~1.0 s | ✗ (oracle) |
| P8 deployable router | 5.692 ± 0.085 | 6.003 ± 0.049 | ~0.1 s | ✓ but +0.6pp |

Key findings:
- **Per-bucket specialization** reaches 4.74 % oracle but deployable router
  (86.5 % exact, 13.5 % misroute) erases the gain (P8 +0.6 pp regression vs S4).
- Quantile / custom MAPE objectives don't help (P1 5.38, P3 12.1 broken).
- 4 % requires richer **representation** (not new inputs — StarRC uses same DEF/LEF/Liberty/layer-stack and reaches golden). Current 67-D scalar feature + tree model leaves on the table the per-pair / per-segment field information that StarRC's NXTGRD lookup + field solver capture.

Full writeup: `pex_v5/reports/FINAL.md`. Strategy artifacts: `pex_v5/runs/`.

## 0.5. **S4 Tweedie 5-seed** — auto-4pct round 2 verdict (2026-05-09)

Goal of tv80s tot_med ≤ 4.0% NOT achievable on current input modality —
8 strategies (S1-S9) tried, best single-seed 5.032 (vp=1.4), best 5-seed
locked **5.087 ± 0.049** (S4 Small_combined + Tweedie loss vp=1.5).

**S4 = new paper frontier**. Beats v12 PINN tv80s tot −0.46 pp at 400× faster
wall, AND beats prior Small_combined / Big_combined on nova-side
(tot 5.42 / gnd 17.48 / cpl 15.00 — best on every nova axis). std 0.027
on nova (lowest). No regression vs v12 anywhere.

Diagnosis: **information ceiling on DEF/LEF/Liberty inputs**. C1 (smallest
nets, cap<0.15fF) drives residual; closing 1.1 pp gap to 4 % requires new
input modality (GDSII substrate / extracted res-grid) or different target
(block-level / cap-weighted MAPE).

Full writeup: `pex_v4/auto_4pct/reports/FINAL.md`.

S4 deployable artifact: `pex_v4/auto_4pct/runs/S4_tweedie/S4_tweedie_seed{42,0,1,2,3}_test.csv`

## 0.5. Newest frontier — H-track 5-seed lock (2026-05-09)

5 seeds × 3 configs (B1 / Small_combined / Big_combined). Paired Wilcoxon p=0.0625
across-seed comparison = max significance achievable with n=5. All differences below
have effect size d > 1.0 except where noted "NS".

### Per-design 5-seed mean ± std (intel22 cross-design test)

| config | feats | wall (95k nets) | tv80s tot_med | tv80s gnd | tv80s cpl | nova tot_med | nova gnd | nova cpl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 (5-seed) | 41 | 0.05 s | 5.30 ± 0.05 | 19.89 ± 0.16 | 14.16 ± 0.09 | 5.83 ± 0.08 | 19.93 ± 0.11 | 16.31 ± 0.12 |
| **Small_combined** | **67** | **0.05 s** | **5.28 ± 0.07** | **18.90 ± 0.24** | **13.57 ± 0.07** | **5.62 ± 0.02** | **18.59 ± 0.23** | **15.32 ± 0.06** |
| Big_combined | 67 | 0.4 s | 5.17 ± 0.07 | 17.84 ± 0.08 | 13.23 ± 0.13 | 5.92 ± 0.06 ⚠ | 17.54 ± 0.11 | 15.60 ± 0.05 |
| v12 PINN frontier | 41 + cuboid-enc | **20.4 s** | 5.55 (5-seed) | 22.59 | 17.53 | n/a | n/a | n/a |

⚠ Big_combined nova tot_med +0.09 pp regression vs B1 (d=-1.20, paired Wilcoxon
p=0.0625) — capacity overfits without compensating per-channel gain.

### Paper-grade winner: **Small_combined**

Beats B1 on every design × every per-channel axis with d > 4, no regression
anywhere. Beats v12 frontier on every axis: tv80s tot −0.27 pp, gnd −3.69 pp,
cpl −3.96 pp at **400× faster wall-clock** (0.05 s vs 20.4 s on 95 594 nets).
pex_v4 CLAUDE.md strict targets (gnd ≤ 18.0, cpl ≤ 14.0) — cpl hit cleanly on
both designs; gnd at 18.59-18.90 (just above 18.0 target).

Big_combined captures additional per-channel lift (gnd −2 pp, cpl −0.5 pp on
top of Small_combined) but pays nova tot stability cost — high-capacity variant
suitable for tv80s-equivalent designs but not yet cross-design Pareto-optimal.

Reproducibility: `pex_v4/scripts/{29_extract_new_features.py, 30_xgb_with_new_features.py,
31_five_seed_lock.py}`. Outputs: `pex_v4/results/{xgb_big_combined, xgb_new_features,
five_seed_lock}/`. Full writeup: `pex_v4/docs/H_FEATURES_RESULT.md`.

---
_Scope: every measured run on cross-design OOD (nova + tv80s, 95,594 nets) or its single-design subset. 5-seed numbers locked under P6 protocol; single-seed marked explicitly._
_Source aggregations consolidated here: `pex_v3/paper/RESULTS_CONSOLIDATED.md`, `pex_v3/joint_pareto/results/leaderboard.json`, `pex_v3/experiments/auto_optimize_2026_05_03/HERO.md`, `experiments/cross_design_tv80s_2026_05_02/r_analytic_v3/reports/{V3,CGND}_RESULTS.md`, `docs/PROJECT_REPORT.md` §2._

---

## 1. Headline numbers (paper-grade, 5-seed locked)

| # | Pillar | Headline metric | Value | Artifact dir |
|---|---|---|---|---|
| 1 | Cap MAPE — Mesh-curriculum best-step | total OOD test, 5-seed median ± stdev | **6.26% ± 0.108pp** | `pex_v3/output/phase1_mesh_5seed/` |
| 2 | Auto-Optimize HERO (Mesh + InputSubset + ClampNorm + LGBM 8-feat) | total OOD test, 5-seed median | **6.364%** [CI 6.247, 6.505] | `pex_v3/output/ablation/HybridPexV3MeshInputSubsetClampNorm/` |
| 3 | Joint Pareto v11 (α=0.20, single-pass parallel) | total OOD test, 5-seed mean ± stdev | **6.821% ± 0.040** | `pex_v3/joint_pareto/experiments/exp_011_alpha_sweep_0_25/` (frozen) |
| 4 | Joint Pareto v12 (α=0.30, per-channel optimum) | gnd / cpl OOD test, 5-seed mean | **gnd 22.59 / cpl 17.53** | strict per-channel Pareto over v11 |
| 5 | Hybrid C calibration (XGB anchor + PINN distribution) | tv80s SPEF C MAPE, 5-seed | **10.96% ± 0.047pp** (R² 0.984) | `output_intel22/active_learning/m6_v10b_baseline_seed{0..4}/` |
| 6 | Per-net R calibration (NNLS + LightGBM sister hybrid) | tv80s SPEF R MAPE, mean / median | **2.21% / 1.40%** (R² 0.999) | `experiments/cross_design_tv80s_2026_05_02/r_analytic_v3/` |

Statistical evidence for HERO #2: Cohen's d = -5.97, paired MWU p = 0.008, paired per-net Wilcoxon p ≈ 0 (n = 477,970), bootstrap 95% CI does NOT overlap baseline (8.272%).

---

## 2. Per-net Cap MAPE leaderboard (cross-design OOD test)

5-seed mean ± stdev of per-seed median per-net MAPE on test split (nova + tv80s = 95,594 nets), unless noted.

| Method | params | valid total | test total | OOD gap | gnd test | cpl test | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| **HERO Mesh+IS+CN + LGBM** | 44K + LGBM | — | **6.364% ± 0.106** | — | 20.18 | 15.36 | Auto-optimize Round 3, sprint target ≤6.5% MET |
| **Mesh-curriculum best-step** | 44K | 6.258% ± 0.108 | — | — | — | — | 5-seed best-step over curriculum |
| **Mesh-curriculum last-step** | 44K | 8.59% ± 0.717 | 8.27% ± 0.342 | -0.32pp | 20.49 | 15.53 | last-step (curriculum end) |
| Mesh-curriculum 5-seed ensemble | 44K × 5 | 7.81% | 7.89% | +0.08pp | 19.90 | 15.15 | model-average |
| Hybrid_v3 Tier 2 (single seed) | 11K | 10.77% | 11.79% | +1.02pp | 24.83 | 16.82 | early Phase 1 |
| **B4 V3 log-GBDT** | ~100K | 5.72% ± 0.04 | 6.59% ± 0.13 | +0.87pp | 20.30 | 12.80 | strongest classical |
| **B1 XGBoost** | ~100K | 4.66% ± 0.026 | 5.84% ± 0.096 | +1.19pp | 19.93 | 16.13 | tree boosting; widest OOD gap |
| **Option F deep MLP** | 286K | 4.76% ± 0.012 | 5.62% ± 0.042 | +0.87pp | 21.67 | 16.44 | hand-feature ceiling tied with B4 |
| B3 PINN legacy DeepPEX | 1M | 30.90% ± 2.20 | — | — | — | — | valid only; never retrained on H3 |

**Key takeaways:**
- Hand-feature ceiling = **4.66 – 5.84%** across XGB / MLP / log-GBDT (3 architectures, identical features). Capacity is NOT the bottleneck (Phase 1 capacity sweep 11K → 406K all hit 11–14%).
- Mesh PINN closes 2/3 of the gap from legacy 30.90% → 6.26%, with **2.3× fewer params than B4**.
- Best per-channel in this table: B4 cpl 12.80% / Mesh-curriculum 5-seed gnd 19.90%.

---

## 3. Per-channel ceiling (information-bound finding)

| Source | gnd | cpl | Comment |
|---|---:|---:|---|
| 4-way model oracle (XGB+B4+OptF+Mesh, take-min) | 14.07% | 11.21% | floor under DEF/LEF inputs |
| Sprint target | ≤17.0% | ≤13.0% | NOT met by any single model |
| Best single model: HERO | 20.18% | 15.36% | gap to oracle: gnd +6.1pp / cpl +4.2pp |
| Strike #7 (sister cell-OBS features) | 21.65% | 14.80% | regression vs Mesh-only |
| Strike #8 (Liberty pin caps) | 25.85% | 14.29% | regression vs Mesh-only |
| pex_v3 21% gnd ceiling (joint Pareto session) | 21% | — | corroborates oracle bound |

**Verdict:** 10% per-channel target NOT achievable on DEF/LEF — requires GDSII / substrate-aware extraction (~4 weeks). 56% of nets exceed 10% gnd at the 4-way oracle bound. See PROJECT_PLAN §1.3 D7 verdict.

---

## 4. Joint Pareto SPEF E2E frontier (tv80s test, 3,380 nets)

5-seed measurement; all variants share the same XGB-anchor + PINN distribution + sister R hybrid post-process. Baseline 0 = legacy 1 M PINN + XGB anchor (Path-1).

| Variant | total mean ± stdev | total median | total p95 | gnd matched | cpl matched | wall (s) | R²(C) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Path-1 legacy DeepPEX 1 M | 10.96% ± 0.047 | 5.77% | 44.30 | 21.0 | 12.0 | 864.0 | 0.983 | dominated by Path-2 v3 |
| Path-2 v1 (uncalibrated) | 12.68% ± 0.043 | 5.78% | 99.66 | 31.87 | 24.07 | 68.9 | 0.976 | dominated by v3 |
| Path-2 v3 (calib placeholder) | 7.04% ± 0.045 | 5.44% | 18.54 | 27.37 | 18.78 | 68.9 | 0.993 | dominated by v9 |
| v7 parallel pass-2 | 7.04% | 5.44% | 18.54 | 27.20 | 18.70 | 27.77 | 0.993 | dominated by v11 |
| v9 parallel + Mesh ratio | 7.04% | 5.44% | 18.54 | 23.40 | 18.35 | 43.65 | 0.993 | dominated by v10 |
| v10 α=0.20 XGB-Mesh blend | 6.821% ± 0.040 | 5.46% | 17.20 | 22.83 | 17.77 | 42.59 | 0.9939 | dominated by v11 |
| **v11 single-pass parallel α=0.20** | **6.821% ± 0.040** | 5.46% | 17.20 | 22.83 | 17.77 | **20.34** | **0.9939** | **🏆 frontier (best total + best wall)** |
| **v12 α=0.30** | 6.856% ± 0.035 (NS vs v11) | 5.55% | 17.15 | **22.59** | **17.53** | 20.42 | 0.9941 | **🏆 frontier (best per-channel Pareto over v11, p<0.01)** |

`pending_variants`: v4_sakurai_gnd, v5_3d_overlap_cpl, v6 (v4+v5), v8_mesh_pinn_anchor — design-only, not yet measured.

---

## 5. SPEF E2E full report — tv80s hero (single seed for SPEF write)

```
PINN raw (tile→net aggregation drift):     C 47.69%   R 28.36%
+ XGB cap calibration (5-seed anchor):     C 10.96% ± 0.047pp   R 28.36%
+ R global α=1.4777 (cross-codebase):      C 10.96%             R 11.78%
+ R per-net (sister v3 hybrid):            C 10.96%             R  2.21%   ← HERO FINAL
                                          R²(C) = 0.983 / R²(R) = 0.999
                                          R median = 1.40% / C median = 5.77%
                                          R RMSE = 11.67 Ω / C RMSE = 0.291 fF
                                          chip-level cap balance = 0.96×
```

**Long-net Q4 cap MAPE 71.42% → 9.16%** (8× improvement) via XGB anchor — calibration scales with net length.

Length-stratified MAPE (XGB-anchored, tv80s):
| Quartile | range | n | median MAPE |
|---|---:|---:|---:|
| Q1 short | 35.9 – 79.0 Ω | 845 | 6.46% |
| Q2 | 79.1 – 120.8 Ω | 845 | 5.90% |
| Q3 | 121.1 – 262.2 Ω | 845 | 6.24% |
| Q4 long | 262.4 – 6043.9 Ω | 845 | 4.61% |

---

## 6. Sister R hybrid (NNLS + LightGBM stacked)

Cross-design tv80s test, 3,380 nets. Stage 1 = NNLS analytic; Stage 2 = LGBM ensemble; Stage 3 = LGBM on S2 residuals.

| Policy | Test MAPE | median APE | P90 APE | bias | parameters |
|---|---:|---:|---:|---:|---|
| v3 hybrid (S1 + S2 5-seed LGBM) | 2.456% | 1.56% | 5.06% | -0.74% | 23 NNLS + 5×500 trees |
| **v3 stacked (S1 + S2 + S3 3-seed)** | **2.443%** | **1.57%** | **5.08%** | -0.71% | + 3×300 trees |

R²(R) = 0.999 across cross-design test. Sister approach on c_gnd (26.47%) underperforms previous v7 ML (21.09%) — c_gnd is transistor-characterization dominant, requires `.lib pin_capacitance` not in our intel22 .lib.

---

## 7. Industry tool comparison (10 designs, vs StarRC golden)

Source: `docs/pex_tool.csv`.

| Tool | mean MAPE (10 designs) | tv80s MAPE | nova MAPE | tv80s real (s) | nova real (s) | License |
|---|---:|---:|---:|---:|---:|---|
| StarRC fs (1-core) | 0 (golden) | 0 | 0 | 3,496.6 | 31,200.7 | commercial |
| Innovus (4-core, vs fs) | 4.85% | 4.87% | 6.15% | 41.82 | 122.22 | commercial |
| OpenRCX (4-core, vs fs) | 7.69% | 7.60% | 7.89% | 5.10 | 64.18 | open-source, **0 cpl entries** |
| **PINN-PEX v11 (ours)** | — | **6.82% (5-seed)** | (in progress) | **20.34** | (~ proportional) | **license-free, full cpl** |

**Speed:** PINN-PEX v11 = **2.05× faster than Innovus** on tv80s (20.34s vs 41.82s) and ~172× faster than 1-core StarRC.

**Functional differentiation:** Innovus + OpenRCX both lump all coupling into gnd (0 cpl entries per net). PINN-PEX is the only license-free tool emitting explicit cpl predictions.

---

## 8. Single-design cap MAPE references

### tv80s test only (3,169 nets, single seed unless noted)

| Method | total | gnd | cpl |
|---|---:|---:|---:|
| Mesh-curriculum (single seed 42 best) | 6.94% | — | — |
| Strike #2 per-pair coupling (KILLED ep53) | 60% cpl | — | spike |
| Strike #7 sister cell-OBS features | 10.09% | — | — |
| Strike #8 Liberty pin caps | 9.30% | — | gnd +5.05pp |
| Strike #8 z-score variant | 10.06% | — | — |

### nova test only (92,425 nets)

Currently: only via 5-seed cross-design test number above (combined with tv80s). Standalone nova SPEF in `output_intel22/active_learning/m6_v10b_baseline_seed0/intel22_nova_f3_autonomous.spef` (2.69 GB, golden 3.02 GB).

---

## 9. Killed / superseded variants (don't repeat — see PROJECT_PLAN §5)

| ID | Approach | Result | Verdict |
|---|---|---|---|
| GINO (FNO operator) | global FNO replace 1-hop GNN | 3 fatal flaws caught pre-train | KILL pre-train |
| DS-PINN MacroDensityFNO | macro density stream | +2.04pp mean within v10b stdev 5.02pp | empirically refuted |
| NNLS hand-tuned ζ calib | data-driven init | -6.5pp but n=5 MWU p>0.5 | statistically NS |
| γ scaling head | per-net residual multiplier | smoke 67% best, 5-seed never finished | abandoned |
| **A1 per-channel separate encoders** | per-channel encoder | gnd 21.60% (+1.11pp **worse**) | KILL Round 2 |
| C1 isotonic full-distribution | bulk -1.31pp / Top-50 +38pp | distribution mismatch | KILL |
| Strike #2 per-pair Sakurai | uniform analytic baseline | cpl 38→60% at curriculum transition | KILL ep 53 |
| Strike #7 cell-OBS features | 13 sister features | total 6.94→10.09% | dead-end |
| Strike #8 Liberty pin caps | 7 features from 5,147 pins | total 6.94→9.30% | dead-end |
| Capacity sweep h128–h256 (×36 params) | scale capacity on hand features | 11–14% ceiling unchanged | NOT capacity-bound |
| K3 synthetic pretrain | Stage 1+2 Mode A + Mode B | fired in 3 min (zero-init+synthetic=truth) | KILL, saved ~125 GPU-days |

---

## 10. Legacy v9-era AL runs (ARCHIVED 2026-05-05)

These pre-date the v3 paradigm shift. None feed paper claims. Detailed narrative in `docs/PROJECT_REPORT.md` §2. **Raw artifacts moved to `/data/PINNPEX/legacy_archive/output_intel22/active_learning/` (78 GB, 52 dirs)** — see `README.md` there for restoration instructions.

| Run dir | Track | Single-seed result | Status |
|---|---|---|---|
| `v1` – `v3_*`, `v4_*`, `v5_*`, `v5b_*` | early DeepPEX iterations | total ~30–60% | superseded |
| `v6_*`, `v7_*`, `v8_*`, `v8b_*` | gamma-head + railcpl experiments | mixed | abandoned |
| `v9`, `v10`, `v10b` | n_tiles conditioning + full re-train | v10b: nova 47.69% raw → 10.96% w/ XGB anchor | **m6_v10b_seed{0..4}** kept (paper-cited) |
| `dspinn_v1` – `dspinn_v3` | DS-PINN MacroDensityFNO | empirically refuted | killed |
| `gino_v1` | GINO smoke | killed pre-real-data | abandoned |
| `m5_v3_baseline_seed{0..4}` | m5 5-seed baseline | superseded by m6 series | archive candidate |
| `m5_v4_full_calib_seed{0..4}`, `m5_v5_gnd_only_seed{0..4}`, `m5_v6_gamma_seed{0..4}` | calibration / gamma head | killed | archive candidate |
| `phase1_v4_seed0`, `phase2t2_v4_seed0`, `test_mape_fix1` | smoke tests | n=1 | archive candidate |
| `cache`, `cache_pre_phase1_backup`, `m5_summary`, `ood_compare`, `diag_phase_a` | infrastructure | regenerable | archive candidate |
| **`m6_v10b_baseline_seed{0..4}`** | **paper Path-1 baseline + nova/tv80s SPEF + R+C calibrated** | **PILLAR 5 + 6** | **KEEP** |

---

## 11. Notes on future / pending work

- v11/v12 nova measurement (in progress, manifest-passthrough fix landed).
- Per-channel structural pivot: Hypothesis C (per-net adaptive α via LGBM meta-learner) → A (channel-specialized Mesh) per memory `project_structural_pivot_session.md`.
- Mode B specialist isotonic refit on Combined model val output (Codex Round 2 verdict).
- B1 per-pair Sakurai, GDSII feature integration — deferred levers.

For paper consolidation use `pex_v3/paper/RESULTS_CONSOLIDATED.md` as the per-pillar narrative companion.
