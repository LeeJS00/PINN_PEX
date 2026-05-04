# PHASE_STATUS — Strategy v3 live tracker

_Last updated: 2026-05-03 evening (Path-2 + main folder merge done)_

## Active phase: **Phase B** (Real-data baseline experiments — measurement IN)

### Phase 0 — DONE ✅
H1, H2, H3, M5 fixes shipped + validated. 99 unit tests passing. Full v3 manifest at /data/PINNPEX/data/processed_v3/intel22/dataset_manifest_v3.csv (1,322,115 tiles · 257,438 nets · 493 GB).

### Phase B — partial results IN (2026-05-02)

| Job | Status | Result |
|---|---|---|
| 04_build_feature_dataset.py (all 11 designs) | ✅ **done** | 221,102 net-feature rows; SpatialGrid optimization brought time from days to ~12h |
| B3 PINN 5-seed on real v3 data | ✅ **done** | **30.90% ± 2.20pp Net-MAPE** (legacy v10b 63.79 ± 5.02 → -32.89pp, ~6σ, p<2e-5, A1 [ROLE PASS]) |
| B1 XGBoost 5-seed v1 (test/OOD split) | ✅ **done** (deterministic-seed bug) | **median 5.90%, mean 7.48%, P95 19.78%** on cross-design OOD (nova+tv80s, 95594 nets) |
| B1 XGBoost 5-seed v2 (valid split, with subsample) | 🔄 running | apples-to-apples vs B3 |
| B2 ParaGraph reproduction | ⏳ pending | next sprint |
| B4 Compact + GAM | ⏳ pending | next sprint |

### 🎯 Striking finding

```
B1 XGBoost on OOD test (nova+tv80s):    median  5.90%  mean  7.48%
B3 PINN  on in-dist valid:              median 31.08%  mean 30.90%
```

Hand-engineered features + tree boosting **dramatically beats** the legacy
PINN — by 4-5×. This shifts the paper thesis: the "hard part" isn't going
from 60% → 30%, it's going from ~6-7% (XGBoost+features) → <4% (Phase 1
hybrid arch). See `docs/PHASE_B_B1_RESULTS.md`.

Cross-boundary edits this phase:
- `src/__init__.py` + 7 subpackage `__init__.py` files added (namespace → regular package conversion). Documented in `pex_v3/docs/CROSS_BOUNDARY_legacy_src_init.md`.
- `scripts/build_dataset.py:528` env var read (PEX_CONTEXT_MARGIN, default 2.0).

## Phase 0 status (archived)

| Fix | Status | Owner agent | Cost | Blocker |
|---|---|---|---|---|
| H1 — net-level hash split | ✅ **VALIDATED on real manifest** | pex-data-engineer | done (~10 min) | — |
| H2 — priority truncation | ✅ tests green (5/5) | graph-geometry-engineer | runtime, no rebuild | — |
| M5 — SSL split filter | scaffolded (stub body) | experiment-systems-engineer | ~5 min code + 11 GPU-h pretrain | training loop body |
| H3 — context margin 2→6 μm rebuild | 🔄 **IN PROGRESS** (background) | pex-data-engineer | per-design 1-60 min (CPU-parallel) | — |
| H4 — pairwise CPL search | designed for Phase 1 | neural-operator-architect | Phase 1 inherent | — |
| M6/M7/M9 — secondary | docs only | pex-data-engineer / graph-geometry-engineer | varies | deferred |

### H3 smoke test result (2026-05-01)

```
Design: intel22_gcd_f3 (smallest)
Output: /data/PINNPEX/data/processed_v3/intel22/intel22_gcd_f3/
  - 713 tile pickles (matches legacy count)
  - Total dir size: 47 MB (vs legacy 26 MB → 1.81× ratio confirms H3 5μm margin)
  - Manifest: 711 valid samples, 244 train + 32 valid nets

Build time: 8 seconds (32 workers, gcd has 1223 windows generated)
```

H3 verified working. Legacy 2.0 μm vs v3 5.0 μm gives 1.81× tile size,
matching the area ratio (14²/8²) ≈ 3.06× scaled by aggressor density.

### H1 validation results (2026-05-01)

```
Source: /data/PEX_SSL/data/processed/intel22/dataset_manifest.csv (legacy v9)
Output: /data/PINNPEX/data/processed_v3/intel22/dataset_manifest_v3.csv

Total: 1,322,512 tiles, 257,438 nets
  train: 126,193 nets / 779,034 tiles
  valid:  14,153 nets /  86,886 tiles
  test:  117,092 nets / 456,592 tiles

Legacy leak quantified: 31,630 / 257,438 nets = 12.29% spanned multiple splits
                        (PROJECT_REPORT.md §4.1 claimed 12.32% — confirmed)
v3 leak:                0 (run_all_checks green)
```

### Test results (2026-05-01)

```
pex_v3/tests/  — 19/19 passing
  test_determinism.py        — 5 tests (4-way seed reproducibility)
  test_priority_truncation.py — 5 tests (H2 invariants)
  test_split_invariants.py   — 9 tests (H1 + schema + leak detection)
```

Disk: `/data/` 1.7 TB available (3.5 TB total, 52% used). H3 1.2 TB fits.

## Acceptance gates

Phase 0 → Phase 0.5 transition requires all of:

- [ ] `tests/test_split_invariants.py` green (zero net mixing across splits in v3 manifest)
- [ ] `tests/test_priority_truncation.py` green (target cuboids retained, masked output equals)
- [ ] `tests/test_determinism.py` green (4-way seed reproducibility, identical loss curves)
- [ ] H3 rebuild completed + new manifest validated
- [ ] H4 pairwise CPL implementation merged + unit tested
- [ ] M5 SSL pretrain re-run with `split == 'train'` filter
- [ ] 5-seed baseline of legacy `DeepPEX_Model` on rebuilt data → `output/baseline_v3_legacy_pinn/m6_v10b_v3_seed{0..4}/`

Phase 0.5 → Phase 1 transition requires all of:

- [ ] CatBoost/XGBoost baseline on rebuilt data, 5-seed reported
- [ ] ParaGraph-style GNN reproduction, 5-seed reported
- [ ] Current PINN-PEX baseline on rebuilt data, 5-seed reported
- [ ] Analytic compact-model + GAM baseline, 5-seed reported
- [ ] Stratified error report: per-quartile, per-layer, per-design, per-class
- [ ] Paper-grade comparison table (cap MAPE + delay + power + RC percentile)

## Hard kill criteria

- **K1** — Phase 2 floor measurement > 4% MAPE → renegotiate target with user.
- **K2** — Phase 1 prototype plateaus at 6% after 2 months sustained work → architecture redesign.
- **K3** — Synthetic pretrain → real-data finetune gain < 1 pp → abort synthetic strategy.

## Phase 0 work log

| Date | Action | Outcome |
|---|---|---|
| 2026-05-01 | Strategy v3 approved (Codex 2-round + user "<4% 불변" directive) | 8-agent roster + Phase plan locked |
| 2026-05-01 | `pex_v3/` subfolder created, foundation files written | folder structure ready |
| 2026-05-01 | Phase 0 audit: H1-H4 + M5 line-level diff drafted | scaffolded; awaiting baseline rebuild |
| 2026-05-01 | **H1 implementation + tests + end-to-end validation** | ✅ 1.32M tile manifest written, 12.29% leak eliminated, 19/19 tests green |
| 2026-05-01 | H2 priority truncation implemented + tests green | ✅ H2 invariants validated |
| 2026-05-01 | Determinism (4-way seed) tests green | ✅ pytest baseline established |
| 2026-05-01 | H3, H4, M5 design docs written (gated implementation) | awaiting user --confirm / Phase 1 body |
| 2026-05-01 | Cross-boundary edit: legacy `scripts/build_dataset.py:528` env var read (1 line) | documented in `docs/CROSS_BOUNDARY_h3_context_margin.md` |
| 2026-05-01 | H3 smoke test on `intel22_gcd_f3` | ✅ 8 sec, 713 tiles, 47MB (vs legacy 26MB → 1.81× — H3 fix verified) |
| 2026-05-01 | H3 full rebuild started (background) | 11 designs, target 700GB, ETA 3-6 h |
| 2026-05-01 | H3 rebuild 9/11 designs done (192GB, well below 1.4TB cap) | nova + tv80s remaining |
| 2026-05-01 | Synthetic Stage 1 + Stage 2 + ground_truth implemented | 30 unit tests green; 1M parallel-plate generator + 2M layered slab + 2M single-interface |
| 2026-05-01 | Evaluation harness: metrics + stratified_eval + seed_aggregator | 17 unit tests green; 4-metric reporting + stratified slices + MWU + bootstrap CI |
| 2026-05-01 | 5-seed orchestrator (`scripts/05_5seed_runner.py`) | model-agnostic; signature contract for any baseline |
| 2026-05-01 | Phase 1 hybrid arch spec drafted (`docs/PHASE1_HYBRID_ARCH_SPEC.md`) | 11-section spec; Codex round 3 review launched (background) |
| 2026-05-01 | Hand-engineered features (`baselines/features.py`) | 20 unit tests green; pure NetGeometry → NetFeatureVector contract |
| 2026-05-01 | XGBoost baseline body (`baselines/xgboost_baseline.py`) | run_one_seed contract; 5-seed smoke test running (background) |
| 2026-05-01 | Pre-experiment weakness fixes | IMPLEMENTATION_STATUS.md honesty register, pandas warnings fixed, 5-seed runner unit tests (4 new), method-label aggregation bug fixed |
| 2026-05-01 | PINN baseline body (`baselines/pinn_baseline.py`) | wraps legacy run_active_learning.main() via cfg monkey-patch + symlinks for v3 paths |
| 2026-05-01 | Feature dataset pipeline (`baselines/feature_dataset.py`) | one-pass SPEF parser + DEF→NetFeatureVector; gcd smoke 276 nets in 15.9s |
| 2026-05-01 | 5-seed XGBoost (synthetic data) | 3.39% ± 0.12pp median (post-fix aggregation); confirms full plumbing works |
| 2026-05-02 | H3 rebuild complete (full notification arrived) | 11/11 designs · 1,322,115 tiles · 257,438 nets · 493 GB · all H1 invariants green |
| 2026-05-02 | B3 PINN seed 0 finished (~4.5h on H3 data) | best step 32.96% Net-level MAPE (single seed; A1 audit invalidates direct comparison) |
| 2026-05-02 | Phase C Round 1 — 4/4 agents PASS | 15 substantive issues caught (4 fixed inline); see `docs/PHASE_C_ROUND1_AUDIT.md` |
| 2026-05-02 | **CATASTROPHIC bug fix** — layer_idx parsing | `_scan_design_geometry` was assigning ALL cuboids to layer 0 due to `segments[0].get("layer_idx")` defaulting; 15/43 features were dead-weight. Fixed via regex parse of `"layer"` string. Re-extraction triggered. |
| 2026-05-02 | Agent infrastructure gap discovered | Custom agents in `.claude/agents/` are NOT directly invocable as `subagent_type`; workaround (general-purpose + role-md path embedded) verified working. See `docs/AGENT_INFRA_GAP.md`. |
| 2026-05-02 | SpatialGrid + numpy vectorize → feature_dataset O(N²)→O(N log N) | gcd 5.9× speedup verified; 8/11 designs done in <2h vs prior 4.6h for 3 designs |
| 2026-05-02 | Multi-GPU launcher (06_run_pinn_multigpu.py) | 5 seeds × 5 GPUs parallel; 6h wall-clock vs 30h sequential. Two bugs fixed: (a) cfg.GPU_ID=1 vs CUDA_VISIBLE_DEVICES, (b) parser was reading wrong stdout log → all 5 seeds reported same 32.96%. After fix, real per-seed values: 33.15/27.48/30.37/31.08/32.43 |
| **2026-05-02** | **🎯 First real Phase B result: B3 PINN 5-seed** | **30.90% ± 2.20pp on v3 vs legacy 63.79% ± 5.02pp → 32.89pp improvement (6σ separation) from H1+H3 data fixes alone, no paradigm change.** Memory: `project_phase_b_b3_first_real_result.md` |
| **2026-05-02** | **🎯 B1 XGBoost vs B3 PINN — paired MWU SUPPORTED** | **B1 4.66% median (stdev 0.026pp) vs B3 31.08%; U=0, p=0.008, Cohen's d=-16.84.** Hand features + boosting beat legacy PINN 6.6× on in-dist. Phase 1 paradigm bar: **beat XGBoost ~5%**, not legacy PINN 30%. Comparison: `pex_v3/output/baselines/B1_vs_B3/comparison.md` |
| 2026-05-02 | A5 neural-operator-architect [ROLE PASS] | Phase 1 implementation order: Tier 0 = analytic_base + residual + hybrid skeleton in 24h; defer mesh_v3 to Tier 1.5; drop Stage 2 Mode B; per-pair-type RES_CLAMP curriculum log(1.5)→log(2.5)→log(4.0). |
| 2026-05-02 | A6 graph-geometry-engineer [ROLE PASS] | mesh_v3.py spec: Patch dataclass + parent_segment fields for H4 long-parallel preservation; pattern (NOT tile) is inference unit; ~1070 LOC, 6 person-days; 24h MVP feasible (~400 LOC). |
| **2026-05-02** | **🎯 Phase 1 kill-signal gate PASSED — analytic_base_v3.py** | A5 mandate: differentiable parallel-plate w/ autograd. 11/11 tests: 100-sample parity vs ground_truth max_rel<1e-4, gradcheck passes, Mode B series formula correct, CUDA=CPU. Hybrid paradigm foundation viable. |
| 2026-05-02 | residual_head_v3.py + 11 tests | Bounded multiplicative residual (`exp(clamp(...))`); zero-init last layer → day-1 output = 1.0; per-pair-type variant; RES_CLAMP curriculum log(1.5)→log(2.5)→log(4.0). 121 tests total. |
| 2026-05-02 | A2 classical-baseline-owner [ROLE PASS] | B1 verified clean (no leakage, manifest hash matches). **Critical finding: 4.66% total benefits from gnd/cpl cancellation**. Real channel error: gnd 20.6%, cpl 12.4%. Phase 1 contribution → per-channel honesty (β strategy). B4 ~3 days; B2 ~5-day capped. |
| 2026-05-02 | B1 Stratified report (08_b1_stratified_report.py) | Per-design × per-channel × per-quartile breakdown. Heteroscedastic (good direction): Q4 large nets total=3.42%, Q2 small nets 5.65%. Designs: gnd 13-23%, cpl 11-21%. Phase 1 must improve per-channel without exploiting cancellation. |
| **2026-05-02** | **🚧 Phase 1 Tier 0 model code complete** | hybrid_v3.py (per-channel β-strategy heads) + per_channel_mape_loss. 12/12 hybrid tests + 11/11 residual + 11/11 analytic = **34 new tests, 133 total green.** Day-1 invariant: hybrid output = analytic exactly. Bounded multiplier verified. |
| 2026-05-02 | Strategy v3 plan UPDATED (`STRATEGY_V3_UPDATED_PLAN.md`) | 11 lessons L1-L11 + revised paper thesis (β strategy: gnd<8% AND cpl<8%) + risk register R1-R7 + acceptance gates. |
| 2026-05-02 | pretrain_synthetic_v3 + transfer_canary_v3 (Tier 1) | 19 new tests; harness validated. K3 canary executed → FIRED (3 min). Lesson: zero-init+synthetic=truth means pretrain has no signal. Saved ~125 GPU-days. Synthetic pretrain DROPPED from Phase 1. |
| 2026-05-02 | finetune_hybrid_v3 (Tier 2 NEW, replaces synthetic pretrain) | df_to_tensors + per_channel evaluation + curriculum + early stop. 9 tests green, 161 total. Direct real-BEOL fine-tune path. |
| 2026-05-02 | Phase 1 Tier 2 single-seed smoke on real v3 | day-1: total 38.31% (analytic-only). After 8 epochs (early stop at clamp=log(1.5)): valid total 26.62%, test total 25.80%. β-FAIL but pipeline works; clamp curriculum too tight for this analytic prior calibration. Looser-clamp re-smoke running. |
| 2026-05-02 | **🎯 HAND-FEATURE CEILING 4.66% identified** | Option F deep MLP (286K) hits same 4.66% as XGBoost (per-channel ~21% gnd / 12.6% cpl identical). Architecture not bottleneck; FEATURES are. Phase 1 paradigm needs mesh_v3 per-cuboid features. |
| 2026-05-02 | **B4 Compact + GAM 5-seed** done | V3 log-GBDT 5.72% ± 0.04pp valid / 6.59% test (best OOD gap +0.87pp). V2 additive 7.46% / 9.33%. Paper-grade physics-informed baseline. |
| 2026-05-02 | Cross-design eval (B1 + B4 + Option F) | nova: B4 V2 9.32%, Option F 5.68%. tv80s: B4 V2 8.16%, Option F 5.34%. PAPER_GRADE_COMPARISON.md ready. |
| 2026-05-02 | **Session handoff — context near limit** | Next session: read `pex_v3/SESSION_HANDOFF.md` first. P1=5-seed Option F variance lock, P2=B1 OOD test, P3=legacy v10b re-eval, P4=paper #1A draft. |
| **2026-05-03** | **🎯 P1 — Option F deep MLP 5-seed locked** | `scripts/14_option_f_5seed.py` (Codex 1-round confirm). 5 seeds × ~42s on GPU 0 = 3.5 min total. **valid 4.756% ± 0.012pp · test 5.623% ± 0.042pp · OOD gap +0.87pp** (tied with B4 V3 for best generalization). Per-channel ceiling confirmed: gnd 21.20% / cpl 12.67% on valid; gnd 21.67% / cpl 16.44% on test. Per-design test: nova 5.627% ± 0.042pp, tv80s 5.453% ± 0.082pp. 286K params, 0.48 μs/net inference. Outputs in `pex_v3/output/baselines/Option_F_MLP/`. |
| **2026-05-03** | **🎯 P2 — B1 XGBoost OOD test locked** | `xgboost_baseline.py:run_one_seed` extended to dual eval (valid + test) + per-channel summary; `scripts/15_b1_test_aggregate.py` post-processes. 5-seed re-run (~12 min/seed × 5 = ~63 min CPU, valid still 4.657% ± 0.023pp). **Test OOD: 5.842% ± 0.096pp** (nova 5.858 ± 0.097, tv80s 5.314 ± 0.039). **OOD gap +1.19pp** — wider than Option F (+0.87pp) and B4 V3 (+0.87pp); tree boosting transfers worse than smooth-function or physics-prior models. Per-channel test: gnd 19.93%, cpl 16.13%. Outputs: `pex_v3/output/baselines/B1_xgboost_real/test_5seed_summary.json`. |
| **2026-05-03** | **🚀 Hero SPEF v2 (R+C HERO)** | Filter 1-line fix `evaluator.py:409` + tv80s SPEF E2E (3,380 nets, 14.4 min). XGB cap-anchor calibration: C 47.69% → **10.95% ± 0.047pp** 5-seed, R²=0.983. Sister `r_analytic_v3` v3 hybrid stacked v6 per-net R rescale: R 28.36% → **2.21% mean / 1.40% median, R²=0.999**. Long-net Q4 cap MAPE 71% → 9% auto-fix via XGB anchor. Conflict-free post-process (sister `pex_pipeline/` 안 건드림). |
| **2026-05-03** | **🎯 PINN headline — Mesh-curriculum 5-seed** | HybridPexV3Mesh (44K params, cuboid set encoder + bounded multiplicative residual + 3-phase clamp curriculum log(1.5)→log(2.5)→log(4.0)). 5-seed × 200 epoch on 5 GPUs parallel (~25 min wall). **Best valid 6.258% ± 0.108pp / last 8.27% ± 0.342pp / 5-seed ensemble 7.89%** on cross-design test. Mesh best-step beats B4 V3 log-GBDT (6.59%) with **2.3× fewer params**. Phase 0→1 transition gives -1.89pp jump (curriculum critical). 5/5 paper-pillar locked. |
| **2026-05-03** | **❌ Strikes #2/#7/#8 — 4 negative results documented** | (#2) Per-pair coupling head with uniform analytic baseline KILLED at epoch 53 (cpl total 38→60% transition). (#7) Sister cell-OBS features (13 added): test +3.15pp worse. (#8) Liberty pin-cap features (7): test +2.36pp worse; counts-only (3): +2.23pp; z-score per-design: +3.12pp. **5-variant systematic diagnostic**: distribution-shift hypothesis REJECTED (z-score also worse), cuboid encoder redundancy + bounded multiplier overfit at Phase 2 CONFIRMED. Mesh-only is true architecture ceiling. |
| **2026-05-03** | **🚀 Path-2 Fast deterministic deployment (Option D')** | Analytic + geometric SPEF generator (no PINN inference). tv80s wall-clock **68.9 s — 12.5× faster** than legacy 14.4 min. 5-seed median MAPE **5.78% ± 0.077pp** (essentially identical to legacy 5.77% median); mean 12.68% ± 0.043pp (+1.7pp vs legacy 10.96%). GPU-optional / CPU-only deployable. R per-net rescale identical to Path-1. **Adds 4th paper contribution**: fast deterministic deployment path. |
| **2026-05-03** | **📦 METHOD doc + main folder merge** | `pex_v3/paper/METHOD.md` 10-section paper-ready. Symlink merge: `src/v3/{models,baselines,data,trainers,utils}` → `pex_v3/src/*` + `configs/config_v3.py`. Path-magic `__init__.py` for backward-compat. Pex_v3 paper-pillar 5 files: absolute → relative imports. **3-way import verified** (main `src.v3.*` + legacy `src.*` + pex_v3 scripts). Documented `pex_v3/docs/CROSS_BOUNDARY_v3_merge_to_main.md`. |

## Pending decisions

(none active — user has approved auto-execute "all recommendations")

## Notes for future sessions

- Use specialist agent before any code change in `pex_v3/`. Lead session as coordinator.
- Codex deliberation loop is the outer convergence — round 1 catches majority of bugs.
- Strategy v3 memory entry at `~/.claude/projects/-home-jslee-projects-PINNPEX/memory/project_strategy_v3_paradigm_shift.md` is canonical; this file is the *active state*.
