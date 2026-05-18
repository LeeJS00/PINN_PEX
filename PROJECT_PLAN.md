# PINN-PEX PROJECT PLAN

_Last updated: 2026-05-11_
_Status: v3 paradigm shift LOCKED — 5/5 paper pillars paper-grade, joint Pareto v11/v12 frontier, 2.05× faster than Innovus, license-free vs StarRC. **2026-05-11**: Phase F (ASAP7 7nm cross-PDK sprint) added — see §7._
_Order: **두괄식** — Task → Policies → Method → Results → Previous approaches (mistakes)._
_Canonical references: `docs/PROJECT_REPORT.md` (deep history), `PEX_FRAMEWORK.md` (pipeline definition), `pex_v3/paper/` (paper drafts)._

---

## 1. TASK — what we are building

### 1.1 Headline (one sentence)
Build a **physics-informed neural extractor** that turns routed layout (DEF + tech LEF + Liberty + layer stack) into **StarRC-quality SPEF in seconds, license-free**, beating commercial pattern-matching tools (Innovus, OpenRCX) on cross-design OOD.

### 1.2 Concrete deliverables (paper-grade)
| # | Deliverable | Current state |
|---|---|---|
| D1 | DEF→SPEF E2E pipeline (parsing → cuboid tile → mesh model → calibration → SPEF write) | **Locked.** `experiments/cross_design_tv80s_2026_05_02/scripts/predict_spef_e2e.py` |
| D2 | 5-seed cross-design OOD measurement (TV80s + Nova) | **Locked.** v11 total 6.821 ± 0.040, v12 6.856 ± 0.035 |
| D3 | Industry tool comparison (vs Innovus, OpenRCX, StarRC) | **Locked.** Innovus 6.96 / OpenRCX 8.83 / PINN v11 6.82 — and 2.05× faster than Innovus |
| D4 | Per-net R calibration (sister NNLS+LightGBM) | **Locked.** R MAPE 2.21% mean, 1.40% median, R² 0.999 |
| D5 | Hybrid C calibration (XGB anchor + PINN distribution + α-blend) | **Locked.** C MAPE 10.96% (5-seed), R² 0.984 |
| D6 | Paper draft (METHOD + RESULTS + EXPERIMENTS) | In progress. `pex_v3/paper/METHOD.md` done; RESULTS_CONSOLIDATED + OUTLINE in place. |
| D7 | Per-channel gnd / cpl reduction below 17% / 13% | **REFRAMED 2026-05-05** (pex_v4 sprint, Codex R2). Per-net unweighted matched per-channel 10–17% NOT achievable via current architecture (4-way oracle bound = gnd 14.07 / cpl 11.21). **Block-level balance ±2.5% / cap-weighted total MAPE ≤5% IS achievable on DEF/LEF inputs without GDSII** — pex_v4 targets sign-off-grade IR-drop / power agreement license-free vs StarRC via 3-tier downstream-aware framework (Tier 1 block balance / Tier 2 weighted MAPE / Tier 2.5 crosstalk noise / Tier 3 per-net envelope). Original 19.5/15.0 unweighted matched targets retained as superseded exploratory goals; new sanity envelope = gnd≤24, cpl≤18, gnd_heavy_cpl≤35. See `pex_v4/CLAUDE.md` + `pex_v4/docs/PHASE_A_RESULTS.md`. |

### 1.3 Scope boundaries
- **In scope:** DEF + tech LEF + Liberty + layer stack inputs only; SPEF (Voltus / StarRC compatible) outputs only; cross-design OOD (held-out designs) is the headline metric. **As of 2026-05-11, cross-PDK OOD (intel22 + ASAP7 7nm) is also in-scope** — see Phase F (§7).
- **Out of scope:** GDSII / substrate-aware extraction, transient simulation, parasitic-aware timing, in-design fine-tune. (D7 ceiling is hit precisely because of this scope choice.)

---

## 2. POLICIES — critical operational rules

**Cut-off principle:** this section lists ONLY rules whose violation causes (a) system damage, (b) data loss, or (c) invalidates a paper-grade claim. Soft guidelines, design heuristics, and architecture-specific lessons live elsewhere (see §2.9 Reference pointers) and are NOT enforced here. This keeps the model free to make context-dependent judgments outside the 8 core invariants below.

### P1 — `tool` path: project-local only
- **NEVER** write or create anything at the system-root path `/tool`. It is a root-owned read-only empty directory; writes fail or pollute the system.
- When the user says "tool" / "tool 폴더" / "tool 디렉토리", they ALWAYS mean `/home/jslee/projects/PINNPEX/tool/` (project-local PnR tooling).
- *User-flagged 2026-05-04. Mirrored in `docs/PROJECT_REPORT.md` §0.1.1 and memory `feedback_tool_path_policy.md`.*

### P2 — Scratch directory: never `/tmp`
- All large cuboid pkls, intermediate SPEF outputs, and e2e measurement scratch live under `/data/PINNPEX/scratch/`.
- `/tmp` is **forbidden** for project I/O.
- *Past incident: 2026-05-04 nova run dumped 258 GB into `/tmp` and bricked the host.*

### P3 — Legacy manifest: read-only
- **NEVER** overwrite `/data/PEX_SSL/data/processed/intel22/dataset_manifest.csv`. It is the legacy v9 manifest and is irreplaceable post-hoc.
- v3 work writes to `/data/PINNPEX/data/processed_v3/intel22/dataset_manifest_v3.csv` only.
- *Source: `pex_v3/CLAUDE.md` §Data path discipline.*

### P4 — Net-centric splits & sampling
- All splits, sampling, validation, and statistics group by `(design, net)` first. **Never** split or `head(N)` on tile rows.
- Tile-level splits cause net-leak across train/valid/test, invalidating any OOD claim.
- *Past incident: legacy net leak quantified at 12.29% on real manifest (Phase 0 H1).*

### P5 — TEST_DEFS held out from AL pool
- Designs in `cfg.TEST_DEFS` (currently `nova`, `tv80s`) are **never** seen during AL pool selection or training, regardless of `AL_SAMPLING_METHOD` (`Predefined` / `SSL` / `Sorted`).
- Any sampling method that admits a test design silently invalidates every cross-design OOD number reported in the paper.

### P6 — 5-seed protocol before any *paper-grade* claim
- A claim like "X beats Y" must pass **paired Mann-Whitney U + Cohen's d + bootstrap CI** on a 5-seed run before it appears in the **paper / leaderboard / memory / PR description / external report**.
- Iteration-time smoke runs (single-seed, exploratory) are exempt — but their result is **not** a claim and must not be surfaced as one.
- *Past evidence: single-seed smoke consistently beats 5-seed median by 0.5–1.0pp across auto_optimize rounds 2/3.*

### P7 — StarRC oracle: full-chip DEF only, never on tiles
- `FullChipPEXOracle.generate_golden_spef` and any future oracle wrapper run StarRC on the full chip DEF.
- **Never** invoke StarRC on cropped tiles, sub-nets, or partial layouts — the result is not the same physical answer (boundary conditions differ) and the cost is ~10 min/design.
- *Source: `run_active_learning.py` design + `pex-data-engineer` role.*

### P8 — Stage 2 inference: manifest passthrough required
- The DEF→SPEF E2E Stage 2 (`build_features_inference`) must consume a manifest (CSV) directly. **Never** auto-discover via `rglob + gzip.open + pickle.load` over per-net pkls.
- For new entrypoints: pass the Stage 1 `cuboids_map.csv` through to Stage 2; do not implement disk-walk fallbacks.
- *Past incident: nova rglob path stalled >4 hours on 684K pkls (2026-05-04). Fixed in `predict_spef_e2e.py:160-167`.*

### 2.9 Reference pointers (NOT enforced here)
The following are useful guidelines but live in their source documents and are **not** elevated to policy. Read them when relevant; do not treat them as hard invariants in PROJECT_PLAN review.
- **Codex deliberation loop** for non-trivial design changes → global `~/.claude/CLAUDE.md`.
- **`pex_v3/` boundary rule** (no edits to legacy `src/` / `scripts/` / `configs/` without process) → `pex_v3/CLAUDE.md`.
- **`TORCH_COMPILE_DISABLE=1` for paper runs** → `pex_v3/docs/STRATEGY_V3_UPDATED_PLAN.md`.
- **Custom agent invocation pattern** (use `general-purpose` + role-md path) → `feedback_agent_invocation_pattern.md`.
- **Loss design rules 1-6** (MAPE-aligned, heteroscedastic cap_weight, KCL `.detach()`, global curriculum step, no bundling correlated changes) → `feedback_loss_design_principles.md`.
- **Per-channel β strategy** (no `total = gnd + cpl` only training) → `pex_v3/SESSION_HANDOFF.md` + `pex-gnd-allocator-owner` role.
- **Anti-pattern catalog** (Strikes #2/#7/#8 scalar features, A1 per-channel encoder, synthetic pretrain K3 fired) → §5 below + `joint_pareto/README.md`.
- **Joint-Pareto runtime cap (≤75 s on tv80s)** → `pex_v3/joint_pareto/README.md` (workspace-local).
- **Hard kill criteria K1/K2/K3** → `pex_v3/PHASE_STATUS.md` (live phase tracker).
- **Anti-overclaim publishing rules** (baselines + Innovus + OpenRCX co-measured, per-channel breakdown) → `benchmarking-statistician` role.
- **`RUN_NAME` / `--model_name` consistency, single-GPU only** → project `CLAUDE.md`.

---

## 3. METHOD — current architecture & pipeline

### 3.1 Pipeline overview (7 stages)
```
[DEF + tech LEF + Liberty + layer stack]
        │
        ▼  (Stage 1)  per-net cuboid tiling     →  *.pkl.gz + cuboids_map.csv
        ▼  (Stage 2)  feature build (manifest fast-path mandatory)
        ▼  (Stage 3)  HybridPexV3Mesh forward   →  per-net (gnd, cpl) raw
        ▼  (Stage 4)  α-blend (XGB anchor + PINN distribution, α∈[0.20, 0.30])
        ▼  (Stage 5)  LGBM 8-feat residual calibration
        ▼  (Stage 6)  per-net R sister hybrid (NNLS + LightGBM)
        ▼  (Stage 7)  SPEF write (Voltus / StarRC compatible)
[SPEF]
```

### 3.2 Model — `HybridPexV3Mesh` (44.7K params)
- Cuboid set encoder (3-pool: mean / max / sum); shared MLP across cuboid set.
- Bounded multiplicative residual on top of analytic compact prior.
- Curriculum on the residual clamp: 0.405 → 0.916 → 1.386 over 200 epochs.
- Per-channel heads share the encoder (per-channel separate encoders = A1, KILLED).
- Training: AdamW, lr 1e-3, bs 256, no torch.compile (Mesh-curriculum stays).

### 3.3 Calibration — α-blend (Path-2 v10/v11/v12)
- Per-net total = α · XGB_total + (1−α) · PINN_total (per-channel ratio inherited from PINN's mesh-only ratio).
- α=0.20 ⇒ v11 (best total, 6.821 ± 0.040), α=0.30 ⇒ v12 (best per-channel Pareto, gnd 22.59 / cpl 17.53).
- Single-pass calibration, no isotonic. (C1 isotonic full-distribution variant FAILED Top-50: +38pp; killed.)

### 3.4 Per-net R calibration (sister)
- NNLS + LightGBM trained on 9 cross-design TRAIN_SPEFS, applied as SPEF post-process.
- Mean MAPE 2.21%, median 1.40%, R² 0.999. Achieves <1% mean fundamental DEF/LEF limit (sister-reported).

### 3.5 Datasets & splits (canonical)
- **TRAIN (9 designs):** `aes_cipher_top, gcd, ibex_core, ldpc_decoder_803_3an, mc_top, spi_top, usbf_top, vga_enh_top, wb_conmax_top`.
- **TEST (OOD, 2 designs):** `nova` (92,425 nets) + `tv80s` (3,169 nets) = 95,594 test nets.
- Blacklisted: `mpeg2_top` (instability — auto-skip in `AL_SAMPLING_METHOD = "SSL"`).
- Cuboid pkls live under `/data/PINNPEX/data/processed_v3/intel22/per_net_cuboids` (H3 rebuild).
- **Corner / temperature (added 2026-05-11 for Phase F):** all StarRC golden SPEFs are **typical corner, 25 °C, StarRC S-2021.06-SP2**, both PDKs. intel22 SPEF headers leave `CORNER_NAME` blank but use the `tttt.nxtgrd` (TT/TT/TT/TT) file; ASAP7 SPEF headers explicitly name `typical` + `asap07_x1.nxtgrd`. Cross-PDK A/B is corner-matched — never mix corners in a single table.

### 3.6 Tooling baselines (verified via `docs/pex_tool.csv`)
- StarRC (golden, MAPE = 0): 1-core real time tv80s 3,496s (~58 min), nova 31,200s (~8.7 hr).
- Innovus (4-core): MAPE tv80s 4.87% / nova 6.15% — real time tv80s 41.8s / nova 122s.
- OpenRCX (4-core): MAPE tv80s 7.60% / nova 7.89% — real time tv80s 5.1s / nova 64s.
- PINN-PEX v11: MAPE tv80s 6.82% (5-seed) — runtime ~20s standalone (2.05× faster than Innovus, vs 1-license restriction).

---

## 4. RESULTS — locked headline numbers

### 4.1 Joint Pareto frontier (5-seed, cross-design TV80s OOD)
| Variant | Total MAPE | gnd | cpl | Wall (s) | Notes |
|---|---|---|---|---|---|
| **v11 (α=0.20)** | **6.821 ± 0.040** | 22.83 ± 0.07 | 17.77 ± 0.03 | 20.34 | Best total |
| **v12 (α=0.30)** | 6.856 ± 0.035 (NS vs v11) | **22.59 ± 0.06** | **17.53 ± 0.03** | 20.42 | Best per-channel Pareto over v11 (p<0.01 both channels) |
| Innovus 4-core | 4.87 (tv80s only) | n/a | n/a | 41.82 | License-bound |
| OpenRCX 4-core | 7.60 | n/a | n/a | 5.10 | License-free, but 0 cpl entries |
| StarRC 1-core | 0.0 (golden) | 0.0 | 0.0 | 3,496.6 | License-bound + slow |

### 4.2 Per-channel SPEF analysis (PINN-PEX vs pattern matchers)
- PINN gnd MAPE 18.84% / cpl MAPE 13.96% (decomposed from total).
- Innovus / OpenRCX: emit **0 cpl entries per net** (lump all coupling into gnd) — no per-channel comparison possible. PINN is the only license-free tool with explicit cpl prediction.

### 4.3 Resistance pillar
- Sister NNLS + LightGBM hybrid: R MAPE 2.21% mean / 1.40% median / R² 0.999 — ~7× better than Innovus (14.93%), ~26× better than OpenRCX (58.39%).

### 4.4 Hybrid C calibration breakthrough
- tv80s full-chip SPEF MAPE: 47.69% (raw) → **10.95% ± 0.047pp** (5-seed) via XGB anchor + PINN distribution.
- Long-net Q4 71% → 9% auto-fix. R² 0.984.

### 4.5 5-pillar paper status
1. PINN improvement (Mesh-curriculum 6.26% best-step, 2.3× fewer params than B4 V3) — **paper-grade**.
2. Full-chip SPEF E2E (Voltus-compatible, 7-stage pipeline) — **paper-grade**.
3. License-free vs StarRC (no per-license bottleneck, 2.05× faster) — **paper-grade**.
4. Hybrid C calibration (XGB anchor + PINN distribution) — **paper-grade**.
5. Per-net R calibration (NNLS + LightGBM, R² 0.999) — **paper-grade**.

### 4.6 Information-bound finding (D7 verdict)
- 4-way model oracle (XGB + B4 + Option F + Mesh) bound: gnd 14.07% / cpl 11.21%.
- 56% of nets exceed 10% gnd at oracle — DEF/LEF inputs are insufficient for 10% per-channel.
- 10% per-channel target requires GDSII / substrate-aware extraction (~4 weeks).

### 4.7 Refinement sprint v3 LOCK (2026-05-18, post-pivot TreePEX) ✅

User pivot 2026-05-18: "필수 요소만 남기기 — model engineering". Ablation-driven
pruning of TreePEX stack. Phase A (inference toggle, cached features, ~30s) +
Phase B (5-seed retrain, B1/B2/B3) + intel22 ablation parallel.

| Lever ablation | intel22 ΔMAPE | ASAP7 ΔMAPE | Decision |
|---|---|---|---|
| L5 calibration off | **−0.10 / −0.14 IMPROVE** | net 0 | 🗑 **DROP both PDKs** |
| Specialist d9 n750 → d8 n500 | N/A | ≤−0.04 improve | ✅ **SIMPLIFY** (3× smaller weights) |
| L6 noise σ=0 retrain | not measured | +0.6 / +0.57 pp | ✅ KEEP (B1) |
| V4 H3-off (V3-only) retrain | not measured | +1.36 / +1.83 pp | ✅ KEEP (B3) |
| Ridge ← XGB fanout proxy | +0.04 / +0.11 sig | +0.36 / +0.29 sig | ✅ KEEP (A2) |
| L11 specialist off (ASAP7) | N/A | nova R² −0.033 | ✅ KEEP (A4) |

**Post-sprint canonical numbers (MAPE_med, 5-seed ensemble)**:

**⚡ Warm path** (features cached → inference; label-leak fanout):
- intel22 tv80s_f3 **4.95 % / R² 0.9936** (warm 11.27 s) — was 4.98 %
- intel22 nova_f3 **5.34 % / R² 0.9914** (warm 82.10 s) — was 5.28 %
- ASAP7 tv80s_x1 **6.72 % / R² 0.9854** (warm 9.68 s) — was cold 7.00 %

**❄️ Cold path** (DEF→features→inference; proxy fanout, StarRC fair scenario):
- intel22 tv80s_f3 **4.95 % / R² 0.9933** (cold 68.31 s) — Δ vs warm +0.002 pp (proxy near-perfect)
- intel22 nova_f3 **5.47 % / R² 0.9895** (cold 4767 s / 80 min) — Δ vs warm +0.14 pp
- ASAP7 tv80s_x1 **7.00 % / R² 0.9827** (cold ~70 s) — Δ vs warm +0.28 pp
- ASAP7 nova_x1 **7.93 % / R² 0.9699** (cold ~54 min) — was 7.90 %, ablation runner verified

**Path separation 의무** (2026-05-18 user directive): warm vs cold는 fanout source +
feature 추출 wall 모두 다름; 절대 같은 표에 섞지 말 것. See `~/.claude/.../memory/feedback_warm_cold_path_separation.md`.

Commit: PINNPEX `93f7c45` / TreePEX `1d57f42`. Memo: `~/.claude/.../memory/project_refinement_sprint_v3_lock.md`.

### Next priority (post-sprint, advisor-facing)
1. Push commits to GitHub (LeeJS00/PINNPEX + LeeJS00/TreePEX)
2. Paper draft propagate new ablation table (PAPER_TABLES_v2.md rev 2 already has sprint sub-section)
3. Phase F ASAP7 7nm sprint — minimal canonical already cross-PDK locked, transfer matrix F2a/b/c/d 시도
4. Long-tail (decile-9 bimodal) lever: L15 hierarchical 2-tier (gold>3fF→{3-15 mid, >15 mega})는 cosmetic — paper-grade가 아니라 rebuttal-tier

### Drift watch (post-sprint)
- **Specialist swap 자체가 paper claim**: 단순화 + 3× smaller weights + 동등 accuracy — 이게 "model engineering" 핵심 narrative. PINN 비교의 simplicity advantage에 직접 기여.
- D9 bimodal symmetric error (FE_* over / n_961xx under): monotone post-hoc 처리 불가능 (L11.b NEG로 증명). Long-tail accretion-by-lever 재시도 금지.
- 모든 새 lever 시도 시 3-gate (paired Wilcoxon + Holm + bootstrap-BCa CI + per-decile no-regress > 0.5pp) 의무.

---

## 5. PREVIOUS APPROACHES — what failed and why (don't repeat)

These are recorded so neither the user nor a future agent re-runs them. Each entry: *approach → failure → root cause → archived path*.

### 5.1 GINO (FNO operator) — never trained
- **Idea:** Replace 1-hop GNN with global FNO to handle Poisson non-locality.
- **Failure:** Critical analysis caught 3 fatal flaws (irregular grid mismatch, no boundary handling, memory blowup) before training.
- **Lesson:** Codex round 1 saved ~30 GPU-days. Do not revive without resolving all 3 flaws.

### 5.2 DS-PINN (MacroDensityFNO) — empirically refuted
- **Idea:** Macro density stream + flux head conditioning for long-range screening.
- **Failure:** 5-seed ablation showed +2.04pp mean (within v10b stdev 5.02pp), variance +56%. Net effect = noise.
- **Archived:** `src/models/_archive/dspinn/`, `src/data/_archive/dspinn/`.

### 5.3 Data-driven calibration init (NNLS hand-tuned ζ) — statistically NS
- **Idea:** Replace hand-tuned ζ with NNLS-fit values from TRAIN_SPEFS.
- **Failure:** mean MAPE −6.5pp, IQR halved — but n=5 Mann-Whitney p>0.5. Effect within seed noise.
- **Lesson:** "Looks like a real improvement" without paired test = anecdote.

### 5.4 γ scaling head (per-net residual) — measurement abandoned
- **Idea:** multiplicative scale to fix per-net heteroscedasticity.
- **Failure:** smoke step-1000 BEST 67%, but 5-seed step-5000 measurement never completed; project pivoted.

### 5.5 Per-channel separate encoders (A1) — KILLED in auto-optimize round 2
- **Idea:** Separate encoder per channel (gnd / cpl) on top of Mesh-v3.
- **Failure:** test gnd 21.60% (+1.11pp **worse** than shared encoder).
- **Codex verdict:** Forbidden. Any "per-channel projection" must be implemented as InputSubset zero-mask, NOT separate encoder. Marked the 4th capacity-add strike.

### 5.6 Strike #7 — sister cell-OBS features
- **Idea:** Join 13 cell-OBS features from sister codebase.
- **Failure:** All metrics worse (test 6.94 → 10.09%). Sister features are R-optimal (routing-length), C_gnd needs substrate-area (GDSII / SPICE only).

### 5.7 Strike #8 — Liberty pin caps
- **Idea:** Parse 5,147 Liberty pins → 7 per-net features → Mesh retrain.
- **Failure:** All metrics worse (test 6.94 → 9.30%, gnd +5.05pp). Same overfit pattern as Strike #7. 5-variant systematic diagnostic confirmed `mesh-only 6.07%` is the architecture ceiling on these features.

### 5.8 Strike #2 — per-pair coupling head with uniform analytic baseline
- **Failure:** cpl(total) jumped 38 → 60% at curriculum transition; killed at epoch 53. Mesh-curriculum stays as the PINN final.

### 5.9 Mode B isotonic full-distribution calibration (C1)
- **Failure:** Bulk −1.31pp but Top-50 +38pp **worse** — val/test over-prediction populations mismatched.
- **Codex Round 2 fix:** Refit isotonic on Combined model val output specifically. Pending re-test.

### 5.10 Capacity sweep on hand features (Phase 1 Tier 3)
- **Finding:** 11K → 71K → 406K (36×) parameters all hit 11–14% ceiling on hand features. **Capacity is NOT the bottleneck.** Mesh per-cuboid features ARE the critical path.

### 5.11 Cuboid 9-channel truncation regression
- **Failure:** `cuboids[..., :9]` guard in 6 places stripped channel 9 (net_type), making `is_power` always fall back to `w>2μm`. Fixed in v9 truncation patch.

### 5.12 Tile-vs-net split leak (legacy)
- **Failure:** 12.29% legacy net leak quantified on real manifest (Phase 0 H1). Tiles of the same net could land in different splits.
- **Fix:** v3 net-centric split + `prepare_net_centric_validation`. Always pull all tiles of a chosen net together.

### 5.13 `/tmp` cuboid dump
- **Failure:** 2026-05-04 nova run dumped 258 GB to `/tmp` and bricked the host.
- **Fix:** Policy 2.2 — `/data/PINNPEX/scratch/` only. Ensure `--temp_dir` always points there.

### 5.14 Stage-2 rglob slow path
- **Failure:** `build_features_inference` auto-discovery used `rglob + gzip.open + pickle.load` over 684K nova pkls → >4 hours, never finished.
- **Fix:** `predict_spef_e2e.py` lines 160-167 auto-build manifest from `cuboids_map.csv`. Verified 2× faster on tv80s.

---

## 6. POINTERS

**Read `NAVIGATION.md` (root) first** — it is the single doc map for this repo, organized in tiers from "first 5 min" to "historical archive". The most important pointers are repeated below for convenience:

- **Canonical leaderboard (all runs as a single table)**: `RESULTS.md` ← read this when you need numbers
- Deep narrative + 12-section history: `docs/PROJECT_REPORT.md`
- Pipeline definition: `PEX_FRAMEWORK.md`
- Most recent session state: `pex_v3/SESSION_HANDOFF.md`
- Live phase tracker: `pex_v3/PHASE_STATUS.md`
- Paper drafts: `pex_v3/paper/{METHOD,RESULTS_CONSOLIDATED,OUTLINE,EXPERIMENTS}.md`
- Live joint Pareto frontier: `pex_v3/joint_pareto/PARETO.md`
- Latest hero report: `pex_v3/experiments/auto_optimize_2026_05_03/HERO.md`
- Next-session structural plan (external workspace): `/data/PINNPEX/joint_pareto_workspace/NEXT_SESSION_STRUCTURAL_PLAN.md`
- Memory index (auto-loaded): `~/.claude/projects/-home-jslee-projects-PINNPEX/memory/MEMORY.md`

---

## 7. PHASE F — ASAP7 7nm cross-PDK sprint (added 2026-05-11)

### 7.1 Motivation
- Current frontier (v11/v12 + TreePEX ensemble) is **one-PDK (intel22)**. Any "license-free SPEF beats Innovus / OpenRCX" claim is single-PDK; reviewers will ask cross-PDK transfer evidence.
- ASAP7 (open academic 7nm PDK, Synopsys ARM collaboration) is now available with StarRC golden + Innovus + OpenRCX comparator SPEFs on the **same 12-design suite** already used for intel22. This converts the paper from "1 PDK × 2 OOD designs" → "2 PDKs × 2 OOD designs each (nova_intel22, tv80s_intel22, nova_asap7, tv80s_asap7)".
- Zero asset-procurement cost: all data is on disk; the question is purely whether the pipeline survives a PDK swap.

### 7.2 Available assets (verified 2026-05-11)
- **DEFs (13 designs):** `/home2/hyshin/ICCAD2026/results/def/asap7/asap7_<design>_x1.def`. Suite = intel22 12 designs **+ TinyRocketCore**.
- **Golden StarRC SPEFs (12 designs; no TinyRocketCore):** `/home2/hyshin/ICCAD2026/results/spef/asap7_starrc_fs/asap7_<design>_fs_en_starrc.spef.typical` (corner = `typical`, T = 25 °C, nxtgrd = `asap07_x1.nxtgrd` per SPEF header).
- **Comparator SPEFs (10 designs each):** `*_innovus.spef` + `*_openrcx.spef` siblings — missing for `mpeg2_top`, `nova`, `TinyRocketCore` (those 3 are StarRC-only A/B).
- **PDK `/home/jslee/projects/PEX_SSL/tool/pdk/7nm/`:**
  - `layers/layers.info` — 10 conductor layers (M1–M9 + Pad), 10 via layers (V0–V9), IMDa/b dielectric pairs ε = 3.7 / 4.2, FOX/ILD substrate stack, total z = 0 → 2.86 μm to Pad top.
  - `lef/asap7_tech_1x_201209_JS.lef` (tech), `lef/asap7sc7p5t_27_1x_201211_JS.lef` (cells), `fakeram7_*.lef`, `sram_asap7_*.lef` (memory macros).
  - `qrc/ASAP7.tch`, `rcx_patterns.rules`, `asap7.map`, `asap7.corners`.

### 7.3 Deltas vs intel22 (verified by F0 audit 2026-05-11)
| Item | intel22 | ASAP7 | Action |
|---|---|---|---|
| Conductor layers | **8 metal (M1–M8) + ce1 + ce2 = 10** (paper-grade run uses BEOL only) | **9 metal (M1–M9) + Pad = 10** | Same total count → no `n_layers=13` audit needed; **no hard-coded site found** in active `src/` (`grep -r '13\|n_layers' src/` returns only MLP-depth refs and archived `_archive/macro_density_fno.py` which uses `z_anchors.numel()` — dynamic). |
| BEOL active stack height | 0 → 9.569 μm (M8 top), or 12.569 μm incl. C4 bumps | **0 → 2.098 μm (Pad top, parsed)** | `WINDOW_SIZE=(4,4,20)` μm trivially covers both BEOLs in 1 z-slab (NetTiler z-stride 19 μm). `d` feature is a per-PDK constant → log-norm gives identical encoder input. **Locked identical** on both PDKs per §7.4.0; F1 z-retune cancelled. |
| ε range | 2.8 (ild0–ild5 low-k) / 4.0 (ild6+) / 5.5 (etchstops) / 22 (gate) | 3.7 (IMDa ULK on M1–M9) / 4.2 (IMDb + Pad PASS1) | Both within `log(ε)` clamp safely (`log(2.8)=1.03`, `log(22)=3.09`); ASAP7 distribution **narrower** than intel22 → CuboidEncoder ε channel sees **less variance** on ASAP7. Statistical risk: encoder ε head may underweight ASAP7 inputs (mitigated by F2c per-PDK retrain). |
| DEF layer naming | `m1`..`m8`, `via1`..`tv1` | `M1`..`M9` + `Pad`, `V0`..`V9` | Spot-checked `asap7_gcd_x1.def` lines 72-120: layer names match `layers.info` GRD/DB sections exactly. `LayerInfoParser` parses cleanly (10 conductor entries, ε auto-matched, lvl_idx 1-10 populated). |
| DEF naming convention | `intel22_<design>_f3.def` | `asap7_<design>_x1.def` | **Patched** in `configs/config_asap7.py` (F0 done). |
| Existing config | `configs/config.py` (production) | `configs/config_asap7.py` was **broken** (mis-pointed to non-existent `_t1.def`, missing TRAIN_SPEFS/TEST_SPEFS, INPUT_DIM, USE_VSS_AGGRESSORS, NF_PAD_TO_CUBOIDS) | **Patched** 2026-05-11 to mirror intel22 schema; 11/11 paths resolve. |
| Oracle | StarRC TCL template, full-chip | StarRC SPEFs **precomputed** for 12 designs (no `TinyRocketCore`) | First-pass reuses precomputed; StarRC re-run only required for `TinyRocketCore` (which we drop from TRAIN/TEST per F0). `.nxtgrd = asap07_x1.nxtgrd` referenced in SPEF headers. P7 still applies. |
| Cuboid tensor | (N, 10) — `INPUT_DIM=10` per `configs/config.py:129` (CLAUDE.md's "9-channel" doc is stale post-v9 net_type addition) | unchanged | Only ε channel + xy/z scaling vary by PDK. Channel layout, padding mask untouched. |
| `SCALE_FACTOR` | default 2.5 (`getattr(config, 'SCALE_FACTOR', 2.5)` in `neural_field.py:29`); not set in either config | default 2.5 | **Locked identical** on both PDKs per §7.4.0. Resulting z_abs distribution gap (intel22 z_norm up to ~5, ASAP7 up to ~0.84) is treated as a physical input property, not a methodology knob. |
| Compute (verified) | 8 × NVIDIA RTX A6000 (`nvidia-smi --list-gpus`) | same machine | F2c 5-seed runs in **parallel on 5 GPUs**, not serial. AL 1 iter ≈ 8800 s (`output_intel22/.../al_macro_runtime.csv`); 30 iter × 5 seeds × 8800 s ≈ **365 GPU-h serial → ~73 wall-h (~3 wall-days) parallel**. SSL ~30 GPU-h × 5 = **6 wall-days serial → ~1.5 wall-days parallel** assuming 5 GPUs free. **Total F2c wall-clock: 4–5 days**, not 8–10 GPU-days as initially drafted. |
| Disk footprint | intel22 `processed_v3` = **495 GB** (nova alone 300 GB, ldpc 106 GB) | ASAP7 DEFs 1.2 GB raw, SPEFs 8.9 GB raw | Expected ASAP7 `processed_v3` ≈ 200–400 GB (designs ~similar net counts but ~10× shorter z-axis = fewer cuboids/net). `/data` has **887 GB free**, fits with margin. |
| Corner / temperature | intel22 corner (typical, T documented in headers) | `typical`, 25 °C | Document corner pair in any cross-PDK comparison; do not mix corners in a single table. |

### 7.4.0 Paper-correlation lockdown (locked 2026-05-11)

**Contract: the ASAP7 setup is bit-identical to intel22 on every algorithmic knob.** Reviewer-facing claim is "we changed nothing except `LAYERS_INFO_PATH` / `TECH_LEF_PATH` / `CELL_LEF_PATH` / design list" — period.

Verified by `python3 -c "..."` diff of `configs.config` vs `configs.config_asap7` constants: **only `GPU_ID` (1 vs 3, resource allocation, not algorithmic) differs**. All other knobs identical.

| Category | Locked value (both PDKs) | Notes |
|---|---|---|
| Cuboid tensor | `INPUT_DIM = 10`, `NF_PAD_TO_CUBOIDS = 1024`, channel layout per CLAUDE.md | No representation change per PDK. |
| Tiling | `WINDOW_SIZE = (4.0, 4.0, 20.0)` μm, `TILING_OVERLAP = 0.5`, `CONTEXT_MARGIN = 1.0` | z=20 μm trivially covers both BEOL stacks in 1 z-slab. Verified `d` feature is a per-PDK constant → log-norm gives identical scalar at encoder input. Tile count is purely xy-driven → identical methodology + zero efficiency cost on ASAP7. |
| Encoder | `SCALE_FACTOR = 2.5` (default), encoder MLP shape unchanged | `z_abs` distribution differs by PDK (intel22 max ~5, ASAP7 max ~0.84 post-scale) — **this is unavoidable physical input difference, NOT a methodology choice**. F2a directly tests whether the encoder generalizes across this z range. |
| SSL pretraining | `SSL_BATCH_SIZE=2048, SSL_LR=1e-4, SSL_EPOCHS=500, SSL_W_BC=10, SSL_W_ENERGY=0.1, SSL_W_FAR=1.0` | Identical recipe verbatim. |
| Active learning | `AL_BATCH_SIZE=4, AL_LR=5e-5, AL_TRAIN_STEPS_PER_ITER=12000, AL_FINE_ITERS=6, AL_BATCH_NETS=2, AL_MAX_TILES_PER_BATCH=256, AL_MIN_ENTROPY_THRESHOLD=-inf, AL_SAMPLING_METHOD="Predefined"` | Identical recipe. |
| Model architecture | `MODEL_DIM=256, NUM_HEADS=4, BASIS_LAYERS=4, CORRECTION_LAYERS=3, USE_VSS_AGGRESSORS=True, USE_RAIL_COUPLING=False, CUTOFF_RADIUS=4.0` | Identical. |
| TreePEX features | `BASE_FEATURE_COLS + H3_FEATURE_COLS` (41 + 26 = 67 features) — **schema unchanged across PDKs** | ASAP7's M9 wires dump into the existing `layer_hist_M9_plus` bin (intel22 fills that bin with 0 since intel22 has only M1–M8). **We deliberately do NOT add `layer_hist_M9` for ASAP7** — that would break feature-schema parity and reviewer correlation claim. |
| Calibration | α-blend (intel22: α=0.20 v11 / α=0.30 v12) + Stage-5 LGBM residual + per-net R sister hybrid | **α and LGBM coefficients are PDK-specific by construction** (they fit TRAIN_SPEFS). The pipeline structure + feature space + fit procedure are identical across PDKs. F2a/b/c quantify how much PDK-specific these coefficients are. |
| Canonical split | TRAIN 9 (mirror design names: `aes_cipher_top, gcd, ibex_core, ldpc_decoder_802_3an, mc_top, spi_top, usbf_top, vga_enh_top, wb_conmax_top`); TEST OOD 2 (`nova, tv80s`); `mpeg2_top` blacklisted; `TinyRocketCore` dropped (no ASAP7 SPEF) | Same 9+2 partition by design name → cross-PDK A/B is **paired by design**. |
| Corner / temp / oracle | typical, 25 °C, StarRC S-2021.06-SP2 | R6 closed; documented in §3.5. |
| Metrics | 5-seed MAPE (total + gnd + cpl + p95), paired Mann-Whitney U, Cohen's d, bootstrap CI, anti-overclaim discipline (P6) | Identical reporting protocol on both PDKs. |
| Compute | 8×A6000, GPU_ID per-seed staggered, no DDP | `GPU_ID` per-config differs (intel22: 1, ASAP7: 3) — resource allocation only, not algorithmic. |

**Anti-correlation knobs (explicitly forbidden):**
- Tuning `WINDOW_SIZE`, `SCALE_FACTOR`, `SSL_LR`, `AL_LR`, `MODEL_DIM`, or any other hyperparameter per-PDK.
- Per-PDK feature columns in TreePEX.
- Per-PDK loss weights or curriculum schedule (mesh-curriculum 0.405→0.916→1.386 unchanged).
- Per-PDK model architecture branch.
- Re-running intel22 SSL/AL with different settings to "improve" the cross-PDK delta.

**If any of the above becomes necessary to hit success criteria (§7.5), it must be reported as a separate paper claim — "PDK-specific tuning yields X" — not bundled into the headline cross-PDK transfer claim.**

### 7.4 Phase F sprint (target: 2 weeks, parallelized on 8 × A6000)
**F0 — Asset audit + config (Day 0–1)** — `pex-data-engineer` — **DONE 2026-05-11**
- ✅ Symlinked 13 ASAP7 DEFs at `data/raw/def/asap7/ → /home2/hyshin/ICCAD2026/results/def/asap7/`.
- ✅ Symlinked 32 ASAP7 SPEFs (StarRC + Innovus + OpenRCX) at `golden_data/spef_data/asap7/ → /home2/hyshin/ICCAD2026/results/spef/asap7_starrc_fs/`.
- ✅ Rewrote `configs/config_asap7.py` to mirror `configs/config.py` schema (TRAIN_DEFS/TEST_DEFS/TRAIN_SPEFS/TEST_SPEFS, INPUT_DIM=10, NF_PAD_TO_CUBOIDS=1024, USE_VSS_AGGRESSORS=True, AL_PREDEFINED_DESIGNS, etc.). 11/11 paths resolve.
- ✅ `LayerInfoParser` smoke test passes: 10 conductor entries (m1..m9 + pad), ε auto-matched (3.7/4.2), resistance + lvl_idx populated. ASAP7 layers.info schema compatible with intel22 parser.
- ✅ `LayerInfoParser` smoke test confirms ASAP7 BEOL Pad top z = 2.098 μm (used to compute F1 z-tune candidate).
- **Pending:** F1 `compare_spef.py` SPEF parser smoke test on one ASAP7 StarRC file.

**F1 — Dataset rebuild (Day 1–2)** — `pex-data-engineer` + `graph-geometry-engineer`
- **`WINDOW_SIZE` retune is NOT done** — locked at intel22 value `(4.0, 4.0, 20.0)` per §7.4.0 paper-correlation contract. (Earlier R2 concern was misanalyzed; see R2 closure note.)
- `NF_PAD_TO_CUBOIDS=1024` also locked. May be empirically over-budgeted on ASAP7 (smaller per-net tile counts expected), but reducing it would break the lockdown contract. Acceptable to leave headroom.
- **Pre-flight gate (mandatory):** `python3 scripts/verify_pdk_correlation_lockdown.py` (exit 0 required).
- Run dataset build with config swap, now via the patched `--config` flag (R10 closed):
  ```
  python3 scripts/build_dataset_multi.py --config config_asap7 --num_workers 64
  ```
- **F1 smoke result (gcd, 224 KB DEF, 2026-05-11):**
  - First attempt failed with `[StrictError] Failed to create cuboids for VIA 'M9_M8_1'` (R15 — DEF VIA name case-sensitivity bug in `def_parser.py`). Fixed by normalizing case at DEF VIA parse + lookup.
  - Re-smoke (post-fix) passed cleanly: 424 cells loaded, 3,400 nets parsed, **12,028 signal + 2,180 VSS/VDD cuboids**, 767 tiles processed → **580 valid samples**.
  - Output structure verified: cuboid tensor shape `(N, 10)` with INPUT_DIM=10 ✓, map.csv schema matches intel22, padding mask + cuboid_net_names populated.
- **F1 full build (11 designs, 2026-05-11):**
  - First launch crashed on `asap7_nova` with `KeyError: 'DFFASRHQNx1_ASAP7_75t_R'` — v27 cell LEF was incomplete (R17). Swapped to v28 R-only LEF (`asap7sc7p5t_28_R_1x_220121a.lef`, 212 macros, 100 % cell coverage); purged partial 3.6 GB; relaunched.
  - **DONE: 725,920 total samples** = **413,478 train + 45,938 valid + 266,504 test** (test = nova 261,085 + tv80s 5,419).
  - P4 (net-centric 9:1 split) ✓, P5 (TEST DEFs 100 % test, 0 % train/valid) ✓, manifest schema matches intel22 verbatim.
  - Disk: **533 GB** at `/data/PINNPEX/data/processed_v3/asap7/` (intel22 = 495 GB; ASAP7 slightly larger). `/data` free 354 GB after F1.
  - Per-design disk: nova 332 GB / ldpc 119 GB / vga 53 GB / wb_conmax 23 GB / ibex 10 GB / aes 8.7 GB / usbf 3.6 GB / mc 1.7 GB / tv80s 1.6 GB / spi 680 MB / gcd 45 MB.
  - Post-build lockdown re-verify: `python3 scripts/verify_pdk_correlation_lockdown.py` → `[OK] Lockdown intact. All algorithmic knobs bit-identical.` exit 0.
  - Wall-clock from second launch to Build Complete ≈ 90 min (LEF parse amortized, nova dominated).
- Verify manifest row count, per-design tile distribution, `padding_mask` density vs intel22 baseline. Expect ASAP7 tile-per-net counts to be **lower** than intel22 (denser ASAP7 routing in smaller xy footprint per design).
- Disk budget: expect 200–400 GB at `/data/PINNPEX/data/processed_v3/asap7/`; `/data` has 887 GB free.

**F2 — Per-PDK independent training (HEADLINE, simplified 2026-05-11 + pivoted to TreePEX XGBoost 2026-05-11)** — `classical-baseline-owner` + `benchmarking-statistician`

**Pivot 2026-05-11 (after CLAUDE.md canonical update):** PINN paradigm abandoned. The PINNPEX CLAUDE.md (post-pivot section) now defines **TreePEX 5-seed Tweedie XGBoost ensemble as the CANONICAL model** (intel22 tv80s 4.98 % / nova 5.28 %, beats every PINN variant on both accuracy AND wall-clock). All PINN tracks moved to `archive/` (pex_v3 / v4 / v5 / v7 / v8). The Phase F headline must therefore be **TreePEX XGBoost cross-PDK validation**, NOT PINN cross-PDK.

- **intel22 results:** already locked (TreePEX 5-seed XGBoost ensemble, 4.98 % tv80s / 5.28 % nova, 7.10 s tv80s / 70.55 s nova). Reused as-is, no re-run.
- **ASAP7 training (this is the work):** mirror TreePEX intel22 pipeline exactly on ASAP7.
  1. **41-D base features** (`TreePEX/scripts/00_build_asap7_features.py`): per-design DEF + StarRC SPEF → NetFeatureVector (41-D), via the PDK-parameterized `feature_dataset.py`. Output: `/data/PINNPEX/data/processed_v3/asap7/features/all_designs.csv`.
  2. **26-D H3 features** (`archive/pex_v4/scripts/29_extract_new_features.py` with `--manifest-csv` + `--data-root` for ASAP7): per-net top-K aggressor geometry. Output: `TreePEX/inputs/asap7_new_features_per_net.csv`.
  3. **5-seed Tweedie XGBoost** (`TreePEX/scripts/01_train_asap7_models.py`): SAME hyperparameters (depth=8, n_est=500, vp=1.5), SAME 67-feature schema (41 base + 26 H3) as intel22 — only training data differs. Output: `TreePEX/models_asap7/`.
  4. **Eval on ASAP7 OOD** (`TreePEX/scripts/02_inference.py` + `04_compare_golden.py` with ASAP7 paths): per-design MAPE/R²/RMSE vs StarRC; SPEF round-trip; compare to Innovus + OpenRCX.
- **Wall-clock estimate:** 41-D features 1-3 hrs (nova dominates), H3 ~30 min, XGBoost training ~10 min CPU, eval ~30 min. **Total ~3-4 wall-hours**, vs the original PINN F2 plan's 2 wall-days.

**Lockdown applies fully**: same model class, same hyperparameters, same 67-D feature schema, same canonical TRAIN 9 / TEST 2 (nova + tv80s) split. Only paths differ (ASAP7 vs intel22).

**PINN F2a zero-shot results (archived, NOT in paper):** 3/5 seeds completed before pivot on 2026-05-11. Mean MAPE on `tv80s_asap7`: Total **97.96 %** / Gnd **92.34 %** / Cpl **98.85 %**, pred/golden ratio 0.033× gnd / 0.004× cpl — confirms R12 expected encoder-only zero-shot collapse. PINN SSL pretrain at epoch 131/500 also killed; not reported.

**F3 — Tool comparison on ASAP7 (Day 4–7)** — `benchmarking-statistician`
- `src/evaluation/compare_spef.py` for each ASAP7 design with golden = StarRC, pred ∈ {Innovus, OpenRCX, PINN-PEX (each F2 variant), TreePEX ensemble (each variant)}.
- Build `docs/pex_tool_asap7.csv` mirroring `docs/pex_tool.csv`.
- Cross-PDK leaderboard table: per-PDK total + per-channel + p95 + runtime, 5-seed CIs.

**F4 — Paper integration (Day 7–10)** — `benchmarking-statistician`
- Promote cross-PDK table to paper figure. Re-frame deliverable **D3: "Industry tool comparison on _two_ PDKs"**.
- New paragraph: "Cross-PDK generalization of license-free PINN-PEX." Narrative gates:
  - If F2a/b competitive → transfer success story.
  - If F2b ≪ F2c → "encoder portable; calibration PDK-specific and re-fittable in minutes."
  - If only F2c hits target → "method generalizes; retraining required, but at the same cost as intel22."

### 7.5 Success criteria (paper-grade, 5-seed P6, methodology bit-identical to intel22 per §7.4.0)
- **Minimum (must ship):** ASAP7 5-seed AL fine-tune lands at **total MAPE ≤ 7.0 %** on ASAP7 OOD (`nova_asap7` + `tv80s_asap7`) — i.e. within the intel22 v11 6.82 % ± 0.04 band. Beats OpenRCX on ASAP7 mean. **Paper claim:** "Same model architecture and same training recipe, independently trained on two PDKs, both reach OpenRCX-beating MAPE."
- **Stronger (target):** ASAP7 result strictly Pareto-dominates Innovus on ASAP7 (Total MAPE + Wall-clock), mirroring intel22 v11 vs Innovus on intel22.
- **Headline (stretch):** Cross-PDK side-by-side leaderboard (intel22 + ASAP7 each) shows PINN-PEX beats Innovus + OpenRCX **on both PDKs simultaneously** → license-free SOTA on two independent academic+commercial PDKs.
- **Reviewer-defensible framing:** **No cross-PDK transfer claim.** Each PDK is its own measurement. The methodology lockdown (§7.4.0) is the cross-PDK statement; results are reported independently per PDK and compared side-by-side, with the lockdown contract documenting that nothing PDK-specific was tuned. Anti-overclaim P6 still applies — 5-seed CIs on every reported number.

### 7.6 Minimum publishable claim (if only F2c finishes)
> "PINN-PEX methodology generalizes to ASAP7 7nm without architecture change. Same cuboid 9-channel tensor, same HybridPexV3Mesh model, same SSL + AL curriculum; only `BEOLMaterialStack` (ε, z-thickness) and `WINDOW_SIZE` z-axis are PDK-specific. ASAP7 OOD test MAPE = X.XX ± Y.YY % (5-seed), beats OpenRCX (8.83% intel22 → Z.ZZ% ASAP7) and competitive with Innovus."

### 7.7 Risks & known unknowns (post-F0 audit)
- **R1 — Hard-coded layer count: CLOSED.** `grep -rE '13|n_layers|N_LAYERS' src/` returns only MLP-depth `num_layers` arguments (in `src/models/layers.py`) and archived `_archive/macro_density_fno.py` which uses `z_anchors.numel()` (dynamic). No live `n_layers=13` to fix. ASAP7 has 10 conductor layers vs intel22 8 metal + 2 ce — same total count.
- **R2 — z-window over-allocation: CLOSED (re-analyzed 2026-05-11).** `neural_field.py:96-99` confirms `d` (channel 5, tile depth) is a per-PDK constant → identical log-norm scalar at encoder input on both PDKs. NetTiler z-stride = 20-2×0.5 = 19 μm > both BEOL heights → **1 z-slab/net on both PDKs**, identical tile count, zero efficiency loss. Keeping `WINDOW_SIZE=(4,4,20)` is **methodologically optimal** (paper-correlation lockdown §7.4.0) AND efficient. The "padding waste" concern is a misread of the tiling algorithm. **No F1 retune.**
- **R3 — TinyRocketCore: CLOSED.** Dropped from TRAIN/TEST per F0 patch (no golden SPEF, blocking). Keep only as deployable-demo extra; StarRC re-run with `asap07_x1.nxtgrd` deferred to post-paper.
- **R4 — Comparator SPEFs missing** for `nova`, `mpeg2_top`, `TinyRocketCore`. Cross-PDK Innovus/OpenRCX A/B reduces to **`tv80s_asap7` only** + 9 TRAIN designs. `nova_asap7` is StarRC-only A/B, mirrored from intel22 nova.
- **R5 — `mpeg2_top` blacklisted** on both PDKs (intel22 instability assumed forward); commented out in `config_asap7.py`. Re-test post-F1 only with margin.
- **R6 — Corner mismatch: CLOSED.** Both PDKs are **typical corner, 25 °C, StarRC S-2021.06-SP2**. intel22 leaves `CORNER_NAME` blank with `tttt.nxtgrd` (TT/TT/TT/TT) → equivalent to ASAP7's explicit `typical` + `asap07_x1.nxtgrd`. Documented in §3.5.
- **R7 — Memory-macro LEFs: CLOSED.** Scanned all 13 ASAP7 DEFs (`grep '^- ' ... | awk '{print $3}' | sort -u`) — no `fakeram*` or `sram*` macro instantiations in any design. All COMPONENTS use std cells (`*_ASAP7_75t_R` family). Single-LEF loading via current `CELL_LEF_PATH` is sufficient.
- **R8 — TreePEX feature distribution shift: CLOSED (paper-correlation lockdown).** Per §7.4.0, the 67-feature schema (41 base + 26 H3) is **unchanged across PDKs**. ASAP7's M9 wires populate the existing `layer_hist_M9_plus` bin (intel22 fills it with 0). Adding a per-PDK `layer_hist_M9` column would break feature-schema parity and the cross-PDK transfer claim. The PDK-agnostic features (eps/density/compact_*) auto-adapt via `layers.info`. **TreePEX still needs an ASAP7 sister build of `pex_v4/results/new_features_with_ids.csv`** — parametric ~1 wall-day work using identical scripts with `--config config_asap7`; tracked under F3 deliverables, not a blocker for F2c.
- **R9 — StarRC re-run access: NOT NEEDED for primary scope.** All 11 in-scope designs (TRAIN 9 + TEST 2) have precomputed StarRC SPEFs. Only TinyRocketCore would need re-run, and we drop it.
- **R10 — `build_dataset_multi.py` config swap: CLOSED.** Both `scripts/build_dataset.py` and `scripts/build_dataset_multi.py` now accept `--config` (default `config` = intel22). `build_dataset_multi.py` passes `--config` through to its `build_dataset.py` subprocess. Smoke-tested 2026-05-11: `--config config_asap7` resolves `PROCESSED_DIR=/data/PINNPEX/data/processed_v3/asap7`, intel22 default unchanged. Also fixed hard-coded `intel22_pt` to `cfg.PT_DIR`.
- **R11 — `SCALE_FACTOR` not PDK-tuned: CLOSED (paper-correlation lockdown).** Default 2.5 is the shared scaler — locked identical on both PDKs per §7.4.0. The resulting z_abs distribution difference (intel22 z_norm up to ~5, ASAP7 up to ~0.84) is treated as **a property of the physical input**, not a methodology knob. If F2a reveals encoder degradation due to this z-range shift, it counts as evidence about cross-PDK encoder portability — not a license to introduce PDK-specific scaling.
- **R12 — Per-layer `flux_router` parameters are PDK-shaped: OPEN, design choice.** `flux_head.py:51-69` builds `metal_z_anchors` dynamically from `layer_map` z_pos values (dedup at 0.05 μm). Parameters sized by `num_anchors`: `layer_scale_phys_gnd (num_anchors,)`, `cpl_layer_pair_log_scale (num_anchors, num_anchors)`, `gnd_fringe_scale (num_anchors,)`, `vss_gnd_scale (num_anchors,)`. **NOT a lockdown violation** — same construction rule on both PDKs. But intel22 ckpt → ASAP7 model: per-layer scale tensor shape-mismatch, **strict=False loader re-initializes them from `_make_gnd_cap_density_init()` / `_make_gnd_fringe_scale_init()` physics priors**. Affects F2a interpretation: "encoder + per-cuboid MLPs transfer; per-layer physics scales reset to ASAP7 prior init." Report this in the paper alongside F2a — frame as **encoder-only zero-shot**, not full-model zero-shot.
- **R13 — SSL / AL trainers hard-import `configs.config`: CLOSED 2026-05-11.** Patched `src/trainers/train_ssl.py` and `run_active_learning.py` with the same `argparse + importlib.import_module(f"configs.{_cfg_mod}")` pattern as R10. Pre-arg-parse for `--config` happens before `import configs.*` (avoids argparse race with the configs being imported at module level). Smoke-verified: syntax OK + sys.argv pre-parse handles `--config X`, `--config=X`, and default `config`.
- **R14 — `src/evaluation/evaluator.py` config import: CLOSED 2026-05-11.** Same recipe applied to `evaluator.py:26`. `compare_spef.py` does not import configs.
- **R18 — ASAP7 SPEF filename pattern: CLOSED 2026-05-11.** evaluator.py:196 expects `{design_name}_starrc.spef`. ASAP7 SPEFs are `asap7_<design>_fs_en_starrc.spef.typical` (different middle: `_fs_en` vs `_x1`; extra `.typical` suffix). Fixed without changing evaluator code: replaced `golden_data/spef_data/asap7` symlink with a local directory containing both original-name symlinks (for config_asap7's TRAIN/TEST_SPEFS list to keep resolving) AND normalized `asap7_<design>_x1_starrc.spef` symlinks (matching evaluator's expected pattern). 12/12 SPEFs accessible via both names.
- **R19 — ASAP7 SPEF uses `*NAME_MAP` indirection: CLOSED 2026-05-11.** ASAP7 StarRC SPEFs reference nets by numeric id (`*D_NET *3462`) with a `*NAME_MAP` section (`*3462 m1_n`). intel22 SPEFs use direct net names (`*D_NET m1_n ...`) with NO `*NAME_MAP`. `compare_spef.parse_spef_with_coordinates` used `tokens[1]` literally → ASAP7 dict keys ended up as `*<id>` strings instead of actual net names → 0 common-nets match with PINN predictions → Stage 4 `KeyError`. **Fix:** added pass-1 NAME_MAP collector + `_resolve()` helper that maps `*<id>` and `*<id>:<pin>` references to real names in *D_NET, *CONN, *CAP. Both PDKs verified — ASAP7 3,458 nets resolve correctly; intel22 3,380 nets unchanged (no NAME_MAP, no-op). Lockdown-safe.
- **R15 — DEF VIA name case-insensitivity: CLOSED 2026-05-11 (parser bug, not PDK-specific).** `def_parser.py` stored DEF VIAS section names in **original case** (e.g. `M9_M8_1`) but the routing-time lookup did `.lower()` → key miss → `[StrictError] Failed to create cuboids for VIA 'M9_M8_1'`. intel22 DEFs happened to not hit this because their named VIAs were case-aligned with the lookup; ASAP7 DEFs (uppercase `M9_M8_1`, `M4_M3wide...`) tripped it on the first smoke build. **Fix:** normalize `current_via_name = line.split()[1].lower()` at DEF VIA parse (line 75), and `t.lower()` at routing-time VIA lookup (lines 411/669). Lockdown-safe: same construction rule on both PDKs, just case-normalized.
- **R17 — Wrong ASAP7 cell LEF version: CLOSED 2026-05-11.** `asap7sc7p5t_27_1x_201211_JS.lef` (v27, 201211) was **incomplete**: missing `DFFASRHQNx1_ASAP7_75t_R` and other cells used in some designs. F1 first attempt crashed on `asap7_nova` with `KeyError: 'DFFASRHQNx1_ASAP7_75t_R'` at `def_parser.py:148`. Found newer v28 R-only LEF at `/home2/hyshin/OpenROAD-flow-scripts/flow/platforms/asap7/lef/asap7sc7p5t_28_R_1x_220121a.lef` (212 macros) which provides **100 % cell coverage across all 13 ASAP7 DEFs**. Symlinked into `tool/pdk/7nm/lef/`; updated `configs/config_asap7.py:CELL_LEF_PATH`; partial F1 output purged (3.6 GB); F1 relaunched. Lockdown still bit-identical (CELL_LEF_PATH already a permitted per-PDK path).

### 7.8 Non-goals (explicit cut-off)
- **No** cuboid 9-channel tensor layout change per-PDK.
- **No** SSL curriculum / loss weight re-tune per-PDK (reuse intel22 recipe verbatim — that's the portability claim).
- **No** new model architecture per-PDK. Phase F validates **portability of the existing v11/v12 + TreePEX stack**, not a new design.
- **No** GDSII / substrate input on ASAP7 — D7 information-bound finding stands per-PDK.

### 7.9 Owners (existing agents — no new roster)
- `pex-data-engineer` — F0 + F1 (DEF/LEF/SPEF parse, manifest rebuild, LEF audit R7).
- `graph-geometry-engineer` — F1 z-window re-tune review.
- `pex-physics-architect` — Verify `BEOLMaterialStack` against ASAP7 `layers.info`; sanity-check ε values, R1 + R2 audit.
- `neural-operator-architect` — F2 variants (a/b/c/d) design; freeze/unfreeze recipe.
- `experiment-systems-engineer` — F2 5-seed harness (mirror intel22 setup), determinism, AL pool isolation.
- `benchmarking-statistician` — F3 + F4 paired Mann-Whitney U, cross-PDK leaderboard, per-channel breakdown.
- `classical-baseline-owner` — TreePEX ensemble refit on ASAP7 (F2a/b/c parallel track).

### 7.10 Cross-references
- Phase F entry point: this section (§7).
- Existing config skeleton: `configs/config_asap7.py` (needs fix per F0).
- intel22 frontier numbers (cross-PDK baseline anchor): §4.1 + RESULTS.md.
- 5-seed protocol gate: §2 P6.
- Anti-overclaim rules: `benchmarking-statistician` role.
