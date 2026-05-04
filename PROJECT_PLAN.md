# PINN-PEX PROJECT PLAN

_Last updated: 2026-05-04_
_Status: v3 paradigm shift LOCKED — 5/5 paper pillars paper-grade, joint Pareto v11/v12 frontier, 2.05× faster than Innovus, license-free vs StarRC._
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
| D7 | Per-channel gnd / cpl reduction below 17% / 13% | **Information-bound REJECTED** on DEF/LEF inputs. 4-way oracle bound = gnd 14.07 / cpl 11.21. Requires GDSII / substrate-aware extraction (~4 weeks). |

### 1.3 Scope boundaries
- **In scope:** DEF + tech LEF + Liberty + layer stack inputs only; SPEF (Voltus / StarRC compatible) outputs only; cross-design OOD (held-out designs) is the headline metric.
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

- **Canonical leaderboard (all runs as a single table)**: `RESULTS.md` ← read this when you need numbers
- Deep narrative + 12-section history: `docs/PROJECT_REPORT.md`
- Pipeline definition: `PEX_FRAMEWORK.md`
- Paper drafts: `pex_v3/paper/{METHOD,RESULTS_CONSOLIDATED,OUTLINE}.md`
- Latest hero report: `pex_v3/experiments/auto_optimize_2026_05_03/HERO.md`
- Joint Pareto leaderboard + agents: `pex_v3/joint_pareto/`
- Memory index (auto-loaded): `~/.claude/projects/-home-jslee-projects-PINNPEX/memory/MEMORY.md`
