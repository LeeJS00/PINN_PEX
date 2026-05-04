# NAVIGATION — 어디서 무엇을 읽을지

_Last updated: 2026-05-05_
_Audience: this repo's future Claude session, picked up cold._

> **TL;DR for a fresh session:** read in this order — `CLAUDE.md` → `PROJECT_PLAN.md` → `RESULTS.md` → `pex_v3/SESSION_HANDOFF.md` → done. The rest is reference.

---

## Tier 0 — read first (≈ 5 min)

| File | What it is | When to read |
|---|---|---|
| `CLAUDE.md` (root) | AI-assistant guidance for the project (commands, architecture, conventions). | Every session start. |
| `PROJECT_PLAN.md` (root) | Canonical plan in 두괄식 order: **Task → Policies (8 critical) → Method → Results → Previous mistakes**. | Every session start. |
| `RESULTS.md` (root) | Single leaderboard: 5-pillar headline + per-net cap MAPE + Joint Pareto frontier + industry comparison + killed/superseded variants + legacy v9-era runs. | When you need a number. |

## Tier 1 — domain reference (read on demand)

| File | What it is | Use case |
|---|---|---|
| `PEX_FRAMEWORK.md` (root) | 7-stage DEF→SPEF pipeline definition, canonical TRAIN/TEST split, per-target MAPE status. | Pipeline design questions. |
| `docs/PROJECT_REPORT.md` | Deep history (12 sections + §13 v3 paradigm shift). 1132 lines. **§0.1 = the same 8 policies as PROJECT_PLAN §2.** | "Why was X done?" / "What was tried before?" |
| `docs/pex_tool.csv` | Innovus / OpenRCX / StarRC measured runtimes + MAPE on 13 designs. | Industry comparison numbers. |

## Tier 2 — pex_v3 active workspace (current paradigm)

`pex_v3/` is the v3 paradigm shift workspace and contains all paper-pillar work. Everything outside `pex_v3/` (legacy `src/` etc.) is read-only post-mortem.

| File | What it is |
|---|---|
| `pex_v3/CLAUDE.md` | Boundary rule: don't edit outside `pex_v3/`. |
| `pex_v3/README.md` | Strategy summary (one-page). |
| `pex_v3/PHASE_STATUS.md` | Live phase tracker — phase progress + hard kill criteria K1/K2/K3. |
| `pex_v3/SESSION_HANDOFF.md` | **Most recent session state — read this AFTER PROJECT_PLAN to know what was last touched.** |
| `pex_v3/IMPLEMENTATION_STATUS.md` | Honesty register — what's stub vs production-ready code (May 1, partly stale; `PHASE_STATUS` is more current). |

## Tier 3 — paper drafts

| File | What it is |
|---|---|
| `pex_v3/paper/RESULTS_CONSOLIDATED.md` | Per-pillar narrative (matches `RESULTS.md` numbers but with prose). |
| `pex_v3/paper/METHOD.md` | 10 sections, paper-ready. |
| `pex_v3/paper/OUTLINE.md` | Section structure for ICCAD/DATE submission. |
| `pex_v3/paper/EXPERIMENTS.md` | Stub for the experiments section. |
| `pex_v3/paper/{CGND_ERROR_ANALYSIS,SPEF_COMPATIBILITY_REPORT}.md` | Subordinate technical analyses. |

## Tier 4 — joint Pareto workspace (active runtime × gnd × cpl optimization)

| File | What it is |
|---|---|
| `pex_v3/joint_pareto/README.md` | Mission, frozen baseline, agent-invocation guide for the 4 specialists (runtime / gnd-allocator / cpl-allocator / pareto-architect). |
| `pex_v3/joint_pareto/PARETO.md` | **Live Pareto frontier table** — current = v11 (best total) + v12 (best per-channel). |
| `pex_v3/joint_pareto/docs/PROBLEM.md` | Three levers + IN/OUT scope. |
| `pex_v3/joint_pareto/docs/BASELINE.md` | Frozen Path-2 v3 baseline + reproduction one-liner. |
| `pex_v3/joint_pareto/docs/EXPERIMENTS_LOG.md` | Append-only history (exp_002 → exp_014). |
| `pex_v3/joint_pareto/docs/EVOLUTION.md` | Architecture evolution narrative. |
| `pex_v3/joint_pareto/experiments/exp_*/` | Per-experiment PLAN.md + verdict.md / VERDICT.md. **Most recent: exp_013 per-pair (paper-grade STA, NOT admitted to frontier) + exp_014 pattern-matching tool comparison.** |
| `pex_v3/joint_pareto/results/leaderboard.json` | Machine-readable Pareto frontier (consumed by `admit_to_frontier.py`). |

## Tier 5 — auto-optimize sprint (HERO ≤6.5%)

| File | What it is |
|---|---|
| `pex_v3/experiments/auto_optimize_2026_05_03/HERO.md` | **Final 5-seed locked stack** (HybridPexV3MeshInputSubsetClampNorm + LGBM 8-feat calibration, test_total 6.364% [CI 6.247, 6.505]). |
| `pex_v3/experiments/auto_optimize_2026_05_03/PLAN.md` | Sprint manifest. |
| `pex_v3/experiments/auto_optimize_2026_05_03/RESULTS.md` | Sweep journal. |
| `pex_v3/experiments/auto_optimize_2026_05_03/variants/*/DESIGN.md` | 4 design docs (input_subset, clamp_norm, input_subset_clamp_norm, c1_cts_isotonic). |

## Tier 6 — external next-session workspace

| Path | What it is |
|---|---|
| `/data/PINNPEX/joint_pareto_workspace/NEXT_SESSION_STRUCTURAL_PLAN.md` | **3 hypotheses for beyond-α-tuning improvement: (C) per-net adaptive α via LGBM meta-learner [active] / (A) channel-specialized Mesh retrain / (B) segment-level GNN.** Created 2026-05-04. |
| `/data/PINNPEX/joint_pareto_workspace/NEXT_SESSION_INVOCATION_PROMPT.md` | Pair invocation prompt for next session. |
| `/data/PINNPEX/joint_pareto_workspace/runs/` | Hypothesis C experiment outputs (v11_5seed, v12_alpha_sweep). |

## Tier 7 — agent role definitions (`.claude/agents/`)

12 specialist role-md files. Custom agent types are **not directly invocable** as `subagent_type` — use `subagent_type="general-purpose"` and embed the role-md path in the prompt. See `feedback_agent_invocation_pattern.md` in memory.

| Role | Owner of |
|---|---|
| `pex-physics-architect` | physics correctness gatekeeper |
| `neural-operator-architect` | NN architecture audit |
| `graph-geometry-engineer` | data representation |
| `experiment-systems-engineer` | infrastructure (5-seed determinism, manifests) |
| `benchmarking-statistician` | measurement + anti-overclaim |
| `pex-data-engineer` | DEF/LEF/SPEF/StarRC pipeline |
| `classical-baseline-owner` | XGBoost / GAM / B4 / Option F / current PINN |
| `synthetic-data-pipeline-owner` | Stage 1-4 layered-media pretrain (currently dropped per K3) |
| `pex-runtime-owner` | wall-clock budget (joint_pareto) |
| `pex-gnd-allocator-owner` | c_gnd per-cuboid physics (joint_pareto) |
| `pex-cpl-allocator-owner` | c_cpl per-aggressor geometry (joint_pareto) |
| `pex-pareto-architect` | joint trade-off arbiter (joint_pareto) |

## Tier 8 — auto-loaded memory

`~/.claude/projects/-home-jslee-projects-PINNPEX/memory/` (≈ 50 entries). The index at `MEMORY.md` is auto-loaded on every session. Each entry is one of:
- **feedback** — user-validated rule (loss design, agent invocation, tool path policy)
- **project** — fact / decision / dated state
- **reference** — pointer to external system

Always check memory before recommending file paths or behavior — entries note dates and may be stale.

## Tier 9 — historical / archived

The following are kept for git history but should NOT inform current decisions. Their content is subsumed by `docs/PROJECT_REPORT.md` §2 and `RESULTS.md` §9.

| Path | Replaced by |
|---|---|
| `docs/_archive/calibration_track_report.md` | `docs/PROJECT_REPORT.md` §2.3 + `RESULTS.md` §9 |
| `docs/_archive/distillation_log.md`, `distillation_effect_report.md` | `RESULTS.md` §9 (DS-PINN row) |
| `docs/_archive/dspinn_development_log.md` | `RESULTS.md` §9 (DS-PINN row) + memory `project_dspinn_ineffective.md` |
| `docs/_archive/gino_report.md`, `gino_architecture_report.md` | `RESULTS.md` §9 (GINO row) |
| `docs/_archive/INTERIM_CONCLUSIONS.md` | `docs/PROJECT_REPORT.md` (whole document) |
| `docs/_archive/ultraplan.md`, `gpu_pex_plan.md` | `PROJECT_PLAN.md` |
| `pex_v3/docs/PHASE0_PLAN.md`, `H3_REBUILD_SPEC.md`, `H4_PAIRWISE_CPL_DESIGN.md` | Phase 0 done — see `pex_v3/PHASE_STATUS.md` |
| `pex_v3/docs/PHASE_C_*.md`, `PROGRESS_REPORT_2026_05_02_END.md` | session-bound, see `SESSION_HANDOFF.md` for current state |
| `/data/PINNPEX/legacy_archive/output_intel22/active_learning/` | 78 GB v9-era AL outputs (50 dirs); see archive's own `README.md` |

If an archived doc claims a number that contradicts `RESULTS.md`, **trust `RESULTS.md`** — it is current.

---

## How to update this navigation

When you add a major new doc / workspace / paper section, append a row to the appropriate Tier above and link the doc. Do NOT delete rows; mark replaced files in Tier 9 instead.

When a doc becomes stale (its content is now wrong, or another doc supersedes it), add a 1-line deprecation notice at its top:
```markdown
> **DEPRECATED (YYYY-MM-DD)**: superseded by `<path>`. Kept for git history.
```
Then add a Tier 9 row pointing future readers to the canonical replacement.
