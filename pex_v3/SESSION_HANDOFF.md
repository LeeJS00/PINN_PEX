# Session Handoff — Resume from 2026-05-03 evening

_Updated: 2026-05-03 evening, post Path-2 fast deployment + main folder merge_
_Next session: read this FIRST, then `pex_v3/PHASE_STATUS.md`, then proceed_

---

## 🎯 Current state (2026-05-03 evening)

**5/5 paper-pillar LOCKED** + Path-2 fast deployment + main folder merge done.
Pattern: user explicitly defers paper draft until separate instruction.

**Recent done**:
- Mesh-curriculum 5-seed: best 6.26% / last 8.27% / ensemble 7.89% (44K params)
- Hero SPEF v2: C 10.95% (XGB 5-seed) / R 2.21% (sister per-net), R²=0.999/0.983
- Path-2 fast deterministic: tv80s 68.9 s (12.5× speedup), median MAPE 5.78% ≈ legacy
- Main folder merge: `src/v3/*` symlinks + `configs/config_v3.py`, 3-way import OK
- METHOD.md (10 sections paper-ready) + 4 negative-strike diagnoses

## ⏳ Background processes left running (do NOT kill on next session — let them finish)

**Nova SPEF write Path-1 — ✅ COMPLETED at session-end** (PID 2757573 finished, 13h 24m total):
  - File: `output_intel22/active_learning/m6_v10b_baseline_seed0/intel22_nova_f3_autonomous.spef` **2.69 GB** (golden 3.02 GB)
  - Confirmed: "✅ Autonomous SPEF Streamed successfully" in log
  - Runtime breakdown (eval_spef_runtime.csv):
    - GPU_Inference: 28,177 s (7.83 h, avg 4.04 s/batch over 6,979 batches)
    - Tensor_to_Node_Mapping: 64.6 s
    - **CPU_KDTree_SPEF_Assembly: 18,985 s (5.27 h)** ← dominant CPU cost
  - vs tv80s (14.4 min): nova ~55× longer wall-clock (3,380 → 118,974 nets = 35×, plus heavier SPEF assembly)
  - **READY FOR HERO VERIFICATION** on next session — just apply:
    ```bash
    # 1. Cap calibration (XGB 5-seed mean anchor on nova)
    python3 pex_v3/scripts/16_xgb_calibrate_spef.py \
        --in-spef output_intel22/active_learning/m6_v10b_baseline_seed0/intel22_nova_f3_autonomous.spef \
        --xgb-csv pex_v3/output/baselines/B1_xgboost_real/seed0/eval_predictions_test.csv \
        --design intel22_nova_f3 \
        --out-spef /tmp/intel22_nova_f3_xgb_cap.spef
    # 2. R per-net calibration (sister v6_s3 — confirm sister has nova predictions first;
    #    if not, sister train+predict on nova or use Path-1 R as-is)
    # 3. compare_spef.py vs golden → lock nova hero MAPE/R²
    ```

**PID 3200544** — nova fast autonomous SPEF (different session, 24 workers parallel):
  - cmd: `pex_v3/scripts/40_fast_autonomous_spef.py --design intel22_nova_f3 --out-dir pex_v3/output/spef_fast_nova_imap`
  - log: `pex_v3/output/spef_fast_nova_imap/nova_fast.log`
  - DO NOT INTERFERE — likely user's other session's work (Path-2 nova run)

## 🎯 Next-session priorities (if both nova SPEFs complete)

1. **Nova hero verification** — apply XGB cap + R per-net calibration to
   nova autonomous SPEF. Compare vs golden (`compare_spef.py`). Lock the
   "Hero on 100K-net chip" claim with R² + MAPE numbers.
2. **Nova vs tv80s scaling** — confirm calibration scales (linear in nets)
   and that R² stays ~0.999 / 0.983 on the larger circuit.
3. **Update PHASE_STATUS.md + paper RESULTS_CONSOLIDATED.md** with nova
   numbers as the second canonical reference (alongside tv80s).
4. **Paper draft** ONLY when user explicitly requests (user 2026-05-03:
   "paper draft 는 아직 생각하지 말고, 내가 별도의 지시를 내리면해").

## 🎯 What you (next-session-Claude) should do FIRST

```bash
# 1. Read this handoff (you're here)
# 2. Read the live tracker (latest 2026-05-03 entries)
cat /home/jslee/projects/PINNPEX/pex_v3/PHASE_STATUS.md

# 3. Skim consolidated paper artifacts
cat /home/jslee/projects/PINNPEX/pex_v3/paper/RESULTS_CONSOLIDATED.md
cat /home/jslee/projects/PINNPEX/pex_v3/paper/METHOD.md
cat /home/jslee/projects/PINNPEX/pex_v3/paper/OUTLINE.md

# 4. Check session-final memory
cat /home/jslee/.claude/projects/-home-jslee-projects-PINNPEX/memory/MEMORY.md
cat /home/jslee/.claude/projects/-home-jslee-projects-PINNPEX/memory/project_method_consolidated_main_merge.md
```

---

## 📋 Where things stand at session end

### ✅ Done

| Phase | Item | Status |
|---|---|---|
| 0 | H1 net-level split | ✅ on real 1.32M-tile manifest |
| 0 | H2 priority truncation | ✅ tested |
| 0 | H3 14×14μm rebuild | ✅ 11/11 designs, 493 GB |
| 0 | M5 SSL split filter | scaffolded (training body deferred) |
| 0.5 | feature_dataset.py | ✅ + layer_idx bug fix (A3) |
| 0.5 | B1 XGBoost 5-seed | ✅ 4.66% ± 0.03pp valid, 5.90% v1 OOD |
| 0.5 | B3 PINN 5-seed (multi-GPU) | ✅ 30.90% ± 2.20pp valid (32.89pp ↓ vs legacy) |
| 0.5 | **B4 Compact+GAM 5-seed** | ✅ V3 log-GBDT **5.72% ± 0.04pp valid / 6.59% test** |
| 0.5 | Cross-design eval (B1+B4+Option F) | ✅ paper-grade comparison table |
| 1 | analytic_base_v3.py | ✅ 11 tests, gradcheck pass |
| 1 | residual_head_v3.py | ✅ 11 tests, day-1=analytic invariant |
| 1 | hybrid_v3.py | ✅ 12 tests, per-channel β strategy |
| 1 | Synthetic pretrain harness | ✅ but K3 fired → DROPPED |
| 1 | finetune_hybrid_v3.py (Tier 2) | ✅ best 7.19%, β-FAIL |
| 1 | calibration_v3.py (Tier 3) | ✅ 9 tests, but ceiling not broken |
| 1 | **Option F deep MLP single-seed** | ✅ **4.66% valid / 5.67% test** — confirms ceiling |
| 0.5 | **Option F deep MLP 5-seed (P1)** | ✅ **4.756% ± 0.012pp valid / 5.623% ± 0.042pp test** — variance LOCKED, OOD gap +0.87pp ties B4 V3 (2026-05-03) |
| C | 7 of 8 specialist agents validated | ✅ R1: A1+A3+A4+A7; R2: A2+A5+A6 |
| Plan | Strategy v3 plan updated 4× | ✅ data-driven, 11 lessons learned |

### 🔄 Running / pending background — none active

All background jobs completed before session end. No polling needed.

---

## 🎯 Paper-grade leaderboard (for paper #1A)

| Method | Architecture | params | valid total | valid gnd | valid cpl | OOD test | OOD gap |
|---|---|---:|---:|---:|---:|---:|---:|
| **B1 XGBoost** (5-seed) | tree boosting | ~100K | **4.657% ± 0.023pp** | 20.6% | 12.4% | **5.842% ± 0.096pp** (nova 5.86 / tv80s 5.31) | +1.19pp |
| **Option F deep MLP** (5-seed) | 286K MLP | 286K | **4.756% ± 0.012pp** | 21.2% | 12.7% | **5.623% ± 0.042pp** (nova 5.63 / tv80s 5.45) | **+0.87pp** ← tied best OOD |
| **B4 V3 log-GBDT** (5-seed) | compact + multiplicative residual | ~100K | **5.72% ± 0.04pp** | 20.3% | 12.8% | **6.59% ± 0.13pp** | **+0.87pp** ← best OOD |
| B4 V2 GBDT (5-seed) | compact + additive | ~100K | 7.46% ± 0.05pp | 22.3% | 14.4% | 9.33% (nova 9.32 / tv80s 8.16) | +1.87pp |
| Hybrid_v3 calibrated (1-seed) | 11K bounded | 11K | ~9.5% | 21.5% | 13.8% | — | — |
| B4 V1 linear | affine | 4 | 34.33% | — | — | — | — |
| **B3 PINN legacy** (5-seed) | DeepPEX 1M | ~1M | **30.90% ± 2.20pp** | — | — | — | — |

**4 paper-grade findings**:
1. **32.89pp data-fix gain** (legacy v10b 63.79% → B3 PINN 30.90%)
2. **B1 vs B3 paired MWU SUPPORTED** (Cohen's d=-16.84, p=0.008)
3. **Hand-feature ceiling 4.66%** (XGBoost = MLP = 4.66% same)
4. **K3 canary saved 125 GPU-days** (synthetic pretrain useless given zero-init)

**Per-channel reality** across all flexible baselines: gnd ≈ 21%, cpl ≈ 12-14%.
The 4.66% is gnd/cpl cancellation artifact. Phase 1 paradigm contribution
must improve **per-channel** (β strategy), not just total.

---

## 🚦 Next actions in priority order

### P1 — Lock the ceiling number ✅ DONE 2026-05-03

`pex_v3/scripts/14_option_f_5seed.py` shipped + run on GPU 0 (3.5 min).
Locked numbers:
- valid: **4.756% ± 0.012pp** (range 4.734–4.770%, 12,594 nets)
- test:  **5.623% ± 0.042pp** (range 5.547–5.667%, 95,594 nets)
- per-design test: nova 5.627% ± 0.042pp · tv80s 5.453% ± 0.082pp
- per-channel valid: gnd 21.20% / cpl 12.67% (cancellation pattern intact)
- runtime: 286K params · 42.0s ± 0.8s train · 0.48 μs/net inference
- OOD gap **+0.87pp** ties B4 V3 log-GBDT for best generalization

Outputs: `pex_v3/output/baselines/Option_F_MLP/seed{0..4}/` + `five_seed_summary.json`.

### P2 — Add B1 OOD numbers ✅ DONE 2026-05-03

`xgboost_baseline.py:run_one_seed` extended to always evaluate on BOTH
valid + test, write per-split CSVs + per-channel summary. 5-seed re-run
(~12 min/seed × 5 = ~63 min total). Aggregator `15_b1_test_aggregate.py`
post-processes per-seed summaries into `test_5seed_summary.json`.

Locked numbers:
- valid: **4.657% ± 0.023pp** (gnd 20.58%, cpl 12.41%) — preserved from prior run
- test (OOD): **5.842% ± 0.096pp** (gnd 19.93%, cpl 16.13%)
- per-design test: nova 5.858% ± 0.097pp · tv80s 5.314% ± 0.039pp
- OOD gap **+1.19pp** — wider than Option F MLP (+0.87pp) and B4 V3 (+0.87pp).
  Tree boosting transfers worse to unseen designs than the bounded
  multiplicative residual or even capacity-rich MLP. Both physics prior
  (B4 V3) and gradient-trained smooth function class (Option F) help OOD.

Outputs:
- `pex_v3/output/baselines/B1_xgboost_real/seed{0..4}/eval_predictions_{valid,test}.csv`
- `pex_v3/output/baselines/B1_xgboost_real/seed{0..4}/per_channel_summary.json`
- `pex_v3/output/baselines/B1_xgboost_real/test_5seed_summary.json`

### P3 — Close A1 caveat #1+#2 (1 day)

```bash
# Re-evaluate legacy v10b checkpoints on v3 valid set.
# Per A1 audit `PHASE_C_A1_REVALIDATION_VERDICT.md` §1.

# Legacy ckpts at:
ls /home/jslee/projects/PINNPEX/output_intel22/active_learning/m6_v10b_baseline_seed{0..4}/best_model.pth

# Approach: subprocess legacy `src/evaluation/evaluator.py` with
# monkey-patched cfg (output_dir=legacy, processed_dir=v3) so it
# loads m6_v10b ckpt and evaluates on v3 valid manifest.
# Compatible with cross-boundary edits already in place.

# Output: pex_v3/output/baselines/legacy_v10b_v3val/seed{0..4}/metrics_row.csv
# Then 07_b1_vs_b3_comparison.py-style MWU on raw 5-seed pairs.
```

### P4 — Paper #1A draft kickoff (1-2 weeks)

Outline already in `pex_v3/docs/PROGRESS_REPORT_2026_05_02_END.md` §"Paper #1A".
Write `paper/sections/{intro,data,methods,results,discussion}.md`.

### P5 — Mesh_v3 (paper #1B, 6 days for MVP)

Per A6 spec at `pex_v3/docs/PHASE1_HYBRID_ARCH_SPEC.md` §11 + A6 [ROLE PASS]
output. NOT next sprint — paper #1A first.

---

## 🗂️ Critical file locations

### Live status
- `pex_v3/PHASE_STATUS.md` — live tracker (update at every milestone)
- `pex_v3/IMPLEMENTATION_STATUS.md` — honesty register (✅/⚠️/🟡/❌)
- `pex_v3/docs/PROGRESS_REPORT_2026_05_02_END.md` — last session's summary
- This: `pex_v3/SESSION_HANDOFF.md`

### Key results
- `pex_v3/output/cross_design_eval/PAPER_GRADE_COMPARISON.md` — final leaderboard
- `pex_v3/output/baselines/B1_xgboost_real/per_method.csv` — B1 5-seed
- `pex_v3/output/baselines/B3_pinn_real/per_method.csv` — B3 5-seed
- `pex_v3/output/baselines/B4_compact_gam/five_seed_summary.json` — B4 5-seed
- `pex_v3/output/baselines/B1_vs_B3/comparison.md` — paired MWU SUPPORTED

### Strategy + lessons
- `pex_v3/docs/STRATEGY_V3_UPDATED_PLAN.md` — current plan (4× revised)
- `pex_v3/docs/PHASE1_TIER3_CEILING_FINDING.md` — hand-feature ceiling
- `pex_v3/docs/PHASE_C_A1_REVALIDATION_VERDICT.md` — A1 statistician
- `pex_v3/docs/PHASE_C_A2_STRATIFIED_FINDINGS.md` — A2 baseline-owner
- `pex_v3/docs/PHASE_C_ROUND1_AUDIT.md` — agent findings
- `pex_v3/docs/PHASE1_K3_CANARY_LESSON.md` — K3 design discovery
- `pex_v3/docs/AGENT_INFRA_GAP.md` — custom agents NOT directly invocable

### Code modules
- `pex_v3/src/models/{analytic_base,residual_head,hybrid}_v3.py` — Phase 1 Tier 0
- `pex_v3/src/trainers/{finetune_hybrid,pretrain_synthetic,transfer_canary}_v3.py` — Phase 1 trainers
- `pex_v3/src/baselines/{features,xgboost,compact_gam,calibration}_v3.py` (mostly _v3 suffix)
- `pex_v3/src/baselines/{features,xgboost_baseline,compact_gam_v3,calibration_v3,pinn_baseline,paragraph_baseline,gam_baseline,feature_dataset}.py`
- `pex_v3/src/evaluation/{metrics,stratified_eval,seed_aggregator}.py`
- `pex_v3/src/synthetic/{stage1_parallel_plate,stage2_layered_image,ground_truth,transfer_canary}.py`

### Scripts (numbered = phase order)
- `01_resplit_manifest.py` — H1 fix
- `02_rebuild_dataset_h3.py` — H3 rebuild (gated)
- `04_build_feature_dataset.py` — DEF→features
- `05_5seed_runner.py` — generic 5-seed orchestrator
- `06_run_pinn_multigpu.py` — B3 PINN multi-GPU
- `07_b1_vs_b3_comparison.py` — paired MWU
- `08_b1_stratified_report.py` — per-design × per-channel × per-quartile
- `09_pretrain_and_canary.py` — K3 canary (FIRED, not for re-run)
- `10_finetune_hybrid_smoke.py` — Phase 1 single-seed smoke
- `11_finetune_hybrid_5seed.py` — Phase 1 multi-GPU 5-seed
- `12_b4_compact_gam_eval.py` — B4 5-seed
- `13_cross_design_acc_runtime.py` — cross-method comparison

---

## 🧠 Critical context (don't forget!)

### Decisions that already happened
1. **<4% per-net hard target stays** but reframed: paper #1A reports ceiling, paper #1B targets <4% via mesh_v3
2. **Synthetic pretrain DROPPED** (K3 fired in 3 minutes; saved ~125 GPU-days)
3. **Stage 2 Mode B [HYPOTHESIS]-level** (A4 audit) — do not use without vector-fitting replacement
4. **Custom agents NOT directly invocable** as `subagent_type` — use `general-purpose` + embed role-md path
5. **Cross-boundary edits authorized**:
   - `scripts/build_dataset.py:528` — env var read (H3 fix)
   - `src/__init__.py` + 9 subpackages — torch introspection compat
   - Both documented in `pex_v3/docs/CROSS_BOUNDARY_*.md`

### Patterns to honor
- **Per-channel β strategy** — never train on `total = gnd + cpl` alone (causes cancellation learning)
- **5-seed protocol** for any improvement claim — single seed is suspicion not signal
- **TORCH_COMPILE_DISABLE=1** for paper runs (A7 #12 determinism)
- **Last-step checkpoint** preferred over best-step (A1 anti-overclaim)
- **Boundary rule** — work only inside `pex_v3/`; legacy is read-only post-mortem

### Things NOT to do
- ❌ Don't re-attempt synthetic pretrain (K3 fired)
- ❌ Don't use Stage 2 Mode B without vector-fitting replacement
- ❌ Don't claim "X beats Y" without paired n=5 MWU
- ❌ Don't try `subagent_type: "<custom>"` — only built-ins work
- ❌ Don't optimize hybrid_v3 with hand features (ceiling reached; capacity not bottleneck)

### Things to do
- ✅ Use general-purpose agent + embed `/home/jslee/projects/PINNPEX/.claude/agents/<role>.md` path
- ✅ Save provenance.json on every run (manifest hash + git SHA + seed + cuda env)
- ✅ Apply per-channel loss (gnd + cpl separately, then sum)
- ✅ Update PHASE_STATUS at every milestone
- ✅ Save memory entry for any new finding

---

## 📊 Test count + invariants

```
170 unit tests passing as of session end.
Run: $PYBIN -m pytest pex_v3/tests/ --tb=no -q

Critical invariants verified by tests:
  - H1 net-level split: 0 (design,net) overlap (test_split_invariants.py)
  - H2 priority truncation: targets always retained (test_priority_truncation.py)
  - 4-way seed determinism (test_determinism.py)
  - Stage 1+2 closed-form parity (test_synthetic_stages.py)
  - Hybrid_v3 day-1 == analytic (test_hybrid_v3.py)
  - Per-channel loss does NOT collapse to total (test_hybrid_v3.py)
  - K3 canary verdict logic (test_transfer_canary_v3.py)
  - Compact+GBDT residual variants (compact_gam_v3 imports lock schema)
```

---

## 🤖 Available tools recap

### Background jobs
- Multi-GPU multi-seed via `06_run_pinn_multigpu.py` template (5 GPUs available: 0,2,3,4,7)
- Use `run_in_background=True` in Bash tool

### Codex deliberation
```bash
# For non-trivial design decisions:
# Use Skill tool: skill: "codex:rescue", args: "--wait <prompt>"
# Or in agent form via codex:codex-rescue subagent_type
```

### Specialist agents (via general-purpose wrapper)
```python
Agent(
  subagent_type="general-purpose",
  prompt="""You are being invoked as the **<role>** specialist agent.
            Read your role at /home/jslee/projects/PINNPEX/.claude/agents/<role>.md
            FIRST and operate strictly within that role.

            [task]"""
)
```

8 roles available:
- pex-physics-architect
- neural-operator-architect
- graph-geometry-engineer
- experiment-systems-engineer
- benchmarking-statistician (A1)
- pex-data-engineer (A3)
- classical-baseline-owner (A2)
- synthetic-data-pipeline-owner (A8 — not yet invoked)

### Memory
```
/home/jslee/.claude/projects/-home-jslee-projects-PINNPEX/memory/
  MEMORY.md (index, READ AT SESSION START)
  + 27 individual entries
```

---

## 💬 What to tell the user when resuming

```
"세션 핸드오프 받았습니다. 현재 상태:
 - Paper #1A 자료 거의 완성 (B1, B3, B4, Option F 5-seed 결과 + cross-design)
 - 핵심 발견 4개 (32pp data-fix, β cancellation, 4.66% ceiling, K3 saved 125 GPU-days)
 - 다음 우선순위: P1 5-seed Option F (~5분), P2 B1 OOD test 추가, P3 legacy v10b 재평가

 SESSION_HANDOFF.md에 상세 내용. 어느 priority부터 진행할까요?"
```

또는 auto mode 유지면:
```
"P1 (5-seed Option F)부터 자동 진행합니다. ~5분 소요 예상."
```

---

## 🎯 One-line summary

**Hand features hit a 4.66% ceiling (XGBoost = deep MLP). B4 V3 log-GBDT
(5.72% valid, 6.59% test, +0.87pp OOD gap) is the best physics-informed
baseline. Phase 1 paradigm needs mesh_v3 per-cuboid features to break
the ceiling. Paper #1A (methodology, ~2 weeks) ready to draft.**
