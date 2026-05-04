# Phase C — Agent Role Validation Plan

_Date: 2026-05-01_
_Trigger: Phase B real-data experiments complete (B1 XGBoost + B3 PINN 5-seed measured)_

8 specialist agents were defined in `.claude/agents/` at session start but
have not been invoked. Per user directive (실험 후 agents 호출하여 각 역할이
잘 작동하는지 확인), each agent gets a domain-specific task once Phase B
data is available. Goal: verify each agent (a) reads its own role, (b)
produces useful domain-specific output, (c) catches issues a generalist
might miss.

## Pre-conditions

Phase B outputs that the validation tasks consume:
- `pex_v3/output/baselines/B1_xgboost_real/per_method.csv` (5-seed B1)
- `pex_v3/output/baselines/B1_xgboost_real/per_run.csv`
- `pex_v3/output/baselines/B3_pinn_real/per_method.csv` (5-seed B3)
- `pex_v3/output/baselines/B3_pinn_real/per_run.csv`
- `/data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv` (feature dataset)
- `/data/PINNPEX/data/processed_v3/intel22/dataset_manifest_v3.csv` (v3 manifest)

## 8 agent validation tasks

### A1. benchmarking-statistician

**Task**: Audit the B1 vs B3 5-seed comparison and produce a paper-grade
comparison table.

**Inputs**:
- per_method.csv for both B1 and B3
- per_run.csv (for MWU + Cohen's d)
- mwu_pairs.csv if present

**Expected output**:
- Combined paper-grade table (1 row per method, all 4 metrics: cap MAPE,
  delay error, power error, RC chip-ratio)
- MWU + Cohen's d B1-vs-B3 with verdict ('supported' / 'small_effect' / 'ns')
- Bootstrap 95% CI on median MAPE for each
- Anti-overclaim check: does the data support any "X beats Y" claim?

**Validates**: 5-seed protocol enforcement, anti-overclaim discipline,
paper-grade reporting, MWU + bootstrap correctness.

---

### A2. classical-baseline-owner

**Task**: Given B1+B3 results, decide priority for B2 (ParaGraph) and
B4 (GAM) baseline implementation.

**Inputs**:
- B1 + B3 per_method.csv + per_run.csv
- `pex_v3/src/baselines/README.md` (4-baseline plan)
- The literature analysis user shared (CNN-Cap, NAS-Cap, ParaGraph 30%+, ResCap)

**Expected output**:
- Recommended priority: B2 first or B4 first?
- Justification (which fills the bigger gap in the comparison story?)
- Implementation effort estimate for each
- Anticipated reviewer pushback if either baseline is omitted

**Validates**: knowledge of literature SOTA, awareness of paper-grade
baseline requirements, anticipation of reviewer questions.

---

### A3. pex-data-engineer

**Task**: Audit `feature_dataset.py` output for sanity. Specifically:
distribution checks on per-net features and per-design coverage.

**Inputs**:
- `/data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv`
- v3 manifest

**Expected output**:
- Per-design row counts vs manifest (catch any drops)
- Feature distribution sanity (e.g., n_aggressor_nets capped at 256 — too
  low for ldpc?)
- Missing/NaN values per column
- Any anomalies (e.g., negative wire length, impossible eps values)
- Recommendation: should max_aggr_per_net be raised?

**Validates**: parser discipline, manifest hygiene, leak invariants
verification on real data.

---

### A4. pex-physics-architect

**Task**: Audit `ground_truth.py` + Stage 1/2 generators for physics
correctness. Review the analytic formulas against literature.

**Inputs**:
- `pex_v3/src/synthetic/ground_truth.py`
- `pex_v3/src/synthetic/stage1_parallel_plate.py`
- `pex_v3/src/synthetic/stage2_layered_image.py`
- `pex_v3/tests/test_synthetic_stages.py`

**Expected output**:
- Formula correctness (parallel plate, stacked dielectric series, image
  charge correction)
- Limit-case verifications (do the formulas reduce correctly?)
- Citations for each formula (Sakurai-Tamaru, image method, FastCap)
- Any subtle units / sign / factor-of-2 bugs
- Recommendation: are the synthetic stages physically grounded enough to
  pretrain a Phase 1 model?

**Validates**: physics correctness gate, "cite or refuse" discipline,
literature awareness.

---

### A5. neural-operator-architect

**Task**: Read Codex round 3 review of Phase 1 spec (when complete) and
propose concrete next implementation steps for the hybrid model.

**Inputs**:
- `pex_v3/docs/PHASE1_HYBRID_ARCH_SPEC.md`
- Codex round 3 review output (background task `task-momwfda2-qvugod`)

**Expected output**:
- Spec changes to incorporate from Codex review (P1 must-fix items)
- Implementation order for the 7 files in `PHASE1_HYBRID_ARCH_SPEC.md` §11
- Parameter budget + activation memory for the hybrid model
- Risks to address before any AL training

**Validates**: architecture leadership, integration of external review,
inductive-bias-first thinking.

---

### A6. graph-geometry-engineer

**Task**: Design the conductor surface mesh format for Phase 1
(`mesh_v3.py`). Specify representation, generation algorithm, and
invariants.

**Inputs**:
- `pex_v3/docs/PHASE1_HYBRID_ARCH_SPEC.md` §2 (representation discussion)
- Existing cuboid format docs in `pex_v3/src/data/datasets.py`

**Expected output**:
- Patch dataclass spec
- Mesh generation algorithm (Manhattan routing edge case handling)
- Invariants test set (translation, reflection, layer-permutation)
- Storage format proposal (numpy arrays in npz, or torch tensors)

**Validates**: representation-design discipline, geometric invariant
awareness, sparse 3D handling expertise.

---

### A7. experiment-systems-engineer

**Task**: Audit the reproducibility infrastructure on a real Phase B run.

**Inputs**:
- `pex_v3/output/baselines/B1_xgboost_real/seed{0..4}/provenance.json`
- B3 equivalents
- `pex_v3/src/utils/seeds.py`, `manifest_hash.py`
- v3 manifest schema_version stamping

**Expected output**:
- Are all 4 RNG sources seeded per run?
- Does manifest hash match across all 5 seeds (it must)?
- Is git SHA logged + dirty state flagged?
- Run-twice reproducibility test plan
- Any silent format drift between seeds

**Validates**: determinism enforcement, provenance discipline, leak
invariant verification.

---

### A8. synthetic-data-pipeline-owner

**Task**: Plan Stage 3+ Q3D oracle integration. Stage 1+2 are analytic;
Stage 3 onward needs commercial 3D solver labels.

**Inputs**:
- `pex_v3/src/synthetic/README.md` (5-stage curriculum)
- `pex_v3/docs/PHASE1_HYBRID_ARCH_SPEC.md` §6 (transfer canary)
- User context: which 3D solvers are available?

**Expected output**:
- Stage 3 sample budget + per-sample cost
- Q3D / FastCap setup requirements
- Cross-validation plan (1000 samples both oracles)
- Hard kill K3 specifics — what exact metric triggers abort?
- Cost-benefit: do we actually need Stage 3 for Phase 1 paper, or can we
  ship paper #1 on Stage 1+2 only?

**Validates**: cost-aware planning, oracle validation discipline, gating
discipline (don't burn GPU-months without canary).

---

## Invocation order

Run A1 first (statistical foundation), then A3 + A2 in parallel (data +
baseline review). A4 + A7 can also run in parallel (independent domains).
A5 + A6 require Codex round 3 output; defer until that completes.
A8 is lowest priority (Phase 1.5 prep).

```
Round 1 (immediate after Phase B):
  A1 benchmarking-statistician   (validates 5-seed protocol)
  A3 pex-data-engineer           (validates feature_dataset)
  A4 pex-physics-architect       (validates synthetic generators)
  A7 experiment-systems-engineer (validates reproducibility)

Round 2 (after Round 1 + Codex round 3):
  A2 classical-baseline-owner    (next-baseline priority)
  A5 neural-operator-architect   (Phase 1 implementation start)
  A6 graph-geometry-engineer     (mesh_v3.py design)

Round 3 (planning):
  A8 synthetic-data-pipeline-owner (Stage 3+ planning)
```

## Success criteria

Each agent passes its validation if:
1. Output is concretely actionable (not generic platitudes)
2. References specific files / line numbers / data points
3. Catches at least one issue (or explicitly confirms cleanliness)
4. Stays within domain (no scope drift)

Failure: agent produces generic ML-coach-style output without
domain-specific anchor. Such an agent's role is wasted; rewrite its
definition before relying on it for Phase 1+.
