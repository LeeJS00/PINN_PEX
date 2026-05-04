# CLAUDE.md — pex_v3 subfolder rules

This file augments the parent `PINNPEX/CLAUDE.md`. When working inside
`pex_v3/`, read both. Where they conflict, this file wins.

## Boundary rule (CRITICAL)

**Do not modify any file outside `pex_v3/`.** Legacy `src/`, legacy
`scripts/`, legacy `configs/` are read-only post-mortem references.
The four failed tracks (GINO, DS-PINN, NNLS, γ head) live there and the
project memory uses them as the "what didn't work" anchor.

If a Phase 1+ paradigm decision requires changing legacy code (e.g.,
shared parser fix), the procedure is:
1. Discuss with user before any cross-boundary edit.
2. Document the change in `pex_v3/docs/CROSS_BOUNDARY_<topic>.md`.
3. Apply minimum surgical edit to legacy + ensure no Strategy-v3 work
   becomes silently dependent on undocumented legacy state.

Reading legacy code is fine and encouraged.

## Data path discipline

- **Legacy data** (read-only post-mortem): `/data/PEX_SSL/data/processed/intel22/`
  (390 GB, v9 manifest, v8 backup).
- **v3 data root**: `/data/PINNPEX/data/processed_v3/intel22/`
  (created by Phase 0 scripts; will hold rebuilt H3 data ~1.2 TB).
- **v3 manifest** (H1 fix): `/data/PINNPEX/data/processed_v3/intel22/dataset_manifest_v3.csv`
  Phase 0 H1 step rewrites the *split* column only (net-level hash) without
  touching the legacy manifest.

Never overwrite `/data/PEX_SSL/data/processed/intel22/dataset_manifest.csv`.

## Phase 0 acceptance criteria

Before declaring Phase 0 complete:
1. `pex_v3/tests/test_split_invariants.py` — green (no `(design, net)` overlap
   across train/valid/test splits in v3 manifest).
2. `pex_v3/tests/test_priority_truncation.py` — green (target cuboids
   always retained when N > pad_size).
3. `pex_v3/tests/test_determinism.py` — green (same seed → identical loss curve).
4. `PHASE_STATUS.md` updated with H1/H2/H3/H4/M5 fix status + 5-seed
   baseline numbers on rebuilt data.

## Workflow rules (override / extend parent)

- **Korean to user, English in code/commits/docs** (parent rule).
- **Codex deliberation loop is the outer convergence layer** for any
  non-trivial Phase 1+ design decision (parent rule).
- **5-seed protocol before any "improvement claim"** (parent rule).
- **Strategy v3 PHASE_STATUS.md is the live source of truth** for which
  phase is active and what's blocked. Update it at each milestone.
- **Hard kill criteria K1/K2/K3** are check-gates. If any fires, stop and
  report to user; don't push through.
- **Specialist agents have priority for their domain**. Lead session must
  delegate physics review to `pex-physics-architect`, architecture to
  `neural-operator-architect`, etc., before merging Phase 1+ code.

## File organization rules

- Numbered scripts (`scripts/01_*.py`, `scripts/02_*.py`, ...) for
  entrypoints. Number = phase order, easy to discover.
- Tests at `tests/test_<topic>.py`, follow pytest convention.
- Design docs at `docs/<TOPIC>_DESIGN.md` — written *before* implementation,
  reviewed by relevant specialist agent, then code follows.
- Memory entries at `/home/jslee/.claude/projects/-home-jslee-projects-PINNPEX/memory/`
  remain at project level (not duplicated here).

## What lives where

- Strategy summary: `pex_v3/README.md`
- Live tracker: `pex_v3/PHASE_STATUS.md`
- Boundary rules: this file
- Phase plans + design docs: `pex_v3/docs/`
- Code: `pex_v3/src/`
- Entrypoints: `pex_v3/scripts/`
- Tests: `pex_v3/tests/`
- Outputs: `pex_v3/output/`

## When in doubt

- Strategy v3 plan questions → `README.md` + `docs/PHASE0_PLAN.md`
- Why is X archived? → `../docs/PROJECT_REPORT.md` (legacy post-mortem)
- What does the user prefer? → memory at `/home/jslee/.claude/projects/.../memory/`
- How does H4 work? → `docs/H4_PAIRWISE_CPL_DESIGN.md` (Phase 0 spec)
