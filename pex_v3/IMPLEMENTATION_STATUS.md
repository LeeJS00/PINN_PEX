# Implementation Status — Stub vs Implemented

_Updated: 2026-05-01_

This file is the **honesty register**: what's actually working end-to-end vs
what looks like code but is currently a stub. Read this BEFORE assuming any
file is callable on real data.

Companion to `PHASE_STATUS.md` (which tracks phase progress) — this is the
zoomed-in implementation truth-table.

## ✅ Fully implemented + tested (production-ready for v3)

| Module | LOC | Tests | Notes |
|---|---:|---:|---|
| `src/data/manifest.py` | 110 | 9 | H1 hash split + invariants |
| `src/data/leak_check.py` | 80 | (covered by test_split_invariants) | invariant assertions |
| `src/data/datasets.py` | 130 | 5 | H2 priority truncation (pure function) |
| `src/utils/seeds.py` | 60 | 5 | 4-way seed |
| `src/utils/manifest_hash.py` | 100 | 0 (smoke via 5seed-runner) | provenance writer |
| `src/synthetic/ground_truth.py` | 280 | 12 | analytic helpers |
| `src/synthetic/stage1_parallel_plate.py` | 120 | 6 | 1M sample generator |
| `src/synthetic/stage2_layered_image.py` | 210 | 6 | Mode A + Mode B |
| `src/evaluation/metrics.py` | 210 | 6 | 4-metric reporting |
| `src/evaluation/stratified_eval.py` | 190 | 5 | per-axis slicing |
| `src/evaluation/seed_aggregator.py` | 190 | 6 | MWU + Cohen's d + bootstrap |
| `src/baselines/features.py` | 370 | 20 | NetGeometry → NetFeatureVector |
| `scripts/01_resplit_manifest.py` | 95 | E2E (1.32M tile manifest) | H1 fix entrypoint |
| `scripts/02_rebuild_dataset_h3.py` | 230 | E2E smoke (gcd 8s) | H3 rebuild orchestrator |
| `scripts/05_5seed_runner.py` | 180 | E2E smoke (xgboost 645s) | model-agnostic 5-seed |

## ⚠️ Body implemented but ONLY tested with synthetic data — real-data path untested

| Module | LOC | Issue | Blocker |
|---|---:|---|---|
| `src/baselines/xgboost_baseline.py` | 190 | Uses `_make_synthetic_features_df()` — synthetic regression task. Real-data path requires importing from `feature_dataset.py` instead of the synthetic stub. ~10 LOC swap. | **wire up `feature_dataset.py`** |
| `src/baselines/pinn_baseline.py` | 230 | Body wraps legacy AL trainer via cfg monkey-patch + symlinks. Verified at import-level only; not yet run end-to-end on real v3 data. | full H3 rebuild + first run will validate |
| `src/baselines/feature_dataset.py` | 410 | Body works (gcd smoke 276 nets/15.9s); `_layer_eps_array` heuristic uses string match on layer names — may need tuning per PDK. Coupling cap of 256 aggressors per net hits saturation on large nets — bump to 1024 for full run. | extend to all 11 designs once H3 rebuild done |

## 🟡 Stub bodies (function signatures locked, body = `NotImplementedError`)

These files **look complete** — function signatures, docstrings, type hints all there. But every function raises `NotImplementedError`. **Do not call them and expect output.**

| Module | Status | Activation gate |
|---|---|---|
| `src/trainers/train_ssl_v3.py` | training loop body = `NotImplementedError`. Manifest filter, provenance, seed setting are real. | wire legacy SSL training loop OR rewrite for v3 model |
| `src/baselines/paragraph_baseline.py` | every function raises | port ParaGraph (DAC 2020) GNN |
| `src/baselines/gam_baseline.py` | every function raises | implement Sakurai-Tamaru analytic + GBDT residual |
| `src/synthetic/transfer_canary.py` | raises | needs Phase 1 model first |
| `src/synthetic/ground_truth.py:cross_validate_oracles` | implemented | (this one is actually real) |

## ❌ Not yet started (referenced in plan but no file yet)

| What | Where it should go | Phase |
|---|---|---|
| `mesh_v3.py` (conductor surface mesh utility) | `src/preprocessing/mesh_v3.py` | Phase 1 |
| `analytic_base_v3.py` (differentiable Mode A + B Green's function) | `src/models/analytic_base_v3.py` | Phase 1 |
| `residual_head_v3.py` (bounded MLP residual) | `src/models/residual_head_v3.py` | Phase 1 |
| `hybrid_v3.py` (composes analytic + residual) | `src/models/hybrid_v3.py` | Phase 1 |
| `pretrain_synthetic.py` (Stage 1+2 trainer) | `scripts/06_pretrain_synthetic.py` | Phase 1 |
| `train_pattern_v3.py` (real-data finetune) | `scripts/07_train_pattern_v3.py` | Phase 1 |
| `eval_pattern_v3.py` (stratified eval) | `scripts/08_eval_pattern_v3.py` | Phase 1 |

## 🔄 In-flight (background tasks)

| Job | ID | Phase | ETA |
|---|---|---|---|
| H3 dataset rebuild | bash `b8w736j84` | Phase 0 | finishing nova + tv80s; v3 manifest auto-written on completion |
| 5-seed XGBoost (synthetic, validation) | bash `bx6oebuo6` | Phase 0.5 sanity | ~50 min from start; will produce 5-seed aggregates |
| Codex round 3 (Phase 1 spec review) | `task-momwfda2-qvugod` | Phase 1 design | ~5-10 min Codex side; results in spec change recommendations |

## How to call this register

When the user asks "is X working?":
1. Check this file FIRST
2. If "✅ Fully implemented" → yes, call it
3. If "⚠️ Synthetic only" → works on synthetic input; explain real-data blocker
4. If "🟡 Stub" → no, it raises `NotImplementedError`; explain what's needed
5. If "❌ Not yet started" → file doesn't exist
6. If "🔄 In-flight" → background job; wait for completion notification

When implementing something new, add a row here BEFORE calling it "done."
