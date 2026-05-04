# Phase 0 Plan — Foundation rebuild

_Status: in progress (started 2026-05-01)_

## Goal

Rebuild the data + measurement foundation so that any architectural
comparison (Phase 0.5 onward) is statistically sound. Without this, no
paradigm shift can be measured cleanly.

## Sub-tasks (5 fixes + 4 tests)

### H1 — net-level split (manifest only, ~10 min CPU)

**Where**: `src/data/manifest.py`, `scripts/01_resplit_manifest.py`
**Approach**: SHA256 of `(hash_seed, design_name, net_name)` → uniform
[0, 1) → bucket vs `valid_ratio`. TEST_DEFS designs always 'test'.
**Acceptance**: `test_split_invariants.py` green; legacy 12.32% net mixing → 0%.

### H2 — priority truncation (runtime, no rebuild)

**Where**: `src/data/datasets.py:priority_truncate`
**Approach**: Sort by `(is_target, distance_from_center)` ascending; keep
first `pad_size` rows. Targets always retained.
**Acceptance**: `test_priority_truncation.py` green.

### H3 — context margin 2→6 μm rebuild (gated; 2-4 GPU-day, 1.2 TB)

**Where**: `scripts/02_rebuild_dataset_h3.py`, `docs/H3_REBUILD_SPEC.md`
**Approach**: Re-run tiling with `cfg.CONTEXT_MARGIN_V3 = 5.0` so stored
window is 14×14×20 μm. Captures top-metal coupling within 4 μm cutoff.
**Disk available**: 1.7 TB on `/data/` ✅
**Acceptance**: new manifest at `cfg.MANIFEST_PATH_V3` after rebuild;
H1 invariants re-validated; per-design `du -sh` documented.
**Status**: stub script gates with `--confirm`; awaiting Phase 0 H1+H2+M5 stable.

### H4 — pairwise CPL search (model-side, no rebuild)

**Where**: design doc at `docs/H4_PAIRWISE_CPL_DESIGN.md`; implementation
in Phase 1 model code.
**Approach**: Replace `closest_dist`-based edge selection with pairwise
enumeration of (target_cuboid, aggressor_cuboid) pairs whose surface-to-
surface distance ≤ `cutoff_radius`. Edge count grows ~2.25× → bump
`MAX_AGGR_BUDGET` to 768 (already in `cfg.MAX_AGGR_BUDGET_V3`).
**Acceptance**: deferred to Phase 1 — H4 should be designed against the
hybrid analytic+neural arch, not retrofitted into legacy flux_head.

### M5 — SSL split filter (training-side, ~5 min code + SSL re-run 11 GPU-h)

**Where**: `src/trainers/train_ssl_v3.py`
**Approach**: Filter dataset by `manifest['split'] == 'train'` in addition
to design name. Eliminates encoder memorization of valid-net features.
**Acceptance**: training dataset count matches `manifest['split'].value_counts()['train']`
in train designs only.

### Tests (Phase 0 acceptance gate)

- `tests/test_split_invariants.py` — H1
- `tests/test_priority_truncation.py` — H2
- `tests/test_determinism.py` — 4-way seed
- `tests/conftest.py` — shared fixtures

## Sequencing

```
1. Write code: manifest.py, datasets.py, leak_check.py, seeds.py, manifest_hash.py
2. Write tests
3. Run pytest (all green)
4. Run 01_resplit_manifest.py → produces v3 manifest at cfg.MANIFEST_PATH_V3
5. Verify summary report — net mixing eliminated
6. (Gated) Run 02_rebuild_dataset_h3.py --confirm
7. (Gated) Run 03_train_ssl_v3.py for SSL re-pretrain on rebuilt data
8. 5-seed legacy DeepPEX baseline on rebuilt data → output/baseline_v3_legacy_pinn/
9. Update PHASE_STATUS.md, hand off to Phase 0.5
```

## Open work for next session

- [ ] Wire the SSL training loop body in `train_ssl_v3.py` (currently NotImplementedError stub)
- [ ] Port build_dataset.py + build_dataset_multi.py into `02_rebuild_dataset_h3.py` body
- [ ] Run pytest, run 01_resplit, validate summary
- [ ] Decide whether SSL re-pretrain is needed for Phase 0.5 baseline (or use legacy ssl_basis_dspinn_v1 for now and gate re-pretrain on Phase 1)

## Notes

- `cfg.MANIFEST_PATH_V3` will not exist until `01_resplit_manifest.py` runs; loaders that depend on it must check existence and emit a useful error.
- Legacy PT manifest (`/data/PINNPEX/data/processed/intel22_pt/dataset_manifest.csv`) is untouched. v3 will produce its own PT manifest only after H3 rebuild.
