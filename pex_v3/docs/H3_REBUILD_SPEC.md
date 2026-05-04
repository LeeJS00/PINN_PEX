# H3 Rebuild Specification

> **DEPRECATED (2026-05-05)**: H3 rebuild complete (1,322,115 tiles · 257,438 nets · 11/11 designs · 493 GB at `/data/PINNPEX/data/processed_v3/intel22/`). See `pex_v3/PHASE_STATUS.md`. Kept for spec history.

_Status: design — implementation gated by user --confirm_

## Why

Legacy `scripts/build_dataset.py:528` hardcodes `context_margin = 2.0` μm.
Combined with `WINDOW_SIZE = (4.0, 4.0, 20.0)` μm, the stored tile is 8×8 μm.
The model's CPL search radius is `cfg.CUTOFF_RADIUS = 4.0` μm. A target
cuboid near the tile edge needs aggressors within 4 μm — but only 2 μm
margin is saved, so the necessary aggressors are not in the file.

Result: top-metal long-parallel coupling (M7/M8 wires running in parallel
for tens of μm) is *physically uncapturable* by any model on this dataset.

## Fix

Increase `context_margin` to 5.0 μm (≥ `CUTOFF_RADIUS + 1.0` for slack).

| | legacy | v3 |
|---|---|---|
| context_margin | 2.0 μm | 5.0 μm |
| stored window xy | 8.0 × 8.0 μm | 14.0 × 14.0 μm |
| stored window z | 20 μm | 20 μm |
| relative volume | 1× | 3.06× |

## Cost

- **Time**: 2-4 GPU-day for full rebuild (90 designs × ~10 min build per design × 64 workers; CPU-bound, not GPU)
- **Disk**: 390 GB (legacy v9) → ~1.2 TB (v3 H3)
- **Storage path**: `/data/PINNPEX/data/processed_v3/intel22/`
- **Disk available**: 1.7 TB on `/data/` (verified 2026-05-01)

## Procedure

```
1. Backup the legacy manifest (already exists as dataset_manifest_v8_backup.csv)
2. Set cfg.CONTEXT_MARGIN_V3 = 5.0 (already in config_v3.py)
3. For each design in TRAIN_DEFS + TEST_DEFS:
     a. Run legacy build_dataset.py with --out_dir = cfg.PROCESSED_DIR_V3 / <design>
        and override context_margin via env var or argparse extension
     b. Validate per-design tile count matches expectation (tile area shrinks
        because larger windows cover more area per tile, so total tile count
        per design DECREASES proportional to (8/14)^2 ≈ 0.33×)
4. Aggregate per-design maps into v3 manifest using H1 hash split
5. Run H1 invariant checks (run_all_checks)
6. Verify (sample 10 random tiles per design) that stored window contains
   geometry up to 7 μm from tile center in xy — aggressor at 4 μm cutoff
   has 3 μm of buffer
7. Update PHASE_STATUS.md
```

## Implementation port plan

`02_rebuild_dataset_h3.py` will need to either:
1. Call the legacy `scripts/build_dataset.py` via `subprocess.run` with an
   environment variable (`PEX_CONTEXT_MARGIN=5.0`) that the legacy script
   reads at startup. This requires a 1-line change to legacy
   `build_dataset.py:528` to read the env var before falling back to 2.0.
   *Cross-boundary edit* — must follow `pex_v3/CLAUDE.md` boundary protocol.
2. Or: replicate the relevant tiling code in `pex_v3/src/preprocessing/`,
   defaulting to v3 margin. More code; cleaner boundary.

**Decision**: option (2). Even though it duplicates code, it keeps
strategy-v3 work fully isolated from legacy. Phase 1+ will likely rewrite
the data pipeline anyway as part of the conductor-surface-mesh transition,
so duplication is short-lived.

## Acceptance criteria

After H3 rebuild:
- [ ] `du -sh /data/PINNPEX/data/processed_v3/intel22/` ≤ 1.4 TB
- [ ] `cfg.MANIFEST_PATH_V3` exists, valid v3 schema
- [ ] H1 invariants green on rebuilt manifest
- [ ] Sample 10 tiles per design — verify aggressor coverage extends ≥ 7 μm from tile center
- [ ] PHASE_STATUS.md updated

## Risk

- **Disk overflow**: Estimate ~1.2 TB; if actual is > 1.5 TB, abort and
  reconsider context_margin = 4.0 μm fallback (10×10 μm window).
- **Manifest drift**: H3 build invalidates the predefined AL caches at
  `output_intel22/active_learning/cache/predefined_*.csv`. After rebuild,
  v3 needs its own cache regeneration in `pex_v3/output/`.
- **StarRC SPEFs unchanged**: Golden SPEFs at
  `/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22/` are
  full-chip; no rebuild needed there.
