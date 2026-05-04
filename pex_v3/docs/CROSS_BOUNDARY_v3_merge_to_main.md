# CROSS_BOUNDARY: v3 → main folder merge

_Created: 2026-05-03. User-requested merge of validated `pex_v3/` artifacts into the main project folder._

## Context

CLAUDE.md (project) and `pex_v3/CLAUDE.md` (subfolder) state:
> "Do not modify any file outside `pex_v3/`. Legacy `src/`, legacy `scripts/`, legacy `configs/` are read-only post-mortem references."

User explicitly requested merge ("메인 폴더에 병합하는 과정"). This document
records the cross-boundary edits + rationale per CLAUDE.md procedure
(discuss → document → minimum surgical edit).

## Approach: Symlink-based exposure (safest)

**Goal**: Make validated `pex_v3/src/*` modules importable from main
project (`src.v3`, `configs.v3`) without duplicating files.

**Rationale**:
- Single source of truth (no risk of divergence)
- Trivially reversible (`unlink src/v3`)
- pex_v3 boundary preserved per CLAUDE.md
- Legacy `src/` modules unchanged (zero risk)
- User can later choose to physical-move if symlinks become awkward

## What gets merged (validated, paper-grade)

### Models (5/5 paper pillar)
| pex_v3 path | main exposure |
|---|---|
| `pex_v3/src/models/cuboid_set_encoder.py` | `src.v3.models.cuboid_set_encoder` |
| `pex_v3/src/models/hybrid_v3_mesh.py` | `src.v3.models.hybrid_v3_mesh` |
| `pex_v3/src/models/analytic_base_v3.py` | `src.v3.models.analytic_base_v3` |
| `pex_v3/src/models/residual_head_v3.py` | `src.v3.models.residual_head_v3` |
| `pex_v3/src/models/hybrid_v3.py` | `src.v3.models.hybrid_v3` |

### Calibration / Dataset / Trainer
| pex_v3 path | main exposure |
|---|---|
| `pex_v3/src/baselines/calibration_v3.py` | `src.v3.baselines.calibration_v3` |
| `pex_v3/src/baselines/features.py` | `src.v3.baselines.features` |
| `pex_v3/src/baselines/xgboost_baseline.py` | `src.v3.baselines.xgboost_baseline` |
| `pex_v3/src/data/cuboid_set_dataset.py` | `src.v3.data.cuboid_set_dataset` |
| `pex_v3/src/trainers/finetune_hybrid_v3.py` | `src.v3.trainers.finetune_hybrid_v3` |

### Configs
| pex_v3 path | main exposure |
|---|---|
| `pex_v3/configs/config_v3.py` | `configs.config_v3` |

### Scripts (already use absolute paths, no change needed)
- `pex_v3/scripts/14_option_f_5seed.py` (Option F MLP 5-seed)
- `pex_v3/scripts/16_xgb_calibrate_spef.py` (Cap anchor calibration)
- `pex_v3/scripts/19_finetune_hybrid_mesh_smoke.py` (Mesh trainer)
- `pex_v3/scripts/20_r_alpha_calibrate_spef.py` (R α calibration)
- `pex_v3/scripts/23_r_per_net_calibrate_spef.py` (R per-net calibration)
- `pex_v3/scripts/25_verify_starrc_compat.py` (StarRC verification)

## What does NOT get merged (negative results, kept for documentation)

- `pex_v3/scripts/22_finetune_hybrid_perpair_smoke.py` — Strike #2 KILLED
- `pex_v3/scripts/28_finetune_mesh_with_cell.py` — Strike #7 negative
- `pex_v3/scripts/32_finetune_mesh_with_pincap.py` — Strike #8 negative
- `pex_v3/src/models/hybrid_v3_perpair.py` — never converged
- `pex_v3/src/data/per_pair_dataset.py` — only used by Strike #2

These remain in `pex_v3/` as historical / paper-evidence files.

## Cross-boundary edits to legacy main code

Two prior edits from earlier in this session (already documented):
1. `src/evaluation/evaluator.py:409` — env var `SPEF_DESIGN_FILTER` for SPEF write loop
2. `src/evaluation/evaluator.py:431` — env var `SPEF_INFER_BATCH` for batch size override

Both are minimal and backwards compatible (env vars default to no-filter / 256).

## Procedure (this merge)

```bash
# Inside /home/jslee/projects/PINNPEX
mkdir -p src/v3
ln -s ../../pex_v3/src/models   src/v3/models
ln -s ../../pex_v3/src/baselines src/v3/baselines
ln -s ../../pex_v3/src/data     src/v3/data
ln -s ../../pex_v3/src/trainers src/v3/trainers
ln -s ../../pex_v3/src/utils    src/v3/utils

# Add empty __init__.py for src.v3 sub-package
touch src/v3/__init__.py

# configs.config_v3 — symlink the file
ln -s ../pex_v3/configs/config_v3.py configs/config_v3.py
```

After this:
```python
from src.v3.models.hybrid_v3_mesh import HybridPexV3Mesh   # works
from configs import config_v3                                # works
```

## Verification (2026-05-03 ✅ ALL PASS)

After merge:
- ✅ `src.v3.*` imports OK (main folder integration)
- ✅ Legacy `src.*` + `configs.config` work alongside (no shadowing)
- ✅ pex_v3 scripts (old `from src.X` path) still work (backward compat)
- ✅ `HybridPexV3Mesh` forward + parameter_count match earlier runs (44,738 params)

## Critical post-merge code changes (pex_v3 boundary edits)

To enable 3-way import compatibility, the following pex_v3 paper-pillar
files were edited from absolute `from src.X` → relative `from .X` imports:

| File | Change |
|---|---|
| `pex_v3/src/models/hybrid_v3_mesh.py` | `from src.models.X` → `from .X` |
| `pex_v3/src/models/hybrid_v3.py` | `from src.models.X` → `from .X` |
| `pex_v3/src/trainers/finetune_hybrid_v3.py` | `from src.models.X` → `from ..models.X` |
| `pex_v3/src/baselines/xgboost_baseline.py` | `from src.{baselines,evaluation,utils}.X` → relative |
| `pex_v3/src/baselines/feature_dataset.py` | `from src.baselines.X` → `from .X` |

These changes are within the `pex_v3/` boundary (per CLAUDE.md). Non-paper
files (synthetic, transfer_canary, train_ssl_v3, etc.) NOT changed —
they only matter for pex_v3 scripts and not for main folder import path.

## Rollback

```bash
unlink src/v3/models src/v3/baselines src/v3/data src/v3/trainers src/v3/utils
rmdir src/v3
unlink configs/config_v3.py
```

## Memory entries to update post-merge

- `MEMORY.md`: add merge entry + new import paths
- `project_paper_narrative_3pronged.md`: note v3 modules now in main src

## Risk assessment

| Risk | Mitigation |
|---|---|
| Symlinks break on filesystem operations | Documented, reversible |
| Imports break for downstream | Test after merge (script in §Verification) |
| Future divergence between pex_v3/ and src.v3 | Single source = no divergence |
| pex_v3/ accidentally moved/renamed | Symlinks broken; rebuild from same script |
