# pex_v3 — PINN-PEX Strategy v3 workspace

This subfolder is the workspace for **PINN-PEX**: a hybrid physics-informed
neural net + classical-anchored calibration for full-chip parasitic
extraction (SPEF) without commercial PEX licenses.

_Status (2026-05-03 evening): **5/5 paper-pillar LOCKED + Path-2 fast deployment + main folder merge**._

---

## TL;DR — paper-grade results (single chip canonical reference: tv80s)

| Pillar | Result |
|---|---|
| **PINN per-net total MAPE** (Mesh-curriculum 5-seed) | best **6.26% ± 0.108pp** / last 8.27% / ensemble 7.89% (44K params, cross-design OOD) |
| **Hero SPEF cap** (XGB anchor, 5-seed) | mean **10.95% ± 0.047pp**, median **5.77%**, R²=0.983 |
| **Hero SPEF R** (sister NNLS+LightGBM per-net) | mean **2.21%**, median **1.40%**, R²=0.999 |
| **Path-1 wall-clock** (legacy 1M PINN + post-process) | tv80s 14.4 min; nova ~14-19 h cold-start |
| **🚀 Path-2 wall-clock** (fast deterministic, no PINN) | **tv80s 68.9 s — 12.5× faster**, median MAPE essentially identical (5.78% vs 5.77%) |
| **License** | None (StarRC ~$50K-100K/seat/yr avoided) |

---

## 5 paper contributions

1. **Cuboid set encoder + bounded multiplicative residual + clamp curriculum** — physics-informed neural arch (44K params), beats B4 V3 log-GBDT (6.59%) with 2.3× fewer params.
2. **Hybrid per-net calibration** — XGB anchor (cap) + sister NNLS+LightGBM (R) corrects tile→net aggregation drift. Long-net Q4 cap MAPE 71% → 9% (8× improvement).
3. **Full-chip SPEF E2E pipeline** — DEF + LEF in → calibrated SPEF out, structurally StarRC-compatible (3 fixable cosmetic items).
4. **🚀 Fast deterministic deployment path (Path-2)** — analytic + geometric SPEF generator with 12.5× speedup vs Path-1, median MAPE essentially unchanged. GPU-optional.
5. **Honest negative methodology findings** — capacity scaling, per-pair head, 4 cell-internal feature additions, GDSII-only R sub-1% — all paper-grade.

---

## Folder layout

```
pex_v3/
├── README.md                # this file
├── CLAUDE.md                # subfolder boundary rule
├── PHASE_STATUS.md          # live tracker (2026-05-03 latest)
├── SESSION_HANDOFF.md       # next-session context
├── configs/
│   └── config_v3.py         # v3 manifest paths + cfg snapshots
├── src/                     # 5/5 pillar modules (mirrored to ../src/v3/)
│   ├── models/              # cuboid_set_encoder, hybrid_v3_mesh, analytic_base_v3, residual_head_v3
│   ├── baselines/           # calibration_v3 (NNLS), xgboost_baseline (B1), feature_dataset, features
│   ├── data/                # cuboid_set_dataset (per-net loader), per_pair_dataset (Strike #2 negative)
│   ├── trainers/            # finetune_hybrid_v3 (Mesh + curriculum)
│   ├── utils/               # seeds, manifest_hash
│   ├── synthetic/           # Stage 1-2 (legacy, Strike DROPPED per K3 canary)
│   └── preprocessing/       # essentially empty (uses legacy ../src/preprocessing)
├── scripts/                 # numbered entrypoints — 01-34, paper-pillar = 14, 16, 19, 20, 23, 25
│   ├── 01_resplit_manifest.py            # H1 hash split
│   ├── 02_rebuild_dataset_h3.py          # H3 14×14μm rebuild
│   ├── 04_build_feature_dataset.py       # per-net features
│   ├── 14_option_f_5seed.py              # Option F MLP 5-seed
│   ├── 16_xgb_calibrate_spef.py          # 🚀 Cap anchor calibration (PILLAR)
│   ├── 19_finetune_hybrid_mesh_smoke.py  # 🚀 Mesh PINN trainer (PILLAR)
│   ├── 20_r_alpha_calibrate_spef.py      # R global α calibration
│   ├── 23_r_per_net_calibrate_spef.py    # 🚀 R per-net (sister) (PILLAR)
│   ├── 24_mesh_5seed_ensemble_eval.py    # 5-seed ensemble inference
│   └── 25_verify_starrc_compat.py        # StarRC structural compat
├── tests/                   # 170+ pytest invariants (split, determinism, priority)
├── docs/                    # phase plans + cross-boundary docs
│   ├── PHASE0_PLAN.md
│   ├── PHASE1_HYBRID_ARCH_SPEC.md       # A6 mesh_v3 spec
│   ├── CROSS_BOUNDARY_v3_merge_to_main.md  # 2026-05-03 merge plan
│   └── ...
├── paper/                   # paper-ready artifacts
│   ├── METHOD.md            # 10-section method (paper-ready)
│   ├── OUTLINE.md           # ICCAD/DATE structure
│   ├── RESULTS_CONSOLIDATED.md  # 5/5 pillar leaderboards + tables
│   ├── HYBRID_CALIBRATION_FINDING.md
│   ├── SPEF_COMPATIBILITY_REPORT.md
│   └── CGND_ERROR_ANALYSIS.md
└── output/                  # all run artifacts (5-seed dirs, hero SPEFs, calibration JSONs)
```

---

## Main folder integration (2026-05-03 merge)

`pex_v3/src/{models,baselines,data,trainers,utils}` are symlinked into
`../src/v3/` for main-folder import:

```python
# 3-way compatible imports:
from src.v3.models.hybrid_v3_mesh import HybridPexV3Mesh        # main path (NEW)
from src.preprocessing.def_parser import DefStreamParser         # legacy main (unchanged)
from src.models.hybrid_v3_mesh import HybridPexV3Mesh            # pex_v3 scripts (backward compat)

from configs import config_v3                                     # symlinked
import configs.config as cfg                                      # legacy (unchanged)
```

See `pex_v3/docs/CROSS_BOUNDARY_v3_merge_to_main.md` for merge details + rollback.

---

## Quick commands

All run from repo root (`PINNPEX/`):

```bash
source tool.env  # Python 3.11.9, StarRC

# Tests (170+ invariants)
python3 -m pytest pex_v3/tests/

# Phase 0 (one-time, ~12-24 h on first chip)
python3 pex_v3/scripts/01_resplit_manifest.py
python3 pex_v3/scripts/02_rebuild_dataset_h3.py
python3 pex_v3/scripts/04_build_feature_dataset.py

# Train PINN (5-seed × 200 epoch + curriculum, ~25 min wall on 5 GPUs parallel)
python3 pex_v3/scripts/19_finetune_hybrid_mesh_smoke.py --seed 42

# Inference + SPEF write (Path-1)
SPEF_DESIGN_FILTER=intel22_tv80s_f3 \
python3 src/evaluation/evaluator.py --model_name m6_v10b_baseline_seed0 --gpu 0 --spef_write

# Hybrid calibration post-process
python3 pex_v3/scripts/16_xgb_calibrate_spef.py \
    --in-spef <autonomous.spef> --xgb-csv <B1_predictions.csv> \
    --design intel22_tv80s_f3 --out-spef <cap_calibrated.spef>

python3 pex_v3/scripts/23_r_per_net_calibrate_spef.py \
    --in-spef <cap_calibrated.spef> \
    --r-pred-parquet <sister_v6_s3.parquet> \
    --r-pred-col R_pred_v6_s3 --out-spef <hero.spef>

# Verify StarRC structural compatibility
python3 pex_v3/scripts/25_verify_starrc_compat.py \
    --golden <golden.spef> --pred <hero.spef> --out-md compat_report.md
```

---

## Known limits / negative findings

- **C_gnd 19% gnd ceiling** — architecture-bound (4 cell-internal feature additions all hurt; cuboid set encoder already saturates the spatial signal).
- **<1% mean R MAPE** — DEF/LEF info-impossible (sister-confirmed); requires GDSII transistor-internal routing parser.
- **Per-pair coupling head (Strike #2)** — uniform analytic baseline + sample-aggregator high-variance; killed at epoch 53. Per-pair-specific analytic prior required for redo.
- **Cold-start runtime** — 18 min (tv80s) to ~14-19 h (nova); ~3-10× slower than pattern-match StarRC. Bottleneck: PINN inference O(N²) memory in flux_router prevents larger batch.
- **Path-1 vs Path-2 tradeoff**: Path-1 has 1.7pp tighter mean MAPE; Path-2 has 12.5× speedup with median MAPE essentially unchanged.

See `paper/METHOD.md` §7 for full negative findings + diagnostics.
