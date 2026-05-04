# pex_v3/scripts — entrypoints

Numbered scripts run in order. Each is idempotent unless noted.

| # | Script | Phase | Cost | Idempotent |
|--:|---|---|---|:--:|
| 01 | `01_resplit_manifest.py` | Phase 0 H1 | ~10 min CPU | ✅ |
| 02 | `02_rebuild_dataset_h3.py` | Phase 0 H3 | 2-4 GPU-day, 1.2TB | ⚠️ stub gated by `--confirm` |
| 03 | `03_train_ssl_v3.py` | Phase 0 M5 | ~11 GPU-h | ✅ |
| 04 | `04_run_baselines.py` | Phase 0.5 | varies | TBD (Phase 0.5 scaffold) |
| 05 | `05_5seed_runner.py` | All phases | 5× single AL run | TBD |

## How to run

All commands from repo root (`PINNPEX/`):

```bash
source tool.env

# Phase 0 immediate (manifest only, no rebuild)
python3 pex_v3/scripts/01_resplit_manifest.py

# Phase 0 invariant tests
python3 -m pytest pex_v3/tests/

# Phase 0 H3 dry-run (will print plan, won't write data)
python3 pex_v3/scripts/02_rebuild_dataset_h3.py
# After user approval:
python3 pex_v3/scripts/02_rebuild_dataset_h3.py --confirm
```

## Naming convention

`<NN>_<verb>_<noun>.py`:
- `NN` is order of execution within a phase
- `verb` is action (resplit, rebuild, train, eval, ablate)
- `noun` is target (manifest, dataset_h3, ssl_v3)

Each script's first line should be a one-paragraph summary of (a) what it
does, (b) what files it produces, (c) cost estimate, (d) idempotency.
