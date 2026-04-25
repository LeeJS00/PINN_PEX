# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

PINN-PEX is a physics-informed neural field that predicts parasitic capacitance from routed layouts (DEF + tech LEF + layer stack) and emits SPEF. The golden oracle is StarRC; the network learns to mimic it cheaply. The pipeline is four stages: **build dataset → SSL pretrain (`DeepPEX_Model`) → active-learning finetune against StarRC → evaluate / write SPEF**.

## Environment

Shell is `tcsh`, Python is loaded through environment modules. Before running anything, in an interactive shell:

```tcsh
source tool.env   # module load python/3.11.9, starrc/2021.06, license/license
```

`tool.env` contains `module load ...` commands; it is not a shell-agnostic env file.

The active config is `configs/config.py` (imported as `configs.config`). Other configs (`config_intel22.py`, `config_asap7.py`, ...) are alternates — you swap them by editing the imports, there is no `--config` flag. Paths in `config.py` are host-specific (e.g. `PROCESSED_DIR=/data/PEX_SSL/data/processed/intel22`, `SPEF_DIR=/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22`) — verify they exist before blaming code.

## Commands

All commands run from the repo root.

**Build dataset** (all DEFs in `cfg.TRAIN_DEFS + cfg.TEST_DEFS`, skips designs already in the manifest):
```bash
python3 scripts/build_dataset_multi.py
```
Single design (what `_multi` shells out to):
```bash
python3 scripts/build_dataset.py --def_path <DEF> --out_dir <dir> --pt_out_dir <dir> --num_workers 16
```

**SSL pretrain** (writes `output_intel22/checkpoints/<cfg.RUN_NAME>/bem_ssl_ep*.pth`):
```bash
python3 src/trainers/train_ssl.py
```

**Active-learning finetune** (loads the latest `bem_ssl_ep*.pth` as basis, freezes encoder + norm, trains only `charge_basis_mlp / gnd_mlp / cpl_mlp`):
```bash
python3 run_active_learning.py --model_name <run_tag> --gpu <id> [--model_type DeepPEX|GNNCap]
```
Outputs land under `output_intel22/active_learning/<model_name>/` (`model_iter_*.pth`, `best_model.pth`, `al_training_log_*.csv`, `al_session_budget.csv`, `al_macro_runtime.csv`). Note: `--model_type GNNCap` imports `src.models.baselines`, which does not exist in this tree — only the `DeepPEX` path is runnable without adding that module.

**Evaluate** (reads `best_model.pth` from the AL output dir):
```bash
python3 src/evaluation/evaluator.py --model_name <run_tag> --gpu <id> [--spef_write]
```
`--spef_write` streams a predicted SPEF via `AutonomousGraphBuilder` + `SPEFWriter`.

**Compare predicted vs golden SPEF**:
```bash
python3 src/evaluation/compare_spef.py --golden <path.spef> --pred <path.spef> --out_dir <dir>
```

There are no tests, lint, or formatter configured. Reports are CSVs written next to outputs; `plot.py` generates a short-vs-long-net scatter from `spef_comparison_report.csv`.

## Architecture

### Data: cuboid tiles

`NetTiler` (in `src/preprocessing/tiling.py`) splits each net's routing geometry into fixed-size cuboids using `WINDOW_SIZE = (4.0, 4.0, 20.0)` μm and `TILING_OVERLAP = 0.5` μm. Each tile is saved as a gzipped pickle `<design>/<sample>.pkl.gz`. A global `dataset_manifest.csv` in `PROCESSED_DIR` indexes every tile with columns `sample_filename, net_name, design_name, split, tile_idx, ...`. `split` is assigned at build time: test designs (from `cfg.TEST_DEFS`) get `split=test`; train designs get a random 90/10 `train`/`valid` split per design.

Tensor shape fed to the model is `(N, 9)` per tile. The nine channels are:

| idx | meaning |
|-----|---------|
| 0–2 | `x_rel, y_rel, z_abs` |
| 3–5 | `w, h, d` (cuboid extents) |
| 6 | semantic type (1.0 = wire, 0.5 = pin) |
| 7 | logic flag (1.0 = target net, 0.0 = aggressor) |
| 8 | permittivity ε (from layer stack) |

Batches are padded to `cfg.NF_PAD_TO_CUBOIDS` cuboids (default 1024) with a `padding_mask`.

### Model: `DeepPEX_Model` (src/models/neural_field.py)

Two-stage:
1. **CuboidEncoder** — per-cuboid MLP; input scaling is non-trivial (xy/z divided by `SCALE_FACTOR`, w/h/d `log1p`-normalized, ε `log`-normalized with a clamp to 1.0 to avoid `log(0)` from padding).
2. **NeuralFluxRouter** (src/models/flux_head.py) — single unified router that replaces attention + physics head + cap head. It runs KCL, 1-hop context aggregation and sparse shielding/coupling (see `compute_sheilding.py`) inside one module.

The router has three trainable heads: `charge_basis_mlp`, `gnd_mlp`, `cpl_mlp`. `freeze_ssl_layers()` freezes the encoder and `flux_router.norm` and leaves those three MLPs trainable — this is the state used throughout active learning. SSL checkpoints sometimes have `_orig_mod.` prefixes from `torch.compile`; loaders strip them and tolerate shape mismatches (e.g. resized `cpl_mlp`) via `strict=False` filtering.

Layer stack info (thickness, z position, ε per metal/dielectric layer) is parsed once by `LayerInfoParser(cfg.LAYERS_INFO_PATH)` into a dict and wrapped in `BEOLMaterialStack` for permittivity lookups; both the dataset builder and the model instantiate it.

### Active learning loop (run_active_learning.py)

The loop is **net-centric**, not tile-centric:

1. `PhysicsSelector.evaluate_pool` scans up to `MAX_POOL_EVAL=5000` tiles and returns per-tile flux entropy.
2. Entropies are grouped by `(design_name, net_name)` with max reduction, and the top-`NETS_PER_ITER` *nets* are chosen.
3. **All tiles of each chosen net** are pulled from `pool_df` to reassemble full nets (never leave a net partially labeled — mixing tiles across splits was a past bug, see `prepare_net_centric_validation`).
4. `FullChipPEXOracle.generate_golden_spef` returns the precomputed StarRC SPEF for the design if present in `cfg.TRAIN_SPEFS`; otherwise it fills in a TCL template (`cfg.PEX_TEMPLATE_PATH`) and runs StarRC. It **never re-runs StarRC on tiles** — always on the full chip DEF.
5. The net's tiles + golden SPEF are added to `DesignLevelReplayBuffer`; `NetGroupedSampler` (src/data/samplers.py) ensures all tiles of a net stay in the same batch and drops "Mega-Nets" with > `max_tiles_per_batch` tiles to avoid OOM.
6. `NeuralFieldFinetuner.train_steps` runs `AL_TRAIN_STEPS_PER_ITER` (10000) steps against `val_loader` and saves `best_model.pth` when validation improves.
7. Loop stops on `AL_MIN_ENTROPY_THRESHOLD` (currently `-inf`, i.e. disabled) or on the net budget cap `AL_MAX_BUDGET_RATIO` × total nets.

The `USE_FAST_ENGINEERING_MODE = True` flag (hard-coded in `main`) builds a predefined train/valid cache once (`cache/predefined_{train,valid}_subset.csv`) and reuses it on subsequent runs — flip to `False` to use the live `prepare_net_centric_validation` path.

`AL_SAMPLING_METHOD` restricts which designs the AL pool draws from:
- `"Predefined"` — use `cfg.AL_PREDEFINED_DESIGNS` (default).
- `"SSL"` — auto-pick top-3 highest-entropy designs (skipping any design whose name contains `mpeg` — it's blacklisted for instability).
- `"Sorted"` — take the alphabetically first 3.

### Key supporting modules

- `src/preprocessing/{def_parser,lef_parser,cell_parser,layer_parser}.py` — streaming parsers for DEF / LEF / `layers.info`. `DefStreamParser` yields `(net_name, cuboids, segments)`.
- `src/data/tensorizer.py` — `FeatureTensorizer` turns parsed geometry into the (N, 9) tensor.
- `src/utils/spef_writer.py` — `AutonomousGraphBuilder` + `SPEFWriter` stream SPEF nets one at a time (used by the evaluator's `--spef_write` path).
- `src/utils/profiler.py` — `RuntimeProfiler` writes `*_macro_runtime.csv` timing breakdowns (`AL_Cycle`, `Eval_SPEF_Gen`, etc.).

### What lives where on disk

- `configs/` — config modules.
- `scripts/` — one-shot CLI scripts (dataset build, probes).
- `src/` — library code; nothing in `src/` is a top-level entrypoint except `train_ssl.py`, `evaluator.py`, `compare_spef.py`.
- `data/processed/intel22_pt/` — repo-local mirror; the live processed data is at `cfg.PROCESSED_DIR = /data/PEX_SSL/data/processed/intel22`.
- `golden_data/spef_data/intel22/` — golden SPEFs from StarRC. `config.py` currently points `SPEF_DIR` at a sibling project path (`/home/jslee/projects/PEX_SSL/...`), not this repo's copy.
- `output_intel22/` (created at runtime) — checkpoints under `checkpoints/<RUN_NAME>/`, AL artifacts under `active_learning/<model_name>/`.

## Conventions worth knowing

- Paths and run tags go through `cfg.RUN_NAME` (SSL basis dir) and `--model_name` (AL / eval output dir); they are independent — keep them consistent when chaining SSL → AL → eval.
- The codebase assumes a single GPU selected by `cfg.GPU_ID` (default 1) or `--gpu`. There is no DDP/multi-GPU support; `_orig_mod.` prefix stripping exists only because `torch.compile` is enabled in `run_active_learning.py`.
- Comments and user-facing strings mix English and Korean — this is intentional. Preserve the existing language when editing nearby text.
