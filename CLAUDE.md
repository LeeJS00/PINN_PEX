# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project purpose (2026-05-11, post-pivot; refinement sprint v3 lock 2026-05-18)

**PINN-PEX is a parasitic capacitance predictor for routed VLSI layouts.** Given
DEF + tech LEF + cell LEF + Liberty + layer.info as input, it produces a SPEF
file with predicted ground + coupling capacitances. The golden oracle is StarRC.

The **canonical model is `TreePEX/`: a 5-seed Tweedie XGBoost ensemble**, which
matches or beats every PINN paradigm tried in the historical tree on accuracy
AND wall-clock. Other tracks (PINN mesh v3 / substrate physics v4 / auto-4 % v5
/ per-pair v7 / hybrid analytic v8) are archived under `archive/` (gitignored).

### Numbers (post refinement sprint v3, 5-seed ensemble, MAPE_med)

**⚡ Warm path** (features pre-computed CSV → inference + SPEF + compare; fanout = gold-SPEF-derived, label-leak; deployment scenario):

| PDK | Design | n_nets | MAPE_tot | R²_tot | Wall (warm e2e) |
|---|---|---:|---:|---:|---:|
| intel22 22 nm | tv80s_f3 | 3,169 | **4.95 %** | **0.9936** | 11.27 s |
| intel22 22 nm | nova_f3 | 92,425 | **5.34 %** | **0.9914** | 82.10 s |
| ASAP7 7 nm | tv80s_x1 | 3,328 | **6.72 %** | **0.9854** | 9.68 s |
| ASAP7 7 nm | nova_x1 | 125,499 | N/A (no training entry) | — | — |

**❄️ Cold path** (DEF → parse → features → fanout XGB proxy → inference; StarRC-equivalent from-scratch):

| PDK | Design | n_nets | MAPE_tot | R²_tot | Cold wall | vs StarRC FS |
|---|---|---:|---:|---:|---:|---:|
| intel22 22 nm | tv80s_f3 | 3,280 | **4.95 %** | **0.9933** | 68.31 s | 4.1× |
| intel22 22 nm | nova_f3 | 113,812 | **5.47 %** | **0.9895** | 4767 s / 80 min | 1.50× |
| ASAP7 7 nm | tv80s_x1 | 3,328 | **7.00 %** | **0.9827** | ~70 s | 3.9× |
| ASAP7 7 nm | nova_x1 | 125,499 | **7.93 %** | **0.9699** | ~3249 s / 54 min | 2.2× |

Warm vs cold Δ: intel22 tv80s +0.00 pp (proxy near-perfect), intel22 nova +0.14 pp,
ASAP7 tv80s +0.28 pp — fanout proxy OOS quality 가 cold/warm gap을 결정.

Warm vs cold: fanout source 차이 (label leak vs proxy) + feature 추출 wall 차이 (0 vs 1500-3000s).
두 path는 **절대 같은 표에 섞지 말 것** — see `~/.claude/.../memory/feedback_warm_cold_path_separation.md`.

PINN v12 mesh reference (intel22 tv80s/nova): 8.23 % / 7.88 % MAPE at 10.46/91.12 s wall.

See `TreePEX/paper_benchmark/PAPER_TABLES_v2.md` (rev 2) for full breakdown.

## Environment

Shell is `tcsh`, Python is loaded through environment modules:

```tcsh
source tool.env   # module load python/3.11.9, starrc/2021.06, license/license
```

## Canonical pipeline (TreePEX)

### Run inference end-to-end

```bash
# Both test designs, full pipeline (parse → features → predict → SPEF):
python3 TreePEX/scripts/pex_tool.py --all

# One design:
python3 TreePEX/scripts/pex_tool.py --design intel22_tv80s_f3
```

### Run paper benchmark (parse + predict + SPEF + compare, with stage timings)

```bash
python3 TreePEX/paper_benchmark/scripts/bench_e2e.py --skip-pinn   # XGBoost only
python3 TreePEX/paper_benchmark/scripts/bench_pinn.py              # PINN reference (from archive)
```

Outputs land in `TreePEX/paper_benchmark/{results,outputs}/`.

### Train from scratch

```bash
# Build cuboid-tile cache (one-time, offline):
python3 scripts/build_dataset_multi.py

# Train 5-seed XGBoost ensemble (≈10 min on CPU):
python3 TreePEX/scripts/01_train_save_models.py
```

## Architecture (TreePEX frontier)

### Features (67-D per net)

- **41-D base** (`src/baselines/features.py::NetFeatureVector`): wire length,
  metal area, layer histogram, fanout, aggressor counts, overlap stats, spacing
  distribution, VSS shielding, dielectric ε, density per metal stack, compact
  Sakurai-Tamaru-style analytic gnd/cpl estimates.
- **26-D V4 H3 features** (`archive/pex_v4/scripts/29_extract_new_features.py`,
  cached in `pex_v4/results/new_features_with_ids.csv`): top-3 aggressor pair
  geometry (score, overlap, min-xy-dist, mean-dz, agg size, layer-diff flag) +
  aggressor counts within radii.

### Model

- 10 XGBoost regressors: 5 seeds × {gnd, cpl}. Config: depth=8, n_est=500, lr=0.05,
  `reg:tweedie` (variance_power=1.5), `subsample=0.8`, `colsample_bytree=0.8`,
  `early_stopping_rounds=100`. Each weight file ~12 MB JSON.
- Inference: 5-seed prediction-mean (NOT mean of MAPE). CPU-only.
- **Trained with L6 σ=0.2 multiplicative noise on `fanout` column** (regularizes
  to cold-inference fanout proxy distribution, ESSENTIAL — drop costs +0.6 pp).
- ASAP7 only: **L11 large-net specialist** (depth=8, n_est=500 — matched to
  canonical hyperparams post-2026-05-18 sprint; was d9 n750, simplified after
  ablation showed identical or marginal-improve performance with 3× smaller
  weights). Switch: `total_wire_length_um > 15.35 μm` routes 6-9 % of nets.
- ~~L5 3-stage isotonic calibration~~ **DROPPED 2026-05-18** (both PDKs;
  ablation: intel22 −0.10/−0.14 pp IMPROVE; ASAP7 net 0). `calibration.json`
  archived; `pex_cold.py` guards skip automatically when file absent.

### SPEF write

- `src/utils/spef_writer.py::SPEFWriter` + `AutonomousGraphBuilder` stream SPEF
  nets one at a time. Round-trip lossless (max abs err 5e-6 fF).
- IEEE 1481-1999 compatible.

## Key supporting modules

- `src/preprocessing/{def_parser,lef_parser,cell_parser,layer_parser}.py` —
  streaming parsers for DEF / LEF / `layers.info`. `DefStreamParser` yields
  `(net_name, cuboids, segments)`.
- `src/data/tensorizer.py` — `FeatureTensorizer` turns parsed geometry into the
  per-cuboid (N, 9) tensor (only used when training the archived PINN models).
- `src/utils/spef_writer.py` — `AutonomousGraphBuilder` + `SPEFWriter`.
- `src/utils/spef_parser.py` — golden SPEF parser used by `04_compare_golden.py`.
- `src/baselines/features.py` — `NetFeatureVector` extractor.

## What lives where

| Path | Role |
|---|---|
| `TreePEX/` | **Canonical XGBoost frontier + paper benchmark.** |
| `src/` | Core library (parsers, SPEF writer, feature extractor, materials). |
| `scripts/` | One-shot CLI scripts (dataset build, probes). |
| `configs/config.py` | Path constants (DEF / LEF / layer.info / SPEF dir). |
| `tool/pdk/22nm/` | PDK assets (tech LEF, cell LEF, Liberty, layer.info). |
| `golden_data/spef_data/intel22/` | StarRC golden SPEFs for 13 designs. |
| `archive/` | Old paradigm trees (gitignored). pex_v3 PINN, pex_v4 substrate, pex_v5 auto, pex_v7 per-pair, pex_v8 hybrid analytic. All NEG vs TreePEX. |

`TreePEX/inputs/` contains pre-extracted V3+V4 feature CSV pointers; live data
lives at `/data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv`.

## Conventions worth knowing

- The codebase has no tests, lint, or formatter. Reports are CSVs written next
  to outputs.
- Paths in `configs/config.py` are host-specific — verify they exist before
  blaming code.
- Comments and user-facing strings mix English and Korean — this is intentional.
- StarRC uses the SAME inputs we do (DEF + LEF + Liberty + layer.info). The
  remaining ~1 pp gap from TreePEX to StarRC is representation-bound, not
  input-bound. **Never claim "GDSII is needed" — it isn't.** (See
  `TreePEX/REPORT.md` for the correction history.)
- 4-way oracle blend bound = 4.74 % on tv80s. This is the hand-feature ceiling.
  Closing it requires new input modality (voxel CNN over rasterized routing)
  or a fundamentally different paradigm than scalar features + trees.

## Archived (gitignored) — for post-mortem only

Trees under `archive/` represent paradigms that failed to beat TreePEX:

| Tree | Best test MAPE | Why archived |
|---|---:|---|
| `archive/pex_v3/` (PINN mesh + curriculum) | 6.26 % tv80s | Beaten by TreePEX ensemble (4.98 %) at 1/20 wall. |
| `archive/pex_v4/` (substrate physics, auto-4 %) | 5.55 % tv80s | Phase B1 K1 gate failed (Sakurai-Tamaru over-est upper-layer 1.5-2.8×). |
| `archive/pex_v5/` (auto-4 % sprint) | 5.09 % tv80s | Plateau; TreePEX ensemble passed it. |
| `archive/pex_v7/` (per-pair regression N5) | 15.7 % tv80s | Cuboid tile resolution 4×4×20 μm too coarse; pair R² ≤ 0.17. |
| `archive/pex_v8/` (hybrid analytic + residual) | 55.5 % tv80s | analytic_cpl over-estimate exploded the residual loss to cpl 100 % MAPE. |

Memory entries at `~/.claude/projects/-home-jslee-projects-PINNPEX/memory/`
record the specific failure modes for each track. Read those before proposing
to revive any archived path.
