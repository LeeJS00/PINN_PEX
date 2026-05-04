# Joint-Pareto Baseline — Path-2 v3

_Frozen 2026-05-03 late. Any new variant must be measured against this._

## Code reference

- `pex_v3/src/utils/fast_spef_engine.py` (engine)
- `pex_v3/scripts/40_fast_autonomous_spef.py` (driver)
- `pex_v3/scripts/41_nova_post_calibrate.sh` (post-process driver)

## Configuration

```python
fast_spef_engine.write_fast_autonomous_spef(
    design_name="intel22_tv80s_f3",
    topology_dir=Path("/data/PINNPEX/data/processed_v3/intel22/intel22_tv80s_f3/topology"),
    layer_info=LayerInfoParser(cfg.LAYERS_INFO_PATH).parse(),
    tech_lef=LefParser(cfg.TECH_LEF_PATH).parse(),
    out_spef_path=...,
    max_dist_um=5.0,
    top_k=20,
    n_workers=1,  # tv80s small enough for serial
)
```

Placeholder analytic estimate:
```
c_gnd = Σ_segs(length × width × ε_layer × 0.22)         # fF
c_cpl = c_gnd × 1.3                                      # fF
```

XGB anchor calibration: `pex_v3/scripts/16_xgb_calibrate_spef.py` per seed.
Sister R: `pex_v3/scripts/23_r_per_net_calibrate_spef.py` per parquet `R_pred_v6_s3`.

## Frozen numbers (5-seed tv80s, B1 XGB seeds 0..4)

| Axis | Value |
|---|---:|
| Wall-clock | 68.9 s |
| Total wall (incl. post-process) | ~71 s |
| Total cap MAPE mean | 7.035 ± 0.045 pp |
| Total cap MAPE median | 5.441 ± 0.052 pp |
| Total cap MAPE p95 | 18.54 ± 0.35 pp |
| R²(C) | 0.993 |
| RMSE C | 0.181 fF |
| Total cap balance | 1.00× |
| **gnd MAPE mean (matched)** | **27.37 %** |
| **gnd MAPE median (matched)** | **20.00 %** |
| **gnd MAPE mean (unmatched)** | **21.50 %** |
| **cpl MAPE mean (matched)** | **18.78 %** |
| **cpl MAPE median (matched)** | **14.20 %** |
| **cpl MAPE mean (unmatched)** | **27.29 %** |
| R MAPE mean | 2.21 % |
| R²(R) | 0.9991 |

## Per-stage runtime breakdown

| Stage | Wall-clock | % of total |
|---|---:|---:|
| Topology load (3,380 .pkl.gz, serial) | 12.8 s | 18.6 % |
| Global segment KD-tree build | 0.9 s | 1.3 % |
| Per-net assembly + write | 52.4 s | 76.0 % |
| XGB cap calibration | < 1 s | < 1.5 % |
| Sister R per-net rescale | < 1 s | < 1.5 % |
| **Total** | **68.9 s** | 100 % |

## Limitations to address

1. **Per-channel ceiling at XGB level** (matched nets) — gnd 27 % / cpl 19 %
   are XGB's per-net prediction errors. Spatial allocator does not help
   per-net totals; it might help per-segment / per-pair distribution.
2. **Unmatched 211 / 3,380 nets (6.2 %)** — not in XGB CSV. v3 placeholder
   gives reasonable totals (mean 11.87 % MAPE) but per-channel split (gnd
   vs cpl) uses fixed 1.3 ratio.
3. **Geometric c_cpl uses midpoint distance** — ignores 3D overlap area,
   layer-aware coupling physics (lateral vs vertical), shielding effects.
4. **Per-segment c_gnd by length only** — ignores per-cuboid layer ε, top
   vs bottom plate cap, fringe contributions.

## Reproduction (single command)

```bash
cd /home/jslee/projects/PINNPEX
mkdir -p pex_v3/output/spef_fast_repro

# 1. Generate fast autonomous SPEF
python3 pex_v3/scripts/40_fast_autonomous_spef.py \
    --design intel22_tv80s_f3 \
    --out-dir pex_v3/output/spef_fast_repro

# 2. Apply XGB anchor (seed 0)
python3 pex_v3/scripts/16_xgb_calibrate_spef.py \
    --in-spef pex_v3/output/spef_fast_repro/intel22_tv80s_f3_autonomous_fast.spef \
    --xgb-csv pex_v3/output/baselines/B1_xgboost_real/seed0/eval_predictions_test.csv \
    --design intel22_tv80s_f3 \
    --out-spef pex_v3/output/spef_fast_repro/xgb.spef

# 3. Apply sister R per-net rescale
python3 pex_v3/scripts/23_r_per_net_calibrate_spef.py \
    --in-spef pex_v3/output/spef_fast_repro/xgb.spef \
    --out-spef pex_v3/output/spef_fast_repro/HERO.spef \
    --r-pred-parquet experiments/cross_design_tv80s_2026_05_02/r_analytic_v3/outputs/test_predictions_v6_s3.parquet \
    --r-pred-col R_pred_v6_s3

# 4. Compare vs golden
python3 src/evaluation/compare_spef.py \
    --golden /home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22/intel22_tv80s_f3_starrc.spef \
    --pred pex_v3/output/spef_fast_repro/HERO.spef \
    --out_dir pex_v3/output/spef_fast_repro/compare
```

Expected output: `Total Capacitance MAPE: ~7.04 % | R^2: 0.9933`.
