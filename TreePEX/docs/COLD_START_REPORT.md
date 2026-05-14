# TreePEX cold-start report

> Treat every test design as a brand-new circuit. Run the full
> DEF/LEF/layer.info → SPEF pipeline with **only trained model weights** —
> no access to pre-built feature CSVs, no per-test-design fitting. Compare
> all top-3 non-TreePEX baselines under identical cold-start conditions.

Date: 2026-05-13
Author: TreePEX cold-start sprint

---

## 1. Scope

* **Test designs** (held out from every training pipeline):
  * `intel22_tv80s_f3` — 3,380 signal nets, 147,156 cuboids
  * `intel22_nova_f3` — 118,959 signal nets, 5,340,138 cuboids
* **Training designs** (9, used to fit models + offline proxies, never test):
  `aes_cipher_top`, `gcd`, `ibex_core`, `ldpc_decoder_802_3an`, `mc_top`,
  `spi_top`, `usbf_top`, `vga_enh_top`, `wb_conmax_top`.
* **Golden oracle**: StarRC SPEFs in `golden_data/spef_data/intel22/*_starrc.spef`.
* **Models evaluated** (top-3 non-TreePEX + TreePEX reference):

  | Tag | Architecture | Trained on | Source |
  |---|---|---|---|
  | `treepex` | 5-seed Tweedie XGBoost, 67-D V3+V4 features | warm V3+V4 CSV | `TreePEX/models/tweedie_*.json` |
  | `b1_xgb` | 5-seed log-target XGBoost, 41-D V3 features | warm V3 CSV | `archive/pex_v3/output/baselines/B1_xgboost_real/seed{0..4}/model_{gnd,cpl}.json` |
  | `catboost` | 5-seed CatBoost-Tweedie, 67-D V3+V4 features | warm V3+V4 CSV (fit in this sprint) | `TreePEX/models/catboost_*.cbm` |
  | `mesh_pinn` | 5-seed HybridPexV3Mesh PINN + per-layer calibration | warm V3 + cuboid tile cache | `archive/pex_v3/output/phase1_mesh_5seed/seed{0..4}/model.pt` |

## 2. Cold-start pipeline

All four models share the same cold-start feature pipeline. The feature
parquet is produced once per test design by `pex_cold.py` and reused.

```
DEF + tech LEF + cell LEF + layer.info
        │
        ▼
[1] LayerInfo / LEF / CellLEF parse                  (PDK parse, < 1 s)
        │
        ▼
[2] DefStreamParser → per-net cuboid arrays
    (signal nets + PIN/INST_PORT pseudo-nets +       (DEF parse)
     VSS/VDD; preserves training-time all_owner semantics)
        │
        ▼
[3] SpatialGrid (4×4 μm xy bbox bucketing)
    + V3 41-D NetFeatureVector / net  ← parallel (16 workers, fork-Pool)
        │                              ───────────────────────────────
        │   • coupling-edge enumeration via spatial query + cutoff 4 μm
        │   • VSS shielding, eps stats, compact Sakurai-Tamaru analytic
        │     priors, layer histogram, fanout (XGB-Tweedie proxy*)
        ▼
[4] V4 H3 26-D top-K aggressor features              ← parallel
    aggregated from cuboid tile pkl.gz cache         (per-net concat)
        │                                            (matches training
        │                                             distribution where
        │                                             tile overlap inflates
        │                                             target_n_cuboids /
        │                                             top-K scores by 4-100×)
        ▼
[5] feature parquet cached @ outputs/cold_reports/{design}_cold_features.parquet
        │
        ▼ (per model — same parquet shared by 4 models)
        │
[6] Model-specific inference (treepex / b1_xgb / catboost / mesh_pinn)
        │
        ▼
[7] SPEF write (IEEE 1481-1999) + golden compare (per-net MAPE, R², chip cap)
```

### 2.1 Key engineering decisions

1. **`fanout` is SPEF-labeled at training time** — XGB feature_importance 0.81 on cpl.
   Cannot be derived from DEF. Replaced with an **offline-fit XGB-Tweedie regressor**
   on 8 DEF-only structural features, trained on the 9 training designs:
   ```
   feats = [n_aggressor_nets, n_cuboids, n_edges_lt_1um, n_edges_1_to_3um,
            broadside_overlap_total_um2, lateral_overlap_total_um2,
            total_metal_area_um2, bbox_xy_um2]
   ```
   OOS fanout MAPE_med: tv80s 12.4 %, nova 63.6 % (vs Ridge baseline 31 % / not measured).
   Lifting fanout proxy: tv80s tot 5.91 % → 5.10 % (−0.81 pp), nova 5.89 % → 5.54 %.

2. **V4 H3 features must come from the tile cache** — they were trained on
   tile-overlap-aggregated cuboids where the same physical cuboid appears in
   ~4 tile windows (4 μm window, 1 μm stride). Computing them from raw DEF gives
   target_n_cuboids 4× smaller and top1_score ~100× smaller. Mismatch destroyed
   cpl MAPE (167 % in early draft). Tile cache is treated as a pre-built raw-geometry
   asset; for a brand-new design it would be produced once via `build_dataset.py`
   (measured: tv80s 120 s with 16 workers; nova ~60-90 min projected).

3. **V3 must include PIN/INST_PORT pseudo-nets in `all_owner`** to match training-time
   `n_aggressor_nets` (training counted them as separate aggressor identities;
   without them the count was 1/3 of warm).

4. **`compact_*_estimate_fF` analytic priors need per-layer calibration for mesh-PINN**.
   The PINN was trained on calibrated priors so its bounded multiplier output ≈ 1.0.
   Calibration constants pre-fit on 9 training designs (per-layer median ratios,
   ~10 scalars). Without it, gnd MAPE = 55 %; with it, gnd MAPE = 18-19 %.

5. **All shared stages (PDK, DEF, V3, V4) run once and are reused across all 4 models**
   via the cached parquet at `outputs/cold_reports/{design}_cold_features.parquet`.
   Only per-model inference + SPEF write differ between models.

### 2.2 Code surface added in this sprint

| Path | Purpose |
|---|---|
| `TreePEX/scripts/pex_cold.py` | Cold-start orchestrator (DEF → V3 → V4 → cache parquet → predict → SPEF). XGB-Tweedie fanout proxy + parquet save. |
| `TreePEX/scripts/pex_cold_predict.py` | Unified prediction entry. `--model {treepex,b1_xgb,catboost}`. Loads parquet + applies model. |
| `TreePEX/scripts/pex_cold_predict_mesh.py` | mesh-PINN cold-start (loads HybridPexV3Mesh, applies per-layer analytic calibration). |
| `TreePEX/scripts/_fit_fanout_proxy.py` | Pre-fit XGB-Tweedie / XGB-log / LGBM-Tweedie fanout proxy + benchmark. |
| `TreePEX/scripts/_fit_mesh_calibration.py` | Pre-fit per-layer median-ratio calibration on training designs. |
| `TreePEX/scripts/_train_catboost.py` | Train 5-seed CatBoost-Tweedie on warm V3+V4 features. |
| `TreePEX/scripts/summarize_cold_results.py` | Aggregate all per-(design, model) summaries into a Markdown table with full timing. |
| `TreePEX/models/fanout_proxy_xgb_tweedie.json` | Fanout proxy weights. |
| `TreePEX/models/fanout_proxy_meta.json` | Proxy metadata (input cols, kind). |
| `TreePEX/models/mesh_per_layer_calibration.json` | Per-layer compact-prior calibration scalars. |
| `TreePEX/models/catboost_{gnd,cpl}_seed{42,0,1,2,3}.cbm` | CatBoost-Tweedie 5-seed weights. |

---

## 3. Results

### 3.1 Accuracy + chip-level capacitance

| Model | Design | n | MAPE_tot | gnd | cpl | R²_tot | pred chip total fF | gold chip total fF |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **treepex** | tv80s | 3,280 | **5.105 %** | 17.63 % | 13.88 % | 0.9919 | 4,522 | 4,455 |
| **treepex** | nova | 113,812 | **5.538 %** | 15.85 % | 15.94 % | 0.9867 | 210,863 | 205,550 |
| b1_xgb | tv80s | 3,280 | 5.298 % | 19.50 % | 14.76 % | 0.9925 | 4,518 | 4,455 |
| b1_xgb | nova | 113,812 | 5.844 % | 18.42 % | 17.32 % | 0.9842 | 210,472 | 205,550 |
| catboost | tv80s | 3,280 | 5.654 % | 19.44 % | 14.54 % | 0.9930 | 4,493 | 4,455 |
| catboost | nova | 113,812 | 6.101 % | 17.68 % | 16.95 % | 0.9866 | 212,319 | 205,550 |
| mesh_pinn | tv80s | 3,280 | 7.740 % | 18.60 % | 15.34 % | 0.9919 | 4,320 | 4,455 |
| mesh_pinn | nova | 113,812 | 7.461 % | 19.05 % | 16.48 % | 0.9882 | 201,699 | 205,550 |

### 3.2 Full cold-start pipeline time (seconds)

Shared stages (`pdk` + `def` + `v3 feat` + `v4 H3 feat`) are per design,
produced once by `pex_cold.py`. Per-model stages (`infer` + `spef write`)
add to that.

| Model | Design | pdk | def | v3 feat | v4 H3 feat | infer | spef | **TOTAL** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| treepex | tv80s | 0.77 | 3.96 | 69.79 | 87.65 | 7.17 | 0.13 | **169.47** |
| treepex | nova | 0.39 | 93.67 | 5,607.13 | 2,348.97 | 5.28 | 3.72 | **8,059.16** (≈ 2 h 14 m) |
| b1_xgb | tv80s | 0.77 | 3.96 | 69.79 | 87.65 | 8.26 | 0.13 | **170.56** |
| b1_xgb | nova | 0.39 | 93.67 | 5,607.13 | 2,348.97 | 4.42 | 3.70 | **8,058.28** |
| catboost | tv80s | 0.77 | 3.96 | 69.79 | 87.65 | 1.64 | 0.14 | **163.95** |
| catboost | nova | 0.39 | 93.67 | 5,607.13 | 2,348.97 | 0.98 | 3.79 | **8,054.93** (fastest model) |
| mesh_pinn | tv80s | 0.77 | 3.96 | 69.79 | 87.65 | 187.10 (CPU) | 0.12 | **349.42** |
| mesh_pinn | nova | 0.39 | 93.67 | 5,607.13 | 2,348.97 | 2,273.46 (CPU) | 3.86 | **10,328.12** (≈ 2 h 52 m) |

(Run host: gpu-8, 16 worker fork-Pool. mesh-PINN inference on CPU; GPU is
expected to drop that 10-20×. SPEF write is single-threaded.)

### 3.3 Cold vs warm-start gap

| Model | tv80s cold | tv80s warm | Δ | nova cold | nova warm | Δ |
|---|---:|---:|---:|---:|---:|---:|
| TreePEX | 5.10 % | 4.98 % | **+0.12 pp** | 5.54 % | 5.28 % | **+0.26 pp** |
| B1 XGBoost | 5.30 % | 5.31 % | **−0.01 pp** | 5.84 % | 5.86 % | **−0.02 pp** |
| CatBoost (this sprint, no warm baseline) | 5.65 % | — | — | 6.10 % | — | — |
| Mesh-PINN | 7.74 % | 6.26 % (best-step) | +1.48 pp | 7.46 % | — | — |

Warm-start references: TreePEX (canonical, `TreePEX/REPORT.md`), B1 XGBoost
(`archive/pex_v3/output/baselines/B1_xgboost_real/test_5seed_summary.json`),
Mesh-PINN best-step (`memory/project_mesh_curriculum_5seed_locked.md`).

---

## 4. Observations

1. **TreePEX is the strongest cold-start model** on both designs (5.10 % / 5.54 %).
   The 5-seed Tweedie XGBoost ensemble with the 67-D feature set is robust to the
   one labelled feature being replaced by a regression proxy.
2. **B1 XGBoost almost perfectly reproduces its warm-start numbers** (Δ ≤ 0.02 pp).
   This is the cleanest demonstration that the cold-start feature pipeline is
   faithful: when the model's training-time labels (fanout was used during
   training but only as one weak feature) are well-matched by the proxy, cold
   ≈ warm.
3. **CatBoost trained in this sprint is the fastest model end-to-end** (0.98 s
   inference on nova) and stays within 0.6-0.8 pp of TreePEX. It's a useful
   acceleration option if accuracy budget allows.
4. **Mesh-PINN cold-start is reasonable but the worst on accuracy**, +1.5 pp
   from its warm best-step. The two main caveats: (a) per-layer analytic prior
   calibration is required — without it gnd MAPE is 55 % vs 18-19 % with;
   (b) inference is dominated by CPU PyTorch fwd time (2,273 s for nova) and
   would shrink an order of magnitude on GPU.
5. **The shared feature pipeline is >99 % of the wall time for tree models**.
   Speeding up V3 + V4 extraction is the only meaningful win for tree-based
   cold-start. See `FEATURE_SPEEDUP_PLAN.md`.
6. **The XGB-Tweedie fanout proxy was load-bearing**: replacing the Ridge
   baseline dropped tv80s TreePEX cold from 5.91 % to 5.10 %. cpl MAPE moved
   from 14.76 % to 13.88 %, mostly because cpl XGBoost has 0.81
   feature_importance on `fanout` and the proxy noise propagated directly.

## 5. Limitations / honest caveats

1. **Tile cache is treated as a pre-built asset for V4 H3.** A truly first-time
   design would have to build it (measured: 120 s tv80s, projected 60-90 min
   nova) before the cold-start pipeline can run V4. Reported timings DO NOT
   include this step.
2. **`fanout` proxy is fit on labeled training data**. Pure cold-start in the
   strictest sense would either retrain the predictor without `fanout` or
   collect a small in-domain labelled batch. The proxy is a pragmatic compromise.
3. **mesh-PINN inference was measured on CPU**. GPU numbers will be much lower.
4. **CatBoost weights were trained in this sprint** rather than reused from an
   archived run — so there is no warm-start reference number for it.
5. **Only intel22 PDK**. ASAP7 cold-start was out of scope.

## 6. Reproduction

```bash
# Once per host:
source tool.env

# Per design (produces feature parquet + TreePEX prediction):
python3 TreePEX/scripts/pex_cold.py --design intel22_tv80s_f3 --workers 16
python3 TreePEX/scripts/pex_cold.py --design intel22_nova_f3  --workers 16

# Per (design, model):
python3 TreePEX/scripts/pex_cold_predict.py --design intel22_tv80s_f3 --model treepex
python3 TreePEX/scripts/pex_cold_predict.py --design intel22_tv80s_f3 --model b1_xgb
python3 TreePEX/scripts/pex_cold_predict.py --design intel22_tv80s_f3 --model catboost
python3 TreePEX/scripts/pex_cold_predict_mesh.py --design intel22_tv80s_f3 --device cpu

# Aggregate report:
python3 TreePEX/scripts/summarize_cold_results.py > /tmp/summary.md
```

Outputs land in:
* `TreePEX/outputs/cold_reports/{design}_cold_features.parquet`
* `TreePEX/outputs/cold_reports/{design}_{model}_summary.json`
* `TreePEX/outputs/cold_reports/{design}_{model}_per_net.csv`
* `TreePEX/outputs/spef/{design}_{model}_cold_pred.spef`
* `TreePEX/outputs/predictions/{design}_{model}_cold_pred.csv`
