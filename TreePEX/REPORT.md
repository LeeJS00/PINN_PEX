# TreePEX — deployable PEX tool report (2026-05-10 baseline, 2026-05-14 Round 4 njit lock)

End-to-end demonstration: pre-net features → 5-seed Tweedie XGBoost ensemble
inference → SPEF write (IEEE 1481-1999) → cross-check vs StarRC golden SPEF.

## Pipeline (single command per design)

```bash
# End-to-end from raw DEF (cold-start, recommended)
python3 TreePEX/scripts/pex_cold.py --design intel22_tv80s_f3 --v3-algo njit
python3 TreePEX/scripts/pex_cold.py --design intel22_nova_f3  --v3-algo njit

# 또는 cached features (precomputed CSV가 있을 때)
python3 TreePEX/scripts/pex_tool.py --all
```

```
[DEF + tech LEF + cell LEF + Liberty + layer.info]
       ↓
[V3 41-D + V4 26-D features per net]   ← `pex_cold.py:_v3_per_net`, `_v4_net_features`
       ↓
[5-seed Tweedie XGBoost ensemble inference]   ← models/ (10 + 3 fanout proxy weight files)
       ↓
[per-net (pred_gnd, pred_cpl)]   ← outputs/predictions/<design>_pred.csv
       ↓
[SPEF write — IEEE 1481-1999]    ← outputs/spef/<design>_pred.spef
       ↓
[parse pred SPEF + parse golden SPEF + per-net align + metrics]
       ↓
[outputs/cold_reports/<design>_treepex_{per_net.csv, summary.json}]
```

## Headline result

| design | n_nets | tot_med | gnd_med | cpl_med | R² (tot) | R² (gnd) | R² (cpl) | inference | SPEF write |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **tv80s** | 3,280 | **4.98 %** | 18.02 % | 13.27 % | **0.9940** | 0.9685 | 0.9745 | **3.9 s** (incl. 16-worker fork) | 0.1 s |
| **nova**  | 113,812 | **5.28 %** | 17.40 % | 14.96 % | **0.9911** | 0.9609 | 0.9686 | **6.4 s** | 3.8 s |

(원본 `pex_tool.py` cached-feature 경로: tv80s 0.17 s inference, nova 0.19 s inference — feature 추출은 별도 cold pass.)

**End-to-end wall (DEF → SPEF, 16-worker, Round 4 njit)**:
- tv80s: **48.2 s** (V3 3.5 s, V4 38.2 s, infer 3.9 s, SPEF 0.1 s, DEF 2.1 s)
- nova: **4,906 s** (V3 2,352 s, V4 2,384 s, infer 6.4 s, SPEF 3.8 s, DEF 94 s)

Round 0 pre-patch baseline (tv80s 169.5 s / nova 8,059 s) 대비 누적 가속: tv80s **3.52×**, nova **1.64×**. V3 단독 nova **2.38×**. 자세한 round-별 표: `COLD_START_SPEEDUP_REPORT.md`.

## SPEF integrity

- IEEE 1481-1999 compliant — `*SPEF`, `*DESIGN`, `*UNIT` headers correct
- Per-net `*D_NET <name> <total_cap>   *CONN   *CAP   1 <netname>:0 <gnd>   *END`
- Round-trip cap value preservation: max abs error **5e-6 fF** (essentially lossless)
- File size: tv80s 0.2 MB / nova 6.2 MB
- Parser-tested: predicted SPEF re-parsed and per-net cap matched the
  prediction CSV exactly

## Per-cap-decile breakdown (tv80s)

Within-bucket per-net MAPE_med + R² showing the within-bucket variance challenge:

| bucket | n | cap_mean (fF) | MAPE_tot | R²(tot) | R²(cpl) |
|---|---:|---:|---:|---:|---:|
| C1 | 317 | 0.120 | 6.85 % | 0.757 | 0.398 |
| C2 | 317 | 0.177 | 5.68 % | -0.574 | 0.270 |
| C3 | 317 | 0.232 | 5.69 % | -0.154 | 0.352 |
| C4 | 317 | 0.300 | 5.07 % | -0.604 | 0.309 |
| C5 | 317 | 0.391 | 4.99 % | -0.005 | 0.360 |
| C6 | 316 | 0.537 | 4.92 % | 0.399 | 0.352 |
| C7 | 317 | 0.816 | 4.88 % | 0.744 | 0.541 |
| C8 | 317 | 1.458 | 4.02 % | 0.869 | 0.529 |
| C9 | 317 | 2.743 | 4.43 % | 0.892 | 0.685 |
| C10 | 317 | 6.828 | 4.20 % | 0.980 | 0.906 |

C1 (smallest 10 % nets, cap < 0.15 fF) drives the residual MAPE — denominator-noise
dominated. Mid-bucket C8 actually hits **4.02 %** (best per-bucket). Negative
within-bucket R²(tot) on C2-C5 confirms the well-known information ceiling
for DEF/LEF/Liberty inputs (per pex_v4 H-track diagnosis).

## Per-cap-decile breakdown (nova)

| bucket | n | cap_mean (fF) | MAPE_tot | R²(tot) | R²(cpl) |
|---|---:|---:|---:|---:|---:|
| C1 | 9,243 | 0.109 | 6.88 % | 0.642 | 0.283 |
| C5 | 9,243 | 0.343 | 5.14 % | -0.128 | 0.403 |
| C8 | 9,243 | 1.363 | 4.74 % | 0.885 | 0.663 |
| C10 | 9,243 | 13.956 | 4.85 % | 0.970 | 0.903 |

Same pattern as tv80s but on a 30× larger test set; nova confirms the model
generalizes (no per-bucket regression).

## Comparison vs prior

| method | tv80s tot | nova tot | inference (95k) | input |
|---|---:|---:|---:|---|
| **TreePEX (S4 Tweedie 5-seed ensemble)** | **4.979 %** | **5.279 %** | **0.36 s combined** | DEF+LEF+Lib |
| v12 PINN (pex_v3 frontier, 5-seed mean) | 5.55 % | n/a | 20.4 s (tv80s only) | DEF+LEF+Lib |
| B1 XGBoost baseline (5-seed) | 5.30 % | 5.83 % | ~0.05 s | DEF+LEF+Lib |
| Innovus (Cadence pattern matching) | 22-72 % per bucket | similar | ~minutes/chip | DEF+LEF |
| OpenRCX (open-source pattern matching) | 16-72 % per bucket | similar | ~minutes/chip | DEF+LEF |
| StarRC (Synopsys golden) | 0 (reference) | 0 (reference) | minutes/chip | full extraction |

TreePEX frontier:
- Beats v12 PINN by 0.57 pp on tv80s tot at **120× faster wall** (0.17 s vs 20.4 s)
- Beats B1 XGBoost by 0.32 pp / 0.55 pp on tv80s/nova (per-channel improvements)
- Beats Innovus / OpenRCX pattern matching by 4-9× margin (per pex_v4 killer figure)

## Reproducibility

```bash
# 1) Train + save weights (one-time, ~9 min)
PY=/tool/etc/python/install/3.11.9/bin/python3
$PY TreePEX/scripts/01_train_save_models.py
# Saves 10 model files to TreePEX/models/ (≈ 120 MB total)

# 2) Run end-to-end on test designs (~6 min wall, mostly Python startup)
$PY TreePEX/scripts/pex_tool.py --all
# Per-design: ~0.2s inference + ~3s SPEF + ~2 min SPEF parsing/comparison
```

Outputs:
- `TreePEX/outputs/predictions/<design>_pred.csv` — per-net predictions
- `TreePEX/outputs/spef/<design>_pred.spef` — IEEE 1481-1999 SPEF
- `TreePEX/outputs/reports/<design>_report.json` — full metrics
- `TreePEX/outputs/reports/tool_summary.json` — overall summary
- `TreePEX/outputs/reports/<design>_per_net_compare.csv` — per-net pred vs golden

## On the 4 % gap (what's actually limiting us)

Important correction: StarRC golden uses the SAME inputs we do (DEF + tech LEF
+ Liberty + layer stack). It reaches accurate cap via NXTGRD pattern-lookup
tables + 3D field solver. So the gap from 4.98 % → 4.0 % is NOT a
missing-input problem, it is a **representation / model expressivity**
problem:

- Our 67-D scalar feature compresses a 3D routing pattern to ~67 numbers
- DeepSet aggregation (v12 cuboid encoder) symmetrizes per-segment info
- Tree model can't represent per-pair field interactions explicitly
- StarRC's NXTGRD has access to thousands of precomputed 3D pattern → cap
  entries for direct lookup; our model has to extrapolate from training data
  alone.

The path to 4 % is in the model side, not the input side.

## What TreePEX does NOT include

- **Raw DEF→features pipeline**: implemented in pex_v3 preprocessing
  (`pex_v3/src/preprocessing/`, `pex_v3/scripts/04_build_feature_dataset.py`,
  `pex_v4/scripts/29_extract_new_features.py`). TreePEX starts FROM the
  precomputed feature CSVs to focus on the deployment-stage demonstration.
  The feature extraction is heavy (~3-4 h on 8 workers for raw-tile path)
  but a one-time job per design.
- **Per-pin cap distribution**: TreePEX writes per-net lumped `*CAP` only.
  Per-pin distribution would require a pin-level cap model (future work).
- **R prediction (parasitic resistance)**: TreePEX covers C only (g + c
  decomposition). R has separate pipelines (sister `r_analytic_v3` work).
