# Model Card — TreePEX 5-seed Tweedie XGBoost ensemble

**Released**: 2026-05-10 (trained), 2026-05-14 (deployment lock)
**Frontier MAPE**: tv80s 4.98 % · nova 5.28 %
**Status**: production canonical for intel22 PDK

---

## 1. Model architecture

- **Type**: gradient-boosted decision tree ensemble
- **Base learner**: XGBoost regressor (10 models = 2 channels × 5 seeds)
- **Channels**: ground capacitance (`gnd`) + coupling capacitance (`cpl`); per-net `total = gnd + cpl`
- **Aggregation**: prediction-mean across seeds (NOT MAPE-mean) — variance cancels √5 ≈ 2.24×
- **Inference cost**: CPU-only; ~6 s for nova (113,812 nets), ~0.17 s for tv80s (3,280 nets)

### Hyperparameters

| Param | Value | Comment |
|---|---|---|
| `objective` | `reg:tweedie` | compound Poisson-Gamma likelihood matched to non-negative right-skewed C distribution |
| `tweedie_variance_power` | 1.5 | between Poisson (1.0) and Gamma (2.0); MAPE-aligned |
| `max_depth` | 8 | balance underfit vs overfit on 67-D scalar features |
| `n_estimators` | 500 | with early-stop |
| `learning_rate` | 0.05 | |
| `subsample` | 0.8 | row bagging per tree (stochastic gradient boosting) |
| `colsample_bytree` | 0.8 | feature bagging per tree |
| `early_stopping_rounds` | 100 | on validation MAPE |
| `seeds` | 42, 0, 1, 2, 3 | 5-seed sweep |
| `device` | CPU | no GPU dependence (xgb 2.0+) |

Weight files: `tweedie_{gnd,cpl}_seed{42,0,1,2,3}.json`, ~12 MB each (Booster save_model JSON).

### Auxiliary: fanout proxy

학습 데이터의 `fanout` 컬럼은 SPEF의 coupled_caps 개수에서 라벨링됐기 때문에 cold-start (DEF만 가진 신규 design)에서는 **추정 필요**. cpl XGBoost의 feature_importance에서 `fanout`이 0.81로 dominant이라 proxy 정확도가 직접 cpl MAPE를 좌우.

Pre-fit proxy files (이 디렉에 동봉):
- `fanout_proxy_meta.json` — meta (kind + feats + model_file 경로)
- `fanout_proxy_xgb_tweedie.json` — XGBoost-Tweedie 8-feature proxy (tv80s OOS MAPE_med 12%; primary)
- `fanout_proxy_ridge.json` — Ridge baseline (tv80s OOS MAPE_med 31%; fallback)

`pex_cold.py:apply_fanout_proxy` (line 1421)에서 XGB 우선 로드, 없으면 Ridge fallback. 학습/배포 시 둘 다 동봉 권장.

---

## 2. Training data

- **Source**: StarRC golden SPEF on 13 intel22 designs (Synopsys 2021.06)
- **Split**:
  - Train: 11 designs (ldpc, b14, ethernet_top, jpeg_encoder, leon3, opentitan, sha256, simple_spi, swerv_wrapper, vga_lcd, wb_conmax)
  - Test: 2 designs (intel22_tv80s_f3, intel22_nova_f3 — strict cross-design generalization)
- **Total nets**: ~210 k train + 117 k test
- **Targets**: per-net `cap_total_fF` decomposed into `cap_gnd_fF` and `cap_cpl_total_fF`
- **Features**: 67-D scalar — see `docs/FEATURE_SPEC.md`

Build pipeline: `src/baselines/features.py::NetFeatureVector` (41-D base) + `archive/pex_v4/scripts/29_extract_new_features.py` (26-D V4 H3).
Tile cache root: `/data/PINNPEX/data/processed_v3/intel22/`

---

## 3. Validation gates

| Metric | tv80s gate | nova gate | TreePEX |
|---|---|---|---|
| `MAPE_tot_med` ∈ [pp range] | [4.85, 5.30] | [5.30, 5.75] | 4.98 / 5.28 ✅ |
| `MAPE_gnd_med` drift vs prior frontier | ≤ +0.3 pp | ≤ +0.3 pp | within ✅ |
| `MAPE_cpl_med` drift vs prior frontier | ≤ +0.3 pp | ≤ +0.3 pp | within ✅ |
| `R²_tot` ≥ 0.985 | 0.9940 ✅ | 0.9911 ✅ |

(Round 4 njit-path 추가 drift: tv80s tot +0.043 pp / nova tot +0.015 pp — feature-side drift이며 모델 weights는 동일.)

---

## 4. Ensemble math (왜 5-seed prediction-mean인가)

각 seed의 예측 `ŷ_s = y + ε_s`, 노이즈 `ε_s`는 seed별 독립 (subsample row/column 무작위성 + tree split tie-break).
- 단일 seed MAPE: σ
- 5-seed prediction-mean MAPE: σ / √5 ≈ σ × 0.447

직접 측정 (tv80s):
- 단일 seed avg ~5.17 %
- 5-seed prediction-mean **4.98 %** (−0.19 pp 직접 이득)

대조: 만약 5-seed MAPE를 평균만 했다면 (metric-mean), 그 결과는 단일 seed 평균과 같아 노이즈는 줄지 않음. **Prediction-mean이라야** 노이즈가 진짜로 cancel됨.

---

## 5. Known biases / caveats

- **Per-bucket MAPE**: 가장 작은 cap C1 bucket (cap < 0.15 fF)에서 6-7% — denominator-noise dominated. Mid-buckets (C5-C8)에서 4.0-5.0%.
- **gnd MAPE 17-18% 고정 floor**: representation-bound; substrate cap distribution이 hand feature로 안 잡힘 (확인 트랙: archive/pex_v4 H-track diagnosis).
- **22nm intel22 PDK 전용**: ASAP7 cross-PDK 전이는 미검증 (archive/scripts/01_train_asap7_models.py에 초기 시도).
- **Routing density 분포**: training이 8K-50K cuboid net을 본 만큼, 100K+ giant clock spine은 V3-A subsample (512)로 압축됨 — predictor extrapolation에 약간 의존.

---

## 6. Re-train

```bash
# 학습 데이터 build (재실행 필요한 경우):
python3 scripts/build_dataset_multi.py

# 모델 학습 (CPU, ~9 분):
python3 TreePEX/scripts/01_train_save_models.py
# Saves 10 .json files to TreePEX/models/, overwriting in place.

# 검증 (1 design, ~1 분):
python3 TreePEX/scripts/pex_cold.py --design intel22_tv80s_f3
```

학습 디시플린: `01_train_save_models.py` 내부 deterministic seed plumbing 참고. 모델 reproducibility는 `numpy`, `xgboost` 버전에 따라 ~1e-6 perturbation 가능.

---

## 7. Lineage

```
TreePEX (this card) ← Tweedie XGBoost 5-seed (2026-05-10, locked)
 │
 ├── pex_v6 deployable demo (parent, same architecture, slight feature diff)
 ├── S4 Tweedie 5-seed lock (single-config baseline 2026-05-09)
 ├── H5 Big-combined 1-seed (early frontier 2026-05-08)
 │
 └── Alternative paths (all archived, all NEG vs TreePEX):
      pex_v3 mesh-curriculum PINN — 6.26 % tv80s
      pex_v4 substrate physics    — 5.55 %
      pex_v5 auto-4 % sprint      — 5.09 %
      pex_v7 per-pair regression  — 15.7 %
      pex_v8 hybrid analytic+res  — 55.5 %
```
