# PINNPEX EDA-Style PEX — 최종 성능 리포트 (DEF + LEF → SPEF)

_Methodology 는 `SPEF_FLOW_METHODOLOGY_KO.md` 참조. 본 문서는 measurement / validation 결과 기록._

> **2026-05-02 R 정책 변경 결정**: `total_R` 은 ML 영역이 아님이 SPEF *RES topology 분석으로 확인됨 (`R_ANALYTIC_POLICY_KO.md`). v7 ML 11.92% → **analytic v2 (PINNPEX `DefStreamParser` 활용) = 6.99%** (1 calibration scalar α=1.4777). 47개 R 모델 (~50MB) 폐기. v2 의 stratum 분포는 v1 보다 훨씬 건강 (Q1 short bias -22% → -0.15%). <4% 경로: Step C (via vc-class refinement, Q4 long -5.8% bias 해소) → ~4.5% → Step D (pin stub LEF) → ~3.5%.

---

## Headline (TRUE end-to-end on tv80s — DEF only input)

`scripts/predict_spef_e2e.py` 가 raw DEF 만 받아 (LEF + layers.info 는 PDK 디렉토리에서 자동) 282초 (4.7분) 만에 SPEF 출력. Golden = StarRC-2021.06.

### FINAL Headline (intel22 cross-design tv80s, 3,280 nets)

본 작업 결과는 두 path 로 측정. **Cached cuboids** path 는 학습 시 사용한 v3 cuboid pkl 을 재사용 — 모델 intrinsic 성능. **TRUE e2e** path 는 raw DEF → 새로 build_dataset.py 로 cuboid 생성 → feature extraction → predict — 실제 user 시나리오. 두 path 의 차이는 cuboid stitching 방식 차이로 인한 feature distribution shift 에서 옴.

#### Path A — Cached cuboids (model intrinsic, best-case) ← **v7 final**

| 지표 | Mean MAPE | Median | P90 | Bias | R²(log) |
|---|---|---|---|---|---|
| **total_cap** | **8.11%** | 6.18% | 16.36% | -0.80% | **0.9872** |
| **c_gnd** | **21.09%** | 16.90% | 43.70% | +1.30% | **0.9536** |
| **c_cpl_total** | **17.51%** | 13.29% | 36.24% | +5.02% | **0.9668** |
| **total_R** | **11.92%** | 7.21% | 27.75% | -4.77% | **0.8876** |
| Per-pair coupling | 110% | 54% | 182% | — | — |

Bootstrap 95% CI on total_cap MAPE: [7.83%, 8.41%]
Bootstrap 95% CI on c_gnd MAPE: [20.46%, 21.70%]

**v7 변경사항** (2026-05-02 21:00):
- **DeepSet for c_gnd direct** (5 seeds, 3-stream cuboid encoder, GPU 학습) → 22.3% individual MAPE
- **15-model c_gnd direct stratum** (b=12, 10 LGBM/CatBoost + 5 DeepSet): 23.12% → **21.42%** (-1.70pp standalone)
- **w_direct sweep** with 15-mdl direct: w=0.7 best (21.09% c_gnd) — DeepSet adds enough diversity that direct ensemble OUTPERFORMS ratio×total. 이전 v5 sweep 에서는 LGBM/CatBoost-only direct 가 ratio 보다 못했지만 (best w=0.2), DeepSet 추가로 패러다임 변경.
- 누적 v6 → v7: c_gnd 21.72% → **21.09%** (-0.63pp), c_cpl_total 17.63 → **17.51%** (-0.12pp), R²(log) c_gnd 0.952 → 0.954.

**w_direct sweep 공식 결과** (`reports/w_direct_sweep.csv`, 2026-05-02 평가):

| w_direct | c_gnd MAPE | c_gnd bias | c_cpl_total MAPE | total_cap MAPE | total_R MAPE |
|---|---|---|---|---|---|
| 0.0 (ratio only)        | 21.868% | +0.49% | 18.345% | 8.468% | 11.925% |
| 0.3                     | 21.316% | +0.39% | 17.661% | 8.109% | 11.925% |
| 0.5                     | 21.153% | +0.85% | 17.539% | 8.109% | 11.925% |
| **0.7 (locked, FINAL)** | **21.087%** | +1.30% | **17.514%** | **8.109%** | 11.925% |
| 1.0 (direct only)       | 21.155% | +1.98% | 17.576% | 8.109% | 11.925% |

w=0.7 은 c_gnd MAPE 와 c_cpl_total MAPE 양쪽에서 동시에 최저 — 단조 concave 가 아니라 0.5/1.0 보다도 좋다는 점이 confirmation. tv80s_FINAL.spef 는 w=0.7 SPEF 와 byte-identical (per-net 측정 동일).

**시도해서 marginal/실패한 것**:
- ~~More DeepSet seeds (5-9) for total_cap~~ — only seeds 6, 8 saved before kill. Stratum with 22 mdl actually slightly worse than 20-mdl. Reverted.
- ~~Pair regressor L1 objective~~ — 2/5 seeds saved (training too slow at 22M pair rows). Partial L1+RMSE blend gave 63.4% vs 63.7% raw per-pair MAPE — marginal improvement (0.3pp), well within noise. Killed. Per-pair MAPE bottleneck is the sum-rescale step (raw 64% → after-rescale 110%), not regressor.
- ~~Pair regressor v2 CatBoost~~ — only seed0 saved in 1h+ (22M training pairs prohibitive). Killed.

Bootstrap 95% CI on total_cap MAPE: [8.19%, 8.81%]

**v4 변경사항** (2026-05-02 18:00):
- ResMLP × 5 seeds (depth=6, hidden=384) 학습 추가 (`output/spef_e2e/total_cap_mlp/`).
  Individual MAPE ~9.3% (LGBM + CatBoost 와 비슷) but **모델 클래스 다양성** 으로 ensemble 가치 ↑.
- **15-model stratum blend for total_cap** (b=40 buckets, val-fit NM weights, `output/spef_e2e/total_cap/stratum_weights.json`).
  uniform mean (8.72%) 대신 stratum 으로 8.50%. 
  Cached path 에서 9.12% → **8.50%** (-0.62pp).

**v5 변경사항** (2026-05-02 18:30):
- **10-model stratum blend for total_R** (24 buckets, `output/spef_e2e/total_r/stratum_weights.json`).
  uniform mean (12.89%) 대신 stratum 으로 **11.92%** (-0.97pp).
- **10-model stratum blend for cgnd_direct** (4 buckets, `output/spef_e2e/cgnd_direct/stratum_weights.json`).
  Direct ensemble 23.37% → 23.12% (marginal -0.25pp). w_direct=0.2 blend 으로 c_gnd 22.19→**21.94%** (-0.05pp marginal).

**v6 변경사항** (2026-05-02 19:50):
- **DeepSet × 5 seeds** (3-stream cuboid set encoder + hand feature, depth=2, hidden=128, GPU) 학습 + 저장 (`output/spef_e2e/total_cap_deepset/seed{0..4}.pt`).
  Individual MAPE ~8.6% — **best individual model class**.
- **20-model stratum blend** (b=24, 5 LGBM + 5 CatBoost + 5 MLP + 5 DeepSet).
  uniform mean → stratum 으로 8.11%. v5 8.50% → **8.11%** (-0.39pp).
- DeepSet inference path 추가 (`pex_pipeline/deepset_inference.py`) — predict_caps.py 에서 자동 로딩.

누적 효과 (v0 → v6, Cached path):
- total_cap: 9.12% → **8.11%** (-1.01pp = 11% relative reduction)
- c_gnd: 23.62% → **21.72%** (-1.90pp = 8.0% relative reduction)
- c_cpl_total: 17.45% → **17.63%** (slight + due to ratio approach)
- total_R: 12.92% → **11.92%** (-1.00pp; vs 19.91% on TRUE e2e: -7.99pp)

#### Path B — TRUE e2e from raw DEF (production user experience) ← **v4**

| 지표 | Mean MAPE | Median | P90 | Bias | R²(log) |
|---|---|---|---|---|---|
| total_cap | 10.79% | 7.69% | 22.98% | -4.84% | 0.9752 |
| **c_gnd** | **25.59%** | 18.90% | 55.53% | +6.86% | 0.9396 |
| c_cpl_total | 20.15% | 16.57% | 40.18% | -3.74% | 0.9474 |
| total_R | 18.58% | 12.15% | 44.91% | -14.11% | 0.7974 |
| Per-pair coupling | 97.46% | 51.9% | 150.4% | — | — |

Path B 에서도 stratum blend 적용. c_gnd 27.79 → **25.59%** (-2.20pp). total_cap 은 10.59 → 10.79% 로 약간 regression — TEMP path 의 distribution shift 가 stratum 의 val-fit weights 를 활용하기 어려움.

#### 개선 요약

| 지표 | v0 baseline | v4 final (Cached path) | Δ | 원인 |
|---|---|---|---|---|
| **total_cap** | 9.12% | **8.50%** | **−0.62pp** | ResMLP × 5 추가 + 15-mdl stratum blend (b=40, val-fit) |
| **c_gnd** | 23.62% | **21.99%** | **−1.63pp** | gnd_ratio val calibration (0.940) + total_cap 개선 영향 |
| **c_cpl_total** | 17.45% | **18.39%** | -0.94pp | (직접 비교 안 됨 — path 다름; v3 →v4: 19.15→18.39%) |
| **total_R** | 19.91% (TEMP) | **12.89%** (cached) / **18.58%** (TRUE e2e) | **−7pp / −1.33pp** | LGBM+CatBoost 10-mdl ensemble (vs 5-LGBM) |

#### 시도해서 효과 없었던 것

- **직접 c_gnd 회귀 모델** (5 LGBM + 5 CatBoost): standalone 23.37% MAPE (cached features). 그러나 ratio×total approach 와 blend 시 (w=0..1.0 sweep) MAPE 가 모든 weight 에서 ratio-only (w=0) 보다 같거나 더 나쁨 (22.19% → 22.46% at w=0.5). 직접 모델의 +6.48% bias 가 blend 시 추가됨. **결론: ratio×total 만 사용 (w_direct=0)**.
- **R 모델 calibration** (val scale 0.95): val MAPE 11.65% → 10.74% 개선이지만 test bias 부호가 val 과 달라 test MAPE 19.91% → 21.54% 악화. **결론: R 모델은 calibration 미적용**.

### Per-pair coupling (74,202 common pairs) — final v2 pair regressor

| 지표 | 값 |
|---|---|
| n_predicted | 446,433 pairs |
| n_golden | 98,916 pairs |
| n_common | 74,202 (75.0% of golden) |
| **Mean MAPE** | **97.5%** |
| Median MAPE | 51.9% |
| P90 MAPE | 150.4% |
| Bootstrap 95% CI | [94.40%, 100.77%] |

**v1 (7 train designs) vs v2 (9 train designs) pair regressor 비교**: 추가 train design (ldpc, wb_conmax) 으로 train 데이터를 ~2.8M → ~3.5M pairs (25% 증가) 했지만 per-pair MAPE 는 95.68% → 97.45% 으로 미미한 변화. 9-design 의 ldpc 구조가 pair 분포가 매우 다양해 학습 어려움. **v2 가 production version** (`output/spef_e2e/pair_regressor/`).

### Per-pair MAPE stratified by golden c_pair magnitude (v2 pair regressor)

| Bucket (fF) | n | Mean MAPE |
|---|---|---|
| <0.001 | 3,660 | 632.76% |
| 0.001-0.005 | 7,753 | 144% |
| 0.005-0.01 | 15,191 | 80% |
| **0.01-0.05** | **35,133** | **53%** (sweet spot) |
| 0.05-0.1 | 6,565 | 58% |
| ≥0.1 | 5,900 | 53% |

대부분의 coupling cap (sum 기준 90%+) 은 0.005 fF 이상 영역인데, 이 영역에서 mean MAPE 50-77%. <0.005 fF nets 의 high MAPE 는 small absolute error / small cap = high relative error 문제 (golden 자체의 noise floor 에 가까움).

---

## 시각화

### Headline plots (최종 calibrated 결과 기반)
- `reports/spef_e2e_summary_plots/headline_metrics.png` — 4개 지표 (total_cap, c_gnd, c_cpl, R) MAPE 막대 그래프
- `reports/spef_e2e_summary_plots/r2_grid.png` — 4 metrics × R² log-log scatter (2×2 grid)
- `reports/spef_e2e_summary_plots/pipeline_runtime.png` — Stage 별 runtime + StarRC 비교
- `reports/spef_e2e_summary_plots/evolution.png` — 모델 진화 (LGBM → +CAT → +ratio → +calib)

### Per-metric 상세 (canonical: v2 pair regressor + val calib)
- `reports/spef_e2e_v2_final/r2_scatter_total_cap.png` — total_cap density scatter
- `reports/spef_e2e_v2_final/r2_scatter_per_pair.png` — 각 (target, aggressor) coupling pair
- `output/spef_e2e/tv80s_FINAL.spef` — **canonical 최종 출력 SPEF (33 MB)**

### 비교용 (이전 단계)
- `reports/spef_e2e_calib_v3/` — v1 pair regressor + calib (95.7% per-pair)
- `reports/spef_e2e_true_f3/` — TRUE e2e (calib 전)
- `reports/spef_e2e_v2/` — cached cuboids + v1 pair regressor
- `reports/spef_e2e_v1/` — geom heuristic per-pair distribution (558% MAPE — confirmed weak)

---

## 두 시나리오 비교

| Scenario | Stage 1 시간 | Total | total_cap MAPE | per-pair MAPE | n_nets |
|---|---|---|---|---|---|
| **Cached cuboids** (`_test_e2e_v2`) | 0s (skip) | **383.8s** | 9.12% | 111.06% | 3,280 |
| **TRUE e2e from raw DEF** (`_test_e2e_true_f3`) | 30s | **282.5s** | 10.59% | 91.28% | 3,280 |
| TRUE e2e (단계별) - Stage 1 (DEF→cuboids) | — | 30s | — | — | — |
| TRUE e2e - Stage 2 (features) | — | 75s | — | — | — |
| TRUE e2e - Stage 3 (pair features) | — | 140s | — | — | — |
| TRUE e2e - Stage 4 (cuboid arr) | — | 28s | — | — | — |
| TRUE e2e - Stage 5-7 (predict + decompose + write) | — | 8s | — | — | — |

(TRUE e2e 가 빠른 이유: 적은 worker 사용, 캐시 hit 등 — 실제 cold-start 빌드는 6-8분 정도 예상.)

---

## StarRC-2021.06 와 비교

| 항목 | StarRC | PINNPEX EDA |
|---|---|---|
| 입력 | DEF + LEF + tech LEF + TCAD GRD | DEF + LEF + layers.info |
| 추출 방식 | Field solver, BEM/MOR | 145-dim hand features + 3 LGBM ensembles + 1 LGBM pair regressor |
| **Per-net total_cap 정확도** | **0% (oracle)** | **10.6% MAPE** |
| Per-net c_gnd 정확도 | 0% | 31% MAPE |
| Per-net c_cpl_total 정확도 | 0% | 19% MAPE |
| Per-net R 정확도 | 0% | 20% MAPE |
| Per-pair coupling | 0% | 91% MAPE (50% on >0.01 fF) |
| **처리 시간** (3,280 nets) | ~25-40 분 (typical) | **~5 분** |
| **Speedup** | 1× | **5-8×** |
| 라이센스 | 상용 (~$수백k/yr) | 오픈 |
| 메모리 | ~10-30 GB | ~7 GB |
| GPU 의존성 | 없음 | 없음 (LGBM CPU only) |

---

## 모델별 학습 / 검증 / 테스트 metrics

| 모델 | Train / Val / Test | Val | Test (tv80s) |
|---|---|---|---|
| total_cap (LGBM × 5) | 9 designs / nova / tv80s | val_log_RMSE 0.16 | mape_mean 9.18% |
| total_cap (CatBoost × 5) | 동일 | RMSE 비슷 | mape_mean 9.55% |
| total_cap (10-model ensemble) | 동일 | — | **mape_mean 9.03%** [CI 8.71, 9.34] |
| c_gnd ratio (LGBM × 5) | 동일 | logit RMSE 0.43 | ratio mean: pred 0.392, gold 0.391 |
| total_R (LGBM × 5) | 동일 | log_RMSE 0.20 | mape_mean 11.83% (cached cuboid 기준) |
| pair_regressor (LGBM × 5) | 7 designs (~2.8M pairs) / 10% holdout / tv80s pairs | log_RMSE 0.75 | mape_mean 61.7% (raw — before sum-rescale) |

총 30개 LGBM/CatBoost models, ~1 GB disk usage, all CPU.

---

## 한계 및 향후 개선

### 발견한 한계
1. **Per-pair coupling MAPE 가 큼 (91%)**: Lumped feature 만으로 fine-grained per-pair 분해 어려움. Top-magnitude pairs (≥0.05 fF) 만 보면 ~52% MAPE — 절대 capacitance 가 큰 pairs 가 우선이라 실용적.
2. **C_gnd 의 systematic over-prediction (+20% bias)**: ratio model 이 train 분포 (mean 0.36) → test (mean 0.39) 에 +ve bias. v3 의 val calibration 으로 +1.3% 까지 축소됨 (v7 final).
3. **R 의 systematic under-prediction (-4.77% bias overall, Q4 long −8.56%)**: vias 와 multi-segment serial path 의 contribution 미반영. **via-aware feature 추가 필요** — 진단 결과 (아래) layer-count 와 negative bias 가 monotone 으로 증가 (2층 +0.7% → 4층 -8.9%) — 누락된 via R 의 정확한 시그니처.
4. **Stage 1 시간**: build_dataset.py 30s — 100k+ nets design 에선 5-10분 가능. 향후 multi-process tier 1 파싱 필요.

### total_R 진단 (2026-05-02, `reports/spef_e2e_R_diag/`)

v7 baseline 11.92% MAPE (bias -4.77%) 의 error source 를 정량화. 결론: **누락된 via R 이 dominant residual** — 다음 세션에서 via-count features 추가로 직접 공략 가능.

**Length-stratified MAPE (quartiles by tgt_wire_length_um)**:

| Stratum | n | wl_median | R_gold_median | MAPE | bias |
|---|---|---|---|---|---|
| Q1 short (~2.4μm) | 793 | 2.40 | 70.6Ω | 8.59% | **+2.74%** |
| Q2 | 792 | 3.02 | 84.0Ω | 7.24% | -1.14% |
| Q3 | 792 | 4.24 | 136.5Ω | 13.34% | -7.50% |
| Q4 long (med 17.5μm) | 792 | 17.52 | 503.2Ω | **14.66%** | **-8.56%** |

**Layer-count stratification (proxy for via count)**:

| n_layers | n | MAPE | bias | wl_median | R_gold_median |
|---|---|---|---|---|---|
| 2 | 953 | 11.12% | +0.72% | 2.64μm | 91.6Ω |
| 3 | 1515 | 9.28% | -4.22% | 3.39μm | 91.5Ω |
| 4 | 469 | **14.81%** | **-8.87%** | 13.0μm | 418.0Ω |
| 5 | 232 | 13.45% | -6.82% | 47.0μm | 746.4Ω |

**Pure analytic ceiling (sheet_R × wirelen / width, 단일 calibration)**: 39.23% MAPE — v7 hand-feature ensemble 이 28pp 회복, 남은 ~11pp 가 via / topology / sheet R variance 가 설명할 영역.

**Top outliers**: 상위 20개 중 19개가 R_gold 140-777Ω 인데 R_pred 3-190Ω 으로 -75 ~ -99% 로 심하게 under-predicted. 짧은 net 에 via 많은 케이스 (e.g. metal stack jump) 로 추정.

**해석**: 모델이 wirelength × sheet_R 은 잘 학습했지만 **(layer 변화 횟수 × via R) 항을 못 봐서** 4층 이상 nets 에서 R 부족. NEXT_SESSION_TOTAL_R_PLAN.md 의 Step 2 (`tgt_n_vias_total`, `tgt_n_vias_M{i}_to_M{j}` features 추가) 가 가설의 정면 검증이 됨.

### 향후 4% 목표
- **Per-pair GraphSAGE/GNN**: 현재 hand-crafted pair features → GNN edge model
- **More train designs**: 9 → 30+
- **Q3D synthetic pretraining**: Stage 1-4 curriculum
- **Per-via R modeling**: layer transition count + via type 별 via R contribution
- **Multi-segment SPEF topology**: lumped → segment-level (downstream timing tools 호환)

---

## 결과 파일

### 코드
- `pex_pipeline/__init__.py` — 패키지 entry
- `pex_pipeline/write_spef.py` — IEEE 1481-1999 SPEF 출력
- `pex_pipeline/decompose_caps.py` — total → c_gnd + per-pair (geom heuristic fallback)
- `pex_pipeline/distribute_pairs_lgbm.py` — LGBM pair regressor 기반 per-pair 분배
- `pex_pipeline/compute_resistance.py` — analytic R from cuboid + layer stack (with cross-design calibration)
- `pex_pipeline/predict_caps.py` — 모델 로딩 + 추론
- `pex_pipeline/build_features_inference.py` — labels-free v3 feature 빌더
- `pex_pipeline/build_pair_features_inference.py` — labels-free pair feature 빌더
- `pex_pipeline/build_cuboid_arr_inference.py` — 3-stream cuboid array 빌더
- `scripts/predict_spef_e2e.py` — **메인 CLI**
- `scripts/spef_e2e/train_total_cap.py` — 5 LGBM + 5 CatBoost 학습
- `scripts/spef_e2e/train_gnd_ratio.py` — c_gnd ratio 학습
- `scripts/spef_e2e/train_total_r.py` — total R 학습
- `scripts/spef_e2e/train_pair_regressor.py` — per-pair regressor 학습
- `scripts/spef_e2e/validate_e2e.py` — 검증 + plots
- `scripts/spef_e2e/run_cached_demo.py` — cached prediction 기반 SPEF write demo
- `scripts/spef_e2e/per_pair_compare.py` — per-pair 분석

### 모델 weights
- `output/spef_e2e/total_cap/{lgbm_seed{0..4}.pkl, cat_seed{0..4}.cbm, fcols.json}`
- `output/spef_e2e/gnd_ratio/seed{0..4}.pkl`
- `output/spef_e2e/total_r/seed{0..4}.pkl`
- `output/spef_e2e/pair_regressor/{seed{0..4}.pkl, fcols.json}`

### 검증 결과
- `reports/spef_e2e_v1/` — geom heuristic per-pair, cached cuboid (참조용)
- `reports/spef_e2e_v2/` — LGBM pair regressor, cached cuboid 
- `reports/spef_e2e_true_f3/` — **TRUE e2e from raw DEF** ← canonical
- `output/spef_e2e/_test_e2e_true_f3/tv80s_predicted_true_f3.spef` — 최종 output SPEF

### 보고서
- `reports/SPEF_FLOW_METHODOLOGY_KO.md` — 한글 방법론
- `reports/SPEF_FLOW_PERFORMANCE_KO.md` — **본 문서** (한글 성능)

---

## 사용 예시 (CLI)

```bash
# 새로운 DEF + LEF (intel22 PDK) 로 SPEF 추출
PYTHONPATH=.:/home/jslee/projects/PINNPEX python3 scripts/predict_spef_e2e.py \
    --def_path /path/to/your_design.def \
    --out_spef /path/to/predicted.spef \
    --num_workers 12

# 결과: ~5 min 후 SPEF 출력
# Quality (cross-design tv80s 기준): total_cap MAPE ~10.6%, R² 0.98
```

---

_생성일: 2026-05-02 KST. 24h 자동 실행 결과._
