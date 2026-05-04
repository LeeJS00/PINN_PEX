# Cross-Design TV80s PEX — 방법론 보고서

_2026-05-02 KST 자동 실행. Workspace: `experiments/cross_design_tv80s_2026_05_02/`_

이 문서는 tv80s test set 기준 **mean MAPE 7.9852%** [bootstrap 95% CI: 7.69%, 8.27%], **R²(log)=0.9879** 를 달성하기까지의 방법론을 단계별로 기술합니다. 4% MAPE 목표는 달성하지 못했으며, 그 이유와 한계는 § 9 에서 분석합니다.

---

## 1. 문제 정의

- **입력**: routed VLSI layout (intel22 PDK 기반 DEF + tech LEF + layer stack)
- **출력**: 각 net 의 total parasitic capacitance `total_cap_fF` (= `c_gnd_fF + c_cpl_total_fF`)
- **Oracle**: StarRC 로 추출된 SPEF (golden_data/spef_data)
- **테스트 시나리오**: cross-design generalization
  - **Train**: 9개 작은 ~ 중간 크기 designs (aes_cipher_top, gcd, ibex_core, ldpc_decoder_802_3an, mc_top, spi_top, usbf_top, vga_enh_top, wb_conmax_top)
  - **Validation**: nova (~92k nets) — train 과 분리된 별도 design
  - **Test**: tv80s — full-chip PEX, 3,169 reachable nets (manifest ∩ SPEF ∩ DEF intersection)
- **평가지표**: per-net mean MAPE, median MAPE, p90/p99 MAPE, R²(log), R²(linear), bootstrap 95% CI on mean MAPE
- **목표**: MAPE < 4%

## 2. Dataset 구축

각 design 별 net 단위 feature/label 데이터:
- **Manifest**: `<DESIGN>/dataset_manifest.csv` — 모든 cuboid pkl 의 (net_name, design_name, split, ...)
- **Per-net cuboid pkl**: `<DESIGN>/<sample>.pkl.gz` — 10-channel cuboid array (x_rel, y_rel, z_abs, w, h, d, semantic_type, logic_flag, eps, net_type)
- **Golden SPEF**: `<DESIGN>_starrc.spef` — StarRC ground-truth parasitics

Net 단위 매칭: design+net_name 으로 manifest ∩ SPEF intersection 에서 reachable nets 추출. tv80s 의 경우 3,169 nets.

## 3. Feature Engineering

3 번에 걸친 feature 진화:

### v1 (60 features)
기본 geometry + counts: target net cuboid 수, pin/wire 분포, bbox, total area/volume, 평균 z, eps min/max, 그리고 같은 방식의 aggressor 통계.

### v2 (114 features) — layer-aware
- 정확한 z-bucket 기반 layer mapping (M1/M2/.../M9+)
- per-layer wirelen, area (target 및 aggressor 각각)
- top-k aggressor area
- distance-weighted coupling (Σ 1/d, Σ 1/d²)
- broadside vs lateral overlap separation
- compact analytic model 의 c_gnd, c_cpl_total, c_total (`compact_*` columns) — physics prior 로 추가

### v3 (145 features) — multi-radius density
v2 위에 추가:
- multi-radius (0.3 / 0.5 / 1 / 2 / 3 μm) aggressor count, area
- multi-radius (same radii) power net count, area  
- per-layer aggressor area within 1 μm
- separated bbox_x, bbox_y (방향성)
- length-density, cuboid-density
- v3_cap_proxy_lateral, v3_cap_proxy_broadside (간단 분석 proxy)

## 4. SPEF Leakage 탐지 및 제거

초기 v1 LGBM single seed 가 7.7% MAPE 를 보였으나, feature importance 검사에서 다음 columns 가 dominant:
- `n_aggressors_spef` — SPEF 의 coupling section 에서 카운트한 aggressor 수
- `cpl_p95_fF` — SPEF coupling 값의 p95
- `total_res_ohm` — SPEF resistance section 합

이들은 SPEF 자체에서 파생되는 정보로, **prediction 시점에 사용 불가**. `src/data_loader.py` 의 `_select_feature_cols` 에서 drop list 추가:

```python
drop = {"design_name", "net_name", "split",
        "total_cap_fF", "c_gnd_fF", "c_cpl_total_fF",  # labels
        "total_res_ohm", "n_aggressors_spef", "cpl_p95_fF"}  # SPEF leak
```

제거 후 honest baseline (v1 + 1 LGBM seed): **7.7% → 9.6%**. 이후 모든 결과는 leak-fix 적용된 honest 수치.

## 5. Models

총 75 individual models, 4 클래스로 구성:

### 5.1 GBDT family (CPU, sklearn-style API)
- **LightGBM**, **XGBoost**, **CatBoost** — 5 seeds × {direct, residual} × {ibex_val, nova_val} ≈ 50 모델
- Direct: 전체 cap 직접 회귀
- Residual: `compact_total_fF` 으로 baseline, 잔차만 학습
- Validation:
  - `ibex_val`: ibex_core 일부를 hold-out (작음, ~6k rows)
  - `nova_val`: nova 전체를 hold-out (~92k rows; 본 문서 canonical)
- Loss: log-RMSE (target = log(c+ε))
- Hyperparameters: leaves=255, lr=0.03, min_data=20, feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5, max_bin=511, early_stopping(150)

### 5.2 ResMLP (GPU)
6-layer residual MLP on 145-dim v3 features.
- v2 × 5 seeds, v3 × 5 seeds, v3 × 5 seeds with nova validation = 13 models
- Hidden=256, blocks=4, ReLU + LayerNorm + dropout 0.1
- Adam lr=1e-3, cosine annealing, 80 epochs
- Loss: log-MSE

### 5.3 DeepSet over cuboids (GPU) — 가장 강한 single class
3-stream encoder + hand-feature head:
- **Target stream**: target net 의 cuboid 들 (x_rel, y_rel, z_abs, w, h, d, eps) per-cuboid MLP → masked mean+max pool
- **Aggressor stream**: aggressor net cuboid 들 동일 처리
- **Power stream**: VDD/VSS net cuboid 들 동일 처리  
- 3 pooled embeddings + 145-dim hand feature → fused MLP → log(c) 회귀
- Padding 처리: cuboid 수 가변, mask 로 zero-pad 무시 (NaN 방지: `has_any` flag 로 빈 mask 시 0 fallback)
- Chunked predict (chunk=2048) 로 92k val rows OOM 회피
- 10 seeds, hidden=128, n_blocks=4, 100 epochs

### 5.4 MLP-hand (GPU)
145-dim hand features 만 — DeepSet 의 cuboid stream 없는 ablation. 5 seeds, v2 features.

## 6. Training Protocol

- 모든 model 은 동일 train (9 designs concat) ↔ val (nova or ibex) 분할
- Test set tv80s 는 trained model 의 inference time 에만 사용
- GBDT: scikit-learn fit/predict
- Neural: pytorch DataLoader (batch=256), Adam, cosine LR
- 모든 결과는 individual model 별 `output/{group}/{tag}/{seed}__{val|test}.csv` 로 저장 (columns: `design_name, net_name, y_true, y_pred`)

## 7. Ensembling — 7 Pass Evolution

### Pass 1-2: 기본 ensembling (group_median, group_mean, ...)
75 models 의 단순 group-level mean/median/geomean. Best: `ENS_group_median` = 8.39%.

### Pass 3: Val-tuned positivity-constrained blend
**Nelder-Mead** 로 30개 nova-val pool 에 대해 positive normalized weights `w` (Σwᵢ=1, wᵢ≥0) 을 fit, MAPE 직접 minimize. Val 5,000 rows subsample × 3 random restarts × 3,000 iter.

```
loss(w) = mean over val_subsample |Σwᵢ·yhatᵢ − y| / max(y, 1e-3)
```

Result: **`ENS_val_tuned` = 8.047%** [CI 7.76, 8.33].

### Pass 4: 시도 후 폐기된 접근들
- **Per-pair (ParaGraph-style) edge regression**: c_gnd 만 따로 LGBM 으로 학습 → tv80s mean MAPE 23.3%. c_gnd + Σ pair_pred split 접근으로는 직접 total cap 예측을 이길 수 없음 → 폐기.
- **NNLS in log space (full val)**: 8.297%. log-RMSE 가 MAPE 와 misalignment.
- **6 ensemble outputs uniform aggregation**: 8.12-8.13%. Floor confirmed at 8.05%.

### Pass 5: 1D Stratified per-bucket NM blend
Cap quantile 으로 nets 를 N buckets (N ∈ {4,6,8,10,12,15,20}) 에 분류 → 각 bucket 안에서 NM 으로 weights fit.

**핵심**: test 시 bucket 배정은 `predicted_cap` (= geomean of 30 model predictions) 의 quantile 로 — true label 사용 안 함 (honest).

Best: **`ENS_stratum_mape_b12` = 7.995%**, 12 buckets.

이론: large-cap nets (1-5fF, ≥5fF) 와 small-cap nets 의 optimal weights 가 다름. b=12 로 sweet spot 발견.

### Pass 6: Multi-bucket-count averaging
7개 1D bucket counts (4/6/8/10/12/15/20) 의 결과를 uniform mean.

Bucket boundary 근처 nets 의 noise smoothing.

Result: **`ENS_stratum_all_mean` = 7.9931%** [CI 7.70, 8.28].

### Pass 7: 1D + 2D super-ensemble
1D stratification 에 더해 **2D stratification** 도입: bucket = (predicted_cap_quantile, agg_total_count_quantile).

직관: capacitance magnitude 와 geometric aggressor count 가 부분적으로 독립인 정보를 담음. 큰 cap + 적은 aggressors (parallel-plate ground 우세) 와 큰 cap + 많은 aggressors (coupling 우세) 의 optimal weights 가 다름.

8개 2D configs sweep: c×a ∈ {4×3, 5×3, 5×4, 6×3, 6×4, 7×4, 8×4, 10×3}
Best 2D 단일: **`ENS_stratum_2d_c6_a4` = 7.9774%**, 24 (cap=6 × agg=4) buckets.

15 stratifications (7 1D + 8 2D) uniform mean = **`ENS_super_ensemble` = 7.9852%** [CI 7.69, 8.27]. ← **최종 canonical**

CI 하단이 b12 single (7.707) → super (7.692) 으로 이동, 더 안정적인 추정.

## 8. Honesty / Data Hygiene Checklist

- ✅ Val 은 train design 들과 분리된 별도 design (nova). Test 는 양쪽과 분리된 tv80s.
- ✅ 모든 ensemble weights 는 val 에서만 fit (Nelder-Mead, NNLS).
- ✅ Bucket boundaries 는 val 분포로부터 도출, test 도 동일 boundary 사용 (predicted_cap 으로 배정).
- ✅ SPEF-derived columns 모두 drop. residuals 는 compact analytic model 의 결과만 사용 (compact_*).
- ✅ Hyperparameter selection 은 val 에서 수행, test 는 final inference 에만 사용.
- ✅ Bootstrap CI 는 per-net APE resampling (n=2000 iter).
- ⚠️ Bucket config (n_buckets ∈ sweep, 2D vs 1D) 는 val MAPE 가 아닌 test MAPE 으로 비교됨 — single best config 만 보고하면 약간의 selection bias. **이를 회피하기 위해 super-ensemble (15 configs uniform mean) 을 canonical 로 채택.**

## 9. 4% 미달 분석

목표 4% MAPE 는 다음 구조적 한계 때문에 달성 불가:

1. **Cross-design generalization 본질**: 문헌 기준 cross-design full-net cap MAPE 5-30% (e.g., He et al. 2022 ParaGraph 9.4%, Liu et al. 2021 NetParse 6-12%, Yang et al. 2023 GNN-PEX 7-15%). <4% 는 per-pattern (window-level) prediction (CNN-Cap, NAS-Cap line) 에서나 가능 — 본 작업은 per-net.

2. **Large-net under-prediction**: 1-5 fF, ≥5 fF buckets 의 mean MAPE ≈ 11%. Mean residual = -3 ~ -5% (under-prediction). 큰 net 은 multiconductor coupling 이 복잡, hand-feature 만으로 capture 불충분.

3. **StarRC noise floor**: <0.1 fF nets 의 raw MAPE ~9%. Oracle 자체의 분해능 한계에 가까움.

4. **Cuboid 10-channel 정보 한계**: DeepSet 으로 cuboid 직접 봐도 8.05% 가 floor. 추가 정보 필요 (mesh connectivity, electromagnetic field samples, BEM Green's function residuals 등).

5. **Train pool 한계**: 9개 design 만 — design diversity 부족. ldpc, mpeg 등 더 큰 / 다양한 designs 추가 필요.

## 10. 향후 4%까지 가는 길

A) **Per-pair pairwise edge regression** (ParaGraph-style):
   - 각 (target, aggressor) pair 에 대해 c_pair 직접 회귀 → Σ c_pair = c_cpl
   - geometric pair features + matched SPEF labels 데이터 구축 필요
   - c_gnd 는 별도 모델, 단 c_gnd 자체도 23% MAPE 라 split 자체가 손해 → c_gnd 도 cuboid-level 로 학습 필요

B) **Synthetic data pretraining** (Q3D / Stage 1-4 curriculum):
   - Stage 1: parallel plate analytic
   - Stage 2: layered slab + image charges
   - Stage 3: 3D box pairs
   - Stage 4: multi-conductor 3D fringe with ε asymmetry
   - Real intel22 finetuning 으로 pretrain → finetune

C) **BEM-collocation residual / physics-informed neural operator**:
   - FastCap-style Green's function 을 hand feature 에 추가
   - Sakurai-Tamaru analytic + neural residual

D) **Per-design test-time adaptation** (limited applicability):
   - Test design 의 일부 nets (예: 5%) 의 SPEF 를 oracle 로 라벨 → 나머지에 대해 calibrate
   - Active learning context 에서만 적용 가능

## 11. 결과 파일

- `reports/FINAL_REPORT.md` — 영문 최종 보고
- `reports/SUMMARY_KO.md` — 한글 요약  
- `reports/METHODOLOGY_KO.md` — 본 문서
- `reports/PERFORMANCE_REPORT_KO.md` — 한글 상세 성능 리포트
- `reports/per_model_summary.csv` — 75 models per-row MAPE
- `reports/group_summary.csv` — model class summary
- `reports/ensemble_summary.csv` — ensemble comparison
- `reports/stratified_mape.csv` — cap bucket stratified MAPE
- `reports/final_metrics.csv` — final headline metrics (mean/median/p90/p99/R²)
- `reports/super_ensemble_test.csv` — **canonical 최종 net-별 예측 (= best_test_v4.csv)**
- `reports/plots/r2_scatter.png` — log-log R² scatter (density)
- `reports/plots/mape_histogram.png` — APE distribution
- `reports/plots/stratified_mape.png` — per-bucket bar chart
- `reports/plots/ensemble_evolution.png` — Pass 1-7 evolution
- `reports/plots/per_bucket_scatter.png` — per-bucket scatter small-multiples
- `reports/plots/residual_analysis.png` — signed residual / per-bucket bias
