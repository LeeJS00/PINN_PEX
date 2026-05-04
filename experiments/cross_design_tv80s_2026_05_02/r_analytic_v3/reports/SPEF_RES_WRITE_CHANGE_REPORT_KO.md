# SPEF *RES Section 작성 정책 변경 보고서

_작성일: 2026-05-03 KST. v7 ML / v2 analytic / v3 hybrid 정책 변경 이력 + SPEF write 통합 가이드._

---

## 0. 요약

| 항목 | 변경 전 (v7 ML) | v2 analytic (배포 후보) | **v3 hybrid (best)** |
|---|---|---|---|
| Test MAPE (tv80s, 3380 nets) | 11.92% | 6.99% | **2.21%** |
| 학습 파라미터 | 47 models, ~10⁵ weights, 50 MB | 1 scalar α | 23 NNLS coefs + 5×500 + 3×300 LGBM trees (~30 MB) |
| Inference time/3380 nets | 5–7 s | <1 s | <2 s |
| 외부 의존성 | LightGBM, CatBoost | 없음 | LightGBM |
| Calibration 빈도 | per design retrain | 1회 (PDK 설치 시) | 1회 (PDK 설치 시) |
| SPEF *RES segment 형식 | lumped 1-line | lumped 1-line | lumped 1-line (변경 없음) |

**핵심 변경**: *RES section 형식은 그대로 (lumped 단일 segment). `total_r` 값을 만들어내는 **predictor**만 교체.

---

## 1. 현재 SPEF *RES 작성 흐름

`pex_pipeline/write_spef.py:96` — 단일 net 의 lumped R:

```python
# pex_pipeline/write_spef.py:LumpedSPEFWriter._format_net
if total_r > 0:
    res_section = f"\n*RES\n1 {name}:1 {name}:2 {total_r:.6g}\n"
else:
    res_section = ""
```

`total_r` 의 출처:

```python
# scripts/predict_spef_e2e.py:191-207
R_analytic = total_resistance_for_design(cubarr_path)   # ← v2 analytic baseline

if (args.models_dir / "total_r").exists():
    r_pred = predict_total_r(feat, args.models_dir / "total_r")   # ← v7 ML override
else:
    r_pred = np.array([R_analytic.get(n, 0.0) for n in feat["net_name"]])
```

→ `total_r` 가 LumpedSPEFWriter 에 dict 로 전달되고, *RES 한 줄에 기록.

**현재 파일/구성 의존성**:

| 컴포넌트 | 파일 | 역할 |
|---|---|---|
| Analytic R fallback | `pex_pipeline/compute_resistance.py` | 잘못된 sheet R + brute-force `R_CALIBRATION_SCALE=3.5` |
| ML R predictor | `pex_pipeline/predict_caps.py:predict_total_r` | 5 LGBM + 5 CatBoost + 10-mdl stratum |
| Trained weights | `output/spef_e2e/total_r/{lgbm_seed*,cat_seed*,stratum_weights}` | 50 MB |
| SPEF writer | `pex_pipeline/write_spef.py:LumpedSPEFWriter` | per-net lumped *RES line |

---

## 2. 변경 이력

### Phase A — v2 analytic (이미 합의)
참조: `reports/R_ANALYTIC_POLICY_KO.md`

`compute_resistance.py` 의 잘못된 default sheet R (m1=1.5, m2-5=0.42) 와 brute-force scale (3.5) 을 calibrated 값으로 교체:

```python
sheet_R = {"M1": 0.713, "M2": 0.583, "M3": 0.600, "M4": 0.600, "M5": 0.587}
R_via   = {"v1": 11.61, "v2": 13.07, "v3": 13.07, "v4": 13.07}
α       = 1.4777   # global scalar from train designs
```

`R_pred = α × (Σ sheet × L/W + Σ R_via × n_via)` — 6.99% MAPE.

### Phase B — v3 NNLS linear (3.30%)
모든 features 를 NNLS-IRLS 로 동시에 fit:

- `nsq_M{i}` (per-layer L/W sum, i ∈ {1..5})
- `rsq_M{i}` (RECT landing — 결국 ~0 coefficient)
- `nvian_<NAME>` (per DEF-VIA-name count)
- `n_pin_inst` (cell-pin count, captures pin attachment R)
- `one` (per-net intercept)

physically-meaningful coefficients (sheet R, per-via R) + per-net overhead. `pred = X @ c`.

### Phase C — v3 hybrid (2.46%)
1단계 NNLS → 2단계 LightGBM ensemble (5 seeds, depth 4) on relative residual `(R − R_lin) / R_lin`. Per-design 1-hot dummy 추가.

### Phase D — v6 cell-OBS + cell-SIZE features (2.25%)
`/home/jslee/projects/PINNPEX/tool/pdk/22nm/cell_lef/b15_nn.lef` 에서 추가 추출:
- `OBS` 섹션 → cell internal M1/M2 routing의 nsq/area/count
- `MACRO ... SIZE w BY h` → cell footprint
- `PIN ... DIRECTION` → input/output pin 분류
- VCC/VSS pin port area 를 OBS area 에서 차감 → **signal-internal OBS** (power-rail 노이즈 제거)

per-net feature aggregation: signal net 의 모든 (inst, pin) tuple 을 순회하며 cell type 별 OBS/SIZE 합산.

### Phase E — v3 hybrid (final, 2.21%)
v6 features + Stage 3 stacking (residual-of-residual GBT, 3 seeds).

| 단계별 진전 | tv80s test MAPE |
|---|---|
| compute_resistance.py 기존 default+scale | 18.58% |
| **v2 analytic** (calibrated, +α) | **6.99%** |
| v3 NNLS linear | 3.30% |
| v3 hybrid (v4 features) | 2.46% |
| v3 stacked (v4 features) | 2.44% |
| v3 hybrid (v6 cell-OBS + SIZE) | 2.25% |
| **v3 stacked (v6, FINAL)** | **2.21%** |

---

## 3. 제안 변경 (SPEF write 통합 경로)

### 3.1 `pex_pipeline/compute_resistance.py` 교체
- 기존 `DEFAULT_SHEET_R_INTEL22` / `R_CALIBRATION_SCALE=3.5` 삭제
- `r_analytic_v3/outputs/v6_stage3_summary.json` 의 calibrated 상수 + 학습된 LGBM ensemble 로드
- API 유지: `total_resistance_for_design(cubarr_path) → Dict[net_name → R]`

```python
# new compute_resistance.py
import json, lightgbm as lgb
from pathlib import Path
from .feature_extractors import (
    nsq_per_layer, via_count_per_name,    # PINNPEX DefStreamParser 활용
    pin_count, cell_size_sum,
    cell_obs_signal_per_layer,             # cell LEF OBS - power-pin filtered
)

_HERE = Path(__file__).parent
_CALIB = json.load((_HERE / "r_calibration_v6.json").open())

_LIN_COEFS = _CALIB["nnls_coefs"]
_LGB_S2 = [lgb.Booster(model_file=str(_HERE / f"r_lgbm_s2_seed{s}.txt")) for s in range(5)]
_LGB_S3 = [lgb.Booster(model_file=str(_HERE / f"r_lgbm_s3_seed{s}.txt")) for s in range(3)]
_FCOLS_LIN = _CALIB["lin_features"]
_FCOLS_S2  = _CALIB["s2_features"]   # incl per-design 1-hot (test = uniform mean)
_FCOLS_S3  = _CALIB["s3_features"]


def total_resistance_for_design(cubarr_path):
    feats = build_per_net_features(cubarr_path)         # DataFrame indexed by net_name
    X_lin = feats[_FCOLS_LIN].values
    R_lin = X_lin @ np.array([_LIN_COEFS[c] for c in _FCOLS_LIN])

    X_s2 = build_s2_matrix(feats, _FCOLS_S2)            # incl uniform 1/N_designs 1-hot
    z2 = np.mean([gb.predict(X_s2) for gb in _LGB_S2], axis=0)
    R_s2 = np.maximum(R_lin, 1e-3) * (1.0 + z2)

    X_s3 = build_s3_matrix(feats, _FCOLS_S3, R_lin, R_s2)
    z3 = np.mean([gb.predict(X_s3) for gb in _LGB_S3], axis=0)
    R_final = R_s2 * (1.0 + z3)
    return dict(zip(feats["net_name"], R_final.tolist()))
```

### 3.2 LumpedSPEFWriter 변경 없음
`*RES\n1 {name}:1 {name}:2 {total_r:.6g}\n` 형식 그대로. 단 `total_r` 의 정확도가 11.92% → 2.21% 로 5.4× 개선됨.

### 3.3 `predict_spef_e2e.py` 의 fallback chain 단순화

현재:
```python
if (args.models_dir / "total_r").exists():
    r_pred = predict_total_r(feat, args.models_dir / "total_r")  # v7 ML
else:
    r_pred = np.array([R_analytic.get(n, 0.0) for n in ...])      # broken default
```

제안:
```python
# 단일 진입점, 항상 v3 hybrid 사용
r_pred = total_resistance_for_design(cubarr_path)         # v3 inside
# 이 함수는 calibration json + LGBM .txt 만 필요 — total_r/ ML dir 불필요
```

### 3.4 모델 파일 마이그레이션

| 삭제 / 보관 | 신규 |
|---|---|
| `output/spef_e2e/total_r/{lgbm_seed{0..4}.pkl, cat_seed{0..4}.cbm, stratum_weights.json}` (50 MB) | `pex_pipeline/r_calibration_v6.json` (3 KB) |
| `pex_pipeline/predict_caps.py:predict_total_r` (호출 제거, 함수 보존 or archive) | `pex_pipeline/r_lgbm_s2_seed{0..4}.txt` (~5 MB each) |
| | `pex_pipeline/r_lgbm_s3_seed{0..2}.txt` (~1 MB each) |

→ 디스크 ~50 MB → ~30 MB, 그러나 R 정확도 5.4× 향상.

### 3.5 신규 의존성

| 의존성 | 용도 | 위치 |
|---|---|---|
| `src/preprocessing/def_parser.py::DefStreamParser` | per-net wirelength + via count | 이미 PINNPEX 코어에 있음 |
| `src/preprocessing/cell_parser.py::CellLibParser` | cell pin geometry | 이미 있음 |
| (신규) `pex_pipeline/cell_obs_extractor.py` | OBS 섹션 + signal/power 분리 파싱 | `r_analytic_v3/scripts/parse_cell_sizes_and_pins.py` 코드 이식 |
| (신규) `pex_pipeline/feature_builder_v6.py` | per-net feature aggregation | `r_analytic_v3/scripts/build_features_v6_signal_obs.py` 이식 |
| `lightgbm` | LGBM .txt model 로딩 | 이미 있음 |

---

## 4. 검증

### 4.1 SPEF 출력 호환성

기존 SPEF 와 byte-level diff 시:
- `*D_NET <name> <total_cap>` — 변경 없음 (cap 정책 유지)
- `*CAP` 섹션 — 변경 없음
- `*RES` 섹션:
  - `1 {name}:1 {name}:2 <NEW_R>` — `<NEW_R>` 값만 변경
  - format `%.6g`, segment 토폴로지 동일 (lumped 1-segment)

→ downstream STA/timing tool 호환성 유지. 단지 R 값이 더 정확.

### 4.2 Cross-design generalization 검증

| Train/Val | Test | v2 MAPE | v3 hybrid MAPE | v3 stacked MAPE |
|---|---|---|---|---|
| 9 train designs (207K nets) → tv80s (3380 nets) | tv80s | 6.99% | 2.25% | **2.21%** |
| 9 design α stability | — | σ=0.95% (α 1.45-1.50 범위) | (NNLS coefs σ < 1%) | (LGBM bagging seeds σ=0.06%) |

### 4.3 잔여 한계 (1% 미달)

`r_analytic_v3/cache/def_vs_golden_diffs.parquet`:
- DEF parser 가 보지 못하는 cell-internal M1/M2 routing 평균 30 squares/net
- v6 의 signal-OBS 가 그 절반 (~96 squares/cell × per-pin fraction) 회복
- 잔여 ~15 squares/net 은 GDSII 기반 transistor-internal routing 영역 — DEF/LEF 만으로는 도달 불가

---

## 5. 단계적 deploy 계획

### Step 1 — `r_calibration_v6.json` lock (1 일)
- `r_analytic_v3/outputs/v6_stage3_summary.json` 의 NNLS coefs / S2 coefs / S3 coefs 를 `pex_pipeline/r_calibration_v6.json` 으로 이식
- LGBM model 5+3 = 8 개를 `.txt` 형식으로 저장 (`booster.save_model`)
- 인덱스 / metadata 포함 (cell LEF 의 hash, calibration 일자, train design list)

### Step 2 — `compute_resistance.py` rewrite (2-3 일)
- 위 §3.1 코드 적용
- `total_resistance_for_design` API 호환 유지
- unit test: tv80s 검증 → 2.21% MAPE 재현 확인

### Step 3 — `predict_spef_e2e.py` 단순화 (1 일)
- fallback chain 제거
- `total_r/` ML weights 의존 제거

### Step 4 — Old artifacts 정리 (1 일)
- `output/spef_e2e/total_r/` archive 폴더로 이동 (`output/_archive/total_r_v7_ML_legacy/`)
- `pex_pipeline/predict_caps.py` 의 `predict_total_r` 함수 deprecate (warning)

### Step 5 — 회귀 검증 (1 일)
- `scripts/spef_e2e/validate_e2e.py output/spef_e2e/tv80s_FINAL.spef` 재실행
- 기대: total_res MAPE 11.92% → 2.21%
- compare_spef.py 으로 byte-level header 호환성 확인

---

## 6. 위험 요소

| 위험 | 가능성 | 영향 | 대응 |
|---|---|---|---|
| LightGBM 의 .txt 모델이 다른 LGBM 버전에서 호환성 깨짐 | 낮음 | 중간 | `pip install lightgbm==4.6.*` 명시 |
| Cell LEF 변경 (PDK upgrade) → calibration stale | 중간 | 높음 | `r_calibration_v6.json` 에 `cell_lef_md5` 기록, mismatch 시 warning |
| 새 design 이 train 분포와 큰 distribution shift | 중간 | 낮음 | per-design 1-hot 의 test-time uniform encoding 으로 어느 정도 보호; 추가로 monitoring metric (predicted R 분포 vs train) |
| LGBM training infrastructure (sklearn/lightgbm) 부재 환경 | 낮음 | 중간 | Stage 1 NNLS-only fallback (3.30% MAPE, no GBT) 옵션 제공 |

---

## 7. 산출물

### 변경 대상 파일
- `pex_pipeline/compute_resistance.py` (rewrite, ~250 lines)
- `pex_pipeline/predict_caps.py` (predict_total_r 함수 deprecate)
- `scripts/predict_spef_e2e.py` (fallback chain 제거, ~10 lines)
- `pex_pipeline/__init__.py` (exports 정리)

### 신규 파일
- `pex_pipeline/r_calibration_v6.json` (~3 KB)
- `pex_pipeline/r_lgbm_s2_seed{0..4}.txt` (~5 MB each)
- `pex_pipeline/r_lgbm_s3_seed{0..2}.txt` (~1 MB each)
- `pex_pipeline/cell_obs_extractor.py` (~150 lines)
- `pex_pipeline/feature_builder_v6.py` (~200 lines)

### 변경 없음
- `pex_pipeline/write_spef.py` (LumpedSPEFWriter)
- SPEF *RES section format (`1 N:1 N:2 R`)
- downstream tool 호환성

---

## 8. 결론

`*RES` section 의 **format 은 변경 없음**. 단지 `total_r` 값을 만들어내는 predictor 가 v7 ML (11.92%) → v3 hybrid stacked (2.21%) 로 5.4× 정확도 개선되어 *RES 라인의 R 값이 정확해짐.

기존 SPEF 와의 byte-level 호환성 유지 (CAP / CONN / *D_NET 헤더 변경 없음). Downstream STA tool 의 timing 분석은 더 정확한 R 값을 받게 되어 자연히 향상.

코드 변경 범위는 좁음 (~5 파일, ~700 lines net change). 모델 weights 는 30 MB (기존 50 MB 보다 작음).

---

_참조 문서_:
- `reports/R_ANALYTIC_POLICY_KO.md` (v2 analytic 정책)
- `r_analytic_v3/reports/V3_RESULTS.md` (v3 hybrid 진전 timeline)
- `r_analytic_v3/outputs/v6_stage3_summary.json` (best calibration)
- `pex_pipeline/write_spef.py` (현재 SPEF writer)
- `scripts/predict_spef_e2e.py` (현재 진입점)
