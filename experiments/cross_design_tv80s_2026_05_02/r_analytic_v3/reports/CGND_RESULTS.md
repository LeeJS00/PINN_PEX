# c_gnd v3 hybrid 도전 — pex_v3 21% ceiling 비교

_작성일: 2026-05-03 KST. IR drop 정확도의 직접 driver인 c_gnd 정확도 도전 결과._

---

## 결론

| 정책 | tv80s c_gnd MAPE | bias | 비고 |
|---|---|---|---|
| 기존 v7 ML (47 models) | 21.09% | +1.30% | 기존 정책 (참조) |
| **v3 Stage 1 NNLS (v6 features) — best** | **26.47%** | **-3.07%** | linear + cell OBS + cell SIZE |
| v3 Stage 2 LGBM (10 seeds, small trees) | 27.12% | +0.10% | direct y, L1 + 1/y weight |
| v3 Stage 3 blend (linear + LGBM) | 27.12% (α=0) | — | LGBM은 val에서 overfit → blend 효과 없음 |
| **pex_v3 B1 XGBoost (참고)** | **20.6%** | — | 11 designs / 1.32M tiles |
| pex_v3 Option F MLP (참고) | 21.2% | — | 286K params |
| pex_v3 B4 V3 GBDT (참고) | 20.3% | — | compact + multiplicative |

**핵심 발견**:
1. 우리 best (26.47%) > 기존 v7 ML (21.09%) → **이 세션의 v3 hybrid는 c_gnd에서 v7 ML 대비 worse**
2. v7 ML이 21.09%이고 pex_v3가 20.3-21.2%인데, 우리 v3 v6 features는 21% 깨지 못함
3. pex_v3의 결론과 일관: **hand-feature ceiling은 ~21%** — paradigm shift (analytic Green's function + neural residual) 필요

## 왜 R 에서는 잘 되었는데 c_gnd 에서는 안 되었는가

| 측면 | total_R | c_gnd |
|---|---|---|
| 분포 | tight (median 121Ω, mean 250Ω) | heavy-tail (median 0.2fF, max 14fF) |
| 물리 | wire L/W + via R count → analytic 형식 | wire-to-substrate fringe + cell intrinsic Cgg |
| 정보 | DEF + sheet R + via R 만으로 충분 | **cell intrinsic gate cap (.lib 의 pin_capacitance)** 필요 |
| Cell internal | OBS metal routing 으로 capture 가능 | gate cap 은 transistor-level, OBS 무관 |
| 결과 | v3 hybrid 2.21% MAPE | v3 hybrid 26.47% (v7 ML 21.09%) |

R 은 LEF + DEF 만으로 분석식 + ML hybrid 가 작동. c_gnd 는 본질적으로 **transistor characterization (`.lib` pin_capacitance)** 가 dominant — 우리는 intel22 .lib 미보유.

## Stage 1 NNLS 진단 (best v3 결과)

### 학습된 coefs (active)

```
nsq_M2:    0.0028  (per-square wire-to-substrate cap of M2, fF/sq)
nsq_M3:    0.0079
nsq_M4:    0.0040
nsq_M5:    0.0020
n_pin_inst: 0.0312  (per-pin overhead, captures cell intrinsic cap)
v6_obs_signal_nsq_M1: 0.0009  (cell-internal M1 cap)
v6_obs_signal_area_M1: 0.0  (NNLS pushed to 0)
v6_cell_area_sum: 0.0001  (gate cap proxy)
one: 0.0089  (per-net intercept)
```

→ NNLS 가 cell SIZE 와 OBS 를 약하게나마 사용. 그러나 pex_v3 의 21% 도달 못함.

### Length-stratified MAPE (final blend on tv80s)

| Stratum | n | c_gnd_med | MAPE | bias |
|---|---|---|---|---|
| Q1 (smallest) | 845 | 0.057fF | 34.7% | +22.6% (severe over-pred) |
| Q2 | 845 | 0.130fF | 25.8% | +3.0% |
| Q3 | 845 | 0.314fF | 23.1% | -3.5% |
| Q4 (largest) | 845 | 1.168fF | 24.9% | -21.7% (severe under-pred) |

전형적 **regression-to-mean** 패턴: 작은 net 과대평가 / 큰 net 과소평가. heavy-tail 분포의 hand-feature 한계.

## Ablation 진전 (Stage 1 NNLS, target c_gnd)

| 단계 | features | active | train MAPE | test MAPE |
|---|---|---|---|---|
| v4 baseline (wire+via+pins) | 15 | 7 | 30.05% | 28.37% |
| + v6 signal OBS | 21 | 10 | 27.47% | **26.47%** |
| + v6 OBS + cell SIZE | 24 | 10 | 27.47% | 26.47% (no improvement) |
| + all v6 + segment counts | 29 | 10 | 27.47% | 26.47% (plateau) |

**v6 signal OBS 가 -1.9pp 개선 (28.4% → 26.5%)**. 추가 features (cell SIZE, sumL 등) 은 NNLS 에서 무의미.

## Stage 2 LGBM 시도들 (모두 실패)

1. **Residual LGBM (relative residual)**: 30-35% test (Stage 1 보다 worse)
   - 이유: heavy-tail 분포에서 (y - pred) / pred 발산
2. **Log-target LGBM**: 33-34%
   - 이유: log-target → exp() 시 geometric mean bias
3. **Direct y, L1, 1/y weight, large trees**: 31% (cfg2 기준)
   - 이유: overfit
4. **Direct y, L1, 1/y weight, small trees, 10-seed ensemble**: **27.12%**
   - val MAPE 23%, test MAPE 27% → distribution shift
   - blend α = 0.0 (val 에선 LGBM 이 좋아 보이지만 test 에서 overfit 확인)

→ **Stage 1 NNLS 26.47%가 이 hand-feature setup 의 진짜 한계**.

## IR drop 영향 평가

c_gnd MAPE 가 IR drop 에 미치는 영향:
- IR_drop = current × R_path
- current ≈ C × dV/dt × switching_activity
- C ≈ c_gnd (dominant for switching current)
- → c_gnd MAPE 26.47% → IR drop MAPE ~26% (linear propagation)

기존 v7 ML (21.09%) 에서 우리 v3 (26.47%) 로 가면 **IR drop 정확도가 worse**. 따라서 c_gnd 만큼은:
- **v7 ML 21.09% 정책을 유지**하는 것이 IR drop 관점에서 더 정확

## pex_v3 결과와의 관계

pex_v3 (다른 세션, 11 designs, 1.32M tiles):
- B1 XGBoost: 20.6% c_gnd
- Option F MLP (286K params): 21.2%
- B4 V3 log-GBDT: 20.3%

→ 3개 hand-feature 방법 모두 21% 수렴. **Hand-feature ceiling 확정**.

pex_v3 의 결론: Phase 1 hybrid (analytic layered Green's function + bounded neural residual) 가 paradigm-shift 후보. 우리 v3 hybrid 는 그 방향이 아니라 (ML residual on physics linear) 같은 paradigm 안에서의 ensemble — 따라서 같은 ceiling.

**Cell OBS + Cell SIZE 추가도 ceiling 못 깸**: 이 features 가 wire geometry 에서 못 보던 정보를 일부 주지만, **cell intrinsic gate cap** (transistor characterization, `.lib` 영역) 을 모르는 한 21% 가 floor.

## 권고

### 단기 (이번 세션 종료)
1. **c_gnd 는 v7 ML 21.09% 정책 유지** — 우리 v3 가 worse
2. **R 만 v3 hybrid (2.21%) 채택** — analytic 영역, ML hybrid 효과 큼
3. SPEF *RES write 는 v3 (개선), *CAP write 는 v7 ML (유지)

### 중기 (다음 세션 또는 pex_v3 합류)
1. **pex_v3 의 Phase 1** (analytic Green's function + neural residual) — paradigm shift
2. **`.lib` 통합** (intel22 .lib 가 추가 확보 시) — cell intrinsic Cgg 직접 사용
3. **Q3D synthetic pretraining** (pex_v3 docs 의 Stage 1-4 curriculum)

### IR drop 관점 우선순위 재정리
1. **R**: 우리 v3 (2.21%) 채택 → 5.4× 정확도 ↑
2. **c_gnd**: v7 ML (21.09%) 유지, paradigm shift 까지 대기
3. **total_cap**: c_gnd + c_cpl 합 (자동 따라옴)
4. **per-pair coupling**: 110% — 별도 push (낮은 우선순위, IR drop 영향 중간)

## 산출물

| 파일 | 내용 |
|---|---|
| `r_analytic_v3/cache/cgnd_<design>.parquet` | per-net c_gnd_gold targets |
| `r_analytic_v3/scripts/build_cgnd_target.py` | golden SPEF c_gnd 추출 |
| `r_analytic_v3/scripts/fit_cgnd_v3_hybrid.py` | Stage 1+2+3 residual approach (실패) |
| `r_analytic_v3/scripts/fit_cgnd_direct.py` | direct y / log target / small trees (best) |
| `r_analytic_v3/outputs/cgnd_direct_summary.json` | 최종 결과 |
| `r_analytic_v3/outputs/test_predictions_cgnd_direct.parquet` | per-net 예측 |
| **`r_analytic_v3/reports/CGND_RESULTS.md`** | **본 보고서** |

---

## 결론 요약

| 지표 | 변경 전 (v7 ML) | 우리 v3 | 권고 |
|---|---|---|---|
| **total_R** | 11.92% | **2.21%** ✅ | v3 채택 |
| **c_gnd** | **21.09%** ✅ | 26.47% (worse) | v7 유지 |
| **total_cap** | 8.11% | (미시도) | v7 유지 |

R 은 analytic dominant → v3 hybrid 가 큰 이득. c_gnd 는 transistor characterization dominant → 같은 paradigm 안에서는 21% ceiling. **paradigm shift 필요한 영역.**

IR drop 정확도를 위한 다음 단계는 **pex_v3 의 Phase 1 hybrid analytic + neural residual 결과 대기** 또는 **.lib 통합**.
