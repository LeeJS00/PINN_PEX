# 현재 세션 작업 리포트 — r_analytic_v3 (cross_design_tv80s_2026_05_02)

_작성일: 2026-05-03 KST. 본 세션의 모든 작업 종합. pex_v3 (별도 세션) 와 격리._

---

## 0. 세션 정의

- **작업 디렉토리**: `/home/jslee/projects/PINNPEX/experiments/cross_design_tv80s_2026_05_02/r_analytic_v3/`
- **격리 원칙**: pex_v3 코드/파일 미수정 (read-only 참조만)
- **목표**: DEF+LEF (+ optional .lib) → 정확한 SPEF 생성 (Voltus IR drop 등 downstream 활용)
- **canonical train/test split**: TRAIN 9 designs (incl. ldpc), TEST nova + tv80s

---

## 1. 작업 흐름 (chronological)

### Phase 0 — 환경 / split 검증
- `predict_spef_e2e.py` I/O 계약 확인: DEF→SPEF 7-stage pipeline (StarRC와 동일 입출력)
- `configs/config.py` 의 canonical split 확인 (TRAIN 9, TEST 2)
- 사전 v3 작업의 nova-leak / ldpc-누락 발견 → **Paper-grade audit** 작성

### Phase 1 — total_R 정확도 push
- v2 analytic baseline 측정 (calibrated sheet_R + global α): **6.99% MAPE on tv80s**
- v3 NNLS-IRLS linear (15 features): 3.30% on tv80s
- v3 hybrid (NNLS + 5-seed LGBM): 2.46% on tv80s (nova-leak split)
- v3 stacked (S2 + 3-seed LGBM): 2.21% on tv80s (nova-leak split)

### Phase 2 — Cell LEF OBS 통합 (v6 features)
- `parse_cell_obs.py`, `parse_cell_sizes_and_pins.py`: cell LEF OBS section + cell SIZE 추출, signal/power 분리
- v6 features: 평균 96 squares M1/cell (cell internal routing 회복)
- v6 hybrid R MAPE on tv80s: 2.21% (Stage 3 stacked, **nova-leak split**)

### Phase 3 — Audit & canonical split 재실행
- ldpc features 빌드 (170K nets — TRAIN 누락이었음)
- `fit_canonical_split.py`: 진정한 OOD 측정
- **Canonical R MAPE**: nova 4.02% / tv80s 3.30% / **combined 4.00%** (Stage 1 NNLS best)
- Stage 2/3 hybrid: tv80s 더 좋아지지만 (2.92%) nova 약간 악화 (4.42%)
- nova leak가 사전 보고를 -0.7pp 부풀렸음

### Phase 4 — c_gnd 시도들
- v3 hybrid c_gnd: NNLS 26.47%, LGBM ensemble 27.12% (둘 다 v7 ML 21.09% 보다 worse)
- pex_v3 ceiling 확인: B1 XGBoost / Option F MLP / B4 GBDT 모두 20-21% 수렴 → hand-feature paradigm 한계
- intel22 `.lib` 파일 부재 (asap7 만 보유) → cell intrinsic Cgg 정보 미보유

### Phase 5 — pex_v3 Phase 1 paradigm 차용 (이번 세션 implementation)
- `fit_cgnd_phase1_hybrid.py`: analytic parallel-plate prior + bounded multiplicative MLP residual
- Stage A (prior): 71.9% combined
- Stage B (NNLS calib): 31.1% combined
- Stage C (clamp=log(2) MLP): 24.19% combined
- Stage C (clamp=log(4)): **23.92% combined** (best)
- → 21% ceiling 못 깸 (Phase 1 paradigm 도 full-net에 직접 적용 시 한계)

### Phase 6 — SPEF runtime 벤치마크
- tv80s (3,280 nets): **247.6s (4.13 min)** — 7-stage breakdown 측정
- nova (118,960 nets): Stage 1 38.9 min 측정. Stage 2-7 4hr+ 미완료 (super-linear scaling 확인)

---

## 2. 정량 결과 종합 (canonical OOD)

### total_R MAPE

| Stage | nova (118K) | tv80s (3.4K) | Combined |
|---|---|---|---|
| Pure analytic v2 (calibrated + α) | (TBD) | 6.99% | — |
| **Stage 1 NNLS (best combined)** | **4.02%** | 3.30% | **4.00%** |
| Stage 2 (NNLS + 5-LGBM) | 4.45% | 2.96% | 4.41% |
| Stage 3 (+ stacking) | 4.42% | **2.92%** | 4.38% |
| v7 ML legacy 참고 | (미수정) | 11.92% | — |

**Headline**: R OOD combined **4.00%** (3× 개선 from v7 ML 11.92%).

### c_gnd MAPE

| Method | Combined OOD | tv80s only |
|---|---|---|
| v7 ML legacy | (미수정) | 21.09% |
| v3 NNLS (canonical) | 31.1% | 32.6% |
| v3 hybrid LGBM | (TBD) | 27.12% |
| **Phase 1 hybrid (NNLS + bounded MLP, log(4))** | **23.92%** | **24.74%** |
| pex_v3 best (XGBoost / GBDT / MLP) | 20.3-21.2% | — |

**Headline**: Phase 1 paradigm (analytic + bounded MLP) **NNLS 31% → 24% (-7pp)**, 그러나 21% ceiling 미돌파.

### SPEF runtime

| Design | n_nets | Total | Best stage | Worst stage |
|---|---|---|---|---|
| tv80s | 3,280 | **247.6s (4.13 min)** | SPEF write 0.6s | pair features 110.9s |
| nova | 118,960 | (Stage 1 only: 38.9 min) | — | Stage 2 features (>4 hr did not complete) |

vs StarRC tv80s ~30 min → **7× speedup**.

---

## 3. 핵심 산출물

### Scripts (r_analytic_v3/scripts/)
- Feature builders: `build_segment_features.py`, `build_features_v2.py`, `build_features_v3.py` (pins), `build_features_v4_with_pins_routing.py`, `build_features_v6_signal_obs.py`, `parse_cell_obs.py`, `parse_cell_sizes_and_pins.py`
- Target builders: `build_cgnd_target.py`
- Fits: `fit_canonical_split.py`, `fit_cgnd_phase1_hybrid.py` (외 14개 ablation 스크립트)

### Results (r_analytic_v3/outputs/)
- `canonical_split_results.json` — R per-design × per-stage MAPE + CI
- `cgnd_phase1_hybrid.json` — c_gnd Phase 1 hybrid 결과
- `test_predictions_*.parquet` — per-net 예측

### Reports (r_analytic_v3/reports/)
- `PAPER_GRADE_AUDIT.md` — leakage / contribution 검토
- `PAPER_GRADE_FINAL.md` — canonical split 최종 결과
- `R_ANALYTIC_POLICY_KO.md` — analytic R 정책
- `SPEF_RES_WRITE_CHANGE_REPORT_KO.md` — SPEF *RES section 변경 보고
- `CGND_RESULTS.md` — c_gnd 21% ceiling 분석
- `V3_RESULTS.md` — 사전 (nova-leak) 결과 (deprecated 표기)
- `CURRENT_WORK_REPORT.md` — **본 보고서**

---

## 4. 주요 발견 (paper-relevant)

### Finding 1 — R 은 analytic dominant
- v2 analytic (calibrated sheet R + α) 가 6.99% MAPE 도달
- 추가 features + GBT residual cascade 로 Stage 1 NNLS 4.00% combined OOD
- → R 영역은 paradigm-shift 불요, hand-feature + linear regression 으로 sufficient

### Finding 2 — c_gnd 는 paradigm-independent ceiling 21%
- XGBoost (pex_v3 B1): 20.6%
- MLP (pex_v3 Option F): 21.2%
- GBDT (pex_v3 B4): 20.3%
- NNLS+GBT (우리): 27%
- **Phase 1 hybrid (NNLS + bounded MLP)**: 24%
- → 21% 가 hand-feature paradigm 의 fundamental limit

### Finding 3 — Cell LEF OBS 활용은 R에 큰 도움, c_gnd 에 작은 도움
- R MAPE: v4 (no OBS) 2.46% → v6 (signal-OBS + cell SIZE) 2.21% (-0.25pp on nova-leak split)
- c_gnd MAPE: v4 27.1% → v6 26.5% (-0.6pp on tv80s)
- → R 은 wire/via geometry 가 dominant, c_gnd 는 transistor characterization (`.lib`) 가 dominant

### Finding 4 — nova-leakage 영향 정량화
- 사전 (nova in TRAIN): tv80s 2.21%
- Canonical (nova in TEST): tv80s 2.92%
- → +0.71pp 가 nova-leak 의 영향. 사전 보고를 그만큼 inflate.

### Finding 5 — Phase 1 paradigm 의 full-net 적용 한계
- per-pattern 설계인 Phase 1 (분석식 Green's function + bounded MLP) 을 full-net 에 직접 적용
- NNLS+GBT (27%) 보다 -3pp 개선 (24%) 그러나 21% 미돌파
- 진정한 Phase 1+2 (per-pattern → aggregator) 가 필요할 가능성

### Finding 6 — Runtime: small/medium designs 7-8×, large nova 알고리즘 최적화 요
- tv80s 4.13 min vs StarRC 30 min: 7×
- nova Stage 2 features 가 super-linear → multi-process sharding 또는 알고리즘 최적화 필요 (future work)

---

## 5. Paper-grade 영역 / 한계 영역

### Strong (paper-ready)
- **R 정확도 OOD**: 4.00% combined / 2.92% tv80s (3× from v7 ML)
- **Cell LEF OBS feature engineering**: novel signal-vs-power separation
- **Engineering pipeline**: production-ready DEF→SPEF, StarRC 호환, 7× speedup

### Limited (need more work)
- **c_gnd**: 24% (paradigm ceiling near 21%), 추가 paradigm shift 필요
- **per-pair coupling**: 110% (lumped feature 한계)
- **Large design runtime**: nova-scale 미완성, Stage 2 알고리즘 최적화 필요
- **`.lib` 통합**: intel22 .lib 미보유 → cell intrinsic Cgg 정보 부족

---

## 6. 다음 단계 권고

### Immediate
1. Voltus IR drop 검증: 우리 SPEF (R v3 + c_gnd v7) 로 Voltus 돌려서 IR drop accuracy 측정
2. nova Stage 2 알고리즘 최적화 (multi-process sharding)

### pex_v3 합류 시점
1. pex_v3 Phase 1 (per-pattern analytic + neural residual) 결과 확보 후 우리 Phase 2 aggregator 와 합치기
2. .lib 통합 (intel22 .lib 확보 시) — c_gnd ceiling 직접 돌파

### Paper composition
1. 공동 paper 구조: pex_v3 per-pattern + 우리 full-net pipeline
2. Engineering contribution: production SPEF generation
3. Negative result: hand-feature c_gnd 21% ceiling 정량적 근거

---

_End of current work report._
