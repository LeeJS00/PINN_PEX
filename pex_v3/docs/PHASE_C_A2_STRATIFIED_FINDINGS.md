# A2 Audit + B1 Stratified Report — 핵심 발견

_Date: 2026-05-02_
_Source: A2 classical-baseline-owner [ROLE PASS] + scripts/08_b1_stratified_report.py_

## TL;DR

**B1 XGBoost의 4.66% headline은 gnd/cpl 오차 상쇄에 따른 환상** —
per-channel reality는 gnd 20.6% / cpl 12.4%. Phase 1 paradigm은 per-channel
honesty (β strategy)를 목표로 해야 paper-grade contribution이 됨.

## A2 검증 결과 (B1 정직성)

| 검증 항목 | 결과 |
|---|---|
| MAPE 재계산 (12594 rows) | seed0 median 4.6430% — **metrics_row.csv와 정확 일치** (no eps clamp gaming) |
| Zero-target degeneracy | 0 rows with `\|golden_total\|<1e-9` (clean) |
| 5 random spot-checks | APE 1.76% / 4.24% / 5.55% / 5.71% / 10.56% — sane mix |
| `(design,net)` 누수 (train↔valid) | **0** (manifest groupby `nunique(split)>1` empty) |
| Manifest sha256 in provenance | `a18142d0…d5d1888` 라이브 매니페스트와 일치 |

**판정**: B1의 4.66% in-dist는 **leakage 없음, feature가 in-dist 신호를 진짜로 캡처**.
OOD gap (4.66 → 7.48)은 **distribution shift**, NOT overfit.

## 채널-레벨 진실 (Stratified Report 결과)

```
B1 seed 0 on v3 valid (12,594 nets):

  Channel         Median    Mean     P95
  ----------------------------------------
  gnd_fF only:    20.57%    27.81%   83.97%   ← 실제 ground cap 어려움
  cpl_fF only:    12.38%    16.65%   44.75%   ← 실제 coupling cap 어려움
  total_fF:        4.64%     5.95%   15.78%   ← cancellation 효과
```

→ **headline 4.64%는 gnd 양수 오차 + cpl 음수 오차의 partial cancellation**
   에서 비롯. 진짜 채널별 정확도는 ~4× 낮음.

## Per-design 분포

| design | cpl mdn | gnd mdn | total mdn |
|---|---:|---:|---:|
| intel22_aes_cipher_top_f3 | 11.79 | 22.20 | 4.35 |
| intel22_gcd_f3 | 20.93 | 13.12 | 6.38 |
| intel22_ibex_core_f3 | 14.43 | 19.20 | 5.32 |
| intel22_ldpc_decoder_802_3an_f3 | 11.38 | 22.60 | 4.22 |
| intel22_mc_top_f3 | 14.03 | 15.43 | 5.68 |
| intel22_spi_top_f3 | 11.96 | 18.61 | 6.29 |
| intel22_usbf_top_f3 | 17.67 | 16.25 | 6.51 |
| intel22_vga_enh_top_f3 | 14.71 | 15.59 | 5.57 |
| intel22_wb_conmax_top_f3 | 11.32 | 23.40 | 4.40 |

- gnd: 13-23% 범위 (largest design ldpc + wb_conmax 최고 ~22-23%)
- cpl: 11-21% 범위 (gcd 가장 어려움 — small design, sparse aggressors)
- 모든 design에서 total < min(gnd, cpl) → 일관된 cancellation

## Per-quartile heteroscedastic 패턴

```
quartile (by golden_total_fF):
  Q4 (>5fF, 2453 nets):   gnd 17.6%  cpl  8.0%  total 3.42%   ← 큰 net 가장 정확
  Q3 (0.5-5fF, 4477):     gnd 21.2%  cpl 12.3%  total 4.45%
  Q2 (0.05-0.5, 5662):    gnd 21.3%  cpl 15.5%  total 5.65%
  Q1 (<0.05fF, 2 nets):   nonsensical (n=2, 무시)
```

**좋은 방향의 heteroscedasticity**: 큰 net (CTS, power) 가장 정확,
작은 net 가장 부정확. Phase 1이 small-net 정확도를 올리면 큰 절댓값 개선.

## Phase 1 contribution narrative — 3 옵션 (A2 권고)

| Strategy | 정의 | 강점 | 약점 |
|---|---|---|---|
| α | total <4% AND test <6%, beat B1 p<0.01 | 단순 | XGBoost 대비 thin |
| **β** | **gnd <8% AND cpl <8%** (B1 20.6 / 12.4 능가) | **가장 강한 physics story** | ablation 명확 |
| γ | in-dist→OOD gap < +2.82pp | generalization claim | absolute number tie ok |

A2 추천: **β as primary, γ as secondary**. 단독 α는 XGBoost 대비 너무 얇음.

## A2의 B2/B4 권고

- **B2 ParaGraph 재현**: 5-day capped (no tuning). 25-35% 영역으로 떨어질 것;
  pre-feature-engineering paradigm 베이스라인 컬럼.
- **B4 Compact + GAM**: 3-day (Sakurai features 이미 NetFeatureVector에 있음 —
  PHASE_B_B1_RESULTS.md line 95). (i) linear on Sakurai (1h), (ii) GAM on
  ~10 features (4h), (iii) GBDT residual (4h).

## 즉시 후속 작업

1. ✅ Stratified report 완료 (`08_b1_stratified_report.py`)
2. ⏳ Phase 1 hybrid 모델은 **per-channel loss**로 학습 — total만 fitting하면 cancellation 학습
3. ⏳ B4 (Sakurai+GAM) 3 days — Phase 1과 비교할 physics-floor anchor
4. ⏳ B2 (ParaGraph) 5-day capped — reviewer expectation

## 파일

- `pex_v3/scripts/08_b1_stratified_report.py` (재현 스크립트)
- `pex_v3/output/baselines/B1_xgboost_real/stratified_per_design.csv`
- `pex_v3/output/baselines/B1_xgboost_real/stratified_per_quartile.csv`
- A2 audit raw: `pex_v3/output/baselines/B1_xgboost_real/seed0/eval_predictions.csv`
