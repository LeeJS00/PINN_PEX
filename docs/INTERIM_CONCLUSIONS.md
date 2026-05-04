# PINN-PEX 중간 결론 종합 (Interim Conclusions)

_작성일: 2026-05-01_
_상태: 모든 실험 트랙 종료, baseline으로 복귀_

---

## 0. TL;DR

지난 며칠간 PINN-PEX의 잔여 오차를 줄이려고 시도한 **3개 트랙 모두
유의미한 효과 없음**으로 결론. 코드는 baseline으로 복귀하고, 향후
다른 방향 탐색 전에 본 문서로 학습한 것을 정리한다.

| 트랙 | 결과 | 상태 |
|---|---|---|
| **DS-PINN (MacroDensityFNO)** | 효과 없음 | 폐기 |
| **Data-driven calibration init** | mean MAPE −6.5pp이지만 ns (n=5) | 폐기 |
| **γ scaling head** | 단일 smoke만 측정, 5-seed 중단 | 폐기 |

**남기는 것**: 5-seed measurement protocol, OOD evaluation script,
critical analysis 방법론, 모든 측정 보고서.

**버리는 것**: DS-PINN macro stream 코드, calibration JSON loader,
γ head 구현, 모든 hand-tuned ζ values, calibration_init.json files.

---

## 1. 시도했던 트랙들

### 1.1 DS-PINN (MacroDensityFNO + flux head conditioning)

**가설**: CPL 6.5× under-prediction은 long-range field-solve missing
때문이며, per-cuboid에 macro PDN screening context (FNO 2D)를 추가하면
해결된다.

**결과**: `dspinn_v1`/`v2`/`v3` 학습에서 v10b baseline (MAPE 27.30%)
대비 개선 미미 또는 오히려 악화. v3 (β + ζ 추가)도 MAPE 35-50%대
유지.

**원인 추정**:
1. Macro feature가 GND를 hijack (v2 P2의 `z_macro_gnd.detach()`로
   완화 시도했으나 부분적)
2. CPL physics base 자체가 부족 (Sakurai-Tamaru 로컬 공식의 한계)
3. proj_out zero-init에서 FNO blocks가 학습 신호 부족

**최종 결정**: **DS-PINN 효과 없음**. NeuralFluxRouter를 1-hop GNN +
surface physics만 남기고 macro stream 완전 제거 (현재 신규 baseline).

### 1.2 Data-driven Calibration Init (NNLS extraction from SPEFs)

**가설**: v3의 hand-tuned ζ (`softplus_inv(8.0)` diag, `softplus_inv(5.0)`
off-diag for `cpl_layer_pair_log_scale`; per-layer hardcoded ρ for
`layer_scale_phys_gnd`) 값을 TRAIN_SPEFS에서 NNLS로 fit한 값으로
바꾸면 heteroscedastic calibration 개선.

**구현**:
- `src/data/calibration_extractor.py`: phase 1 (geometry+SPEF) +
  phase 2 (model physics-only forward)
- `src/data/calibration_solver.py`: joint NNLS with 10 metal-layer
  buckets (collinearity 회피)
- 빌드 중 2개 critical bugs 발견 + fix:
  1. `A_tgt` (name mask) ≠ `is_target` (channel 7) → A_primary 20%
     overcounting
  2. 29 z anchors collinearity → ρ oscillating (0/30/0/16) → 10
     buckets로 collapse

**결과 (5-seed protocol, n=5/variant)**:

| Variant | Mean MAPE | IQR | OOD MAPE | Slope |
|---|---|---|---|---|
| v3_baseline (hardcoded ζ) | 61.86 | 9.80 | 0.553 | 0.525 |
| **v4_full_calib (NNLS)** | **55.34** (-6.5pp) | **4.52** (절반) | **0.459** (-9.4pp) | 0.510 |
| v5_gnd_only (NNLS ρ + hardcoded CPL) | 59.05 | 8.85 | 0.535 | 0.517 |

**Mann-Whitney U test (모든 비교 ns)**:
- v3 vs v4: p=0.222
- v3 vs v5: p=0.548
- v4 vs v5: p=0.690

**결론**:
- Effect size d ≈ 0.85-0.92 (large) 이지만 n=5 검정력 부족
- IQR 절반은 reproducibility 가치
- **그러나 heteroscedastic motivation 미해결** (slope 0.51 → 0.51)
- 최종 판단: **충분치 않음**. baseline으로 복귀.

### 1.3 γ Scaling Head (per-net multiplicative correction)

**가설**: Heteroscedastic은 per-net residual 현상이므로, layer-wide
ρ로는 못 풀고 per-net γ scaling이 필요.

**구현**:
- `src/models/gamma_head.py`: 14-dim features (size, area, layer dist) →
  3-layer MLP → exp(clamp([-2, 2])) per-net γ
- Codex 검토 후 4개 권고 적용 (γ_gnd만, warmup schedule, 0.1× LR,
  identity regularizer)

**결과**:
- Smoke test (단일 seed): step 1000 BEST MAPE 67.24%
  (v3 step 1000 median 82.66 대비 promising)
- 5-seed step 1000 BEST 분포: median 84% (smoke의 67%는 운 좋은 케이스)
- **5-seed step 5000까지 측정 못함** — 사용자가 전체 트랙 종료 결정

**결론**: 단일 smoke만으로 판단 불가. **측정 미완 상태로 폐기**.
나중에 재시도할 수 있으나 우선순위 낮음.

---

## 2. 정량화된 효과 (5-seed에서 측정된 것)

### 2.1 Validation MAPE (in-distribution)

| Variant | Mean | Median | p25-p75 | Range | IQR |
|---|---|---|---|---|---|
| v3_baseline | 61.86 | 64.17 | 55.70-65.50 | 50.70-73.23 | 9.80 |
| v4_full_calib | 55.34 | 54.50 | 52.67-57.19 | 49.32-63.03 | 4.52 |
| v5_gnd_only | 59.05 | 60.40 | 53.56-62.41 | 48.78-70.08 | 8.85 |

### 2.2 OOD (TEST_DEFS: nova + tv80s)

| Variant | total MAPE | GND chip | CPL chip | slope | Pearson r |
|---|---|---|---|---|---|
| v3_baseline | 0.553 | 0.665 | 1.730 | 0.535 | 0.899 |
| v4_full_calib | 0.459 | 0.678 | 1.544 | 0.534 | 0.885 |
| v5_gnd_only | 0.535 | 0.686 | 1.462 | 0.536 | 0.886 |

### 2.3 핵심 관찰

- **OOD MAPE ≈ in-dist MAPE** for all variants (overfit 없음)
- **v4가 모든 metric에서 marginally best** (mean MAPE, IQR, OOD)
- **모든 variant slope ≈ 0.5** — heteroscedastic 미해결
- **CPL chip ratio 1.4-1.7** (모든 variant over-predict)

---

## 3. Methodology — 학습한 것 (이게 진짜 자산)

### 3.1 N=1 fallacy 정량화

단일 seed run 비교는 stochastic noise에 압도된다.

**실증 데이터**:
- v3 5-seed best_mape range: 50.70-73.23 (22pp spread)
- 단일 seed 비교에서 22% improvement claim이 가능했던 이유 = noise

**최소 권장**: n=5 seeds + Mann-Whitney U test.
**6-9pp difference 검출**에는 n=10+ 필요.

### 3.2 OOD evaluation 필수성

In-distribution validation만 보면 transient artifact에 속는다.

**실증 데이터**:
- 단일 v4 iter0 best vs v3 best:
  - in-dist: v4 -22% (looks great)
  - OOD: v4 +5-8pp WORSE (looks bad)
  - → 잘못된 결론: "in-dist overfit"
- 5-seed 분포에서:
  - in-dist: v4 -6.5pp (median)
  - OOD: v4 -9.4pp (median)
  - → 진짜 결론: "v4 약간 더 robust", but ns

### 3.3 Critical Analysis 사전 수행

빌드 시작 전 자체 reviewer 시점에서 가능한 모든 약점 나열 → 후속
data로 검증. Reviewer가 지적할 게 우리가 이미 알고 있게 됨.

본 프로젝트의 critical assessment 항목들이 OOD/heteroscedastic
측정으로 100% 검증됨 (모두 우려대로 결과 나옴).

### 3.4 Codex deliberation loop

빌드 시작 전 1-3 round deliberation이 critical bugs 사전 발견에 필수.

**실증**:
- Multi-scale distillation plan: round 1에 6 critical bugs 발견 → 폐기 후 redesign
- Calibration extractor: round 1에 3 BUG + 3 WARNING → narrow scope
- γ head design: round 1에 4개 권고 → 즉시 반영

### 3.5 명명의 정직성

처음에 "distillation"으로 명명 → reviewer 시점에서 "live teacher
없음" 지적 → "Data-Driven Physics Calibration"으로 정정. **Overclaim
은 자체 검토에서 잡아야 paper review 통과 가능성 높음**.

---

## 4. 코드 자산 정리

### 4.1 보존 — Methodology 도구

| 파일 | 용도 | 가치 |
|---|---|---|
| `scripts/diag_quartile_heteroscedastic.py` | per-quartile ratio 측정 | high — 어떤 모델이든 적용 |
| `scripts/diag_ood_compare.py` | TEST_DEFS multi-ckpt 비교 | high — 5-seed protocol 필수 |
| `scripts/analyze_5seed.py` | log → distribution analyzer | high |
| `scripts/aggregate_5seed_eval.py` | multi-ckpt OOD/heteroscedastic | high |
| `scripts/run_5seed_remaining.py` | 5-seed launcher | medium |
| `scripts/diag_spef_unit_check.py` | SPEF C_UNIT 검증 | medium |

### 4.2 보존 — 데이터

| 위치 | 내용 | 보존 사유 |
|---|---|---|
| `output_intel22/active_learning/m5_*/best_model.pth` | 15 ckpts (v3/v4/v5 5-seed) | future analysis 필요 시 재참조 |
| `output_intel22/active_learning/m5_summary/*.csv` | 모든 통계 결과 CSV | reference data |
| `/data/PINNPEX/data/processed/intel22/calibration_init*.json` | NNLS extraction 결과 | 재실험 시 inputs |
| `/data/PINNPEX/data/processed/intel22/calibration_extract/phase{1,2}_net2k.pkl` | NNLS 입력 데이터 | 재extraction 회피 |

### 4.3 폐기 — 효과 없는 코드 (이미 user가 revert)

- `src/models/macro_density_fno.py` — DS-PINN macro stream
- `src/models/gino_enricher.py` — GINO (별도 트랙, 효과 없음)
- `src/models/gamma_head.py` — γ head (측정 미완)
- `src/data/calibration_extractor.py` + `calibration_solver.py` —
  NNLS pipeline (값들은 더 이상 init에 사용 안 함)

이 파일들은 git에 있으면 되살릴 수 있으니 삭제하지는 않는다.

### 4.4 신규 baseline (현재 상태)

```
flux_head.py:
  - layer_scale_phys_gnd: zeros (softplus(0) ≈ 0.693 fF/μm² uniform)
  - cpl_layer_pair_log_scale: zeros (softplus(0) ≈ 0.693 uniform)
  - DS-PINN macro_context 제거
  - γ head 제거
  - cpl_modifier MLP가 모든 magnitude correction 담당

neural_field.py:
  - DeepPEX_Model에서 macro_density_fno 제거
  - gamma_head 제거
  - GINO enricher 제거
  - 순수 1-hop GNN + surface physics

run_active_learning.py:
  - --use_dspinn, --use_gamma, --use_gino, --calib_path 제거
  - --seed, --max_iters, --steps_per_iter 유지 (5-seed protocol용)

configs/config.py:
  - SSL_USE_DSPINN, SSL_USE_GINO 제거
  - CALIBRATION_INIT_PATH 제거
```

---

## 5. 보고서 정리 (모두 living docs로 git 보존)

| 파일 | 역할 |
|---|---|
| `docs/INTERIM_CONCLUSIONS.md` (this) | 종합 중간 결론 — 가장 최신 |
| `docs/dspinn_development_log.md` | DS-PINN 트랙 living log (DS-PINN 결과 포함) |
| `docs/distillation_log.md` | Calibration 트랙 living log |
| `docs/calibration_track_report.md` | Calibration 종합 보고서 |
| `docs/distillation_effect_report.md` | Distillation 효과 정량화 보고서 |
| `docs/gino_*.md` | 이전 GINO 트랙 보고서 (참고용) |

---

## 6. 학습한 것 — 재현 가능한 인사이트

### 6.1 PINN-PEX의 본질적 한계

다음은 NEURAL_FIELD + 1-hop GNN + surface physics architecture의
한계로 보이며, calibration/macro/per-net scaling 등 **incremental
fix는 모두 6-10pp 수준**에 그친다:

1. **MAPE 50-65% floor**: v3, v4, v5 모두 mean MAPE 55-62 → 이 수준이
   현재 architecture의 floor일 수 있음
2. **Heteroscedastic slope ≈ 0.5**: 1-hop GNN으로는 long-range
   topology 효과를 못 잡음. per-cuboid local features만으로 net-scale
   property 표현 부족
3. **CPL over-prediction (chip ratio 1.5-1.7)**: edge-level Sakurai-
   Tamaru는 inherently noisy; aggregation 후 systematic bias 누적
4. **Outlier nets**: top-100 worst nets (LDPC, dense CTS)가 ~10pp
   MAPE 차지 — incremental fix로 안 풀림

### 6.2 Architecture 차원에서 진짜 fix가 필요

다음 방향은 incremental fix가 아닌 **fundamental redesign**:
- **Multi-hop GNN** (current: 1-hop): long-range coupling
- **Net-level encoder** (current: cuboid-level): per-net property
  encoding
- **Direct field solver in network** (current: Sakurai-Tamaru proxy):
  더 정확한 physics base
- **Outlier-aware loss** (current: uniform weighting): LDPC 같은
  outlier topology specific handling

### 6.3 Active Learning 자체의 가치

5-seed × 1 iter × 5000 steps에서 v3 mean MAPE 61.86%는 **AL 6 iter
× 12k steps**의 docs §3.x v2 trained MAPE 35%와 비교하면 **AL
multi-iter가 ~25pp 더 좋음**. 즉:

- AL multi-iter는 효과적
- Calibration init은 이걸 교체하지 못함 (단일 iter에서만 측정)
- 진짜 비교: v3 6-iter vs v4 6-iter → 시도 안 함 (시간 부족)

→ **결정적 비교 누락**. 향후 AL multi-iter에서 calibration이 효과
있는지 별도 검증 필요. 지금은 단일 iter 5-seed만 했음.

---

## 7. 향후 방향 추천

### 7.1 단기 (즉시 시도 가능)

1. **AL multi-iter 5-seed 비교** — 가장 누락된 측정. v3 vs v4의 6
   iter 종료 시점 MAPE가 진짜 contribution measure. 단 학습 시간
   매우 김 (각 ~24h × 5 seeds × 2 variants = 240 GPU-hour).
2. **다른 PDK 일반화** — asap7로 NNLS 재extraction + 학습. PDK-specific
   pattern인지 검증.

### 7.2 중기 (architecture 재설계)

3. **2-hop GNN으로 확장**: 1-hop의 한계 검증
4. **Net-level encoder 추가**: per-net summary embedding을 cuboid
   feature와 결합
5. **Outlier-aware sampling**: LDPC/CTS specific bucket으로 sampling
   가중치 조정

### 7.3 장기 (paradigm shift)

6. **Direct numerical solver hybrid**: PINN을 단독으로 보지 않고
   StarRC와 hybrid (예: PINN로 빠른 추정, 일부만 StarRC verification)
7. **Diffusion-based calibration**: noise-aware learning으로 outlier
   robustness

---

## 8. Closing

DS-PINN, calibration init, γ head — 3개 트랙 모두 효과 없음으로
결론. 그러나 무가치하지는 않다:

- **5-seed methodology 확립**: 향후 모든 PINN-PEX 실험 표준
- **Critical analysis loop 검증**: reviewer 시점 사전 검토 효과적
- **OOD evaluation 필수성 입증**: in-dist만 보면 wrong conclusion
- **N=1 fallacy 데이터로 입증**: 22pp spread = single-seed detection
  threshold
- **Negative-but-careful results**: 향후 같은 방향 시도자에게 sound
  baseline 제공

다음 시도는 **incremental fix가 아닌 architectural redesign**이
필요해 보인다.

---

_본 문서는 final summary. 새 실험 시작 시 본 문서 §6, §7 검토 필수._
