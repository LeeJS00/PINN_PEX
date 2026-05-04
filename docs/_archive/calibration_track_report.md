# PINN-PEX Data-Driven Physics Calibration — 종합 보고서

_작성일: 2026-05-01_
_저자 워크플로우: Claude Code + Codex (반복 deliberation)_

---

## 1. 요약 (Executive Summary)

PINN-PEX 모델의 heteroscedastic calibration 문제를 **offline data-driven
init**으로 해결하려는 시도와 그 한계를 정리한다.

**핵심 결론**:

1. **NNLS 기반 calibration init**은 v3 baseline 대비 mean MAPE를 6.5pp
   낮추고 IQR을 절반으로 줄였으나, **n=5 seeds에서 통계적 유의성
   미달** (Mann-Whitney p > 0.5).
2. **Heteroscedastic 문제 (slope ≈ 0.5)는 calibration init만으로 해결
   불가** — slope 0.51-0.53 모든 variant 동일.
3. **단일 seed N=1 비교의 22% improvement claim은 stochastic noise**였음
   (v3 5-seed best_mape range 50.7-73.2, IQR 9.8pp).
4. **OOD 결과는 5-seed로 reverse**: 단일 run에서 v4가 OOD 5-8pp WORSE로
   보였으나, 5-seed에서는 v4가 OOD 9.4pp BETTER (그러나 ns).
5. **γ head (per-net 보정) 진행 중**: smoke test에서 step 1000 BEST
   MAPE 67.24% 관측 (v3/v4 step 1000 대비 15-19pp 향상). 5-seed 측정
   대기 중.

**Methodology lesson**: AI/ML 실험에서 single-seed comparison은
stochastic noise에 압도된다. **5-seed × Mann-Whitney**가 최소 신뢰
수준이며, 6-9pp 차이를 검출하려면 n=10+ 필요.

---

## 2. 배경 (Background)

### 2.1 PINN-PEX의 잔여 오차

`docs/dspinn_development_log.md` § 3.4-3.6 에 따르면, v2-v3 변형 모두
다음 문제를 보임:

- **GND heteroscedastic calibration**: Pearson r = 0.85 (위치 잘 앎),
  slope = 0.6 (크기 못 맞춤). Quartile별 ratio:
  - Q1 (smallest, ≤0.41 fF): 1.58 (over-predict)
  - Q3+ (largest, ≥0.48 fF): 0.72 (under-predict)
- **CPL magnitude**: physics-only baseline에서 6.5x under-prediction.
  Sakurai-Tamaru 로컬 공식이 StarRC의 long-range field-solve를 못 잡음.
- **Outlier dominance**: top-100 worst nets가 ~10pp MAPE 차지 (특히
  LDPC decoder).

### 2.2 처음 시도했던 잘못된 접근 — Multi-Scale Distillation

**오류**: 원인 분석을 건너뛰고 "spatial localization missing"으로 가정.
SPEF의 per-node coords를 활용한 per-tile distillation을 plan.

**Codex 검토에서 발견된 문제** (3 round, 6 critical bugs):
- Tile origin이 corner가 아닌 center — 좌표 투영 오류
- Voxel aggressor merge 미러링 누락
- macro_distill_head (gnd, cpl, total)가 기존 net-total losses와 중복
- 기타

**사용자 검토에서 더 근본적 문제 지적**: "원인 분석이 안 됐다.
docs/dspinn_log를 보라". 결과적으로 **위치 (Pearson r=0.85)는 OK,
크기 (slope=0.6)가 문제** — spatial localization 추가는 잘못된 방향.

### 2.3 방향 전환 — Option X (Data-Driven Calibration Init)

`v3`의 hand-tuned ζ 값 (`softplus_inv(8.0)` diag, `softplus_inv(5.0)`
off-diag for `cpl_layer_pair_log_scale`; per-layer hardcoded `ρ`로
`layer_scale_phys_gnd`)을 **TRAIN_SPEFS에서 NNLS로 fit한 값**으로 대체.

**왜 이게 distillation이 아닌가**: live teacher signal 없음. 학습 시작
시 init만 바꾸고, 모델이 자유롭게 덮어씀. 정직한 명명은
"**Data-Driven Physics Calibration**" 또는 "Statistical Pre-tuning of
Physics-Informed Init".

---

## 3. 방법론 (Methodology)

### 3.1 NNLS 정식화

각 train net `i`에 대해:

```
GND eq:  Σ_j ρ_layer[j] · A[i, j]
       + s_diag · A_power_diag[i]
       + s_cross · A_power_cross[i]
       ≈ golden_gnd[i] − c_vss_pred[i]

CPL eq:  s_diag · B_diag[i, a] + s_cross · B_cross[i, a]
       ≈ golden_cpl[i, a]    for each signal aggressor a
```

- `A[i, j]`: net `i`의 layer `j`에 있는 target wire cuboid들의
  `gnd_area_eff` 합 (geometry-only)
- `A_power_*`: power-net edges의 `w_cpl_base × core_ratio_eff` 합
  (model의 power-net lumping 미러링)
- `B_diag/B_cross`: signal-aggressor edges, same/different layer 분리
- `c_vss_pred`: VSS edges의 contribution (current init에서 알려진 양)

**Joint NNLS**: K + 2 unknowns (K = 10 buckets after collinearity fix).
`scipy.optimize.nnls` 사용, 200k+ equations × 12 unknowns, < 1s 풀이.

### 3.2 빌드 중 발견한 critical bugs

**Bug 1**: `A_tgt` (name-based mask) ≠ `is_target` (cuboid channel 7).
Target net의 pin cuboid는 name match지만 ch7=0이라 모델 prediction에서
제외. Name mask 사용 시 A_primary 20% 과다 계상, c_vss_pred −8015 fF
(negative!) 누출.

**Fix**: cuboid channel 7 기반 mask로 통일.

**Bug 2**: 29개 z anchors 사용 시 NNLS 결과 oscillating (0/30/0/16
교대). 같은 metal layer의 top/bottom이 별도 anchor로 collinearity
유발.

**Fix**: 10 physical-metal buckets로 collapse (pre_M1, M1-M6, upper,
top, others).

**Bug 3**: Sanity check가 tile-centric sampling으로 partial coverage
→ pred 5-10% under로 잘못 보고. Net-centric walk로 수정.

### 3.3 Calibration 결과

| 파라미터 | NNLS-fit | v3 hardcoded | 변화 |
|---|---|---|---|
| s_diag (lateral CPL) | **0.182** | 8.0 | 44× lower |
| s_cross (broadside CPL) | 4.250 | 5.0 | similar |
| ρ_M1 (fF/μm²) | 1.246 | 2.50 | 0.5× |
| ρ_M2 | 0.334 | 3.00 | 0.11× |
| ρ_M3 | 0.386 | 3.00 | 0.13× |
| ρ_M4 | 0.278 | 2.75 | 0.10× |
| ρ_M5 | 0.135 | 2.75 | 0.05× |
| ρ_M6+ | hardcoded fallback | unchanged | 데이터 부족 |

**해석**:
- **Lateral CPL (s_diag)**: Sakurai-Tamaru가 raw geometric base에서
  이미 정확. v3의 8× scale-up은 과도해서 cpl_modifier가 0.11×로
  보상해야 했던 이유.
- **Broadside CPL (s_cross)**: v3의 5.0과 거의 동일. 정확.
- **GND density**: hardcoded보다 5-20× 낮음. v3는 `gnd_modifier`가
  ~0.5×로 수렴할 것을 가정한 보상값. NNLS는 modifier=1 가정 하의
  실효값 직접 fit.

---

## 4. 1차 결과 — Single-Seed 비교 (혼란의 시기)

### 4.1 Validation MAPE (in-distribution)

단일 v4 run의 iter 0 best (step 3000): **MAPE 47.41%**.
단일 v3 run의 iter 0 best (step 8000): **MAPE 60.27%**.

→ "v4가 22% 더 낮다" 같은 강한 주장 가능해 보임.

### 4.2 Heteroscedastic 측정

v4 best와 v3 best에 대해 per-quartile-of-y_gnd ratio:

| Quartile | v3 median ratio | v4 median ratio |
|---|---|---|
| Q1 (smallest) | 1.013 | 1.263 (worse over) |
| Q1-Q2 | 0.627 | 0.633 |
| Q2-Q3 | 0.537 | 0.394 (worse under) |
| Q3+ (largest) | 0.545 | 0.489 (worse under) |

**Slope: v3 = 0.453, v4 = 0.369** ← v4가 1.0에서 더 멀어짐.
**Pearson r: v3 = 0.923, v4 = 0.948** (location 약간 더 좋음).

### 4.3 OOD 측정 (TEST_DEFS: nova_f3, tv80s_f3)

| Metric | v3 nova | v4 nova | v3 tv80s | v4 tv80s |
|---|---|---|---|---|
| Total MAPE | 0.32 | **0.37** (+5pp) | 0.34 | **0.42** (+8pp) |
| GND MAPE | 0.43 | 0.54 | 0.43 | 0.50 |
| CPL MAPE | 0.67 | 0.76 | 0.53 | 0.69 |

**모순**: in-dist에서 v4 22% 좋지만, OOD에서 v4 5-8pp 나쁨.

해석 시도: "v4가 in-dist overfit, heteroscedastic 악화" → 정직한
critical assessment 작성.

### 4.4 그러나 — 비판적 분석에서 우려

- **N=1 single-seed**: stochastic variance bound 없음
- **v4 BEST는 step 3000 (transient), v3 BEST는 step 8000 (more trained)**
  — apples-to-oranges
- **확실한 결론을 내릴 수 없음** — 5+ seeds 필요

---

## 5. 5-Seed Measurement Protocol

### 5.1 사용자 주도 설계

비판적 분석 직후 사용자가 `run_active_learning.py`에 `--seed`,
`--max_iters`, `--steps_per_iter` CLI args를 추가. 의도: **5 seeds
× 1 iteration × 5000 steps**로 distributional comparison.

### 5.2 3 variants × 5 seeds = 15 runs

- **v3_baseline**: hardcoded ζ (no calibration JSON)
- **v4_full_calib**: NNLS-fit ρ + CPL pair (calibration_init.json)
- **v5_gnd_only**: NNLS-fit ρ + hardcoded CPL (calibration_init_gnd_only.json)

각 run: `python3 run_active_learning.py --model_name m5_<variant>_seed<N>
--gpu <id> --use_dspinn --calib_path <json> --seed <N> --max_iters 1
--steps_per_iter 5000`

15 jobs를 GPUs 1-4에서 batch parallel. 총 ~6시간 (학습) + 1.5시간
(aggregator).

### 5.3 Validation 결과 (n=5/variant, full step 5000)

| Variant | Median MAPE | p25-p75 | Range | Mean | IQR |
|---|---|---|---|---|---|
| v3_baseline | 64.17 | 55.70-65.50 | 50.70-73.23 | 61.86 | 9.80 |
| **v4_full_calib** | **54.50** | 52.67-57.19 | **49.32-63.03** | **55.34** | **4.52** |
| v5_gnd_only | 60.40 | 53.56-62.41 | 48.78-70.08 | 59.05 | 8.85 |

**Per-seed best_mape (sorted within variant)**:
- v3: 50.70 / 55.70 / **64.17** / 65.50 / 73.23
- v4: 49.32 / 52.67 / **54.50** / 57.19 / 63.03
- v5: 48.78 / 53.56 / **60.40** / 62.41 / 70.08

### 5.4 Mann-Whitney U test (two-sided)

| Comparison | U | p-value |
|---|---|---|
| v3 vs v4 | 19.0 | 0.222 |
| v3 vs v5 | 16.0 | 0.548 |
| v4 vs v5 | 10.0 | 0.690 |

**모든 비교 p > 0.05 — 통계적으로 구분 불가**.

### 5.5 OOD 결과 (TEST_DEFS, n=5/variant)

| Variant | total MAPE median | GND chip ratio | CPL chip ratio | slope | Pearson r |
|---|---|---|---|---|---|
| v3_baseline | 0.553 | 0.665 | 1.730 | 0.535 | 0.899 |
| **v4_full_calib** | **0.459** | 0.678 | 1.544 | 0.534 | 0.885 |
| v5_gnd_only | 0.535 | 0.686 | 1.462 | 0.536 | 0.886 |

**Mann-Whitney 모두 ns (p > 0.5)**.

**중요**: 단일 seed OOD 결과 (v4 +5-8pp WORSE)가 5-seed에서 **reverse**:
- 단일 seed: v4 OOD MAPE 37-42% (cherry-picked iter 0 best)
- 5-seed: v4 OOD median 0.459 (= 45.9%)
- 5-seed: v3 OOD median 0.553 (= 55.3%)
- → v4가 OOD에서 **9.4pp BETTER** (단 ns)

### 5.6 5-Seed의 핵심 교훈

1. **단일 seed 비교는 stochastic noise에 압도**: v3 5-seed best_mape
   range 50.70-73.23 (22pp spread). 단일 seed로 본 22% improvement는
   이 range 안에 충분히 들어감.
2. **n=5는 6-10pp difference 검출에 부족**: Mann-Whitney 최소
   significance를 위해 n=10+ 필요.
3. **트렌드와 distribution shape는 의미**: v4의 mean 6.5pp lower +
   IQR 절반 — direction은 일관, magnitude는 작음.
4. **OOD ≈ in-dist**: 모든 variant generalization 동일. Calibration
   init이 specific design overfit 안 함.

---

## 6. Heteroscedastic 문제 — 여전히 미해결

### 6.1 5-Seed aggregated heteroscedastic

```
v3:  slope 0.525, Pearson r 0.915, GND chip ratio 0.735
v4:  slope 0.510, Pearson r 0.912, GND chip ratio 0.740
v5:  slope 0.517, Pearson r 0.912, GND chip ratio 0.737
```

**Slope ~0.5 모든 variant 동일**. **Calibration init은
heteroscedastic 문제에 영향 없음**.

### 6.2 왜 calibration init은 못 푸는가

- `ρ_layer`는 layer-wide global multiplier — net topology, density,
  aspect ratio per-net 변화를 흡수 못함
- 같은 layer의 작은 net과 큰 net은 같은 ρ로 곱해짐
- Heteroscedasticity는 **per-net residual** 현상

### 6.3 다음 단계 — γ scaling head

**아이디어**: per-net features → 작은 MLP → multiplicative scale γ_net,
적용: `pred_total_net *= γ_net`.

`docs/dspinn_development_log.md` §7.1의 γ proposal 채택.

---

## 7. γ Scaling Head (현재 진행 중)

### 7.1 Codex-Reviewed 설계

**Input features (14 dims)**:
- log1p(n_target_cuboids)
- log1p(total_gnd_area)
- log1p(total_w_cpl_base)
- n_layers_present (count)
- area_layer_dist (10-bucket distribution)

**제외 (Codex 권고)**:
- `pred_*_pre_gamma` — γ가 learned remapper로 전락
- `dominant_layer_idx` — area_layer_dist와 중복

**Architecture**: `Linear(14, 32) → GELU → Linear(32, 32) → GELU →
Linear(32, 1)`. Output: `γ_gnd = exp(clamp(logit, -2, 2))` ∈ [0.135, 7.39].

**Init**: 마지막 Linear weight/bias zero → output ≈ 1.0 (identity).

### 7.2 Codex 권고 4개 적용

1. **γ_gnd만** 활성 (γ_cpl은 cpl_modifier와 중복, 차후 평가)
2. **Warmup schedule**:
   - step ∈ [0, 2000): mix=step/2000, clamp [-0.5, 0.5]
   - step ∈ [2000, 4000): mix=1.0, clamp [-1.0, 1.0]
   - step ≥ 4000: full clamp [-2.0, 2.0]
3. **Separate optimizer group**: 0.1× base LR, weight_decay 1e-3
4. **Identity regularizer**: `λ × mean(log_γ²)` with λ=0.05

### 7.3 적용 위치 (finetuner.py)

`global_pred_gnd` 계산 + power-net lumping 직후 적용:
```python
delta_gnd = (γ - 1.0) × global_pred_gnd
global_pred_gnd += delta_gnd
global_pred_total += delta_gnd  # consistency 유지
```

### 7.4 Smoke test 결과 (단일 seed, step 1000)

- **Net-level MAPE: 67.24%** (v3 step 1000 median 82.66, v4 step 1000
  median 86.00 대비 15-19pp 향상)
- CPL ratio: 0.0% (warmup mid-range, γ는 거의 1.0)
- Crash 없음, γ probe 정상

### 7.5 5-Seed Protocol (현재 진행 중)

m5_v6_gamma_seed{0..4} 5 seeds, GPUs 2/3/4/5/7 parallel. 학습 ~2:30,
완료 후 analyze_5seed.py + aggregate_5seed_eval.py 동일 방식 비교.

**기대치**:
- Median MAPE: 30-50% (v4의 54.50보다 lower)
- Slope: 0.51 → 0.85+ (heteroscedastic improvement)
- Per-quartile ratio: 1.40 / 1.00 / 0.60 / 0.50 → 1.0 / 1.0 / 1.0 / 1.0
  flatten

**위험**:
- γ가 1.0에 머물러 효과 0 (modifier가 이미 흡수)
- Overfit으로 in-dist만 좋아지고 OOD 악화
- 학습 불안정 (warmup이 충분치 않다면)

---

## 8. Critical Self-Assessment

### 8.1 잘 한 것

- **Critical bug 2개 발견**: A_tgt vs is_target mask, anchor
  collinearity. 둘 다 silent failure 일으킬 수 있던 버그.
- **Codex deliberation loop**: round-1에 6 critical bugs 잡아서
  fundamental redesign 유도.
- **5-seed protocol 채택**: N=1 fallacy를 데이터로 입증, 연구 신뢰도
  대폭 상승.
- **OOD evaluation**: in-dist만 보면 transient artifact에 속을 수 있음.
- **사용자 driven course correction**: "원인 분석 안 됐다" 지적
  → 잘못된 multi-scale distillation에서 root-cause Option X로 전환.

### 8.2 잘못한 것 / Overclaim

- **"Distillation" framing**: live teacher 없음에도 distillation으로 명명
  → Codex/사용자 검토 후 "Data-Driven Calibration"으로 정정.
- **단일 seed 22% improvement claim**: Stochastic noise 안에 들어가는
  값을 "significant"처럼 보고. 5-seed로 정정.
- **OOD 단일 seed 결과 해석**: v4 +5-8pp WORSE를 반증으로 해석했지만
  실제로는 cherry-picked transient. 5-seed에서 reverse.

### 8.3 한계

- **Heteroscedastic 문제 미해결** (slope 0.5 → 0.5). Calibration init
  alone은 정답이 아님.
- **n=5에서 통계적 유의성 부족** — 6-10pp difference 검출 못함.
- **2000 nets/design sample**: full data (1.3M tiles) 안 씀. NNLS pooled
  MAPE 0.71로 제한적 fit.
- **5/10 buckets fallback**: M6, upper, top, others, pre_M1는 hardcoded
  유지. "data-driven" 주장 부분적.
- **Single PDK (intel22)**: cross-PDK transfer 미검증.

---

## 9. 결론 및 권고

### 9.1 Calibration Init만으로는 불충분

5-seed로 명확히 입증: data-driven init은 **measurable but small**
positive effect (mean MAPE -6.5pp, IQR 절반)를 가지나 **n=5에서
ns**, **heteroscedastic motivation NOT solved**.

### 9.2 다음 step

1. **γ head 5-seed 결과 대기** (~3시간 후). Slope 0.85+로 flatten되면
   data-driven init + γ head를 결합한 결과 보고.
2. **n=10 seed protocol**: γ head가 의미 있어 보이면 n=10으로 재측정해
   통계적 유의성 확보.
3. **Cross-PDK validation**: asap7으로 NNLS extraction + 학습 재현.
4. **Outlier net 분석**: top-100 worst nets (특히 LDPC) 별도 분석.
   Per-design Bayesian outlier detector 검토.

### 9.3 논문화 권고

**현재 상태로 논문 단독 contribution 불가**:
- Calibration init alone: 통계적 유의성 미달
- Heteroscedastic motivation 해결 안 됨

**합리적 framing**:
- "Statistical methodology paper": 5-seed protocol, single-seed
  fallacy 데이터로 입증, AI4PEX 분야의 reproducibility 표준 제안
- "Calibration + γ joint contribution" (γ head 결과 수집 후):
  data-driven init은 starting point 개선, γ head는 per-net residual,
  두 기법의 ablation
- Negative-but-careful result: "data-driven calibration alone is not
  sufficient" — 향후 연구에 가치

### 9.4 Methodology takeaways

1. **N=1 comparison은 위험**: 단일 seed BEST는 stochastic variance에
   dominated. 최소 5-seed Mann-Whitney 권고.
2. **OOD evaluation은 필수**: in-dist만 보면 overfit artifact에 속음.
3. **Codex deliberation loop**: 빌드 전 critical bug 사전 발견에 필수.
   특히 forward path mirror가 fragile.
4. **Critical analysis는 논문 review 전에 자체 수행**: 자체 검토에서
   "전부 승인" 되면 reviewer 검토 통과 가능성 높음.

---

## 10. 산출물 목록

### Code
- `src/data/calibration_extractor.py` — phase 1 (geometry) + phase 2 (model fwd)
- `src/data/calibration_solver.py` — joint NNLS with bucketing
- `src/models/gamma_head.py` — γ head module + warmup schedule
- `src/models/flux_head.py` — JSON-aware init plumbing
- `src/models/neural_field.py` — γ head instantiation + freeze hooks
- `src/trainers/finetuner.py` — γ application + identity regularizer
- `run_active_learning.py` — `--use_gamma`, `--calib_path`, seed args
- `scripts/diag_spef_unit_check.py` — Step 0B verification
- `scripts/diag_calibration_check.py` — sanity check
- `scripts/diag_quartile_heteroscedastic.py` — per-quartile analysis
- `scripts/diag_ood_compare.py` — TEST_DEFS evaluation
- `scripts/run_5seed_remaining.py` — 5-seed launcher (v3/v4/v5)
- `scripts/run_5seed_v6_gamma.py` — γ head 5-seed launcher
- `scripts/analyze_5seed.py` — log → distribution analyzer
- `scripts/aggregate_5seed_eval.py` — multi-ckpt heteroscedastic + OOD

### Data
- `/data/PINNPEX/data/processed/intel22/calibration_init.json` (full v4)
- `/data/PINNPEX/data/processed/intel22/calibration_init_gnd_only.json` (v5)
- `/data/PINNPEX/data/processed/intel22/calibration_extract/phase1_net2k.pkl`
- `/data/PINNPEX/data/processed/intel22/calibration_extract/phase2_net2k.pkl`

### Checkpoints (보존)
- `output_intel22/active_learning/m5_<variant>_seed<N>/best_model.pth` (15 ckpts)
- `output_intel22/active_learning/dspinn_v3/best_model.pth` (legacy v3 reference)
- `output_intel22/active_learning/v4_distillinit/best_model.pth` (legacy v4 reference)
- `output_intel22/active_learning/m5_v6_gamma_seed{0..4}/best_model.pth` (γ in progress)

### Analysis outputs
- `output_intel22/active_learning/m5_summary/per_run.csv`
- `output_intel22/active_learning/m5_summary/per_variant.csv`
- `output_intel22/active_learning/m5_summary/mann_whitney.csv`
- `output_intel22/active_learning/m5_summary/eval_per_seed.csv`
- `output_intel22/active_learning/m5_summary/eval_per_variant.csv`
- `output_intel22/active_learning/m5_summary/eval_raw_ind.csv`
- `output_intel22/active_learning/m5_summary/eval_raw_ood.csv`
- `output_intel22/active_learning/{v3,v4_*}/hetero_quartile_*.csv`
- `output_intel22/active_learning/ood_compare/ood_*.csv`

### Documentation
- `docs/distillation_log.md` — 진행 과정 living log (2026-04-30 ~)
- `docs/calibration_track_report.md` — **이 보고서** (종합 정리)
- `docs/dspinn_development_log.md` — parallel DS-PINN track 참조

---

## 부록 A. 5-Seed Per-Seed Detail

| Variant | Seed | Best MAPE | Best Step | CPL ratio (med) |
|---|---|---|---|---|
| v3_baseline | 0 | 73.23 | 5000 | 1.0 |
| v3_baseline | 1 | 55.70 | 2000 | 0.0 |
| v3_baseline | 2 | 64.17 | 5000 | 5.6 |
| v3_baseline | 3 | 50.70 | 5000 | 1.4 |
| v3_baseline | 4 | 65.50 | 5000 | 0.5 |
| v4_full_calib | 0 | 63.03 | 2000 | 4.1 |
| v4_full_calib | 1 | 54.50 | 2000 | 0.0 |
| v4_full_calib | 2 | 52.67 | 5000 | 0.6 |
| v4_full_calib | 3 | 49.32 | 5000 | 0.7 |
| v4_full_calib | 4 | 57.19 | 5000 | 7.4 |
| v5_gnd_only | 0 | 70.08 | 5000 | 6.6 |
| v5_gnd_only | 1 | 60.40 | 2000 | 0.4 |
| v5_gnd_only | 2 | 53.56 | 5000 | 4.0 |
| v5_gnd_only | 3 | 48.78 | 5000 | 2.2 |
| v5_gnd_only | 4 | 62.41 | 5000 | 0.0 |

## 부록 B. 시간선 (Timeline)

- 2026-04-30 16:13 — Step 0 verification (tiling.py, SPEF C_UNIT)
- 2026-04-30 16:15 — Step 1 build (calibration_extractor.py)
- 2026-04-30 16:30 — Step 1 phase 1 smoke (450 tiles)
- 2026-04-30 16:50 — Bug 1 발견 (A_tgt vs is_target)
- 2026-04-30 17:30 — Step 2 NNLS solver + bucketing (Bug 2 fix)
- 2026-04-30 18:00 — calibration_init.json 생성, sanity check pass
- 2026-04-30 18:30 — v4_distillinit launched (long single seed)
- 2026-04-30 21:00 — Critical analysis (over claim 인지)
- 2026-04-30 22:00 — User: "5-seed protocol" suggestion via CLI
- 2026-04-30 23:00 — v3 baseline 5 seeds launched
- 2026-05-01 04:00 — v3 5 seeds 완료
- 2026-05-01 06:30 — v4 5 seeds 완료
- 2026-05-01 11:55 — v5 5 seeds 완료 (15/15)
- 2026-05-01 13:54 — aggregator 완료
- 2026-05-01 14:30 — γ head 설계 + Codex 검토
- 2026-05-01 14:45 — γ head implementation + smoke launch
- 2026-05-01 15:15 — γ smoke step 1000 BEST 67.24%
- 2026-05-01 15:30 — γ 5 seeds launched (in progress)

---

_본 보고서는 living document. γ head 5-seed 결과 도착 시 §7.5 업데이트
+ 최종 conclusion 추가 예정._
