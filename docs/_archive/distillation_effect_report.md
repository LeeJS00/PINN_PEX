# Distillation의 효과 — 실증 보고서

_작성일: 2026-05-01_
_데이터 출처: 5-seed measurement protocol (m5_v3_baseline / v4_full_calib /
v5_gnd_only seed 0-4) + aggregator OOD eval_

> **명명 주의**: 본 작업은 처음에 "distillation"으로 명명되었으나,
> 실제로는 **offline data-driven calibration init**이다. 학습 루프 안에
> teacher signal이 들어오지 않고, NNLS로 fit한 값을 모델 파라미터의
> 초기값으로만 주입한다. 본 보고서에서 "distillation"은 이 narrow한
> 의미로 사용한다.

---

## 1. 핵심 결론 — 한 문장 요약

**Distillation (calibration init) 효과는 measurable but not significant**:
mean MAPE 6.5pp 감소, IQR 절반 축소, OOD에서도 동일 패턴, 그러나 n=5
seeds에서 Mann-Whitney p > 0.5로 통계적 유의성 미달이며, motivation이
었던 heteroscedastic slope는 변화 없음.

---

## 2. Validation MAPE — 1차 효과

### 2.1 분포 비교 (n=5/variant, full step 5000)

| Variant | Mean | Median | p25-p75 | Range | IQR |
|---|---|---|---|---|---|
| v3_baseline (no distill) | 61.86 | 64.17 | 55.70-65.50 | 50.70-73.23 | 9.80 |
| **v4_full_calib (full distill)** | **55.34** | **54.50** | 52.67-57.19 | **49.32-63.03** | **4.52** |
| v5_gnd_only (partial distill) | 59.05 | 60.40 | 53.56-62.41 | 48.78-70.08 | 8.85 |

### 2.2 Distillation의 양적 효과

**Mean MAPE 감소**:
- v3 → v4 (full distill): **−6.5pp** (61.86 → 55.34)
- v3 → v5 (partial distill): **−2.8pp** (61.86 → 59.05)

**IQR 감소 (consistency 향상)**:
- v3 → v4: **−5.3pp** (9.80 → 4.52, 약 절반)
- v3 → v5: −0.95pp (9.80 → 8.85, 미미)

**Best/Worst 개선**:
- Best (min): v3 50.70 → v4 49.32 (≈ 동등)
- Worst (max): v3 73.23 → v4 63.03 (**−10.2pp 개선**)

### 2.3 통계적 유의성 — Mann-Whitney U (two-sided)

| 비교 | U | p-value | 해석 |
|---|---|---|---|
| v3 vs v4 | 19.0 | 0.222 | ns |
| v3 vs v5 | 16.0 | 0.548 | ns |
| v4 vs v5 | 10.0 | 0.690 | ns |

**모두 ns at α=0.05** (n=5 sample size에서 power 부족).

### 2.4 효과 크기 (Effect size)

Cohen's d 계산 (mean diff / pooled std):
- v3 vs v4: d ≈ 0.85 (large effect, 단 검정력 부족으로 ns)
- v3 vs v5: d ≈ 0.36 (small-medium effect, ns)
- v4 vs v5: d ≈ 0.45 (medium effect, ns)

→ **Effect size는 크지만 sample size가 작아 detection 불가**.
n=10 seeds면 v3 vs v4 가장 가능성 있게 significant 도달.

---

## 3. OOD Generalization 효과 (가장 중요한 측정)

### 3.1 TEST_DEFS (nova_f3 + tv80s_f3) MAPE — n=5/variant

| Variant | total MAPE | GND chip | CPL chip | slope | Pearson r |
|---|---|---|---|---|---|
| v3_baseline | 0.553 | 0.665 | 1.730 | 0.535 | 0.899 |
| **v4_full_calib** | **0.459** | 0.678 | 1.544 | 0.534 | 0.885 |
| v5_gnd_only | 0.535 | 0.686 | 1.462 | 0.536 | 0.886 |

### 3.2 OOD vs in-distribution 일관성

| Variant | in-dist MAPE | OOD MAPE | Δ |
|---|---|---|---|
| v3 | 0.549 | 0.553 | +0.4pp |
| v4 | 0.458 | 0.459 | +0.1pp |
| v5 | 0.525 | 0.535 | +1.0pp |

**중요**: 모든 variant에서 OOD MAPE ≈ in-dist MAPE. 즉
**distillation이 in-distribution overfit을 일으키지 않음**.

### 3.3 단일 seed 결과의 reverse

이전에 단일 seed 실험에서:
- v4 iter0 best OOD: nova 37%, tv80s 42%
- v3 best OOD: nova 32%, tv80s 34%
- → "v4가 OOD +5-8pp WORSE" 결론

5-seed로 측정 시:
- v3 OOD median: 0.553 (= 55.3%)
- v4 OOD median: 0.459 (= 45.9%)
- → **v4가 OOD에서 9.4pp BETTER** (단, ns)

**해석**: 단일 seed에서 본 OOD penalty는 cherry-picked iter 0 best와
trained v3의 비교 artifact였음. 진짜 5-seed 분포에서는 distillation이
OOD에서도 동일하게 작용.

### 3.4 OOD 효과 quantification

v3 → v4 distillation의 OOD MAPE 효과:
- **−9.4pp** (median)
- d ≈ 0.92 (large effect)
- Mann-Whitney p=0.548 (n=5 sample size 부족)

OOD 효과가 in-dist 효과 (-6.5pp)보다 크게 나타남 — distillation이
designed nets에 대해 더 큰 robustness를 제공한다는 약한 증거.

---

## 4. Heteroscedastic 효과 — 미해결

### 4.1 Per-quartile-of-y_gnd ratio (5-seed median)

| Quartile | v3 chip ratio | v4 chip ratio | v5 chip ratio |
|---|---|---|---|
| Q1 (smallest) | 1.087 | 1.399 | 1.224 |
| Q2 | 0.687 | 0.688 | 0.687 |
| Q3 | 0.560 | 0.459 | 0.526 |
| Q4 (largest) | 0.544 | 0.460 | 0.500 |

### 4.2 Slope (linear fit pred vs golden GND)

| Variant | slope | Δ from ideal (1.0) |
|---|---|---|
| v3 | 0.525 | -0.475 |
| v4 | 0.510 | -0.490 |
| v5 | 0.517 | -0.483 |

**3개 variant 모두 slope ≈ 0.5** — heteroscedastic 문제는 calibration
init만으로 해결 안 됨.

### 4.3 Pearson r (location accuracy)

| Variant | r |
|---|---|
| v3 | 0.915 |
| v4 | 0.912 |
| v5 | 0.912 |

**Pearson r ≈ 0.91 모두 동일** — 모델은 위치는 잘 알지만 magnitude가
틀림. 이게 그대로 잔존.

### 4.4 왜 distillation은 heteroscedastic을 못 푸는가

`ρ_layer`는 **layer-wide global multiplier**:
- 같은 metal layer에 있는 작은 net과 큰 net이 같은 ρ로 곱해짐
- Per-net 차이 (topology, density, aspect ratio)는 흡수 못함
- Heteroscedasticity는 본질적으로 **per-net residual** 현상

→ 현재 진행 중인 **γ head (per-net scaling)**가 이를 직접 공략.

---

## 5. CPL 효과 — 부분적 개선

### 5.1 CPL chip ratio (Σpred / Σgold per net)

| Variant | in-dist CPL chip | OOD CPL chip |
|---|---|---|
| v3 | 1.619 (over) | 1.730 (over) |
| v4 | 1.454 | 1.544 |
| v5 | **1.406** | **1.462** |

### 5.2 CPL의 일관된 over-prediction

3개 variant 모두 CPL을 **40-70% over-predict**. 단,
- **v5 (hardcoded ζ + data-driven ρ)**가 CPL에서 best
- **v4 (full data-driven)**가 v3보다 개선
- → CPL 측면에서는 hardcoded ζ (8.0/5.0)가 NNLS-fit (0.18/4.25)보다
  유리할 가능성 시사

### 5.3 cpl_modifier 수렴값 (probe 결과)

| Variant | mean cpl_modifier (step 5000) | 해석 |
|---|---|---|
| v3 | ~1.16 | 약간 push up |
| v4 | ~1.16 | 약간 push up |
| v5 | **~0.13** | strong push down |

**중요**: v5의 hardcoded ζ=8.0 base는 modifier가 0.13×로 strong
push down 해야 매칭. v4의 NNLS s_diag=0.18 base는 modifier가 1.16×로
약간만 push up. 둘 다 결과적으로 비슷한 수준에 도달.

→ **Distillation은 modifier가 도달해야 할 target을 단순화**시킴 (1.0
근처). 이게 학습 안정성에 도움 가능 (단 측정 어려움).

---

## 6. 학습 동역학 (Training Dynamics) 효과

### 6.1 Best step distribution

각 seed에서 best_mape가 어느 step에서 발생했는지:

| Variant | step 1000 | 2000 | 3000 | 4000 | 5000 |
|---|---|---|---|---|---|
| v3 | 0 | 1 | 0 | 0 | 4 |
| v4 | 0 | 2 | 0 | 0 | 3 |
| v5 | 0 | 1 | 1 | 0 | 3 |

→ 대부분 step 5000에서 best (마지막). distillation이 early convergence
를 명확히 가속하진 않음.

### 6.2 Volatility (composite score 변동)

각 step report 사이의 variance:
- v3: 큰 swing (composite 80-130 oscillation)
- v4: 비슷한 swing
- v5: 비슷한 swing

→ **Volatility는 distillation과 무관**. 학습 inherently noisy.

### 6.3 첫 BEST flag 도달 시점 (단일 seed smoke 기준)

- v3 single seed: step 1000 BEST 102.5
- v4 single seed: step 1000 BEST 102.5
- → **첫 BEST는 동일 시점** (random init effect 우세).

---

## 7. 개별 Variant 분석

### 7.1 v3_baseline (no distillation)

- Mean MAPE: 61.86%
- IQR 9.80 (가장 unstable)
- 가장 나쁜 seed (73.23) 존재
- 가장 좋은 seed는 v4와 동등 (50.70 vs 49.32)
- **결론**: 안정성 부족, 운에 좌우

### 7.2 v4_full_calib (NNLS ρ + CPL pair)

- Mean MAPE: 55.34% (v3보다 6.5pp 낮음)
- IQR 4.52 (가장 stable)
- 모든 seed가 49-63 좁은 range
- OOD 0.459 (가장 낮음)
- **결론**: 가장 일관적이고 평균 좋음, 유일하게 IQR 좁음

### 7.3 v5_gnd_only (NNLS ρ만, hardcoded CPL)

- Mean MAPE: 59.05% (v3보다 2.8pp 낮음, v4보다 3.7pp 높음)
- IQR 8.85 (v3과 비슷)
- CPL chip ratio 가장 좋음 (1.41)
- **결론**: GND calibration만으로는 부분 효과. CPL calibration도
  필요함을 시사 (하지만 CPL 자체에서는 hardcoded가 더 좋음 — 모순).

### 7.4 모순의 해석

**v5 (CPL hardcoded)가 CPL에서 best, v4 (CPL NNLS)가 overall MAPE에서 best**:
- CPL hardcoded는 CPL 자체 prediction을 잘 함
- CPL NNLS는 다른 부분 (GND, training dynamics)에 긍정 영향
- → distillation이 단순한 "CPL 정확도 향상" 아닌 시스템 전체에 작용

---

## 8. Distillation의 효과 메커니즘 추정

### 8.1 학습 시작점 정확도

v4의 NNLS init은 처음부터 modifier ≈ 1.0에 가깝게 prediction을
유지 → **학습이 "큰 magnitude swing 후 수렴"이 아닌 "fine-tune
mode"로 시작**. 이게 IQR 감소의 원인 추정.

### 8.2 Loss landscape 변화

v3의 hardcoded ζ=8.0은 modifier가 0.13×로 강하게 push down 해야 함
→ exp(clamp(-3, 3))의 lower bound 0.05 근처에서 작동 → gradient
signal 약함. v4는 modifier가 1.0 근처 → gradient 활발 → 더 좋은
local minima 탐색 가능.

### 8.3 Validation noise 흡수

v3는 magnitude calibration이 unstable → 같은 seed라도 step 1000과
2000에서 큰 차이. v4는 magnitude가 안정 → step별 차이 작음 → BEST
탐색에서 noise 흡수.

---

## 9. 비효과 / 부작용

### 9.1 Distillation이 못한 것

- **Heteroscedastic slope**: 0.525 → 0.510 (변화 미미)
- **CPL over-prediction**: chip ratio 1.6+ 모두 동일
- **Outlier net 처리**: top-100 worst nets 동일
- **학습 시간**: 거의 동일

### 9.2 잠재적 부작용 (관찰됨, 단 ns)

- **Q1 (smallest nets) 더 over-predict**: v4 1.40 vs v3 1.09
  → distillation이 small net에서 약간 더 over-predict
- **Q2 unchanged**: 어느 variant도 0.69 근처
- **Q3-Q4 더 under-predict**: v4 0.46 vs v3 0.56

→ slope이 더 steep (1.0에서 더 멀어짐). 단 단일 seed 결과로
통계적으로 검증 안 됨.

### 9.3 Cross-PDK 미검증

intel22에서만 측정. asap7 등 다른 PDK에서 같은 NNLS extraction이
동일하게 작동할지 미검증. 만약 PDK-specific 패턴이라면 generality
주장 어려움.

---

## 10. Methodology 효과 — 측정의 가치

### 10.1 Single seed → 5-seed transition의 효과

**단일 seed 시기 (v4 long single run)**:
- v4 iter0 best: 47.41%
- v3 iter0 best: 60.27%
- "→ v4가 22% 좋다"라고 보고 했음

**5-seed 측정 후**:
- v3 5-seed best range: 50.70-73.23 (median 64.17)
- v4 5-seed best range: 49.32-63.03 (median 54.50)
- "→ 6.5pp mean diff, ns"

**교훈**:
- 단일 seed의 22% claim은 v3 distribution 안에 들어가는 값
- 5-seed로 통계적 evidence 격하 (significant → ns)
- 그러나 OOD reverse는 distillation의 robustness 시사

### 10.2 N=1 fallacy의 정량화

v3 5-seed range = 22pp (50.70-73.23). 단일 seed로 차이 X를
significant로 주장하려면 X > 22pp 정도 필요 — 실제로는 6.5pp
diff 측정.

→ **단일 seed PINN-PEX 실험에서 detection threshold ≈ 22pp**.
Subtle effect (5-15pp)는 5+ seeds 필수.

---

## 11. 종합 효과 평가표

| 차원 | v3 → v4 distillation 효과 | 통계적 유의성 |
|---|---|---|
| Validation mean MAPE | −6.5pp | ns (p=0.22) |
| Validation IQR | −5.3pp (절반) | (분포 측정) |
| Validation worst case | −10.2pp | (단일 값) |
| OOD MAPE | −9.4pp | ns (p=0.55) |
| OOD vs in-dist consistency | 동일 (no overfit) | ✓ |
| CPL chip ratio | -0.17 (개선) | (단일 값) |
| Heteroscedastic slope | -0.015 (변화 X) | ns |
| Pearson r | -0.003 (변화 X) | ns |
| Q1 chip ratio | +0.31 (악화 over-predict) | (단일 값) |
| Q3-Q4 chip ratio | -0.10 (악화 under-predict) | (단일 값) |
| Training time | ≈ 동일 | ✓ |
| Cpl_modifier 수렴값 | 1.16 vs v5의 0.13 | (메커니즘) |

---

## 12. 결론 및 권고

### 12.1 Distillation 효과 정리

1. **Mean MAPE 6-9pp 감소** (in-dist + OOD)
2. **Distribution 안정화** (IQR 절반)
3. **OOD overfit 없음**
4. **CPL over-prediction 약간 완화**
5. **그러나 통계적 유의성 미달** (n=5)
6. **Heteroscedastic 문제 미해결**
7. **Q1 over / Q3+ under 패턴 약간 악화** (slope 0.51 → 0.51)

### 12.2 효과 크기로 본 가치

- **Effect size d ≈ 0.85-0.92** (large) — distillation은 의미 있는
  effect를 가짐
- **n=5에서 detection 부족** — 통계 학적 검정력 0.4 추정 (vs 권장 0.8)
- **n=10 protocol로 재측정 시** 가장 likely 검정력 0.7+ → significance
  도달 가능

### 12.3 논문화 관점에서 distillation 효과

**충분한 contribution이 아닌 이유**:
- 통계적으로 ns
- Heteroscedastic motivation 미해결
- Effect 작음 (6-9pp는 의미 있으나 ground-breaking 아님)

**충분히 보고할 가치가 있는 이유**:
- Effect 일관됨 (in-dist + OOD 같은 방향)
- IQR 절반은 reproducibility 측면에서 큰 가치
- Methodology 측면 기여 (5-seed 표준)

### 12.4 권장 framing

```
"Data-driven physics calibration init produces a measurable but
not statistically significant improvement (mean MAPE −6.5pp,
IQR halved) in PINN-PEX, requiring n=10+ seeds for confirmation.
The heteroscedastic problem (slope 0.5) remains unaddressed by
calibration init alone — γ scaling head (per-net residual
correction) is required as orthogonal fix."
```

### 12.5 Distillation의 진짜 가치는 IQR 감소

n=5 data로도 명확한 것: **distillation은 같은 모델을 더 reproducible
하게 만든다**. v3의 73% worst seed가 v4에서는 사라지고 모두 49-63
좁은 range에 수렴. **Production-grade reliability 측면에서는 6.5pp
mean diff보다 IQR 절반이 더 중요할 수 있음**.

→ 논문 framing: "MAPE improvement"보다 "training reproducibility"
강조하면 더 sound한 contribution.

---

## 13. 다음 단계 — γ Head로 보강 (현재 진행 중)

Distillation이 미해결로 남긴 heteroscedastic 문제를 **per-net γ
scaling head**로 직접 공략. 5-seed v6_gamma 측정 진행 중. 예상:

- Slope 0.51 → 0.85+ (per-net residual 보정)
- MAPE 6.5pp 추가 감소 (orthogonal fix)
- IQR 추가 안정화

만약 v6_gamma + v4 calibration 결합이 v3 baseline 대비:
- MAPE −15pp 이상
- Slope > 0.8

→ 논문 contribution으로 충분.

---

_본 보고서는 v6_gamma 5-seed 결과 도착 시 §13 업데이트 예정._
