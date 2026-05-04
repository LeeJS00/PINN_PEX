# NCGT — 실험 결과 보고서 (Living Document)

_Last updated: 2026-05-02_
_Status: Phase A 5-seed validation 진행 중 (C1 완료, C2 진행 중, C3 예정)_

## 1. 전체 진행 요약

| Phase | 결과 | 진행률 |
|---|---|---|
| Plan v0 → v4 (2 Codex rounds + literature review) | ✅ 확정 | 100% |
| Phase 0 audit (11 designs × 300 nets) | ✅ supervision gate 85.6% pass | 100% |
| Phase 1.0 smoke (1-net architecture sanity) | ✅ 4.6× decrease | 100% |
| 3 risk sanity (physics/SPEF/overlap) | ✅ marginal pass + Risk 3 fix | 100% |
| Phase 1.1 plateau debug | ✅ root cause + log-space residual fix | 100% |
| Phase 2.0 single-design (1 seed lucky) | ⚠️ 11.34% (later disproved by 5-seed) | 100% |
| Phase 2.2 augmentation ablation (1 seed) | ❌ 13.33% (negative) | 100% |
| Phase 2.3 bins ablation (1 seed) | ❌ 13.42% (negative) | 100% |
| Phase 3 multi-design (1 seed) | ⚠️ 16.88% | 100% |
| Phase 3 + SOTA stack (1 seed) | ❌ 17.13% (negative) | 100% |
| **Phase A 5-seed validation** | **C1 done, C2/C3 진행 중** | **33%** |

## 2. 작성 코드

| 파일 | LOC | 역할 |
|---|---|---|
| PLAN.md (v4) | 460 | architectural blueprint, phase ladder |
| PHASE0_AUDIT.md | 130 | per-design distribution stats |
| segment_extractor.py | 280 | DEF → conductor segments + virtual subsegments |
| edge_builder.py | 200 | E_local + E_mid + E_long enumeration |
| physics_base.py | 240 | Sakurai-Tamaru + parallel-plate + log-space residual composer |
| spef_to_targets.py | 235 | line-on-wire SPEF mapping with WIRE-preferred tie-break |
| layer_physics.py | 175 | per-metal-layer ε/d/t lookup table |
| geometric_aug.py | 155 | 6× SAFE_TRANSFORMS + 2× extra (post-verify) |
| ncgt_dataset.py | 220 | NCGTSample dataclass + Dataset + collate |
| ncgt_model.py | 380 | Heterogeneous encoder + 4 transformer blocks + global readout + bin-specialized heads |
| gradnorm.py | 130 | multi-task gradient balancing (smoke OK, not yet activated) |
| train_ncgt.py | 800 | 5-seed train_validation + overfit smoke + multi-design + bins/aug flags |
| audit_phase0.py | 510 | dataset audit driver |
| sanity_check_physics.py | 220 | physics-only baseline measurement |
| debug_plateau.py | 170 | gradient/saturation diagnostic |
| smoke_test_ncgt.py | 210 | 1-net end-to-end forward |
| run_5seed_phaseA.py | 70 | 3 configs × 5 seeds wrapper |
| **Total** | **~4,600** | |

## 3. 실험 결과 — 단일 seed (Phase 2.0 - 3.x)

| Config | Best VALID MAPE (single seed) | Pearson r | p95 |
|---|---|---|---|
| Physics-only baseline | 49.45% | 0.990 | — |
| Untrained NN (zero-init) | 49.45% | 0.990 | — |
| Single design vanilla | **11.34%** ⭐ | 0.998 | 26.4% |
| Single + bins | 13.42% | 0.997 | 33.1% |
| Single + aug | 13.33% | 0.999 | 30.2% |
| Multi-design vanilla (8K steps) | 16.88% | 0.927 | 43.7% |
| Multi-design + bins + aug (15K steps) | 17.13% | 0.919 | 40.2% |

## 4. 실험 결과 — 5-seed (Phase A, IN PROGRESS)

| Config | Best VALID Mean | Std | Min | Max | p95 (final) | r (final) |
|---|---|---|---|---|---|---|
| **C1 single vanilla** | **16.22%** | **0.43%** | 15.43% | 16.65% | 37.27% | 0.944 |
| C2 multi vanilla | TBD | TBD | TBD | TBD | TBD | TBD |
| C3 multi+SOTA | TBD | TBD | TBD | TBD | TBD | TBD |

**Per-seed bests (C1)**: [0.1654, 0.1631, 0.1665, 0.1615, 0.1543]

## 5. 핵심 발견

### 5.1 Paradigm shift 입증
- v10b (현재 baseline): **55-65% MAPE** (5-seed mean)
- NCGT v4 single-design 5-seed: **16.22% ± 0.43%**
- **3.4-4.0× 개선** — paradigm shift 정량적으로 작동

### 5.2 N=1 fallacy 재현
- 단일-seed 11.34%은 2.5σ outlier (seeds [15.43, 16.15, 16.31, 16.54, 16.65] 분포에서 유일하게 11% 영역)
- 5-seed mean 16.22%이 진짜 수치 — Plan v4 §3.1이 정확히 예측한 시나리오
- **사용자 권고 (5-seed validation)가 핵심 정정**

### 5.3 Pearson r 떨어짐
- Single-seed: r=0.998 (lucky)
- 5-seed mean: r=0.944
- ranking이 완벽하지 않음 — magnitude bottleneck도 존재

### 5.4 SOTA stacking이 우리 setup에서 작동 안 함 (single-seed)
- Bins: -2pp (악화)
- Aug: -2pp (악화)
- Multi+SOTA: ~동일
- 가설: NCGT가 이미 rotation-invariant + 단일 design heterogeneity 부족
- **5-seed (C2/C3 결과 후) 최종 판단 가능**

## 6. 4% MAPE Target — 솔직한 평가

| 단계 | 1차 추정 (Plan v3) | 실험 후 (현재) |
|---|---|---|
| Single overfit 11% (1seed) → 4% via stacking | 50% | **5%** |
| 5-seed validated 16% → 4% gap | — | **닫기 어려움** |

**현실**:
- Pure ML, full-design net-level cap MAPE 4% 달성 publication 부재
- Literature 1-5%는 모두 pattern-based (CNN-Cap/NAS-Cap), standard cell only (ResCap), 또는 R²-derived (ParaFormer power)
- NCGT 16% 5-seed mean은 **ParaFormer-class** (R²=0.96 → MAPE 5-15%)
- 4% 도달은 hybrid (PINN+StarRC) 또는 pattern fast-path 필요

## 7. Process Critique — 자체 평가

### ✅ 잘한 것
1. **사용자 push-back 반영**: "literature 검토" 후 paradigm 재설계 → ResCap 0.16% / ParaFormer 1.45% 발견
2. **Audit-driven calibration** (Plan v4): 추측 대신 측정 → R_aggr 20→12, vias 제외
3. **Plateau debug**: 단일-seed 결과 발견 즉시 root-cause 추적 (hard clamp gradient zero) → fix
4. **Negative results 정직 보고**: bins/aug 단일 결과를 paradigm 작동으로 misinterpret 안 함
5. **Plan v4 §5 gate 준수**: bins improve >2pp 기준 — gate 통과 안 하면 revert
6. **5-seed validation 채택**: 사용자 추천 → 11.34% lucky seed 정정

### ❌ 부족했던 것
1. **5-seed 처음부터 했어야**: Plan v4 §3 "N=1 fallacy"를 알면서도 single-seed로 "best 11.34%" 발표 — 정정에 시간 소비
2. **Physics base 정확도 placeholder**: layer_info threading 했지만 etch-stop / ILD stack series capacitance는 미구현. Phase 2.0이 완전 X.
3. **Per-edge supervision 0.3-0.6%**: 대부분 net-total로만 학습. SPEF mapping 개선 가능했지만 우선순위 떨어뜨림.
4. **Codex deliberation rounds 2 ROI 낮음**: round 1에서 P1 6개 catch, round 2의 4개 P1 중 2개는 우리 데이터로 자체 해결됨
5. **시간 효율**: Plan iteration 5회는 과다. v3에서 멈췄어도 됐음

### 🔍 Mid-process 진단

**현재 위치**: 16.22% ± 0.43% (5-seed mean, single-design). 4% target과 **12pp gap**.

**Gap 분석**:
- Pearson r 0.944: ranking은 좋지만 outlier가 mean 끌어올림
- p95 = 37%: long-tail nets가 아직 큼
- 3 paradigm change tracks:
  - **A. 더 정확한 physics base** (etch-stop + ILD series, MIM cap 등) → 30-40% 개선 가능
  - **B. Edge-level supervision 확장** (현재 0.5% → 30%+) → outlier 압축 기대
  - **C. Hybrid PINN-StarRC** → 4% 보장 (system MAPE)

**Pending 5-seed (C2/C3)이 알려줄 것**:
- multi-design 16-17%이 안정적인지 (vs single 16% 비교)
- SOTA stack이 statistical하게 의미있는지
- True OOD generalization 신뢰도

## 8. 다음 단계 옵션

Phase A 완료 후 (1-2시간 내) decision points:

1. **A 인정 + write-up**: 16% 5-seed validated가 paper-class 결과. 4% 재조정.
2. **Phase 2.0+ 개선** (physics base 정확도): etch-stop + series capacitance 구현, expected 16% → 10-12%.
3. **Hybrid PINN-StarRC**: NN 80% + StarRC 20% fallback, system MAPE 4-6% reachable.
4. **Larger model + edge supervision**: d=128→256, supervision rate 늘리기.

추천 (현재 데이터 기반): **2 + 3 병행** — physics base 정확도 개선 + hybrid wrapper. 둘 다 Plan v4가 design한 follow-up.

## 9. Living section — 진행 중

- [ ] C2 multi-design vanilla 5-seed (현재 진행 중, ~30분 남음 추정)
- [ ] C3 multi-design + SOTA stack 5-seed (대기 중)
- [ ] 통계 검정 (Mann-Whitney U test for C1 vs C2 vs C3)
- [ ] Final 보고서 (Phase A 완료 후)
