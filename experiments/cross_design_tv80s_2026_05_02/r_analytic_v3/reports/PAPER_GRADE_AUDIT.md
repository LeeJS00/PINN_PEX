# Paper-grade Audit — v3 R/c_gnd Hybrid Approach

_작성일: 2026-05-03 KST. 최상위 학회 (DAC/ICCAD/DATE/TODAES) 제출 가능성 검토._

---

## 0. 요약 — 현 상태와 paper-grade 갭

| 항목 | 현 상태 | Paper-grade 요구 | 갭 / 작업 |
|---|---|---|---|
| **I/O 계약** | DEF+LEF → SPEF (PEX 툴 동일) | ✅ 동일 요구 | **OK** (`predict_spef_e2e.py`) |
| **Train/Test split** | nova∈TRAIN, ldpc 누락, tv80s만 TEST | TRAIN 9 (incl ldpc), TEST nova+tv80s (canonical) | **🚨 LEAKAGE — 재실행 필수** |
| **PINN 역할** | Stage 1 NNLS (physics 계수) + Stage 2/3 GBT | "PINN" 명칭 정당화 필요 | **명칭 재정의** (physics-anchored ML) |
| **R 성능** | tv80s 2.21% MAPE | nova/tv80s 분리, 각 design baseline 비교 | **재측정 (canonical split)** |
| **c_gnd 성능** | 26.47% (v3 v6) | 21% pex_v3 ceiling 명시 | **re-frame as "matched ceiling"** |
| **Runtime** | "~5min/3380 nets" 주장 (구체 측정 없음) | per-stage breakdown, vs StarRC | **end-to-end 벤치마크 필요** |
| **Baselines** | v7 ML (legacy) | XGBoost, ParaGraph GNN, ResCap | **pex_v3 결과 활용 + 직접 재현** |
| **Contribution** | "분석식 + ML residual로 R MAPE 5.4× 개선" | 명확한 novelty + ablation matrix | **framing 강화** |

**🚨 Blocker**: 우리 모든 v3 결과 (R 2.21% 포함) 는 nova-leakage 영향 받음. canonical split 재실행 후에야 paper-grade.

---

## 1. I/O 계약 — ✅ PEX 툴과 동일 (확인)

`scripts/predict_spef_e2e.py` 는 IEEE 1481-1999 SPEF 출력하며 입력은 raw DEF + LEF + tech files:

```
입력:                                        출력:
  --def_path <path/to/design.def>             --out_spef <path/to/predicted.spef>
  (LEF, layers.info 는 cfg 자동 로드)            (lumped per-net topology)

7 stages:
  [1] DEF/LEF/layers parse → cuboid pkls (build_dataset.py)
  [2] cuboid pkls → 145-dim hand features
  [3] cuboid pkls → per-(target,aggressor) pair features
  [4] cuboid pkls → 3-stream cuboid arrays
  [5] features → predicted total_cap, c_gnd_ratio, total_R (ML)
  [6] split + per-pair distribute (geom heuristic) + analytic R fallback
  [7] write lumped SPEF
```

**동등 EDA 툴**: StarRC, Cadence Quantus. 입력 동일 (DEF + LEF), 출력 동일 (SPEF). Voltus 같은 IR drop 분석 도구가 이 SPEF 를 그대로 consume.

---

## 2. Train / Test Split — 🚨 LEAKAGE 발견

### Canonical split (`configs/config.py`)

```python
TRAIN_DEFS = [aes, gcd, ibex, ldpc, mc, spi, usbf, vga_enh, wb_conmax]   # 9
TEST_DEFS  = [nova, tv80s]                                                # 2
```

이는 pex_v3 (다른 세션) 도 따르는 split.

### 우리 v3 r_analytic_v3 split (현재)

```python
DESIGNS_TRAIN = [aes, gcd, ibex,         mc, spi, usbf, vga_enh, wb_conmax, NOVA]   # 9 (ldpc 누락, nova 포함)
DESIGN_TEST   = [tv80s]                                                              # 1
```

**위반 사항**:
1. **nova 가 TRAIN에 들어가 있음** — canonical split 에서는 OOD test 의 일부
2. **ldpc 누락** — canonical TRAIN의 일부 (큰 design, ~169K nets)
3. **tv80s 만 test** — pex_v3는 nova+tv80s 두 design 평균 OOD test로 보고

### 영향
| 결과 | 우리 보고 | 실제 OOD 가치 |
|---|---|---|
| R MAPE on tv80s = 2.21% | 우리 v3 stacked best | nova-leakage 가능 (nova design pattern 학습) |
| nova MAPE | **미측정** | 실제 OOD 성능 알 수 없음 |
| c_gnd MAPE on tv80s = 26.47% | v3 best | 동일 leakage |
| ldpc 영향 | 학습에 미사용 | ldpc 의 다양한 routing pattern 학습 안 됨 → 잠재 underfit |

**Action**: canonical split 으로 재실행 필요 (Task 22).

---

## 3. PINN의 역할 — 명칭 재정의 필요

### "PINN-PEX" 의 명칭에서 "PINN" 의 의미

문헌 (PINN, Raissi et al. 2019): NN with PDE residual loss = `L = L_data + λ·L_PDE`.

### 우리 v3 hybrid 의 실제 architecture

| 단계 | 형태 | "PINN"인가? |
|---|---|---|
| Stage 1: NNLS-IRLS | linear regression with non-negative constraint | ❌ NN 아님. **physics-interpretable linear regression** |
| Stage 2: LightGBM ensemble | 5-seed depth-4 GBT on relative residual | ❌ NN 아님. tree boosting |
| Stage 3: LGBM stacking | 3-seed GBT on Stage 2 residual | ❌ NN 아님 |

**우리 작업은 PINN 이 아님**. 정확한 명칭:
- "Physics-anchored ML calibration"
- "Hybrid analytic-NNLS + boosted-tree residual"
- "PEX-style ground-truth-calibrated linear+GBT pipeline"

### Legacy `DeepPEX_Model` (PINNPEX 코어 src/models/neural_field.py)

원래 PINNPEX repo 는 NN-based:
- `CuboidEncoder` + `NeuralFluxRouter` (KCL + 1-hop attention)
- 하지만 PDE residual loss 없음 → 진짜 PINN 도 아님
- "Physics-informed" 정도의 설계

### Paper framing 권장
- 제목 / Abstract 에 "PINN" 단어 사용 자제
- 대신: "Physics-anchored hybrid extractor" 또는 "Calibrated analytic + GBT ensemble"
- pex_v3 의 Phase 1 (analytic Green's function + bounded **neural** residual) 이 진짜 "physics-informed neural" 후보

---

## 4. 성능 — 재측정 필요 (canonical split)

### 현 결과 (nova-leakage 가능)

| 지표 | tv80s test | 비고 |
|---|---|---|
| total_R (v3 stacked) | 2.21% | nova-leak 영향 미정량 |
| c_gnd (v3 v6 best) | 26.47% | 같은 leak |
| total_cap (legacy v7 ML) | 8.11% | 미수정 |

### Canonical split 재실행 후 예상

| 지표 | 예상 (보수적) | 예상 (낙관) |
|---|---|---|
| R on tv80s | 2.5–3.5% | 2.21–2.5% |
| R on nova | **미측정** (4-6% 추정) | 3-4% |
| c_gnd on nova/tv80s | 22-26% (pex_v3 21% 와 일관) | 22% |

**가설**: nova-leakage 영향은 작을 것 (디자인 특수 정보보다는 layout 통계가 dominant). 그러나 **반드시 측정해서 보고**.

### Per-channel 분해 (paper essential)

pex_v3 가 명시한 cancellation 패턴:
- total_cap MAPE 4-6% = c_gnd 21% + c_cpl 12-14% **상쇄**
- per-channel 각 21%/13% 가 진짜 학습 한계

우리 paper에 동일하게 reporting 필요:
- total_R, total_cap, c_gnd, c_cpl_total, per-pair coupling — 각각 별도 보고
- median, mean, P90, P95, bias, 95% CI — 모두

---

## 5. Runtime — 측정 필요

### 현재 주장
- `predict_spef_e2e.py` doc: "~1-3 min for typical design"
- 우리 보고: "~5 min/3380 nets vs StarRC 30 min → 5-7× speedup"

### 측정해야 할 것
1. **End-to-end runtime per design** (nova, tv80s)
   - Stage-by-stage breakdown:
     - [1] DEF→cuboid pkl
     - [2] cuboid→145-dim feat
     - [3] pair features
     - [4] cuboid arrays
     - [5] ML inference (5 LGBM + others)
     - [6] decompose + distribute
     - [7] SPEF write

2. **vs StarRC actual time** (단일 머신 같은 조건)
   - tv80s: 우리 ___s vs StarRC ___s
   - nova: 우리 ___s vs StarRC ___s

3. **Memory peak**

4. **Per-net inference time** (μs/net)

5. **Speedup vs StarRC** with statistical reporting (multi-run mean ± std)

Action: Task 23.

---

## 6. Baselines — 부족함

### 현재 우리 비교 대상
- v7 ML (legacy 47-model ensemble) — **자체 baseline, weak**
- v2 analytic (calibrated sheet R + α) — **자체 baseline**

### Paper 에 필수 baselines
| Baseline | 상태 | 가능 여부 |
|---|---|---|
| **Compact analytic** (Sakurai-Tamaru) | 미실시 | 직접 구현 (1-2일) |
| **XGBoost on hand features** | pex_v3 수치 활용 가능 | pex_v3 4.66% B1 결과 인용 |
| **ParaGraph (GNN)** | pex_v3 가 시도 중 | pex_v3 결과 대기 |
| **CNN-Cap / NAS-Cap** | per-pattern 만, full-net 적용 어려움 | 문헌 인용으로 정성 비교 |
| **ResCap (ASPDAC 2025)** | physics-base + ML residual | 우리 방법론과 paradigm 일치 — 직접 비교 |
| **StarRC golden** | 정확도 100% (oracle) | 항상 비교 reference |

### Ablation matrix (paper 필수)
| 변형 | total_R | c_gnd | runtime |
|---|---|---|---|
| (1) Pure analytic (calibrated sheet R + α) | 6.99% | — | <1s |
| (2) NNLS linear (physics features) | 3.30% | 26.47% | <1s |
| (3) (2) + cell OBS + cell SIZE | 2.25% (R) | 24-26% | <1s |
| (4) (3) + LGBM ensemble | **2.21%** (R) | (best v6) | <2s |
| (5) (4) + Stage 3 stacking | 2.21% | — | <2s |

### 통계적 검정 (paper 필수)
- 5-seed × 2 (nova, tv80s) bootstrap CI
- paired Mann-Whitney U test (vs strongest baseline)
- Cohen's d effect size

---

## 7. Contribution — Framing 강화 필요

### 현재 contribution 명세
- "v7 ML 11.92% → v3 hybrid 2.21% on tv80s" — **OOD leak 가능성 있음**

### Paper-grade contribution (재구성)

**Primary contribution** (R 영역, 강함):
> *Calibrated analytic-base + GBT-residual* 구조로 cross-design OOD test 에서 **R MAPE < 3%** 달성 — 기존 ML approach 의 ~12% 대비 **4-5× 개선**. 본 정확도는 (a) per-segment golden RES 로부터 sheet R / via R 을 직접 calibrate, (b) cell LEF OBS section 에서 cell-internal routing 추출 (signal-power 분리), (c) NNLS 의 physics-interpretable coefficient + LGBM 의 비선형 residual 결합으로 달성됨.

**Secondary contribution** (c_gnd 영역, ceiling 분석):
> Hand-feature ML 의 **per-channel c_gnd ceiling = 21%** 를 cell OBS 추가로도 깨지 못함 (multiple ML 방법 수렴). 이는 c_gnd 가 transistor characterization (`.lib`) 영역의 정보가 dominant 함을 시사하며, paradigm shift (analytic Green's function + bounded neural residual) 가 필요함을 정량화.

**Engineering contribution**:
> Production-ready DEF+LEF→SPEF pipeline (`predict_spef_e2e.py`), StarRC 의 1/5-1/8 시간으로 동등 SPEF 출력. Cell LEF OBS-aware feature extraction + 비파괴 calibration update 메커니즘.

### Novelty 차별화
| Prior work | 우리 |
|---|---|
| ResCap (ASPDAC 2025) | similar paradigm (physics base + ML residual) for cap; we extend to **R + cell-OBS-aware** |
| CNN-Cap | per-pattern only; we do **full-chip lumped SPEF** |
| ParaGraph | GNN; we use **interpretable NNLS + tree** (lighter, deployable) |
| StarRC | golden field solver; we **mimic at 5-7× speed** |

---

## 8. 즉시 실행 작업 (Re-run for paper-grade)

### Task 21 — Build ldpc features
- ldpc DEF + golden SPEF available
- Run `build_segment_features.py`, `build_features_v2.py`, `pins`, `feat_v6` for ldpc
- ~30 min (대형 design, 169K nets)

### Task 22 — Canonical split re-run
- TRAIN: aes, gcd, ibex, ldpc, mc, spi, usbf, vga_enh, wb_conmax
- TEST: nova + tv80s
- Re-run **all** v3 fits with this split
- Report nova / tv80s / combined separately
- Expected: ~30-60 min (cached features 활용)

### Task 23 — Runtime benchmark
- Cold start: `predict_spef_e2e.py` on nova (raw DEF) + tv80s
- Stage breakdown table
- vs StarRC same-machine reference

### Plus: 명칭 / framing 정리
- Paper draft 에서 "PINN" 단어 사용 신중
- Contribution 재구성 (Primary R, Secondary c_gnd ceiling, Engineering pipeline)

---

## 9. 산출물 / 다음 단계

### 이 audit 결과 즉시 시정
1. ✅ Task 20 — 본 audit 보고서
2. ⏳ Task 21 — ldpc features 빌드
3. ⏳ Task 22 — canonical split re-fit
4. ⏳ Task 23 — runtime 측정

### Paper-ready 완료 후 산출
- `r_analytic_v3/reports/PAPER_GRADE_FINAL.md` — 모든 측정 + contribution + ablation
- `r_analytic_v3/outputs/canonical_split_results.json`
- per-design (nova, tv80s) per-channel (R, c_gnd, c_cpl, total_cap) MAPE table
- runtime breakdown CSV

---

## 10. 결론

**현 상태로는 top venue 제출 어려움**:
- nova-leakage (TRAIN에 nova 포함)
- ldpc 누락
- nova test 미실행
- runtime 측정 부재
- "PINN" 명칭 부정확

**시정 후 strong contribution 가능**:
- R 영역: 4-5× MAPE 개선 (3-4% target on full OOD)
- c_gnd: ceiling 정량화 (paradigm-shift 의 명확한 동기 제공)
- Engineering: 5-8× speedup w/ standard SPEF I/O

**다음 immediate action**: Task 21-23 진행. 1-2일 내 완료 가능 (cached features 활용).
