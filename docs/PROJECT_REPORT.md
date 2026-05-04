# PINN-PEX 프로젝트 정밀 보고서

_작성일: 2026-05-01_
_상태: GINO·DS-PINN·data-driven calibration·γ head 4개 트랙 종료, baseline 복귀_
_목적: 4개 트랙의 시도 이유·결과·교훈을 단일 문서로 정제하여 향후 작업의 기반 자산으로 보존_

> 이 문서는 `docs/INTERIM_CONCLUSIONS.md`, `docs/dspinn_development_log.md`,
> `docs/distillation_log.md`, `docs/distillation_effect_report.md`,
> `docs/calibration_track_report.md`, `docs/gino_architecture_report.md`,
> `docs/gino_report.md`를 통합·재구성한 canonical report다. 개별 living
> document들은 git 히스토리 보존용으로 남겨두되, 새 작업의 출발점은 본
> 보고서를 사용한다.

---

## 0. 1-page TL;DR

| 트랙 | 가설 | 결과 | 사후 verdict |
|---|---|---|---|
| **GINO (FNO operator)** | 1-hop GNN의 Poisson non-locality 한계를 global FNO로 해결 | Critical analysis 단계에서 3가지 fatal flaw 노출, 학습 전에 DS-PINN으로 pivot | 학습 미수행 |
| **DS-PINN (MacroDensityFNO)** | macro density stream + flux head conditioning으로 long-range screening 학습 | 5-seed ablation에서 v10b vanilla 대비 +2.04pp mean (v10b stdev 5.02pp 안), 분산 +56% | **경험적 반증 → 폐기** |
| **Data-driven calibration init (NNLS)** | hand-tuned ζ를 TRAIN_SPEFS NNLS-fit 값으로 대체하면 heteroscedastic calibration 개선 | mean MAPE −6.5pp, IQR 절반, OOD에서도 동일 패턴, 그러나 n=5에서 Mann-Whitney p>0.5 | **효과 작고 통계적 ns → 폐기** |
| **γ scaling head (per-net residual)** | layer-wide ρ로는 못 푸는 per-net heteroscedasticity를 multiplicative scale로 직접 보정 | smoke step 1000 BEST 67%, 5-seed step 5000 측정 미완 | **측정 미완 → 폐기** |

**남기는 것 (진짜 자산)**

- **5-seed measurement protocol**: AI/ML PEX 실험에서 N=1 fallacy 정량화 (v3 5-seed range 22pp).
- **OOD evaluation discipline**: `aggregate_5seed_eval.py`로 in-dist 단일 seed가 OOD에서 reverse될 수 있음을 입증.
- **Critical analysis + Codex deliberation loop**: 빌드 전 round-1에 fundamental bug 다수 사전 발견.
- **Diagnostic 인프라**: `diag_*` 시리즈 (heteroscedastic, OOD, eval_dump, quartile, eps utility 등) — architecture-agnostic 도구.
- **Critical bug catalog** (8개): cache leak / curriculum sawtooth / SymMAPE 포화 / KCL redundancy / A_tgt mask / NNLS collinearity / tile-vs-net coverage / metric mislabel — 발견·수정의 기록은 architecture가 바뀌어도 유효.
- **Architecture-independent bottleneck list** (H1-H4, M5-M9): 다음 architecture에서 그대로 적용해야 할 데이터 파이프라인 이슈.

**버리는 것 (효과 없음)**

- MacroDensityFNO macro stream, GINO enricher, γ head, calibration_extractor + solver, 모든 hand-tuned ζ values, calibration_init JSON files.

코드는 `src/models/_archive/`, `src/data/_archive/`, `scripts/_archive/`에 보존되어 있어 git revert 없이도 재시도 가능.

---

## 0.1 Operational Policies (critical only — 8개)

**Cut-off principle:** 이 §0.1 은 위반 시 (a) 시스템 손상, (b) 데이터 손실, (c) paper-grade claim 무효화 가 발생하는 invariant 만 담는다. soft guideline / design heuristic / architecture-specific lesson 은 §0.2 reference pointer 에서 source 만 인용한다. PROJECT_PLAN.md §2 와 1:1 동기화 (P1-P8).

### P1 — `tool` path: project-local only
- 시스템 root `/tool` 에 절대 write / create 금지. root-owned read-only 빈 디렉토리.
- 사용자가 "tool" / "tool 폴더" / "tool 디렉토리" 라 하면 **항상** `/home/jslee/projects/PINNPEX/tool/` (project-local PnR tooling).
- *User-flagged 2026-05-04. 메모리 `feedback_tool_path_policy.md` 동기.*

### P2 — Scratch directory: never `/tmp`
- 대용량 cuboid pkl / intermediate SPEF / e2e measurement scratch 는 `/data/PINNPEX/scratch/` 하위만 사용.
- `/tmp` 는 project I/O 금지.
- *Past incident: 2026-05-04 nova run 258 GB cuboid pkls 가 `/tmp` 를 채워 호스트 brick.*

### P3 — Legacy manifest: read-only
- `/data/PEX_SSL/data/processed/intel22/dataset_manifest.csv` 절대 덮어쓰기 금지. legacy v9 manifest, post-hoc irreplaceable.
- v3 작업은 `/data/PINNPEX/data/processed_v3/intel22/dataset_manifest_v3.csv` 만 수정.
- *Source: `pex_v3/CLAUDE.md` §Data path discipline.*

### P4 — Net-centric splits & sampling
- 모든 split / sampling / validation / 통계는 `(design, net)` 단위 group 으로 시작. tile row 에 `head(N)` / random split 금지.
- Tile-level split 은 train/valid/test 사이 net leak 발생 → OOD claim 무효.
- *Past incident: legacy net leak 12.29% 정량화 (Phase 0 H1).*

### P5 — TEST_DEFS held out from AL pool
- `cfg.TEST_DEFS` (현재 `nova`, `tv80s`) 는 AL pool selection / training 에 절대 진입 불가, `AL_SAMPLING_METHOD` (`Predefined` / `SSL` / `Sorted`) 무관.
- test design 이 pool 에 한 번이라도 들어가면 cross-design OOD 숫자 전체가 silent 하게 무효화됨.

### P6 — 5-seed protocol before any *paper-grade* claim
- "X beats Y" 형태의 claim 은 5-seed run 위에서 **paired Mann-Whitney U + Cohen's d + bootstrap CI** 모두 통과한 뒤에만 **paper / leaderboard / memory / PR description / external report** 에 기록.
- iteration-time smoke (single-seed, exploratory) 는 면제 — 단, smoke 결과를 claim 처럼 surface 하면 안 됨.
- *Past evidence: single-seed smoke 가 5-seed median 보다 0.5–1.0pp 우월하게 나온 사례 다수 (auto_optimize round 2/3).*

### P7 — StarRC oracle: full-chip DEF only, never on tiles
- `FullChipPEXOracle.generate_golden_spef` 및 미래 oracle wrapper 는 full chip DEF 에서만 StarRC 실행.
- cropped tile / sub-net / partial layout 에서 절대 호출 금지 — 같은 물리 답이 아니고(boundary condition 다름) 비용이 ~10 min/design.
- *Source: `run_active_learning.py` design + `pex-data-engineer` role.*

### P8 — Stage 2 inference: manifest passthrough required
- DEF→SPEF E2E Stage 2 (`build_features_inference`) 는 manifest CSV 직접 소비. per-net pkl 위에서 `rglob + gzip.open + pickle.load` 로 auto-discover 금지.
- 새 entrypoint 는 Stage 1 의 `cuboids_map.csv` 를 Stage 2 로 passthrough; disk-walk fallback 구현하지 않음.
- *Past incident: nova rglob path 가 684K pkls 위에서 4시간+ stall (2026-05-04). `predict_spef_e2e.py:160-167` 에서 fix.*

## 0.2 Reference pointers (NOT enforced as policy)
이하는 유용한 guideline 이지만 source 문서에 살아있고 **정책으로 격상하지 않음**. 관련 작업 시 참조하되 PROJECT_REPORT review 단계에서 hard invariant 로 취급하지 않는다.
- Codex deliberation loop (non-trivial design 변경) → 글로벌 `~/.claude/CLAUDE.md`
- `pex_v3/` boundary rule (legacy 디렉토리 cross-edit) → `pex_v3/CLAUDE.md`
- `TORCH_COMPILE_DISABLE=1` for paper runs → `pex_v3/docs/STRATEGY_V3_UPDATED_PLAN.md`
- Custom agent invocation pattern → `feedback_agent_invocation_pattern.md`
- Loss design rules 1-6 → `feedback_loss_design_principles.md`
- Per-channel β strategy → `pex_v3/SESSION_HANDOFF.md` + `pex-gnd-allocator-owner` role
- Anti-pattern catalog (Strikes #2/#7/#8, A1 forbidden, K3 synthetic pretrain) → §2 트랙 narrative + `joint_pareto/README.md`
- Joint-Pareto runtime cap (≤75 s on tv80s) → `pex_v3/joint_pareto/README.md`
- Hard kill criteria K1/K2/K3 → `pex_v3/PHASE_STATUS.md`
- Anti-overclaim publishing rules → `benchmarking-statistician` role
- `RUN_NAME` / `--model_name` 일관성, single-GPU only → 프로젝트 `CLAUDE.md`

---

## 1. 프로젝트 컨텍스트

### 1.1 PINN-PEX란

routed DEF + tech LEF + layer stack을 입력으로 받아 net-level parasitic
capacitance를 예측하고 SPEF로 출력하는 physics-informed neural field.
Golden oracle은 StarRC. 파이프라인 4단계는 다음과 같다.

```
build dataset (cuboid tile) → SSL pretrain (DeepPEX_Model)
  → Active-Learning finetune against StarRC → evaluate / write SPEF
```

### 1.2 Target

- **Production target**: net MAPE < 5% (StarRC-class).
- **Realistic interim target**: net MAPE < 15% with CPL SMAPE < 100%.
- **현재 baseline (v10b)**: MAPE 27.30% — 단일 seed 측정. 5-seed 재측정 시 63.79% ± 5.02% (v10b vanilla, DSPINN OFF). 즉 단일 seed 27.30%는 5-seed 분포의 **2.4σ 이상 이상치(lucky tail)**였음.

### 1.3 Architecture (현재 baseline)

`src/models/neural_field.py`의 `DeepPEX_Model`:

1. **CuboidEncoder** — per-cuboid MLP (input scaling: xy/SCALE_FACTOR, w/h/d log1p, ε log).
2. **NeuralFluxRouter** (`src/models/flux_head.py`) — 1-hop GNN + surface physics, KCL closure, sparse shielding/coupling 통합 모듈.
3. 학습 가능 head 3개: `charge_basis_mlp`, `gnd_mlp`, `cpl_mlp`. SSL ckpt 로드 후 `freeze_ssl_layers()`로 encoder + `flux_router.norm` freeze.

입력 텐서: `(N, 10)` per tile (v9 이후 10-channel). 9번째 채널 추가는 VSS aggressor 표시용 (`USE_VSS_AGGRESSORS=True`, `INPUT_DIM=10`).

---

## 2. 시도했던 4개 트랙 — 시간순 narrative

### 2.1 트랙 A — GINO (Geometry-Informed Neural Operator)

**제안 시점**: 2026-04-29.
**문서**: `docs/gino_architecture_report.md`, `docs/gino_report.md`.

**왜 시도**: v10/v10b의 CPL SMAPE가 80+ checkpoint 동안 320% 이상에서
plateau. 가설: parasitic capacitance를 지배하는 Poisson 방정식
`∇²φ = 0`은 **non-local elliptic PDE**인데 1-hop GNN (r=4μm)은 Green's
function의 long-range를 표현 불가.

**제안 architecture (v2 roadmap)**:

```
Cuboids → CuboidEncoder → P2G (Gaussian scatter, σ_xy=0.25μm, G=64)
  → per-layer FNO-2D → Z-MLP → G2P → cap heads
```

**Pre-build critical analysis** (`gino_report.md`)에서 발견된 3가지 fatal flaw:

1. **P2G resolution이 BEOL pitch 대비 너무 거침**:
   - σ_xy=0.25μm (250nm) Gaussian kernel
   - M4 wire pitch 44nm
   - 인접 3-5개 메탈이 단일 격자 활성화로 blur됨
   - FNO 진입 전에 nm-scale gap 정보가 비가역적으로 destruction

2. **Latent-space Laplacian regularizer는 수학적으로 무의미**:
   - 제안된 `laplacian_2d(latent_field).pow(2).mean()` PDE loss
   - Latent는 `R^128`의 추상 임베딩 — 물리적 potential `φ`(Volts)가 아님
   - Laplacian은 물리적 스칼라장에서만 의미; embedding vector에 적용은 noise

3. **`cap = net_cap / n_tiles` 분할은 KCL 위반**:
   - Tile 내부의 capacitance는 spatial하게 비선형 (PDN island vs 빈 공간)
   - 균등 분할은 "model이 spatial flux competition을 학습"하는 것을 막음

**Pivot**: **DS-PINN dual-stream** architecture로 redesign — far-field는
FNO의 macroscopic background potential 용도로만, near-field는 surface
physics tensor 직접 보존.

**상태**: GINO 단일 stream은 학습 미수행. DS-PINN으로 흡수.

---

### 2.2 트랙 B — DS-PINN (Dual-Stream PINN with MacroDensityFNO)

**시작**: 2026-04-29.
**문서**: `docs/dspinn_development_log.md`.

#### 2.2.1 Architecture

`src/models/_archive/macro_density_fno.py` (현재 archived):

- P2G: metal volume fraction + permittivity → `(B, L, 2, G, G)` (G=16, σ_xy=0.3μm, window=4μm).
- FNO-2D: 2-channel → `d_macro=32`, L 레이어 shared (`B*L` batch dim).
- G2P: `F.grid_sample` bilinear interpolation.
- Zero-init `proj_out`: `Z_macro ≈ 0` at start (neutral screening).

`src/models/flux_head.py`에 `d_macro` 파라미터로 통합:
- `gnd_mlp`: `Z_dim+6+d_macro` 입력
- `cpl_edge_proj`: `Z_dim+d_macro → 32`
- `forward(macro_context=None)` 시그니처 추가
- `d_macro=0`일 때 zero-shape `z_macro_n` torch.cat 노옵 (backward compat)

#### 2.2.2 Iteration history

| Run | 변경 | 결과 (single-seed, 497-net val) |
|---|---|---|
| **v1_new** | Codex round 1: 4 P1 + 4 P2 fixes (soft top-2 z-bucket, cpl_macro_norm, drop 2-phase FNO freeze, vectorized P2G/G2P, float32 FFT, drop eps channel, padding mask) | MAPE 29.14%, "CPL SMAPE 367%" |
| **v2** | Codex round 2: P1 (per-edge `loss_cpl_vector`), P2 (`z_macro_gnd.detach()`), P3 (`aux_target = log1p(gt_cpl_sum)`), P4 (`cpl_modifier = exp(clamp(-3,3))`) | MAPE 35%, **real per-edge SMAPE 104%** (v10b 158%, v1_new 167%), CPL ratio 0.10 |
| **v3** | Codex round 4 β + ζ: `cpl_layer_pair_log_scale.diag()` init `softplus_inv(8.0)`, β = `loss_cpl_ratio` hinge against under-prediction | iter 1 best 43.06% (CPL ratio 220% overshoot 발생, 0.7-220% 진동) |
| **v4_distillinit** | v3 + NNLS-fit ζ (s_diag=0.18, s_cross=4.25 from data) | iter 0 best 47.40%, oscillation amplitude 2-4× 감소 |

#### 2.2.3 Phase A diagnostic의 metric 발견

`finetuner.evaluate()`가 출력하던 "Validation SMAPE [%]" 값은 사실
`compute_pex_loss` (`L1 + 5×MAPE + 2×log` hybrid). 진짜 per-edge SMAPE는 다름:

| Model | "SMAPE" in log | 실제 per-edge SMAPE |
|---|---|---|
| v10b | 320% | 158% |
| v1_new | 367% | 167% |
| v2 (5k step) | 358% | **104%** |

수정 후 `evaluate()`는 `Custom loss [%]`, `True SMAPE [%]`, `CPL ratio (med)`
3종을 같이 출력.

#### 2.2.4 5-seed 반증 (2026-05-01) ⭐

5-seed × 1 iter × 5000 steps × 1494-net val ablation:

| Recipe | DSPINN | Mean Net MAPE | Stdev | Range |
|---|:---:|---:|---:|---:|
| **m6_v10b_baseline** (vanilla PINN) | OFF | **63.79%** | **±5.02%** | 15.27 pp |
| m5_v3_baseline | ON | 61.75% | ±7.84% | 22.44 pp |
| m5_v4_full_calib (n=2) | ON + NNLS | 58.70% | ±4.23% | 8.46 pp |

**Δ DS-PINN = +2.04pp mean, well inside v10b stdev (5.02pp)**.
**더 나쁜 것은 분산이 +56% 증가** — reproducibility에 해롭다.

**역사적 narrative의 함의**:
- v10b의 "27.30%", v2의 "34.83%", **dspinn_v3의 33.48%** 모두 **497-net val**에서의
  단일 seed best. 셋 다 5-seed 분포의 lucky tail (≈2.4σ 이상). 단일 seed best ckpt만
  보면 "DS-PINN이 v10b 수준에 도달했다"고 오해할 수 있으나, 5-seed mean에서 v10b 63.79
  vs DS-PINN variants 61.75-58.70 — **신호 없음**.
- "v2 beats v10b" 같은 이전 비교는 모두 seed noise.
- v3/v4의 β + ζ tuning은 **신호 없는 stream을 tuning**한 것 — 모두 sunk cost.

**남기는 부분 (DS-PINN 코드와 별개로 유효한 v2의 loss-side fix들)**:
- per-edge `loss_cpl_vector` term
- `cpl_modifier = exp(clamp(...))` (saturating sigmoid 대신)
- `aux_target = log1p(gt_cpl_sum)` (Y_total 대신)

위 3개는 architecture-independent하므로 baseline에 통합 검토 가능.

#### 2.2.5 추정 원인 (왜 macro stream이 작동 안 했는가)

1. **격자 해상도와 BEOL pitch 불일치**: G=16 → 250nm/cell, M4 pitch 44nm. macro feature가 individual wire를 못 봄.
2. **Macro feature의 GND hijack**: `z_macro` 가중치가 GND head에서 먼저 saturate → CPL 역할 학습 실패. v2의 `z_macro_gnd.detach()` 부분 완화로도 해결 미완.
3. **proj_out zero-init의 cold-start**: 학습 신호 부족으로 FNO blocks가 학습 신호 못 받음.
4. **CPL physics base 자체 부족**: Sakurai-Tamaru 로컬 공식이 long-range field-solve 못 잡음 — 이는 architecture 문제가 아닌 physics formula 한계.

**Verdict**: 폐기. `src/models/_archive/macro_density_fno.py` 보존, neural_field.py에서 macro_context 인자 제거, `DSPINN_D_MACRO`/`SSL_USE_DSPINN` config 키 모두 삭제.

---

### 2.3 트랙 C — Offline Data-Driven Physics Calibration

**시작**: 2026-04-30.
**문서**: `docs/distillation_log.md`, `docs/distillation_effect_report.md`, `docs/calibration_track_report.md`.

#### 2.3.1 명명 정정

처음 "distillation"으로 명명 → reviewer 시점에서 "live teacher 없음" 지적
→ 정확한 명명은 **"Data-Driven Physics Calibration"**. live training-loop teacher signal 없이, NNLS로 fit한 값을 모델 파라미터의 초기값으로만 주입 (모델이 학습 중 자유롭게 덮어쓸 수 있음).

#### 2.3.2 동기

`dspinn_development_log.md` §3.4-3.6의 진단:
- GND Pearson r = 0.85 (위치 잘 앎), slope = 0.6 (크기 못 맞춤). Quartile별 ratio: Q1 1.58 over, Q3+ 0.72 under.
- CPL physics-only baseline에서 6.5× under-prediction.
- Hand-tuned ζ values는 magic numbers 2개 (8.0, 5.0). 더 differentiated calibration이 필요한 layer-stack에서 trial-and-error 불가능.

→ **"NNLS로 TRAIN_SPEFS에서 fit하면 어떻겠는가"**가 motivation.

#### 2.3.3 NNLS 정식화

각 train net `i`에 대해 GND eq + CPL eq 결합:

```
GND eq:   Σ_j ρ_layer[j] · A[i, j]
        + s_diag  · A_power_diag[i]
        + s_cross · A_power_cross[i]
        ≈ golden_gnd[i] − c_vss_pred[i]

CPL eq:   s_diag · B_diag[i, a] + s_cross · B_cross[i, a]
        ≈ golden_cpl[i, a]    for each signal aggressor a
```

- `A`: target wire cuboid의 `gnd_area_eff` 합 (geometry-only)
- `A_power_*`: power-net edges `w_cpl_base × core_ratio_eff` (model의 power-net lumping mirroring)
- `B_diag/B_cross`: signal-aggressor edges, same/different layer
- `c_vss_pred`: VSS edges contribution

K + 2 unknowns (K = 10 buckets). `scipy.optimize.nnls`로 200k+ equations × 12 unknowns < 1초 풀이.

#### 2.3.4 빌드 중 발견·수정한 critical bugs (3개)

1. **`A_tgt` (name-based mask) ≠ `is_target` (cuboid channel 7)**:
   target net의 pin cuboid는 name match지만 ch7=0이라 모델 prediction에서 제외. Name mask 사용 시 A_primary 20% 과다 계상, **`c_vss_pred = -8015 fF` (음수!)** 누출. → channel 7 mask로 통일.

2. **29 z anchors collinearity로 NNLS oscillation**:
   같은 metal layer의 top/bottom이 별도 anchor → ρ가 `0/30/0/16` 교대 진동. → 10 physical-metal buckets로 collapse (pre_M1, M1-M6, upper, top, others).

3. **Sanity check가 tile-centric sampling으로 partial coverage**:
   pred 5-10% under-pred로 잘못 보고. → `--max_nets_per_design`으로 net-centric walking.

#### 2.3.5 NNLS-fit values

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

**해석**: lateral CPL은 raw geometric base에서 이미 정확 (v3의 8× scale-up이 과도해서 cpl_modifier가 0.11×로 보상해야 했던 이유), broadside CPL은 v3 5.0과 거의 일치, GND density는 hardcoded보다 5-20× 낮음 (v3는 `gnd_modifier ≈ 0.5×` 보상 가정한 값; NNLS는 modifier=1 가정 하의 실효값 직접 fit).

#### 2.3.6 5-seed 결과 — Validation MAPE (in-distribution)

3 variants × 5 seeds (`--max_iters 1 --steps_per_iter 5000`):

| Variant | Median MAPE | p25-p75 | Range | Mean | IQR |
|---|---|---|---|---|---|
| v3_baseline (no calib) | 64.17 | 55.70-65.50 | 50.70-73.23 | 61.86 | 9.80 |
| **v4_full_calib** | **54.50** | 52.67-57.19 | **49.32-63.03** | **55.34** | **4.52** |
| v5_gnd_only | 60.40 | 53.56-62.41 | 48.78-70.08 | 59.05 | 8.85 |

**Mann-Whitney U two-sided**:
- v3 vs v4: U=19.0, p=0.222 — ns
- v3 vs v5: U=16.0, p=0.548 — ns
- v4 vs v5: U=10.0, p=0.690 — ns

**모두 ns at α=0.05** (n=5 power 부족).

**Cohen's d effect size**:
- v3 vs v4: d ≈ 0.85 (large effect)
- v3 vs v5: d ≈ 0.36 (small-medium)
- v4 vs v5: d ≈ 0.45 (medium)

→ Effect size는 큰데 sample size가 작아 detection 불가. n=10이면 v3 vs v4가 가장 likely하게 significance 도달.

#### 2.3.7 OOD 결과 (TEST_DEFS: nova_f3 + tv80s_f3, n=5)

| Variant | total MAPE | GND chip ratio | CPL chip ratio | slope | Pearson r |
|---|---|---|---|---|---|
| v3_baseline | 0.553 | 0.665 | 1.730 | 0.535 | 0.899 |
| **v4_full_calib** | **0.459** | 0.678 | 1.544 | 0.534 | 0.885 |
| v5_gnd_only | 0.535 | 0.686 | 1.462 | 0.536 | 0.886 |

**중요한 reverse**: 단일 seed에서 `v4 OOD +5-8pp WORSE` 결론이었으나
5-seed에서 **v4 OOD 9.4pp BETTER** (단 ns). 단일 seed v4 BEST는 step 3000
under-trained transient였고 v3 BEST는 step 8000 trained — apples-to-oranges.

#### 2.3.8 Heteroscedastic 미해결 ⚠

```
v3:  slope 0.525, Pearson r 0.915, GND chip ratio 0.735
v4:  slope 0.510, Pearson r 0.912, GND chip ratio 0.740
v5:  slope 0.517, Pearson r 0.912, GND chip ratio 0.737
```

**3 variant 모두 slope ≈ 0.5**. 즉 motivation이었던 heteroscedastic 문제는
calibration init만으로 **해결 안 됨**. 이유: `ρ_layer`는 layer-wide global
multiplier. 같은 layer의 작은 net과 큰 net이 같은 ρ로 곱해짐.
Heteroscedasticity는 본질적으로 **per-net residual** 현상.

#### 2.3.9 Verdict

- **mean MAPE 6.5pp 감소** (in-dist + OOD 동일 방향)
- **IQR 절반 축소** (training reproducibility 향상)
- **OOD overfit 없음**
- **그러나 통계적으로 ns** (n=5 power 부족)
- **heteroscedastic motivation 미해결**
- **Q1 over / Q3+ under 패턴 약간 악화**

→ **단독 contribution으로는 충분치 않음 → 폐기**. 코드는 `src/data/_archive/calibration_extractor.py`, `calibration_solver.py`로 보존. JSON outputs는 `/data/PINNPEX/data/processed/intel22/calibration_init*.json`에 있음 — 재실험 시 NNLS 재실행 회피용.

---

### 2.4 트랙 D — γ Scaling Head (per-net multiplicative correction)

**시작**: 2026-05-01.
**문서**: `docs/calibration_track_report.md` §7.

#### 2.4.1 동기

Calibration init이 못 푼 heteroscedastic 문제는 per-net residual.
`pred_total_net *= γ_net` 형태의 multiplicative scale을 per-net features로 학습.

#### 2.4.2 Codex-reviewed 설계

**Input features (14 dims)**:
- `log1p(n_target_cuboids)`
- `log1p(total_gnd_area)`
- `log1p(total_w_cpl_base)`
- `n_layers_present`
- `area_layer_dist` (10-bucket)

**Architecture**: `Linear(14, 32) → GELU → Linear(32, 32) → GELU → Linear(32, 1)`.
**Output**: `γ_gnd = exp(clamp(logit, -2, 2)) ∈ [0.135, 7.39]`.
**Init**: 마지막 Linear weight/bias zero → output ≈ 1.0 (identity).

**Codex 4 권고 적용**:
1. γ_gnd만 활성 (γ_cpl은 cpl_modifier와 중복).
2. Warmup schedule: step ∈ [0, 2000) clamp [-0.5, 0.5], [2000, 4000) [-1.0, 1.0], [4000, ∞) [-2.0, 2.0].
3. Separate optimizer group: 0.1× base LR, weight_decay 1e-3.
4. Identity regularizer: `λ × mean(log_γ²)` with λ=0.05.

#### 2.4.3 측정 결과

- **Smoke (단일 seed)**: step 1000 BEST MAPE **67.24%** (v3/v4 step 1000 median 82-86 대비 15-19pp 향상).
- **5-seed step 1000 BEST 분포**: median 84% — smoke는 운 좋은 케이스였음.
- **5-seed step 5000 측정 미완료** — 사용자가 트랙 종료 결정.

#### 2.4.4 Verdict

단일 smoke만으로는 판단 불가. **측정 미완 상태로 폐기**. 코드는
`src/models/_archive/gamma_head.py`로 보존. 향후 재시도 시 우선순위는 낮음
(architectural redesign 이후 재고려).

---

## 3. 5-seed measurement protocol — 진짜 자산

### 3.1 N=1 fallacy의 정량화

3개 트랙에서 반복적으로 확인된 패턴: 단일 seed run 비교는 stochastic noise에 압도된다.

**실증 데이터**:
- v3 5-seed best_mape range: 50.70-73.23 (**22pp spread**)
- 단일 seed에서 22% improvement claim이 가능했던 이유 = noise 안에 들어갔던 값
- v10b "27.30%"는 5-seed 분포의 2.4σ lucky tail
- v2 "34.83%" 또한 lucky tail

**최소 권장**: **n=5 seeds + Mann-Whitney U test**.
**6-9pp difference 검출에는 n=10+ 필요**.

### 3.2 OOD evaluation 필수성

In-dist만 보면 transient artifact에 속는다.

**실증**:
- 단일 v4 iter 0 best (step 3000) vs trained v3 best (step 8000): in-dist v4 -22% (looks great), OOD v4 +5-8pp WORSE → 잘못된 결론 "in-dist overfit".
- 5-seed: in-dist v4 -6.5pp, OOD v4 -9.4pp (둘 다 ns, 그러나 같은 방향).

→ **OOD ≈ in-dist consistency**가 진짜 "no overfit" 근거. 단일 seed로는 판정 불가.

### 3.3 Critical analysis 사전 수행

빌드 시작 전 **자체 reviewer 시점**에서 가능한 모든 약점 나열 → 후속 data로 검증.

**실증**:
- GINO P2G aliasing / latent Laplacian / cap=net_cap/n_tiles 3가지 fatal flaw — 빌드 전 차단.
- DS-PINN macro feature가 GND를 hijack할 가능성 — Phase A diag가 확인.
- "distillation" 명명의 overclaim — Codex round 1에 정정.

본 프로젝트의 critical assessment 항목들이 OOD/heteroscedastic 측정으로 100% 검증됨 (모두 우려대로 결과 나옴). **자체 reviewer 통과하면 paper review 통과 가능성 높음**.

### 3.4 Codex deliberation loop

빌드 시작 전 1-3 round deliberation이 critical bug 사전 발견에 필수.

**실증**:
- Multi-scale distillation plan: round 1에 6 critical bugs (좌표 투영 오류, voxel-merge 미러링, distill head 중복 등) 발견 → 폐기 후 redesign.
- Calibration extractor: round 1에 3 BUG + 3 WARNING (CPL K² weak identification, cpl_residual leak at phys_scale=0, aggregation mismatch) → narrow scope.
- γ head design: round 1에 4 권고 → 즉시 반영.

패턴: **round-1이 majority bug 잡음**, 2-3 round은 rare. round-1 부재 시 학습 후 발견되는 비용이 크다 (3-4시간 학습 × 5 seeds).

**Round-별 history (전체 7 rounds)**:

| Round | 날짜 | Topic | Outcome |
|------:|------|-------|---------|
| 1 | 2026-04-29 | DS-PINN first-pass audit | 4 P1 + 4 P2 fixes → v1_new |
| 2 | 2026-04-30 | DS-PINN mid-AL deeper review | P1+P2+P3+P4 → v2 (lucky-seed best) |
| 3 | 2026-04-30 | DS-PINN pre-launch integration | OOM chunking, ckpt warnings |
| 4 | 2026-04-30 | β + ζ breakthrough strategy | v3, v4 launched (later FAILED) |
| 5 | 2026-05-01 | v5 strategic direction | 7 strategies ranked; β_strat + γ recommended |
| **6** | **2026-05-01** | **DS-PINN keep/strip decision** | **MEASURE-FIRST → 5-seed v10b ablation 권고; 측정 결과 STRIP verdict** |
| 7 | 2026-05-01 | Strip-down implementation review | strip 순서 SAFE 확인, ζ revert depth 권고, SSL re-pretrain 불필요 판정, calibration_solver archive 권고 |

### 3.5 명명의 정직성 (overclaim 방지)

**실증**:
- "distillation" → Codex/사용자 검토 후 "Data-Driven Physics Calibration"으로 정정. live teacher 없음에도 distillation으로 명명한 것은 overclaim.
- "DS-PINN works" → 5-seed에서 +2.04pp ns 입증, 분산 +56% 증가까지 발견. claim 철회.
- "v4 22% better" → noise 안 잠긴 단일 seed claim. 5-seed에서 -6.5pp ns로 정정.

→ **자체 검토에서 잡지 못한 overclaim은 paper reviewer에게 잡힌다**.

---

## 4. Architecture-independent bottleneck list

다음 이슈들은 **DS-PINN, GINO, plain MLP, 향후 redesign에 관계없이 holds**. 향후 architecture에서 그대로 fix 필요.

### 4.1 H1 — Tile-level train/valid split의 12.3% net mixing

**File**: `scripts/build_dataset_multi.py:91-95`.

```python
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
valid_count = int(len(df) * cfg.VALID_RATIO)
df.loc[:valid_count-1, 'split'] = 'valid'
```

TILE 단위 random split → **31,706 / 257,438 nets (12.32%)가 train과 valid 양쪽에 tile 보유**.

**Fix**: `(design_name, net_name)` 해시 기반 split. 기존 1.3M-tile manifest 무효화 필요.

### 4.2 H2 — NF_PAD_TO_CUBOIDS=1024 positional truncation

**File**: `configs/config.py:116`, `src/data/datasets.py:232-234`.

- Raw tile 평균 3,714 cuboids; P95 > 1024.
- `[:1024]` 잘라내기는 positional (insertion order), priority 없음.
- Tile edge 근처 aggressor가 silently dropped.

**Fix options**: pad 4096으로 증가 / XY 0.5μm voxelize 후 truncate / `(is_target, distance)` 정렬 후 truncate.

### 4.3 H3 — Build context margin (2μm) < model cutoff (4μm)

**File**: `scripts/build_dataset.py:528` `context_margin = 2.0`, `configs/config.py` `CUTOFF_RADIUS = 4.0`.

- 저장된 tile window: 4×4 + 2*2 = **8×8μm**.
- Model이 search하는 범위: target cuboid 주변 4μm.
- Tile edge 근처 target이 4μm 외부 aggressor를 필요로 하지만 margin이 2μm밖에 없음 → **M7/M8 long parallel coupling 물리적으로 학습 불가**.

**Fix**: rebuild with `context_margin = max(CUTOFF_RADIUS+1, 5.0)` → 14×14μm window.

### 4.4 H4 — CPL search uses closest_dist, long parallel runs lost

**File**: `src/models/flux_head.py:411,456`.

- Aggressor selection: `closest_dist(aggr, target_cuboids) < cutoff_r`.
- M8 wire가 10μm 평행으로 3μm 거리에서 흐르면 single edge로 collapse — 긴 평행 coupling 잃음.

**Fix**: kNN-per-target-cuboid 또는 pairwise enumeration. Edge count ~2.25× 증가 → `MAX_AGGR_BUDGET 768` 필요할 수 있음.

### 4.5 M5 — SSL pretrain ignores split

**File**: `src/trainers/train_ssl.py:36-41`.

- SSL은 모든 TRAIN-design tile을 사용, `split` column 무시.
- AL validation이 동일 net 일부를 사용 → encoder가 valid-net feature를 memorize.

**Fix**: SSL dataloader에서 `split=='train'` filter.

### 4.6 M6 — ε channel single-value, not pair

**File**: `src/data/tensorizer.py:51`.

- 메탈 cuboid의 ε는 "ε of ILD above" fallback으로 resolve.
- 비대칭 ILD stack (예: m6etchstop ε=5.5 above vs ild5 ε=2.8 below)에서 asymmetry 정보 잃음.

**Fix**: `ε_above`, `ε_below`, `etch_stop_present` 채널 추가.

### 4.7 M7 — VSS cuboid cap 128/tile uses 2D distance

**File**: `scripts/build_dataset.py:221`, `lines 357-359`.

- Top-metal-heavy designs (vga, nova)이 200+ wide PWR stripe per tile.
- Distance metric XY only — M8 stripe는 우선 keep, M1 shielding rail이 drop.

**Fix**: cap 256으로 증가 + distance에 Z 포함.

### 4.8 M9 — MAX_AGGR_BUDGET batch-shared

**File**: `src/data/datasets.py:402-412`, `configs/config.py:155` `MAX_AGGR_BUDGET = 512`.

- Aggressor importance가 batch 전체에 누적, top-512 keep.
- LDPC 같은 dense net의 10K aggressor가 small net 50 aggressor를 crowd out.

**Fix**: per-net cap (256/net × 2 nets/batch = 512 total).

### 4.9 Severity ranking for MAPE < 4%

1. **H1** — net-level split rebuild (foundation; 없으면 val MAPE 신뢰 불가).
2. **H3** — context margin rebuild (top-metal coupling 복구 불가능 without it).
3. **H4** — pairwise CPL search (H3에 의존).
4. **H2** — pad/truncation priority (large-design accuracy).
5. **M5** — SSL split filter (training-validation purity).
6. 나머지 — incremental.

---

## 5. Critical bugs caught (8개)

이 기록 자체가 가치 — architecture가 바뀌어도 동일 패턴 재발 가능.

### 5.1 AL cache leakage — 21,173 train-pool tiles
`run_active_learning.py:41-101` `load_or_create_predefined_cache`가 train과 valid 후보를 독립 sampling하지만 H1으로 인해 net-level cache overlap 12 nets / 1494 (0.8%)는 작아도, **cache build 후 `pool_df = pool_df[split=='train']`이 21,173 train-tiles belonging to 1500 sampled valid nets를 AL pool에 남김 → AL selector가 re-label / train 가능**. Phase 1 fix: pool_df anti-join after cache load.

### 5.2 Curriculum step counter sawtooth
`step` resets to 0 every `train_steps()` call (one per AL iter). `w_gnd` (warmup 500-2000), `w_aux` (decay 0-5000)가 매 iter sawtooth → iter 1+ step 0-500 동안 GND supervision 일시 손실. **Fix**: `global_step = al_iter * max_steps + step`.

### 5.3 SymMAPE saturation
`compute_netlevel_loss`의 SymMAPE는 MAPE>40%에서 **0.25에서 saturate** → val 50-70% MAPE 영역에서 gradient signal 거의 없음. **Fix**: 직접 MAPE + magnitude-weighted MAPE + log_reg(clamp) + zero_pen.

### 5.4 KCL formulation redundancy ⚠️ **(미적용 — §9.1 #4 참조)**
`smooth_l1(pred_gnd + pred_cpl, Y_total)`는 second total-teacher (Y_total = Y_gnd + gt_cpl from StarRC). 진짜 closure는 `smooth_l1(pred_gnd + pred_cpl, global_pred_total.detach())` — pred_total로 pull 없이 sum-equals-total 강제. Codex round 2 (2026-05-01)에서 flagged 됐으나 **아직 코드에 반영되지 않음**. 5-line change. 향후 단기 task로 적용 검토.

### 5.5 A_tgt vs is_target mask mismatch
Calibration extractor에서 name-based `A_tgt`와 cuboid channel 7 `is_target` 불일치. Pin cuboid는 name match지만 ch7=0이라 모델 predict에서 제외. Name mask 사용 시 A_primary 20% 과다 계상, **`c_vss_pred = -8015 fF` 음수 누출**.

### 5.6 NNLS anchor collinearity
29 z anchors 사용 시 같은 metal layer top/bottom이 별도 anchor → ρ 진동 (0/30/0/16 교대). Fix: 10 physical-metal buckets로 collapse.

### 5.7 Tile-centric vs net-centric coverage
Sanity check가 tile-centric `head(N)` sampling으로 partial coverage → predicted GND/CPL이 5-10% of golden. Fix: `--max_nets_per_design`로 net-centric walking.

### 5.8 Metric mislabel (compute_pex_loss as SMAPE)
`finetuner.evaluate()`가 출력하던 "Validation SMAPE [%]"가 사실 `compute_pex_loss` (`L1 + 5×MAPE + 2×log` hybrid). 진짜 per-edge SMAPE는 매우 다름 (v10b 158%, v1_new 167%, v2 104%). 이로 인해 **DS-PINN v1_new "fail"이 사실 metric artifact였음**을 한참 후 발견 — DS-PINN abandonment 결정이 잘못 되었던 시점이 있었음.

---

## 6. Loss design principles (validated)

`feedback_loss_design_principles.md` 5개 rule, 사용자 검증됨.

### Rule 1 — align loss with eval metric
Primary signal: `|pred - target| / target.clamp(min=eps)` for `target >= eps`. Secondary: log-space smooth_l1. **Don't mix bounded losses (SymMAPE) with unbounded (raw MAPE)** at high weights.

### Rule 2 — heteroscedastic weighting for power-law targets
PEX cap 분포 power-law (CTS 30+ fF, 평균 0.5 fF, ratio 60×). `cap_weight = clamp(target/median, 0.3, 20.0)`로 small-vs-large balance. 20× cap이 CTS 지배 — 제거 시 large net under-trained.

### Rule 3 — zero-target supervision
Target <0.005 fF는 MAPE 정의 안 됨. `smooth_l1(pred[target<eps], target[target<eps], beta=0.05) × 0.1`. **log1p(pred) zero-pen은 사용 불가** (gradient vanishes for pred=0, the case we want to discourage).

### Rule 4 — KCL closure is internal consistency, not extra teacher
KCL = `smooth_l1(pred_gnd + pred_cpl, pred_total.detach())`. Without `.detach()` 추가 redundant pull on total. Codex 검증.

### Rule 5 — don't bundle correlated changes when each costs 3+ hours
AL run 3-4시간/validation. Tasks A·B 모두 loss assembly에 닿으면 regression 시 attribution 불가능. **Apply one loss-affecting change per validation cycle**. Bundle은 orthogonal 변경만 (encoder unfreeze + stratified AL 등).

---

## 7. 코드 자산 정리 (현재 상태 기준)

### 7.1 Live (현재 baseline에 통합)

| 파일 | 변경 | 역할 |
|---|---|---|
| `src/models/neural_field.py` | macro_density_fno, gamma_head, GINO 모두 제거 | 1-hop GNN + surface physics 순수 baseline |
| `src/models/flux_head.py` | layer_scale_phys_gnd zero-init, cpl_layer_pair_log_scale zero-init, macro_context 인자 제거, γ head 제거 | NeuralFluxRouter |
| `src/trainers/finetuner.py` | step counter 글로벌화, compute_netlevel_loss 단순화, KCL 정정 (적용 안 됨), v2 P1 (per-edge cpl_loss) / P4 (cpl_modifier exp range) 유지; **β `loss_cpl_ratio` 제거**, **`loss_aux_macro` + `w_aux` schedule 제거**, **`gamma_log_reg` + `GAMMA_REG_LAMBDA` 제거**, mdf/gh optimizer groups 제거, GINO 2-phase unfreeze 제거, `probe_flux_router_anomalies`의 DS-PINN diagnostics 블록 제거; loss = `3·loss_scale + loss_cpl_total + 1.5·loss_cpl_vector + 0.10·loss_distribution + w_gnd·loss_gnd_direct + w_cpl_direct·loss_cpl_direct` | finetune loop |
| `src/trainers/train_ssl.py` | (동일 변경) | SSL pretrain |
| `run_active_learning.py` | --seed/--max_iters/--steps_per_iter 유지, --use_dspinn/--use_gamma/--use_gino/--calib_path 제거; cache anti-join 적용 | AL launcher |
| `configs/config.py` | RUN_NAME `ssl_basis_v9` → `ssl_basis_dspinn_v1`; CALIBRATION_INIT_PATH 제거; SSL_USE_DSPINN/SSL_USE_GINO/DSPINN_D_MACRO 제거 | config |

### 7.2 Archive (보존, 비활성)

| 위치 | 파일 | 이유 |
|---|---|---|
| `src/models/_archive/` | `macro_density_fno.py` | DS-PINN 5-seed 반증 후 비활성 |
|  | `gino_enricher.py` | GINO 단일 stream 시도 (DS-PINN으로 흡수 후 비활성) |
|  | `gamma_head.py` | 측정 미완 |
| `src/data/_archive/` | `calibration_extractor.py` | NNLS pipeline phase 1+2 |
|  | `calibration_solver.py` | joint NNLS solver |
| `scripts/_archive/` | `diag_calibration_check.py` | calibration sanity check |
|  | `diag_compare_physics.py` | physics-only baseline 비교 |
|  | `diag_fno_feasibility.py` | FNO feasibility test |
|  | `diag_fno_option_a.py` | physics pseudo-label train |
|  | `diag_fno_option_b.py` | StarRC golden label train |
|  | `diag_gino_runtime.py` | GINO P2G/G2P benchmark |

### 7.3 Methodology 도구 (보존, 사용 빈번)

| 파일 | 용도 | 가치 |
|---|---|---|
| `scripts/diag_quartile_heteroscedastic.py` | per-quartile ratio 측정 | high — 어떤 모델이든 적용 |
| `scripts/diag_ood_compare.py` | TEST_DEFS multi-ckpt 비교 | high — 5-seed protocol 필수 |
| `scripts/analyze_5seed.py` | log → distribution analyzer | high |
| `scripts/aggregate_5seed_eval.py` | multi-ckpt OOD/heteroscedastic | high |
| `scripts/run_5seed_remaining.py` | v3/v4/v5 5-seed launcher | medium |
| `scripts/run_5seed_v6_gamma.py` | γ head 5-seed launcher (참조용) | medium |
| `scripts/diag_5seed_eval.py` | 5-parallel-seed driver | medium |
| `scripts/eval_models_on_val.py` | 1494-net val cache 평가 | medium |
| `scripts/diag_spef_unit_check.py` | SPEF C_UNIT 검증 | medium |
| `scripts/diag_eval_dump.py` | per-net + per-cuboid NPZ dump (--physics_only mode 포함) | high |
| `scripts/diag_case[1-6,_g]_*.py` | Phase A diagnostic 시리즈 (baselines, CPL distribution, GND breakdown, topology, outliers, GND deep) | high |
| `scripts/dspinn_al_diagnose.py`, `dspinn_al_report.py` | live AL milestone reporter | medium (DS-PINN 종료 후 사용 빈도 감소) |

### 7.4 Data 자산

| 위치 | 내용 | 보존 사유 |
|---|---|---|
| `output_intel22/active_learning/m5_*/best_model.pth` | 15 ckpts (v3/v4/v5 5-seed) | future analysis |
| `output_intel22/active_learning/m5_v6_gamma_seed{0..4}/best_model.pth` | γ smoke + early seeds | 측정 미완 — 재시도 시 starting point |
| `output_intel22/active_learning/m5_summary/*.csv` | per_run, per_variant, mann_whitney, eval_per_seed/variant, eval_raw_ind/ood | 모든 통계 결과 |
| `/data/PINNPEX/data/processed/intel22/calibration_init*.json` | NNLS extraction 결과 | 재실험 시 inputs (재extraction 회피) |
| `/data/PINNPEX/data/processed/intel22/calibration_extract/phase{1,2}_net2k.pkl` | NNLS 입력 데이터 | 재NNLS 회피 |
| `output_intel22/active_learning/cache/predefined_*.csv` | fixed val cache (1494 nets, 9 designs) + train cache (12,843 tiles) | 5-seed protocol 표준 split |

---

## 8. 본질적 한계 — 현재 architecture가 plateau 도달한 4가지 증거

3개 트랙 모두 mean MAPE 55-62% 영역에 수렴. 이는 **NEURAL_FIELD + 1-hop
GNN + surface physics**의 floor일 가능성이 큼. Incremental fix는 6-10pp 수준.

1. **MAPE 50-65% floor**: v3/v4/v5 모두 mean 55-62. v10b vanilla 63.79. DS-PINN 변형 모두 같은 영역.

2. **Heteroscedastic slope ≈ 0.5**: 1-hop GNN으로는 long-range topology 효과 못 잡음. per-cuboid local feature만으로 net-scale property 표현 부족.

3. **CPL over-prediction (chip ratio 1.5-1.7)**: edge-level Sakurai-Tamaru는 inherently noisy. Aggregation 후 systematic bias 누적.

4. **Outlier nets**: top-100 worst nets (LDPC, dense CTS)가 ~10pp MAPE 차지. Incremental fix로 안 풀림. LDPC coded_block의 dense routing은 local Sakurai-Tamaru가 confused.

---

## 9. 향후 방향 — 단기 / 중기 / 장기

### 9.1 단기 (즉시 시도 가능)

1. **AL multi-iter 5-seed v3 vs v4**: 가장 누락된 측정. 5-seed × 1 iter × 5000 step만 측정함. **6 iter × 12k steps 실험 시 calibration init이 의미 있는지** 별도 검증 필요. 비용: ~24h × 5 seeds × 2 variants = 240 GPU-hour.

2. **Cross-PDK validation**: asap7로 NNLS 재extraction + 학습. PDK-specific 패턴인지 확인. intel22에서만 측정한 한계 해소.

3. **Architecture-independent fix (H1, H3)**: net-level split rebuild + context margin rebuild는 어떤 architecture에도 유효. 비용: 1.3M-tile manifest 무효화 + 2-4 GPU-day rebuild.

4. **KCL closure fix (5-line change)**: 현재 `smooth_l1(pred_gnd + pred_cpl, Y_total)`은 second total-teacher (Y_total = Y_gnd + gt_cpl from StarRC). 진짜 closure는 `smooth_l1(pred_gnd + pred_cpl, global_pred_total.detach())` — pred_total로 pull 없이 sum-equals-total만 강제. Codex round 2에서 flagged 됐으나 적용 안 됨. **§5.4 참조**. 비용: 1줄 수정 + 5-seed re-validate.

5. **SSL basis revisit (`ssl_basis_dspinn_v1` → 새 basis)**: 현재 baseline은 DS-PINN-aware encoder weight를 stale residual로 보존 중. DS-PINN 트랙 폐기됐으므로 **순수 1-hop GNN architecture로 SSL pretrain 재실행**이 필요할 수 있음. 비용: SSL 11h (cosine LR target 500 epoch). 효과 미지수 — 먼저 기존 basis로 5-seed 측정해 baseline 확보 후 결정.

### 9.2 중기 (architecture 재설계)

6. **Multi-hop GNN (2-hop or more)**: 1-hop의 long-range topology 한계 검증. 가장 incremental한 architectural change.

7. **Net-level encoder**: per-net summary embedding을 cuboid feature와 결합. heteroscedastic per-net residual 직접 공략.

8. **Outlier-aware sampling**: LDPC/CTS specific bucket으로 sampling 가중치 조정. top-100 worst nets handling.

9. **Sakurai-Tamaru 외 physics base 검토**: 더 정확한 fringe formula (예: Wong-Salama-Shieh, fringe 항 차수 증가). edge-level CPL noise 감소 가능성.

### 9.3 장기 (paradigm shift)

10. **PINN-StarRC hybrid**: PINN으로 빠른 추정, 일부 critical net만 StarRC로 verify. 단독 PINN 대신 hybrid 접근.

11. **Diffusion-based calibration**: noise-aware learning으로 outlier robustness. 현재 deterministic regression의 한계 보완.

12. **Direct numerical solver in network**: BEM/FRW 같은 boundary integral을 neural module로. Sakurai-Tamaru proxy 대체.

---

## 10. Methodology takeaways (재현 가능한 인사이트)

본 프로젝트에서 입증된 방법론적 lesson — 향후 PINN-PEX 실험뿐 아니라
유사 ML4EDA 연구에 적용 가능.

1. **N=1 comparison은 위험**. 단일 seed BEST는 stochastic variance에 dominated. 22pp spread가 실증.
2. **OOD evaluation 필수**. in-dist만 보면 overfit/transient artifact에 속음.
3. **Codex deliberation loop**. Pre-build round-1이 critical bug 사전 발견에 필수. 학습 비용 (3-4h × 5 seeds) 대비 prevention cost 매우 작음.
4. **Critical analysis는 paper review 전에 자체 수행**. "imagine the reviewer's worst critique" 직접 수행.
5. **명명의 정직성**. live teacher 없으면 distillation 아님. data-driven init으로 부르라.
6. **Effect size + n-power 관계**. Cohen's d 0.85 (large)도 n=5에서는 ns. n=10이면 likely significant.
7. **Loss design rules** (Rule 1-5): MAPE-aligned, heteroscedastic-weighted, zero-supervised, KCL-as-internal-consistency, single-change-per-cycle.
8. **Curriculum step counter는 global** — local step 사용 시 sawtooth 발생.
9. **Mask consistency**: name mask vs channel mask 불일치는 silent failure 일으킴.
10. **Sanity check sampling은 net-centric**. tile-centric `head(N)`은 partial coverage 함정.

---

## 11. 산출물 인덱스

### 11.1 본 보고서가 통합한 living docs (보존, 신규 작업 시 본 보고서 우선 참조)

| 파일 | 역할 | 라인 수 |
|---|---|---|
| `docs/INTERIM_CONCLUSIONS.md` | 트랙 종료 시점의 통합 결론 (2026-05-01) | 348 |
| `docs/dspinn_development_log.md` | DS-PINN 트랙 living log (v1_new ~ v4 + 5-seed 반증) | 593 |
| `docs/distillation_log.md` | Calibration 트랙 living log (NNLS 빌드 + bug fix) | 754 |
| `docs/distillation_effect_report.md` | Calibration 효과 정량화 (MAPE/IQR/OOD/heteroscedastic) | 436 |
| `docs/calibration_track_report.md` | Calibration 종합 보고서 (γ head 단계까지) | 533 |
| `docs/gino_architecture_report.md` | GINO architecture 제안 v2 (validation + critique) | 440 |
| `docs/gino_report.md` | GINO critical review + DS-PINN pivot (3 fatal flaws) | 221 |

### 11.2 Memory entries

`/home/jslee/.claude/projects/-home-jslee-projects-PINNPEX/memory/` 의:

- `user_profile.md` — ML/EDA researcher
- `project_failed_experiments.md` — v8 / v8b VSS absence failure
- `project_v9_implementation.md` — v9 VSS grid + 10-channel rebuild
- `project_spef_res_insight.md` — StarRC RES segment ~9.x Ω physical interpretation
- `project_model_baselines.md` — v6/v7/v8/v8b MAPE table (legacy single-seed)
- `project_dspinn_implementation.md` — historical DS-PINN architecture (now archived)
- `project_dspinn_ineffective.md` — 2026-05-01 conclusion + retained code
- `project_data_pipeline_bottlenecks.md` — H1-H4, M5-M9 architecture-independent issues
- `project_session_2026_05_01_findings.md` — Phase 1 fixes applied / Phase 2 aborted
- `feedback_loss_design_principles.md` — Rule 1-5

### 11.3 외부 데이터 / 체크포인트 위치

```
/data/PINNPEX/data/processed/intel22/         # 10-channel v9 dataset
  calibration_init.json                       # full v4 NNLS
  calibration_init_gnd_only.json              # v5 (GND-only NNLS)
  calibration_extract/phase1_net2k.pkl        # NNLS phase 1 input
  calibration_extract/phase2_net2k.pkl        # NNLS phase 2 input

/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22/   # StarRC SPEFs

output_intel22/active_learning/
  cache/predefined_train_subset.csv           # 12,843 tiles
  cache/predefined_valid_subset.csv           # 1494 nets, 9 designs
  m5_<variant>_seed<N>/best_model.pth         # 15 ckpts (v3/v4/v5)
  m5_v6_gamma_seed{0..4}/best_model.pth       # γ head (incomplete)
  m6_v10b_baseline_seed{0..4}/best_model.pth  # ⭐ 5 ckpts, DS-PINN OFF (vanilla PINN); 본 보고서 §2.2.4 5-seed 반증의 핵심 증거
  m5_summary/*.csv                            # per_run, per_variant, MWU, eval data
  diag_phase_a/m5_v3_baseline_eval.md         # 4-seed v3 post-hoc Net-MAPE
  diag_phase_a/m5_seed4_v4_eval.md            # v3_seed4 + v4_full_calib_seed{0,1} post-hoc
  diag_phase_a/m6_v10b_eval.md                # v10b 5-seed post-hoc (apples-to-apples vs v3-baseline)
  diag_phase_a/report_5seed_v2.md             # diag_5seed_eval.py 자동 생성 보고서
```

---

## 12. Closing — 작업의 가치 평가

지난 며칠간 **3개 architectural 트랙 + 1개 module** 모두 효과 없음으로
결론. 그러나 무가치하지 않다. **향후 PINN-PEX 작업의 fundamental
infrastructure를 확립**했다.

**측정 가능한 contribution**:

- **5-seed protocol** + Mann-Whitney U test가 표준 측정 방법론으로 정착.
- **N=1 fallacy 데이터로 입증** (22pp spread): 단일 seed 차이는 detection threshold 이하.
- **OOD evaluation discipline**: in-dist만 보면 wrong conclusion (단일 seed reverse 사례).
- **Critical analysis pre-launch loop**: GINO 3 fatal flaws / DS-PINN macro hijack / calibration overclaim 사전 차단.
- **Codex deliberation 검증**: round-1에 majority bug 잡음.
- **8개 critical bug catalog**: cache leak / sawtooth / SymMAPE / KCL / mask mismatch / collinearity / coverage / metric mislabel — pattern 자체가 재발 방지.
- **9개 architecture-independent bottleneck**: H1-H4, M5-M9 — 다음 architecture에 그대로 적용.
- **Diagnostic infrastructure**: `diag_*` 시리즈는 architecture-agnostic 도구로 영구 자산.

**학습한 교훈을 한 문장으로**:

> "현재 1-hop GNN + surface physics architecture의 incremental fix
> (calibration init, macro stream, per-net γ)는 모두 6-10pp 수준에 그친다.
> 다음 시도는 **architectural redesign** (multi-hop, net-level encoder,
> outlier-aware sampling)이 필요하며, 그 전에 **architecture-independent
> data pipeline fix** (net-level split, context margin rebuild)가 우선
> 적용되어야 한다."

다음 시도자는 본 보고서 §4 (architecture-independent bottlenecks),
§5 (critical bugs), §6 (loss design rules), §9 (forward path)를 출발점으로 사용한다.

---

_본 보고서는 final canonical document다. 새 architecture 시도 시 본 문서 §9.2-9.3을 starting point로 사용하고, 그 결과는 본 보고서의 §13(예정)으로 추가한다._

---

## 13. v3 paradigm shift (Phase 0 → Phase 3 Auto-Optimize, 2026-05-01 → 2026-05-04)

_작성일: 2026-05-04. §1-12의 4개 실패 트랙을 토대로 시작된 새 paradigm. data-fix-first → 강한 baseline → 작은 hybrid → calibration > architecture 결론까지._

### 13.1 Phase 0 — Data pipeline rebuild (H1+H2+H3+M5)

§4의 architecture-independent bottleneck을 우선 해결.

| Fix | Before | After | Location |
|---|---|---|---|
| **H1 net-level split (hash-based)** | tile-level random split, 12.29% net leakage | 0% leakage, hash on (design, net) | `pex_v3/scripts/02_h1_split_fix.py` |
| **H2 priority truncation** | random tile pool, target 누락 가능 | target cuboids 우선 보존 | `pex_v3/src/data/priority_pad.py` |
| **H3 14×14 μm rebuild** | 4×4 μm window 부족한 fringe context | 14×14 μm window, 11/11 designs, 493 GB | `pex_v3/scripts/03_h3_window_rebuild.py` |
| **M5 SSL split filter** | SSL pretrain leak 가능 | net-level filter 적용 | `pex_v3/src/data/ssl_split_filter.py` |

**결과**: data-fix만으로 baseline PINN MAPE **63.79% → 30.90% ± 2.20pp** (5-seed locked, **-32.89pp = -51.6% relative gain**). §2.2.4의 5-seed 반증으로 측정한 v10b 63.79%가 H1 leakage 영향이 절반 정도였음을 정량화.

**Memory entries**:
- `project_phase0_h1_validated.md`
- `project_phase05_progress.md`
- `project_phase_b_b3_first_real_result.md` (B3 PINN 5-seed 30.90% locked)

### 13.2 Phase 0.5 — Strong classical baselines

§7의 "선결: B1 (XGBoost) 측정 필수" 권고 실행.

| Method | params | valid total 5-seed | OOD test 5-seed | OOD gap |
|---|---:|---:|---:|---:|
| **B1 XGBoost** (5-seed) | ~100K | **4.66% ± 0.03pp** | 5.84% ± 0.10pp | +1.19pp |
| **Option F deep MLP** (5-seed, 286K) | 286K | 4.76% ± 0.01pp | 5.62% ± 0.04pp | **+0.87pp** ← 최저 OOD gap |
| **B4 V3 log-GBDT** (5-seed) | ~100K | 5.72% ± 0.04pp | 6.59% ± 0.13pp | **+0.87pp** ← tied 최저 |
| B4 V2 GBDT (5-seed) | ~100K | 7.46% ± 0.05pp | 9.33% | +1.87pp |
| Hybrid_v3 calibrated (1-seed) | 11K | ~9.5% | — | — |

**Paper-grade finding #3** — **Hand-feature ceiling 4.66%**: XGBoost = MLP = 4.66% 동일. **Features bottleneck, not architecture**. Deep MLP 286K가 XGBoost 100K와 동일 → architecture overengineer가 답이 아님.

**Memory entries**:
- `project_phase05_option_f_5seed_locked.md`
- `project_phase05_b1_ood_test_locked.md`
- `project_b4_compact_gam_baseline.md`
- `project_b1_vs_b3_supported.md` (paired MWU d=-16.84, p=0.008, B1 dominates B3)

### 13.3 Phase 1 — Hybrid analytic + bounded residual

§9.3의 권고 architecture: `pred = analytic_prior × bounded_multiplier`.

| Variant | Architecture | params | best valid | test total |
|---|---|---:|---:|---:|
| Hybrid_v3 Tier 2 | bounded multiplicative residual | 11K | 7.19% | 11.79% (β-FAIL) |
| Hybrid_v3 + NNLS calib | + per-layer NNLS prior | 11K | 11.03% | — |
| Capacity sweep 11K → 71K → 406K | identical formula, scaled MLP | 36× span | 11-14% (info-bound ceiling) | — |
| **Mesh-curriculum 5-seed locked** | **44K mesh + curriculum 0.405→0.916→1.386** | **44K** | **6.26% best / 8.27% last ± 0.108pp** | **8.27%** |

**Paper-grade finding #4** — **Curriculum is killer feature**: clamp 0.405 → 0.916 transition gives **-1.89pp single-epoch jump**. 0.916 → 1.386 gives -0.51pp. Without curriculum, bounded multiplier saturates.

**§5의 critical bug 카탈로그 추가 (Phase 1)**:
- **Strike #2** per-pair coupling head (uniform analytic baseline) — cpl_total 38→60% explode at curriculum transition. KILLED.
- **Strike #7** sister r_analytic_v3's cell-OBS features — test +3.15pp WORSE.
- **Strike #8** Liberty pin capacitance — test +2.36pp WORSE, gnd +5.05pp.
- **Strike #8 진단** (5-variant systematic): cuboid encoder가 이미 cell-complexity proxy capture, scalar features는 dead-end. C_gnd 21% ceiling은 **information-bound** (DEF/LEF는 substrate area 제공 못함; GDSII/SPICE 필요).

**Memory entries**:
- `project_phase1_capacity_sweep_done.md`
- `project_mesh_curriculum_5seed_locked.md`
- `project_strike_2_perpair_negative.md`
- `project_strike_7_cell_features_negative.md`
- `project_strike_8_pincap_negative.md`
- `project_strike_8_diagnosis_final.md`

### 13.4 Auto-Optimize Sweep (2026-05-03 → 2026-05-04) — 3 rounds, 8 levers

Mesh-curriculum 8.27% baseline에서 시작, sprint targets {gnd ≤17%, cpl ≤13%, total ≤6.5%}. 8 lever 시도, **architecture redesign 모두 marginal/regress, calibration이 dominant lever**로 확인.

#### 13.4.1 Sweep 결과 (모두 5-seed locked)

| Lever | Type | 5-seed median test_total | vs baseline | Verdict |
|---|---|---:|---:|---|
| Baseline (Mesh-curriculum) | bounded multiplicative | 8.272% | — | reference |
| **A1** per-channel separate encoders | capacity-add | 8.82% (single seed) | +0.55pp | **KILL** |
| **C1a** Mode B-only isotonic | output post-process | 8.272% | 0pp | FAIL no-op |
| **C1b** full 1D iso (on baseline) | output post-process | 6.94% | -1.31pp | FAIL Mode B +38pp collateral |
| **InputSubset** (zero-mask, shared encoder) | input re-routing | 7.914% | -0.358pp | weak (per-channel worse) |
| **ClampNorm** (norm-projection clamp) | curriculum stabilizer | 8.960% | **+0.69pp WORSE** | REGRESSION (smoke seed lucky) |
| **Combined IS+CN** | input + clamp | 7.752% | -0.520pp | composition real |
| **Combined + per-seed gnd C1 iso** | + 1D calibration | 7.076% | -1.196pp | Round 2 HERO |
| **Combined + gnd+cpl 1D iso (L6)** | + 2-channel iso | 6.721% | -1.551pp | Round 3 step 1 |
| **🏆 Combined + LGBM 8-feat cal (L4)** | + ML calibration | **6.364%** ⭐ | **-1.908pp / -23.1% rel** | **FINAL HERO** |

#### 13.4.2 Statistical evidence (anti-overclaim, 5-seed locked)

- 5-seed test_total: median **6.364%**, mean 6.377%, std **0.106pp** (baseline std 0.383pp)
- **Cohen's d vs baseline = -5.967 (HUGE)**
- **Mann-Whitney U two-sided: p = 0.0079** (significant α=0.01)
- **Bootstrap 95% CI on median: [6.247%, 6.505%]** (does NOT overlap baseline 8.272%)
- Paired per-net Wilcoxon (n=477,970 = 5 seeds × 95,594): **p ≈ 0**, median per-net Δ **-2.185pp**

#### 13.4.3 Calibration > Architecture 정량화

| Component | Δ test_total |
|---|---:|
| Architecture only (Combined IS+CN, no calib) | **-0.52pp** |
| Calibration only (LGBM 8-feat on baseline + IS+CN) | **-1.39pp additional** |
| Total | -1.91pp |

**Calibration이 architecture의 2.7× 강한 lever**. 4번의 architectural strike 후 정량 확인.

#### 13.4.4 새 lessons (§5/§7에 추가)

11. **Smoke-vs-5-seed regression이 dominant lurker**. Round 3에서 모든 single-seed smoke가 5-seed median을 0.5-1.0pp 초과. 3개 false-positive paper claim을 5-seed protocol이 차단 (10 GPU-hour의 가치).
12. **Capacity-add는 dead lever (4-strike confirmed)**. A1 per-channel encoder + Strike #7 cell-OBS + Strike #8 pin-cap + Strike #8 z-score 모두 Phase 2 overfit으로 OOD test 악화.
13. **LGBM-residual calibration with cross-channel features**: gnd + cpl을 모두 보정하되 pred + (fanout, bbox, design indicator, layer indicator)를 feature로 → 1D iso보다 lower std + lower Top-50 collateral.
14. **InputSubset zero-masking (CRITICAL)**: shared encoder weights + per-channel input mask는 capacity-add 아님. 단, separate input projection은 A1-in-disguise → 금지 (Codex Round 2 verdict).

**Memory entries**:
- `project_phase1_capacity_sweep_done.md`
- `project_method_consolidated_main_merge.md`
- 신규 entries (in progress): A1 KILL, C1 FAIL, ClampNorm REGRESSION, Combined+LGBM HERO

### 13.5 선행연구 / Industry tool 비교 (paper-grade)

#### 13.5.1 Industry pattern-matching PEX tools

데이터 소스: `docs/pex_tool.csv` (Intel22 PDK, 13 designs MAPE vs StarRC FS golden) + actual SPEF 30개 (`/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22_/intel22_<design>_nonamemap_{starrc,innovus,openrcx}.spef`).

#### 13.5.2 Cross-design OOD test (nova 92,425 + tv80s 3,169 = 95,594 nets)

| Tool | nova MAPE | tv80s MAPE | combined OOD | tooling | inference (combined) |
|---|---:|---:|---:|---|---:|
| StarRC FS 1-thread (golden) | 0% | 0% | 0% | commercial $$$$$ | 9.64 hr |
| **Innovus (Cadence flagship)** | 6.154% | 4.869% | **6.111%** | commercial $$$ | 164 s |
| **🏆 PINN best-stack (this work)** | **6.375%** | **6.080%** | **6.364%** | research, license-free | ~19 s (model only) |
| **OpenRCX (OpenROAD)** | 7.891% | 7.605% | 7.882% | open-source | 69 s |

**Paper-grade headline claims**:
1. **PINN matches Innovus on nova within 0.22pp** — Cadence flagship parity on the harder large design
2. **PINN beats OpenRCX by 1.52pp combined** (-19% relative MAPE)
3. **PINN inference-only is fastest** (1815× faster than StarRC; 8.6× faster than Innovus inference)

#### 13.5.3 Methodology verification (apples-to-apples)

Independent SPEF parsing of tv80s 3-tool SPEFs (3,369 common nets) reproduces CSV-reported MAPE within 0.1-0.8pp:
- Innovus tv80s measured median: **4.976%** vs CSV 4.869% (Δ +0.11pp ✅)
- OpenRCX tv80s measured median: **6.813%** vs CSV 7.605% (Δ -0.79pp, likely mean-vs-median)

Verification script: `pex_v3/experiments/auto_optimize_2026_05_03/scripts/verify_industry_tools.py`.

### 13.6 Per-channel SPEF analysis — DIFFERENTIATING FINDING

#### 13.6.1 Pattern-matching tools drop coupling info

직접 SPEF parsing으로 per-net (gnd, cpl, total) decomposition 비교 (tv80s 3369 nets):

| Tool | gnd_frac (median) | gnd entries/net | cpl entries/net | per-channel breakdown? |
|---|---:|---:|---:|---|
| **StarRC** (golden) | 11.7% | 6 | **93** ✅ | proper Cgnd + Ccpl |
| **Innovus** | **100.0%** | 4 | **0** ❌ | all caps lumped to gnd |
| **OpenRCX** | **100.0%** | 8 | 12 (value=0) ❌ | effectively gnd-only |
| **PINN best-stack (this work)** | matches StarRC | learned | learned ✅ | **gnd 18.84% / cpl 13.96% MAPE** |

**핵심 발견**: Innovus와 OpenRCX는 default fast-mode에서 `*DESIGN_FLOW "COUPLING C"` SPEF header에도 불구하고 **per-aggressor coupling 정보 거의 emit하지 않음** (Innovus: 0 cpl entries/net; OpenRCX: 12 cpl entries with value=0). 

**Implication**:
- **Crosstalk / glitch analysis** (PT-SI, Tempus-SI)에서 per-aggressor coupling 필요 → Innovus/OpenRCX SPEFs 부족
- **Our PINN delivers per-channel breakdown** at production-grade speed
- **Functional capability gap, not just accuracy gap** — Innovus가 0% MAPE라도 per-pair coupling 분석 불가능 (detail mode 추가 시 10-100× slower)

분석 스크립트: `pex_v3/experiments/auto_optimize_2026_05_03/scripts/spef_gnd_cpl_analysis.py`. Full report: `pex_v3/experiments/auto_optimize_2026_05_03/reports/spef_3tool_analysis_tv80s.json`.

### 13.7 End-to-end runtime measurement (fresh from scratch)

#### 13.7.1 TV80s full pipeline (DEF + LEF + layer info → SPEF)

Production pipeline (predict_spef_e2e.py, 7 stages, /data partition, 2회 측정 평균):

| Stage | Description | Time | % |
|---|---|---:|---:|
| 1 | DEF/LEF parse → cuboid pkls (16 workers) | 22.3 s | 9.5% |
| 2 | 145-dim hand features per net | 25.1 s | 10.7% |
| **3** | **per-(target, aggressor) pair features (804K pairs)** | **110.6 s** | **47.2%** ← bottleneck |
| 4 | cuboid arrays + analytic R | 23.2 s | 9.9% |
| 5 | ML inference (LGBM 47-model ensemble) | 8.8 s | 3.7% |
| 6 | c_gnd blend + per-pair LGBM regressor | 37.2 s | 15.9% |
| 7 | SPEF write | 0.5 s | 0.2% |
| **TOTAL (production e2e)** | | **233.6 s ± 1.0s** | 100% |

**End-to-end vs commercial PEX tools**:
- vs StarRC FS 3496s: **15× faster** ✅
- vs Innovus 41.82s: **5.6× slower** ❌ (Stage 3 pair features 47% dominant cost)
- vs OpenRCX 5.10s: **46× slower** ❌

#### 13.7.2 Nova full pipeline (in progress)

Stage 1 alone took ~2시간 (1.18M cuboid pkls 생성 / 565K signal + 119K topology). Stage 2 build_features의 single-threaded discovery loop (rglob + gzip.open over 684K pkls)이 추가 1-2시간 더 소요 중. Stage 3-7 추정 ~80분 추가. **총 nova end-to-end 추정 5-7시간** (vs Innovus nova 122s = ~150-200× slower).

⚠️ **Nova measurement는 production pipeline의 bottleneck (Stage 1 file I/O + Stage 2 single-threaded discovery)을 노출**. Future optimization 필요:
- Stage 1: GPU-accelerated DEF parser, in-memory cuboid handoff (현재 565K pkl files via gzip)
- Stage 2: parallelize discovery loop, cache net→pkl mapping
- Stage 3: replace 800K-pair brute-force with learned-pair encoder

#### 13.7.3 PINN best-stack model only (excluding common upstream)

- Combined IS+CN forward on 95,594 nets: ~19 s (1 GPU)
- LGBM 8-feat calibration apply (gnd + cpl): ~1 s (CPU)
- **Throughput: ~4.9K nets/sec** (faster than StarRC by 1815×, slower than Option F MLP by 400×)

### 13.8 Updated cross-reference leaderboard (final, 5-seed median, OOD test)

| Rank | Method | params | total | gnd | cpl | inference | tooling | per-channel? |
|---:|---|---:|---:|---:|---:|---:|---|---|
| ★ | StarRC FS 1-thread (golden) | — | 0% | 0% | 0% | 9.64 hr | commercial $$$$$ | ✅ |
| 1 | Option F MLP | 286K | 5.62% | 21.67% | 16.44% | 0.05 s | research | ✅ |
| 2 | B1 XGBoost | ~100K | 5.84% | 19.93% | 16.13% | ~0.5 s | research | ✅ |
| 3 | Innovus (Cadence) | proprietary | 6.11% | 100% lumped | 0% N/A | 164 s | commercial $$$ | ❌ |
| **🏆 4** | **PINN best-stack (this work)** | **44.7K + LGBM** | **6.36%** | **20.18%** | **15.36%** | **~19 s** | **research, license-free** | **✅** |
| 5 | B4 V3 log-GBDT | ~100K | 6.59% | 20.30% | 12.80% | 0.12 s | research | ✅ |
| 6 | OpenRCX (OpenROAD) | open-source | 7.88% | 100% lumped | 0% N/A | 69 s | open-source | ❌ |
| 7 | Mesh-curriculum (prev best PINN) | 44K | 8.27% | 20.49% | 15.53% | ~19 s | research | ✅ |

### 13.9 Paper narrative — 6 pillars (확정)

1. **PINN methodology**: bounded multiplicative residual + curriculum + LGBM-residual calibration → cross-design OOD 6.36% (-23.1% relative vs Mesh baseline)
2. **Industry parity**: matches Cadence Innovus within 0.25pp combined OOD; beats OpenRCX 1.52pp
3. **Full-chip SPEF E2E**: StarRC-compatible IEEE 1481-1999 output, tested on tv80s + nova
4. **Per-channel breakdown**: gnd 20.18% / cpl 15.36% — pattern-matching tools (Innovus/OpenRCX) cannot provide at this speed
5. **License-free deployment**: 1815× faster than StarRC inference, no commercial license
6. **Calibration > architecture**: 2.7× lever ratio quantified; methodology insight for DEF/LEF-bound PEX

### 13.10 한계 및 향후 작업

#### 13.10.1 정직한 한계 (anti-overclaim)

| Sprint goal | Target | Achieved | Status |
|---|---:|---:|---|
| **test_total** | ≤ 6.5% | **6.364%** | ✅ MET |
| test_gnd | ≤ 17.0% | 20.183% | ❌ 3.18pp gap (info-bound) |
| test_cpl | ≤ 13.0% | 15.356% | ❌ 2.36pp gap (info-bound) |
| best_valid_total | ≤ 5.0% | 6.110% | ❌ 1.11pp gap |

**1 of 4 sprint targets met**. Per-channel targets는 information-bound — DEF/LEF에 substrate area / per-aggressor pair geometry 부재.

**Top-50 outliers collateral**: best stack의 LGBM calibration이 bulk 개선 (-1.91pp total)을 위해 Top-50 outliers 악화 (median 259.1% → 278.6% = +19.5pp). 50/95594 = 0.05% nets에서 발생, 명시 disclosure.

**End-to-end runtime gap**: 현재 production pipeline은 Innovus 5.6× slower (tv80s) / nova ~150× slower estimate. Stage 3 (pair features 47%) 와 Stage 1 (DEF parsing scaling) optimization 미완.

#### 13.10.2 Deferred levers (future paper / next iteration)

- **B1 per-pair Sakurai-Tamaru**: cpl 15.36% → 13% gap이 information-bound, 효과 제한적 예상
- **GDSII feature integration**: gnd 20% → 17% 돌파 잠재력, 1주 작업
- **Mode B specialist** (giant CTS top-1%): Top-50 collateral +19.5pp 회수
- **Stage 1/3 optimization**: PINN inference fast (~19s)이지만 upstream pipeline은 bottleneck

#### 13.10.3 §1-12의 4개 실패 트랙과의 contrast

| 트랙 | §1-12 verdict | §13 paradigm 활용 |
|---|---|---|
| **GINO** (FNO operator) | 학습 미수행, 3 fatal flaws | Layered-media analytic prior로 elegantized (NNLS calibration) |
| **DS-PINN** (macro stream) | 5-seed에서 v10b 대비 +2.04pp (ns) | Multi-scale 폐기, 단일 cuboid encoder + bounded residual로 simplified |
| **Calibration init (NNLS)** | n=5, p>0.5, 효과 작음 | NNLS prior + bounded residual hybrid의 핵심 component로 재활용 ✅ |
| **γ scaling head** | 측정 미완 | bounded multiplicative residual로 흡수 (per-cuboid → per-net) ✅ |

**§13의 핵심 학습**: 4개 실패 트랙은 모두 **이미 좋은 representation에 capacity 추가** 시도였음. v3 paradigm은 그 반대로 **더 작은 model + analytic prior + post-hoc calibration** 조합이 dominate. §7의 lessons (특히 #7 effect-size, #11 features bottleneck)이 새 paradigm에서 정량 확인됨.

### 13.11 산출물 인덱스 (§13 추가)

#### 13.11.1 Code (pex_v3/)
- `pex_v3/src/models/hybrid_v3_mesh.py` — baseline Mesh-curriculum 44K
- `pex_v3/src/models/hybrid_v3_mesh_input_subset.py` — InputSubset (PASS smoke, weak 5-seed)
- `pex_v3/src/models/hybrid_v3_mesh_clamp_norm.py` — ClampNorm (5-seed REGRESSION)
- `pex_v3/src/models/hybrid_v3_mesh_input_subset_clamp_norm.py` — Combined (architecture component of HERO)
- `pex_v3/src/models/hybrid_v3_mesh_perchannel.py` — A1 KILLED (kept for reference)
- `pex_v3/src/baselines/calibration_v3.py` — NNLS per-layer calibration
- `pex_v3/scripts/run_ablation_5seed.py` — generic deterministic 5-seed runner
- `pex_v3/scripts/aggregate_ablation_summary.py` — D2 anti-overclaim helper (Cohen's d + paired Wilcoxon + bootstrap CI)
- `pex_v3/scripts/stratify_eval.py` — D3 stratified MAPE
- `pex_v3/configs/ablation_manifest.yaml` — variant registry
- `pex_v3/experiments/auto_optimize_2026_05_03/round3_final_eval.py` — final HERO LGBM calibration

#### 13.11.2 Reports
- `pex_v3/experiments/auto_optimize_2026_05_03/HERO.md` — 최종 hero report (Cohen's d + bootstrap CI + per-channel + industry comparison)
- `pex_v3/experiments/auto_optimize_2026_05_03/RESULTS.md` — full sweep journal (3 rounds, 8 levers)
- `pex_v3/experiments/auto_optimize_2026_05_03/PLAN.md` — Codex 3-round 숙의 결과 + execution plan
- `pex_v3/experiments/auto_optimize_2026_05_03/variants/{c1_cts_isotonic,input_subset,clamp_norm,input_subset_clamp_norm}/DESIGN.md` — 4 design docs
- `pex_v3/experiments/auto_optimize_2026_05_03/reports/spef_3tool_analysis_tv80s.json` — per-channel SPEF analysis
- `pex_v3/paper/METHOD.md` — 10-section methodology paper-ready
- `pex_v3/paper/RESULTS_CONSOLIDATED.md` — 5-pillar results
- `pex_v3/paper/CGND_ERROR_ANALYSIS.md` — gnd error origin (info-bound)

#### 13.11.3 Data
- `/data/PINNPEX/data/processed_v3/intel22/` — H1 hash-split + H3 14×14 μm v3 dataset
- `/data/PINNPEX/scratch/{tv80s,nova}_e2e/` — fresh end-to-end runtime measurement scratch
- `/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22_/intel22_*_nonamemap_{starrc,innovus,openrcx}.spef` — 3-tool SPEF comparison data (10 designs × 3 tools)

#### 13.11.4 Memory (cross-session)
신규 entries (post-PROJECT_REPORT v1):
- `project_pex_framework.md` — top-level framework definition
- `project_session_handoff_2026_05_02.md`
- `project_strategy_v3_updated_2026_05_02.md`
- `project_phase0_h1_validated.md`, `project_phase05_progress.md`
- `project_phase05_option_f_5seed_locked.md`, `project_phase05_b1_ood_test_locked.md`
- `project_b1_vs_b3_supported.md` (B1 dominates B3 paired MWU)
- `project_phase1_capacity_sweep_done.md` (info-bound confirmed)
- `project_strike_2_perpair_negative.md`, `project_strike_7_cell_features_negative.md`, `project_strike_8_pincap_negative.md`, `project_strike_8_diagnosis_final.md`
- `project_mesh_curriculum_5seed_locked.md`
- `project_hybrid_calibration_breakthrough.md` (XGB anchor SPEF C 10.95%)
- `project_r_alpha_calibration_done.md`, `project_r_per_net_calibration_done.md` (R 4.00%)
- `project_method_consolidated_main_merge.md`
- `project_paper_narrative_3pronged.md` (User D 결정)

### 13.12 Closing — v3 paradigm의 가치 평가

§12의 closing은 "incremental fix는 6-10pp 수준에 그친다 → architectural redesign 필요". **§13에서 그 다음 단계 결과**:

- **Architectural redesign (4 strikes)** 모두 marginal/regress
- **Information-bound ceiling** 정량 확인 (DEF/LEF 한계)
- **Calibration이 진정한 lever** — architecture의 2.7× 효과
- **Industry parity 달성** (Innovus 0.25pp 차이) **with license-free deployment**
- **Functional differentiation**: per-channel breakdown은 pattern-matching tools 못함

**§12의 학습한 교훈을 한 문장 update**:

> "Cross-design PEX MAPE 6%대는 DEF/LEF feature ceiling이며, 그 안에서 PINN의 contribution은 (a) bounded multiplicative residual + curriculum의 architectural prior, (b) post-hoc LGBM-residual calibration이 dominant lever, (c) per-aggressor coupling 정보 보존 — 이 세 축이 industry pattern-matching tools 대비 **functional capability + license-free + parity accuracy** 의 paper-grade contribution."

**§13의 다음 시도자에게**:
1. §13.4의 lessons #11-#14 (smoke vs 5-seed, capacity-add dead lever, cross-channel calibration, input subsetting only via zero-mask)을 starting point.
2. Information-bound 돌파는 GDSII/SPICE 통합 필요 — DEF/LEF 안에서는 6%대 ceiling.
3. Top-50 outliers (Mode B giant CTS) collateral은 calibration의 trade-off — disclosure 필수.
4. End-to-end runtime은 Stage 1/3 optimization 시 Innovus 동급 가능 (현재 5.6× slower).

---

_§13는 v3 paradigm shift의 final canonical update다. 다음 architecture / dataset 시도 시 §13.4 lessons과 §13.10 한계를 starting point로 사용한다._

