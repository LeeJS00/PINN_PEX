# PINNPEX EDA-Style PEX Pipeline — 방법론 (DEF + LEF → SPEF)

_2026-05-02 24h 자동 실행. Workspace: `experiments/cross_design_tv80s_2026_05_02/`_

이 문서는 routed VLSI layout (DEF + LEF + tech stack) 만 입력으로 받아 IEEE 1481-1999 호환 SPEF 를 출력하는 ML 기반 PEX 추론 파이프라인의 설계와 실행을 기록합니다. 비교 기준은 Synopsys StarRC-2021.06.

---

## 1. 입력 / 출력 계약 (EDA contract)

### 입력
- **DEF** (`*.def`): routed layout (placed cells + routed wires + vias + pins + ports)
- **Tech LEF** (`*.lef`): layer definitions, routing rules, via geometry
- **Cell LEF** (`*.lef`): standard cell pin geometry
- **Layers info** (`layers.info`): TCAD GRD-style layer stack with z-position, thickness, ε per layer

### 출력
- **SPEF** (`*.spef`): IEEE 1481-1999 compatible subset
  - Per-net `*D_NET <name> <total_lumped_C>`
  - `*CONN`: 단일 port + sink node (lumped topology)
  - `*CAP`:
    - 1-노드 ground 캡: `<id> <net>:1 <c_gnd>`
    - 2-노드 coupling 캡: `<id> <net>:1 <agg>:1 <c_pair>`
  - `*RES`: lumped 1-segment R: `<id> <net>:1 <net>:2 <r_total>`
  - `*END`

### CLI

```bash
python3 scripts/predict_spef_e2e.py \\
    --def_path  <DEF> \\
    --out_spef  <SPEF> \\
    [--cuboid_pkl_dir <pre-built pkl dir>]   # skip stage 1
    [--manifest <CSV>]                        # speed up pkl discovery
    [--num_workers 8]
```

PDK 경로 (LEF + layers.info) 는 `configs/config.py` 의 `LAYERS_INFO_PATH`, `TECH_LEF_PATH`, `CELL_LEF_PATH` 로 고정. (StarRC와 동일 — TCL 스크립트에서 PDK 디렉토리를 한 번만 지정.)

---

## 2. 파이프라인 단계 (7-stage)

### Stage 1 — DEF/LEF/layers → cuboid pkls
- `scripts/build_dataset.py` (PINNPEX 내장) 호출
- DEF stream parser → 각 net 의 routing geometry → 4×4×20 μm³ cuboid 로 tile
- 각 cuboid 는 (x, y, z, w, h, d, semantic_type, logic_flag, eps, net_type) 10채널
- per-net pkl.gz 출력

### Stage 2 — cuboid pkls → 145-dim hand features
- `pex_pipeline/build_features_inference.py`
- target net 의 geometric features (per-layer wire 길이/면적, bbox, multi-radius aggressor density 등)
- aggressor + power net stats
- compact analytic prior: `compact_gnd_fF`, `compact_cpl_fF`, `compact_total_fF`
- multi-radius aggressor proximity (0.3/0.5/1/2/3 μm)
- Output: per-net 145-feature parquet

### Stage 3 — cuboid pkls → per-aggressor pair features
- `pex_pipeline/build_pair_features_inference.py`
- 각 (target, aggressor) pair 에 대해:
  - min/mean/p25/p75 distance
  - lateral / broadside overlap area
  - aggressor metal area, layer
  - target/aggressor ε mean
  - sum_inv_d, sum_inv_d2 (coupling proxies)
- Cutoff = 4 μm (default `CPL_CUTOFF_UM`)
- Output: per-pair parquet (target_net, aggressor_net + 17 features)

### Stage 4 — cuboid pkls → 3-stream cuboid arrays
- `pex_pipeline/build_cuboid_arr_inference.py`
- 각 net 별로:
  - target stream (T_max=128, 10 ch)
  - aggressor stream (A_max=256, 10 ch)
  - power stream (P_max=128, 10 ch)
- Distance-priority truncation, fixed-size padding
- Output: `<design>.npz`

### Stage 5 — features → predictions (4 모델 ensemble)
- **Total cap predictor**: 5-seed LGBM + 5-seed CatBoost ensemble. Target = log(total_cap_fF). 
  Cross-design test MAPE: **9.029%** [CI 8.71, 9.34].
- **C_gnd ratio predictor**: 5-seed LGBM. Target = logit(c_gnd / total).
  Cross-design test ratio RMSE: 0.090 (mean prediction = 0.392, golden = 0.391).
  Val-fit calibration scale = 0.940 (reduces c_gnd MAPE on cached path 23.62% → 22.19%).
- **Total R predictor v2**: 5-seed LGBM + 5-seed CatBoost ensemble. Target = log(total_res_ohm + 0.1).
  Cross-design test R MAPE: **11.98%** (training-reported) / 12.89% (cached cuboid path).
  v1 (LGBM-only): 11.83% (training-reported) / 19.91% on TRUE e2e — v2 path 일관성 ↑.
- **Direct c_gnd predictor (5 LGBM + 5 CatBoost)**: 학습은 했으나 e2e blend 에서 효과 없음 (모든 weight 에서 ratio-only 보다 같거나 나쁨). 향후 distribution-aligned features 학습 시 활용 가능. 코드는 유지 (w_direct=0 default).

모든 모델은 동일 145-dim v3 features 사용.

### Stage 6 — total → c_gnd + c_cpl + per-pair distribution
- **Split**: `c_gnd = total × ratio_calibrated`, `c_cpl_total = total × (1 − ratio_calibrated)`
  - `ratio_calibrated = clip(LGBM_ratio_pred × val_calibration_scale, 0.05, 0.95)`
  - val_calibration_scale = 0.940 (val-fit on nova; reduces c_gnd MAPE 31.06% → 27.79%)
- **Per-pair distribution** (LGBM pair regressor):
  - 17-feature per-pair regressor 가 raw `c_pair_pred` 출력
  - Per target net: `c_pair_final = c_pair_raw × (c_cpl_total / Σc_pair_raw)`
  - Σ(per-pair final) = c_cpl_total → total cap consistency 보장
- **R**: LGBM-predicted total R (analytic R from cuboid 합 + 3.5x calibration 은 fallback)
  - R 모델은 calibration 적용 안 함 (val/test bias 부호가 달라 calibration 이 overfit val)

### Stage 7 — assemble + write SPEF
- `pex_pipeline/write_spef.py` 의 `LumpedSPEFWriter`
- 각 net 별 `D_NET → CONN → CAP → RES → END` 순서
- IEEE 1481-1999 헤더 + units (FF / OHM / NS / HENRY)
- `src/evaluation/compare_spef.py` 호환 (per-net `sum_gnd_cap`, `sum_cpl_cap`, `total_res` 합산 가능)

---

## 3. 학습 데이터 / 모델 학습

### Cross-design split
- **Train**: 9개 small-mid intel22 design (aes, gcd, ibex, ldpc, mc, spi, usbf, vga, wb_conmax)
- **Validation**: nova
- **Test**: tv80s (3,280 nets, 본 문서의 전 metric 의 기준)

### 학습 모델 (저장된 weights)
- `output/spef_e2e/total_cap/`: 5 LGBM `.pkl` + 5 CatBoost `.cbm` + `fcols.json`
- `output/spef_e2e/gnd_ratio/`: 5 LGBM seeds
- `output/spef_e2e/total_r/`: 5 LGBM seeds
- `output/spef_e2e/pair_regressor/`: 5 LGBM seeds + `fcols.json`

총 disk usage: ~1 GB (LGBM dominates due to many trees).

### 누수 점검
- SPEF-derived columns 모두 `_select_feature_cols` drop list 에 명시:
  - `total_cap_fF`, `c_gnd_fF`, `c_cpl_total_fF`, `total_res_ohm`
  - `n_aggressors_spef`, `cpl_p95_fF`
- C_gnd ratio + total R 모델 학습 시 SPEF label 외부 누수 방지.

---

## 4. 성능 (vs Golden StarRC SPEF)

자세한 수치는 `SPEF_FLOW_PERFORMANCE_KO.md` 참조. 핵심:

- **Per-net total_cap MAPE**: 9.12% (CI 8.78, 9.45)
- **Per-net c_gnd MAPE**: ~23%
- **Per-net c_cpl_total MAPE**: ~17%
- **Per-net total_R MAPE**: ~13%
- **Per-pair coupling MAPE**: TBD (LGBM regressor 적용 후 30-50% 추정)
- **Runtime**: 5-10 분 / 3,280 nets (vs StarRC 25-40 분) — **3-6x 빠름**

---

## 5. EDA 도구와의 비교

| 항목 | StarRC (Synopsys) | PINNPEX |
|---|---|---|
| 입력 | DEF + LEF + tech stack + TCAD GRD | 동일 |
| 추출 방식 | Field solver / model order reduction | 145-dim hand features + LGBM ensemble |
| Per-net total cap 정확도 | 0% (ground truth) | 9.12% MAPE |
| Per-pair coupling 정확도 | 0% | 30-50% MAPE (estimated) |
| Resistance | Layer-based field model | LGBM on geometry |
| 처리 시간 | ~30 min per ~3k-net design | ~5-10 min |
| 라이센스 | 상용 | 오픈 |

---

## 6. 한계와 향후 개선

1. **Per-pair MAPE 가 큼**: 골든 StarRC 가 분해하는 fine-grained per-pair 결과를 lumped feature 만으로 재구성하기 어려움. 향후:
   - GraphSAGE / GNN 으로 pairwise interaction 직접 모델링
   - 학습 데이터 추가 (현재 7-9 train designs → 30+)

2. **Resistance 는 vias 까지 모델링 안 함**: 현재 LGBM 이 wirelength + layer 기반으로만 학습. via R contribution 은 amounts 로만 잡힘 → 13% MAPE.

3. **SPEF 토폴로지는 lumped (per-net 1-port)**: StarRC 의 실제 multi-segment topology 와는 다름 — downstream tools (timing, IR drop) 가 segment-level R 를 요구하면 미달. compatibility 확인 필요.

4. **PDK 종속**: 현재 intel22 PDK 학습. 다른 노드 (asap7, gf12 등) 적용 시 retrain 필요.

5. **DEF parser 가 PINNPEX 자체 구현**: 일부 DEF SPECIALSECTION (group, route guide) 는 미지원. 표준 routed DEF 는 OK.

---

## 7. 향후 4% 목표를 향해

- Per-pair coupling 의 ML 기반 정밀 분해 (현재 단순 sum-rescale → 절대값 학습)
- Q3D synthetic pretraining curriculum (Stage 1-4: parallel plate → layered → 3D box → multi-conductor fringe)
- BEM-collocation residual feature (FastCap Green's function)
- 더 많은 다양한 train designs (현재 9개 → 30+)
- Per-net node decomposition (lumped → multi-segment SPEF for timing tools)

---

## 8. 결과 파일

- `pex_pipeline/` — 추론 시 모듈 (decompose, predict, write_spef, R, build_*_inference)
- `scripts/predict_spef_e2e.py` — 메인 CLI
- `scripts/spef_e2e/train_*.py` — 모델 학습
- `scripts/spef_e2e/validate_e2e.py` — 검증
- `output/spef_e2e/` — 저장된 모델 weights
- `output/spef_e2e/_test_e2e/tv80s_predicted.spef` — 검증 예측 SPEF
- `reports/SPEF_FLOW_PERFORMANCE_KO.md` — 상세 성능 리포트
- `reports/SPEF_FLOW_METHODOLOGY_KO.md` — 본 문서
