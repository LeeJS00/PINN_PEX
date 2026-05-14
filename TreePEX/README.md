# TreePEX — VLSI 기생 capacitance 예측기 (deployable)

**Input**: routed DEF + tech LEF + cell LEF + Liberty + layer.info  
**Output**: per-net ground + coupling capacitance SPEF (IEEE 1481-1999 호환, StarRC schema)  
**Model**: 5-seed Tweedie XGBoost prediction-mean ensemble on 67-D hand features  
**Hardware**: CPU only (16-core fork-Pool 권장; GPU 불필요)

---

## 1. 한 줄 소개

License-free PEX 도구로, StarRC와 **동일한 입력**(DEF / LEF / Liberty / layer stack)을 받아
**4.98% MAPE / 7 s (tv80s)** 또는 **5.28% MAPE / 70 s (nova)** 정확도/속도로 SPEF를 생성한다.
Cadence Innovus (6.96% / 분 단위)와 OpenRCX (8.83% / 분 단위) 대비 정확도 + 속도 모두 우위.

---

## 2. 빠른 시작

### 2.1 환경 준비

```bash
# Python 3.11+, 16-core CPU (32 GB RAM 권장; nova의 경우 64 GB+).
pip install -r TreePEX/requirements.txt

# (선택, V3 njit 가속용)
pip install --user numba
```

또는 사이트 표준 환경:

```tcsh
source tool.env   # /tool/etc/python/install/3.11.9 + StarRC + license
```

### 2.2 사전 학습 모델 위치 확인

배포 디렉에 동봉:
```
TreePEX/models/tweedie_{gnd,cpl}_seed{42,0,1,2,3}.json    # 10 weight files, 총 ~120 MB
TreePEX/models/FEATURE_ORDER.txt                          # feature column 순서
```

### 2.3 신규 design 전체 파이프라인 (DEF → SPEF, end-to-end)

```bash
python3 TreePEX/scripts/pex_cold.py --design intel22_tv80s_f3
python3 TreePEX/scripts/pex_cold.py --design intel22_nova_f3
```

기본값으로 V3 njit kernel (Round 4) + V4 per-design cache(있으면) + 16-worker fork-Pool 사용.
Single command — DEF parse → 67-D feature 추출 → 5-seed 예측 → SPEF write → (optional) golden SPEF 비교.

### 2.4 V3 backend 선택 (성능 vs 정확도 trade-off)

```bash
--v3-algo legacy      # numpy broadcast (가장 안정, 가장 느림)
--v3-algo auto        # threshold-gated (Round 3 기본; long-tail만 per-target)
--v3-algo per_target  # numpy per-target-cuboid 항상 사용
--v3-algo njit        # Numba JIT kernel (Round 4 추천, ~2× V3 빠름)
```

---

## 3. 입력 / 출력 명세

### 3.1 입력

| 항목 | 형식 | 용도 |
|---|---|---|
| `<design>.def` | LEF/DEF v5.8 | net 라우팅 geometry |
| `tool/pdk/22nm/tech_lef/p1222_js.lef` | LEF | metal layer 정의 |
| `tool/pdk/22nm/cell_lef/b15_nn.lef` | LEF | cell pin geometry |
| `tool/pdk/22nm/layers/layers.info` | text | layer stack (ε_r, 두께, 간격) |
| `tool/pdk/22nm/lib/*.lib` | Liberty (선택) | cell pin capacitance (현재 미사용) |

경로는 `TreePEX/scripts/pex_cold.py:66-68` + `configs/config.py`에서 호스트 고정.

### 3.2 출력

```
TreePEX/outputs/spef/<design>_pred.spef          # IEEE 1481-1999 SPEF
TreePEX/outputs/predictions/<design>_pred.csv    # per-net 예측값
TreePEX/outputs/cold_reports/cold_summary.json   # 파이프라인 timing + MAPE
TreePEX/outputs/cold_reports/<design>_treepex_per_net.csv  # 비교용 (golden 존재 시)
TreePEX/outputs/cold_reports/<design>_treepex_summary.json # per-net MAPE summary
```

SPEF schema (per net):
```
*D_NET <net_name> <total_cap_fF>
  *CONN
    *I <pin1> ...
  *CAP
    1 <net_name>:0 <ground_cap_fF>
    2 <net_name>:1 <agg_net_1>:0 <coupling_cap_fF>
    ...
  *END
```

---

## 4. 67-D feature pack

자세한 설명: [`docs/FEATURE_SPEC.md`](docs/FEATURE_SPEC.md) 참고 (또는 코드 `pex_cold.py:_v3_per_net` + `_v4_net_features`).

| 그룹 | 차원 | 설명 |
|---|---:|---|
| V3 wire geometry | 3 | n_cuboids, total_wire_length_um, total_metal_area_um2 |
| V3 net bbox | 3 | bbox_xy_um2, bbox_z_um, aspect_ratio |
| V3 layer histogram | 10 | layer_hist_M1..M8, layer_hist_M9_plus, n_layers_present |
| V3 aggressor pair | 12 | n_aggressor_nets, broadside/lateral overlap × {total, p95}, spacing × {min, p25, p50, p95}, n_edges × {lt1um, 1to3um, 3to4um} |
| V3 dielectric | 3 | eps_min, eps_max, eps_mean |
| V3 metal density | 3 | density_M1_M3, M4_M5, M6_plus |
| V3 VSS shielding | 5 | vss_n_cuboids, vss_total_area, vss_shield × {M1_M3, M4_M5, M6_plus} |
| V3 analytic priors | 2 | compact_gnd_estimate_fF (Sakurai-Tamaru), compact_cpl_estimate_total_fF |
| **V3 subtotal** | **41** | |
| V4 self check | 1 | target_n_cuboids_check |
| V4 aggressor counts | 5 | agg_n_distinct, agg_count × {above_z, below_z, within_{1,3,5}um} |
| V4 top-3 pair geometry | 18 | top{1,2,3} × {score, overlap_um2, min_xy_dist_um, mean_dz_um, agg_size_um2, layer_diff_flag} |
| V4 concentration | 1 | topk_score_concentration |
| **V4 subtotal** | **26** | |
| **TOTAL** | **67** | |

---

## 5. 성능 (golden = StarRC)

### 5.1 정확도

| Design | tot MAPE | gnd MAPE | cpl MAPE | R²_tot |
|---|---:|---:|---:|---:|
| **intel22_tv80s_f3** (3,280 nets) | **4.98 %** | 18.02 % | 13.27 % | 0.9940 |
| **intel22_nova_f3** (113,812 nets) | **5.28 %** | 17.40 % | 14.96 % | 0.9911 |

### 5.2 Wall-clock (DEF → SPEF, 16-worker, Round 4 njit)

| Design | Pipeline | DEF parse | V3 features | V4 features | Inference | SPEF write |
|---|---:|---:|---:|---:|---:|---:|
| tv80s | **48.2 s** | 2.1 s | **3.5 s** | 38.2 s | 3.9 s | 0.1 s |
| nova  | **4,906 s** | 94 s | **2,352 s** | 2,384 s | 6.4 s | 3.8 s |

Round 0 (pre-patch) 대비 누적 가속: tv80s **3.52×**, nova **1.64×**. V3 단독으로는 nova **2.38×**.
세부 내역: [`COLD_START_SPEEDUP_REPORT.md`](COLD_START_SPEEDUP_REPORT.md).

### 5.3 경쟁 도구 비교 (10-design mean MAPE)

| Tool | License | tot MAPE | tv80s wall |
|---|---|---:|---:|
| **TreePEX** | **free** | **4.98 %** | **7 s** |
| v12 PINN (archive) | free | 8.23 % | 10 s |
| Cadence Innovus | commercial | 6.96 % | ~120 s |
| OpenRCX | free | 8.83 % | ~60 s |
| StarRC (oracle) | commercial | reference | minutes |

---

## 6. 디렉 구조

```
TreePEX/
├── README.md                    ← (this file)
├── REPORT.md                    ← Paper-style 종합 보고서
├── COLD_START_SPEEDUP_REPORT.md ← Round 1-4 엔지니어링 deep-dive
├── requirements.txt             ← pip dependencies
├── run.sh                       ← bash wrapper for new designs
├── docs/                        ← 세부 plan/report
│   ├── FEATURE_SPEC.md          ← 67-D feature 정의 + 출처
│   ├── FEATURE_SPEEDUP_PLAN.md  ← Round 1-4 working plan (history)
│   ├── COLD_START_REPORT.md     ← 초기 cold-start report (legacy)
│   └── PROGRESS_REPORT.md       ← 초기 progress (legacy)
├── models/
│   ├── MODEL_CARD.md                                ← 학습 데이터 / 설정 / 검증
│   ├── FEATURE_ORDER.txt                            ← XGBoost feature 순서
│   ├── tweedie_{gnd,cpl}_seed{42,0,1,2,3}.json      ← 10 main predictors
│   ├── fanout_proxy_meta.json                       ← fanout proxy 메타
│   ├── fanout_proxy_xgb_tweedie.json                ← fanout XGB Tweedie proxy (cpl 의존도 0.81)
│   └── fanout_proxy_ridge.json                      ← fanout Ridge fallback
├── scripts/
│   ├── pex_cold.py              ← ★ Main entry: DEF → SPEF end-to-end
│   ├── pex_tool.py              ← Split-stage runner (cached features)
│   ├── 01_train_save_models.py  ← Offline training (5-seed Tweedie)
│   ├── 02_inference.py          ← Stage 1 (split, cached features)
│   ├── 03_write_spef.py         ← Stage 2 (split)
│   ├── 04_compare_golden.py     ← Stage 3 (split)
│   ├── dump_features.py         ← Diagnostic: per-net feature dump
│   ├── compare_features.py      ← Diagnostic: feature drift report
│   └── summarize_cold_results.py ← Aggregate cold reports across models
├── outputs/
│   ├── spef/                    ← Predicted SPEF
│   ├── predictions/             ← Per-net CSV
│   └── cold_reports/            ← Pipeline summaries, feature dumps
├── paper_benchmark/             ← e2e benchmark scripts (PAPER_TABLE.md)
├── presentation/                ← Figures + PPT
└── archive/                     ← Failed/superseded scripts and models
    ├── scripts/                 ← N1-N4 experiments, ASAP7, mesh PINN, etc.
    ├── models/                  ← catboost / fanout_proxy / mesh calibration
    └── runs/                    ← Failed experiment outputs
```

---

## 7. 재학습 (rare; 새 PDK 또는 새 design 추가 시)

```bash
# 1) 학습 데이터 build (한 번)
python3 scripts/build_dataset_multi.py
# → /data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv

# 2) 5-seed Tweedie 학습 (~9 min)
python3 TreePEX/scripts/01_train_save_models.py
# → TreePEX/models/tweedie_{gnd,cpl}_seed*.json (10 files)

# 3) 검증
python3 TreePEX/scripts/pex_cold.py --design intel22_tv80s_f3
```

학습 hyperparameters: `01_train_save_models.py` 상단 SEEDS + XGBoost params 참고.
- objective: `reg:tweedie`, variance_power=1.5
- depth=8, n_est=500, lr=0.05, subsample=0.8, colsample_bytree=0.8
- early_stopping=100 rounds on validation MAPE
- seeds: 42, 0, 1, 2, 3

---

## 8. 알려진 한계 (Known limits)

- **MAPE 천장 ~4.66%**: hand-feature 4-way oracle bound (XGB + Optimized features + B4 compact + Mesh PINN). 그 이하로 가려면 새 input modality (voxel CNN over rasterized routing) 또는 fundamentally 다른 paradigm 필요.
- **DEF/LEF/Liberty 입력의 정보 천장**: gnd MAPE ~17-18%는 representation-bound, NOT input-bound. StarRC도 동일 입력을 받지만 NXTGRD pattern-lookup + 3D field solver로 정확도를 얻음. 4% 격차 해소 경로는 모델 측에 있음.
- **per-pin cap 분포 없음**: lumped per-net `*CAP`만 emit. Per-pin distribution은 future work.
- **R (parasitic resistance) 미포함**: sister `r_analytic_v3` 트랙에서 처리.
- **22nm intel22 PDK 전용**: ASAP7 (7nm) cross-PDK 전이는 Phase F sprint에서 다룸 (`archive/scripts/01_train_asap7_models.py`).

---

## 9. 모델 / 코드 lineage

| Track | Test MAPE | 상태 | 이유 |
|---|---:|---|---|
| **TreePEX** (5-seed Tweedie XGBoost) | **4.98 %** | ✅ 현재 frontier | 정확도 + 속도 모두 우위 |
| PINN v12 mesh ensemble | 8.23 % | archived | TreePEX가 −3.25 pp / 120× faster |
| pex_v4 substrate physics | 5.55 % | archived | Phase B1 K1 gate fail |
| pex_v5 auto-4% sprint | 5.09 % | archived | TreePEX이 추월 |
| pex_v7 per-pair regression | 15.7 % | archived | cuboid resolution 부족 |
| pex_v8 hybrid analytic+residual | 55.5 % | archived | analytic prior over-estimate |

세부: 루트 `archive/` (gitignored) + 메모리 `~/.claude/projects/.../memory/` 참고.

---

## 10. 인용 / 라이선스

- 사용 PDK: 22nm bulk CMOS (intel22), AS-IS license; ASAP7 7nm은 Phase F에서 추가 예정.
- StarRC golden: Synopsys 2021.06.
- Open dependencies: numpy, pandas, xgboost (Apache 2.0), numba (BSD-2). 자세한 list: `requirements.txt`.

문의: 본 repo의 issue tracker.
