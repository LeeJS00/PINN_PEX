# PINNPEX EDA-Style PEX — CLI 사용법 (한글)

`scripts/predict_spef_e2e.py` — DEF + LEF + tech 만 받아 SPEF 출력.

---

## 1. 빠른 시작 (Quick Start)

### 새로운 DEF + LEF 으로 SPEF 추출

```bash
cd /home/jslee/projects/PINNPEX/experiments/cross_design_tv80s_2026_05_02

PYTHONPATH=.:/home/jslee/projects/PINNPEX python3 scripts/predict_spef_e2e.py \
    --def_path /path/to/your_design.def \
    --out_spef /path/to/predicted.spef \
    --num_workers 12
```

PDK 경로 (LEF + layers.info) 는 `configs/config.py` 의 다음 경로에서 자동 로드:
- `LAYERS_INFO_PATH` = `tool/pdk/22nm/layers/layers.info`
- `TECH_LEF_PATH` = `tool/pdk/22nm/tech_lef/p1222_js.lef`
- `CELL_LEF_PATH` = `tool/pdk/22nm/cell_lef/b15_nn.lef`

**예상 출력**:
- 4-6분 후 `predicted.spef` 파일 생성 (3-10k nets 규모)
- `<temp_dir>/cuboids/`, `<temp_dir>/features.parquet`, `<temp_dir>/pair_features.parquet`, `<temp_dir>/cuboid_arr.npz` (intermediate)

---

## 2. CLI 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--def_path` | (required) | 입력 DEF 파일 |
| `--out_spef` | (required) | 출력 SPEF 경로 |
| `--top_module` | DEF 파일명에서 자동 추론 | 디자인 이름 |
| `--temp_dir` | `<out_spef>/_pex_temp_<top>` | 중간 산출물 디렉토리 |
| `--cuboid_pkl_dir` | None | 미리 빌드된 cuboid pkl 디렉토리 (Stage 1 skip) |
| `--manifest` | None | 사전 빌드 manifest CSV (pkl 검색 가속) |
| `--num_workers` | 16 | parallel processes |
| `--cutoff_um` | 4.0 | aggressor 검색 거리 |
| `--models_dir` | `output/spef_e2e/` | 학습된 모델 디렉토리 |

---

## 3. 사용 예시

### 시나리오 1: 캐시된 cuboid 가 있는 경우 (가장 빠름, ~1분)

```bash
PYTHONPATH=.:/home/jslee/projects/PINNPEX python3 scripts/predict_spef_e2e.py \
    --def_path /home/jslee/projects/PINNPEX/tool/def/intel22/intel22_tv80s_t1.def \
    --out_spef output/test/tv80s.spef \
    --cuboid_pkl_dir /data/PINNPEX/data/processed_v3/intel22/intel22_tv80s_f3 \
    --manifest /data/PINNPEX/data/processed_v3/intel22/dataset_manifest_v3.csv \
    --num_workers 8
```

### 시나리오 2: Raw DEF 부터 (full pipeline, ~5분)

```bash
PYTHONPATH=.:/home/jslee/projects/PINNPEX python3 scripts/predict_spef_e2e.py \
    --def_path /home/jslee/projects/PEX_SSL/data/raw/def/intel22/intel22_tv80s_f3.def \
    --out_spef output/test/tv80s_true.spef \
    --num_workers 12
```

### 시나리오 3: Golden SPEF 와 비교

```bash
# 1. SPEF 생성
python3 scripts/predict_spef_e2e.py --def_path your.def --out_spef pred.spef

# 2. Golden 과 비교
PYTHONPATH=.:/home/jslee/projects/PINNPEX python3 scripts/spef_e2e/validate_e2e.py \
    --predicted_spef pred.spef \
    --golden_spef /path/to/golden_starrc.spef \
    --out_dir reports/comparison \
    --design_name your_design
```

산출물:
- `reports/comparison/per_net_metrics.csv` — total_cap, c_gnd, c_cpl_total, total_R MAPE
- `reports/comparison/per_pair_metrics.csv` — per-pair coupling MAPE
- `reports/comparison/per_pair_stratified.csv` — magnitude 별 stratified MAPE
- `reports/comparison/r2_scatter_total_cap.png` — log-log density scatter
- `reports/comparison/r2_scatter_per_pair.png` — per-pair scatter

---

## 4. 학습 모델 재학습

기존 학습된 모델: `output/spef_e2e/{total_cap, gnd_ratio, total_r, pair_regressor}/`

새로 학습이 필요한 경우 (예: 다른 PDK):

```bash
# 1. 9개 train designs 의 pair features 빌드 (이미 캐시 있으면 skip)
python3 scripts/build_pair_dataset.py \
    --designs intel22_aes_cipher_top_f3 intel22_gcd_f3 intel22_ibex_core_f3 ... \
    --workers 12

# 2. v3 features 빌드
python3 scripts/build_features_v3.py \
    --designs <list> --workers 12

# 3. cuboid_arr 빌드
python3 scripts/precache_cuboid_arrays.py \
    --designs <list> --workers 12

# 4. Total cap 모델 학습 (~15min)
python3 scripts/spef_e2e/train_total_cap.py

# 5. C_gnd ratio 모델 학습 (~5min)
python3 scripts/spef_e2e/train_gnd_ratio.py

# 6. Total R 모델 학습 (~5min)
python3 scripts/spef_e2e/train_total_r.py

# 7. Pair regressor 학습 (~20min)
python3 scripts/spef_e2e/train_pair_regressor.py
```

총 학습 시간: ~45분 (9 train designs 기준).

---

## 5. SPEF 출력 형식

```
*SPEF "IEEE 1481-1999"
*DESIGN "<top>"
*DATE "..."
*VENDOR "PINNPEX"
*PROGRAM "PINNPEX-EDA"
*VERSION "v1.0"
*DESIGN_FLOW "PIN_CAP NONE" "NAME_SCOPE LOCAL"
*DIVIDER /
*DELIMITER :
*BUS_DELIMITER []
*T_UNIT 1.0 NS
*C_UNIT 1.0 FF
*R_UNIT 1.0 OHM
*L_UNIT 1.0 HENRY

*D_NET <name> <total_cap_fF>
*CONN
*P <name>:1 O *C 0.0 0.0
*N <name>:2 *C 0.0 0.0
*CAP
1 <name>:1 <c_gnd_fF>
2 <name>:1 <agg1>:1 <c_pair1_fF>
3 <name>:1 <agg2>:1 <c_pair2_fF>
...
*RES
1 <name>:1 <name>:2 <total_r_ohm>
*END
```

**Format note**: lumped per-net topology (1 port + 1 sink). Multi-segment topology 는 미지원 — downstream timing tools 가 segment-level R 을 요구하면 추가 디컴포지션 필요.

---

## 6. Troubleshooting

| 증상 | 원인 | 해결 |
|---|---|---|
| `KeyError: 'cuboids'` in pkl | 비-cuboid pkl 파일 (e.g., inst_net_map) | inference modules 가 자동 필터; 안 될 시 `--manifest` 사용 |
| `No pkl.gz files found in ...` | Stage 1 출력 경로 mismatch | `--cuboid_pkl_dir` 명시적 지정 또는 absolute path |
| `ModuleNotFoundError: src.data_loader` | PYTHONPATH 순서 | workspace 가 PINNPEX root 보다 앞에 와야 함 |
| Pair regressor 가 비어있음 | `output/spef_e2e/pair_regressor/` 없음 | `train_pair_regressor.py` 실행 또는 `--cuboid_pkl_dir` 으로 fallback |
| Total cap MAPE 가 매우 큼 | 다른 PDK 의 DEF 사용 | intel22 학습 모델은 intel22 에서만 유효; 새 PDK 는 retrain 필요 |

---

## 7. 성능 (intel22 cross-design tv80s 기준)

자세한 수치는 `SPEF_FLOW_PERFORMANCE_KO.md`. 핵심:

| 지표 | MAPE | R²(log) |
|---|---|---|
| total_cap | 10.59% | 0.9806 |
| c_gnd | 27.79% | 0.9372 |
| c_cpl_total | 19.85% | 0.9552 |
| total_R | 19.91% | 0.6969 |

**Runtime**: ~5min (StarRC ~32min → 6× speedup).
