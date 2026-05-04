# total_R Analytic Policy — ML 폐기, 분석식 채택

_작성일: 2026-05-02 KST. 기존 v7 ML ensemble (11.92% MAPE, 47 models, ~1.5 GB) 을 분석식 정책으로 대체._

---

## 핵심 결론

**total_R 은 예측 영역이 아니다.** SPEF의 `*RES` 섹션이 segment 별 `length / width / layer` 를 명시적으로 가지고 있고, layer 변환은 DEF NETS 의 `VIA{n}_*` 토큰으로 결정론적으로 셀 수 있다. 즉 R 은 분석식의 합:

```
R_net  =  α × ( Σ_seg  sheet_R[layer(seg)] × length(seg) / width(seg)
              + Σ_via  R_via[via_lvl(via)] )
```

여기서 segment-level 정보 (per-segment length, width, layer, via top/bottom) 는 PINNPEX 의 기존 `src/preprocessing/def_parser.py::DefStreamParser` 가 이미 추출하고 있다. RECT landing, SPECIALNETS power, LEF-derived via stack 도 모두 처리됨.

ML 모델 (LGBM/CatBoost/DeepSet 47개, 145-dim feature engineering) 은 **틀린 도구** 였다.

| 정책 | tv80s MAPE | bias | parameters | inference cost | 상태 |
|---|---|---|---|---|---|
| v7 ML ensemble | 11.925% | -4.77% | ~10⁵ weights | ~5-7s/3380 nets | (legacy) |
| Analytic v1 (ad-hoc DEF parser) | 6.874% | -2.12% | 1 scalar | — | (deprecated) |
| **Analytic v2 (PINNPEX parser + global α)** | **6.987%** | -1.33% | **1 scalar** | **<1s/3380 nets** | **이 정책** |
| Oracle α (이론 ceiling) | 6.990% | -1.29% | (cheat) | — | (sanity) |

v2 와 v1 의 overall MAPE 는 거의 같지만 (0.11pp 차이), **stratum 분포가 훨씬 건강**:

| Stratum | v1 MAPE / bias | **v2 MAPE / bias** |
|---|---|---|
| Q1 short | 22.2% / -22.2% | **9.6% / -0.15%** |
| Q2 | 18.0% / -18.0% | 5.0% / +0.95% |
| Q3 | 13.4% / -13.4% | 5.8% / -0.33% |
| Q4 long | 8.9% / -8.8% | 7.6% / -5.79% |

v1 은 RECT landing / per-segment width 을 모르고 단일 α 가 4 stratum 의 systematic 편차를 한꺼번에 흡수하는 구조. v2 는 PINNPEX parser 의 segment-level 정확도 덕분에 Q1-Q3 거의 unbiased. Q4 long 의 잔여 -5.8% 는 v3/v4 via R 재calibration (Step C) 으로 해결 가능.

추가 작업으로 **<4% MAPE** 도달 경로는 아래 Step C 참고.

---

## 정책 정의 (v2 — PINNPEX parser 기반)

### 입력

1. **Tech calibration table** (training-design SPEFs 로부터 1회 추출, 디자인 독립):
   - `sheet_R[layer]`: per-metal-layer sheet resistance (Ω/sq)
   - `R_via[via_lvl]`: per-via-stack R (Ω)
2. **Per-net DEF segment list** (`DefStreamParser.parse()` 출력의 `segments`):
   - WIRE: `{type, layer, start, end, width}` — width 는 LEF default 또는 SPECIALNETS explicit
   - RECT: `{type, layer, rect}` — landing/contact 패치 (R≈0 처리)
   - VIA : `{type, name, pos, bot_layer, top_layer, layer}` — via stack
3. **One global calibration scalar** `α` (training-design golden SPEF에 대한 중앙값 비율로 fit).

### 추론 (per-net)

```python
R = 0.0
for s in segments:
    if s["type"] == "WIRE":
        L = abs(s["end"][0]-s["start"][0]) + abs(s["end"][1]-s["start"][1])
        R += sheet_R[s["layer"]] * L / s["width"]
    elif s["type"] == "RECT":
        pass   # landing patches contribute ~0Ω (golden RES has 0.001Ω entries)
    elif s["type"] == "VIA":
        via_lvl = parse_via_layer(s["name"], s["bot_layer"], s["top_layer"])
        R += R_via[via_lvl]
R *= alpha
```

`α` 외에는 학습된 파라미터 없음. 모든 상수는 technology-level (디자인 무관).

### Calibrated 상수 (intel22, _f3 corner, 9 train designs, 207K nets)

```json
sheet_R (Ω/sq):  M1: 0.713,  M2: 0.583,  M3: 0.600,  M4: 0.600,  M5: 0.587
R_via   (Ω):     v1: 11.61,  v2: 13.07,  v3: 13.07,  v4: 13.07
α (v2, PINNPEX parser, median over train nets): 1.4777
α (v2, MAPE-min):                                1.4640
```

`width` 는 PINNPEX `DefStreamParser` 가 segment 별로 LEF/SPECIALNETS 에서 자동 결정 — width_typ 가정 불필요.

저장 위치: `reports/sheet_r_calibration.json`, `reports/alpha_global_v2.json`.

(v1 의 α=1.16 vs v2 의 α=1.48 차이는 v1 의 ad-hoc parser 가 RECT/landing/SPECIALNETS 을 누락해서 raw R 이 24% 더 높게 나왔기 때문. v2 는 더 작은 raw R 을 더 큰 α 로 보정.)

---

## 파이프라인

```
DEF + tech LEF + cell LEF + layers.info
   │
   ▼
DefStreamParser.parse()  ──►  (net_name, cuboids, segments[]) per net
                                            │
                                            ▼
                       R = α × (Σ_WIRE sheet × L/W + Σ_VIA R_via)
```

### 코드

| 단계 | 스크립트 | 역할 |
|---|---|---|
| Calibration (1회) | `scripts/spef_e2e/calibrate_sheet_r_from_spef.py` | golden `*RES` 에서 sheet_R, R_via 추출 |
| α-fit (1회, v2) | `scripts/spef_e2e/analytic_r_v2_pinnpex_parser.py` | PINNPEX parser 사용 + train α 추정 |
| Inference | 동일 스크립트 (compute_R_per_net 함수) | DEF → segment list → R |

**기존 활용**:
- `src/preprocessing/def_parser.py::DefStreamParser` — DEF segment list 추출 (이미 있음)
- `src/preprocessing/lef_parser.py::LefParser` — tech LEF (via 정의)
- `src/preprocessing/cell_parser.py::CellLibParser` — cell LEF (cell pin geometry)
- `src/preprocessing/layer_parser.py::LayerInfoParser` — layer stack (z, ε, thickness)

기존 `pex_pipeline/compute_resistance.py` 는 **하드코딩된 잘못된 sheet_R + brute-force `R_CALIBRATION_SCALE=3.5`** 를 사용 — 이 정책으로 대체.

**v1 (deprecated) 와의 차이**:
- v1: 자체 작성한 `parse_def_via_counts.py` (regex 기반, ROUTED 만)
- v2: PINNPEX `DefStreamParser` 사용 (RECT, SPECIALNETS, per-segment width 정확)

---

## 검증 결과 (intel22_tv80s_f3, 3,380 nets)

### v2 (PINNPEX parser, 채택 정책)

| 단계 | MAPE | median | P90 | bias | 95% CI |
|---|---|---|---|---|---|
| Raw analytic (α=1) | 33.229% | 32.35% | 41.35% | -33.23% | [33.01, 33.43] |
| **+ train-fit α (1.4777)** | **6.987%** | 5.69% | 14.17% | -1.33% | **[6.79, 7.19]** |
| train α_mape-min (1.4640) | 6.968% | 5.44% | 14.62% | -2.25% | [6.76, 7.18] |
| Oracle α (1.4783, cheat) | 6.990% | 5.73% | 14.14% | -1.29% | [6.79, 7.19] |
| v7 ML baseline (참고) | 11.925% | 7.21% | 27.75% | -4.77% | [11.44, 12.41] |

train-α 와 oracle α 가 0.0006 차이 (1.4777 vs 1.4783) → α 가 디자인 간 거의 완벽하게 generalize.

### Per-train-design α 안정성 (v2)

| 디자인 | α_med | n_nets | raw MAPE | post-α MAPE |
|---|---|---|---|---|
| aes_cipher_top | 1.497 | 12,042 | 33.79% | 8.58% |
| gcd | 1.489 | 258 | 33.19% | 8.75% |
| ibex_core | 1.490 | 10,630 | 33.95% | 9.45% |
| mc_top | 1.476 | 3,876 | 33.40% | 7.42% |
| spi_top | 1.462 | 1,697 | 32.92% | 6.60% |
| usbf_top | 1.450 | 7,734 | 32.56% | 7.54% |
| vga_enh_top | 1.478 | 34,459 | 33.50% | **5.86%** |
| wb_conmax_top | 1.483 | 17,704 | 35.04% | 8.98% |
| nova | 1.472 | 118,960 | 34.15% | 8.72% |
| **(global, 207K nets)** | **1.4777** | — | — | **8.14%** (train) |
| **TEST tv80s** | (apply 1.4777) | 3,380 | 33.23% | **6.99%** |

α 분포 [1.450, 1.497], σ ≈ 0.014 (0.95%). v1 (σ=2.2%) 보다도 훨씬 안정 — PINNPEX parser 가 segment-level 정확도가 높아 디자인 간 구조 차이가 α 에 덜 흡수됨.

### Length-stratified MAPE (v2 post-global-α, tv80s)

| Stratum | n | R_med | MAPE | bias | 비고 |
|---|---|---|---|---|---|
| Q1 short | 845 | 69.7Ω | 9.60% | **-0.15%** | (v1: 22.2% / -22.2%) — RECT 처리 효과 |
| Q2 | 845 | 92.7Ω | 5.04% | +0.95% | (v1: 18.0% / -18.0%) |
| Q3 | 845 | 169.9Ω | 5.75% | -0.33% | (v1: 13.4% / -13.4%) |
| Q4 long | 845 | 503.3Ω | 7.56% | -5.79% | (v1: 8.9% / -8.8%) — 잔여 |

Q1-Q3 거의 unbiased. Q4 long 의 -5.8% 는 v3/v4 via R 이 단일값 13.07Ω 으로 calibrate 되어 있는데 실제로는 thicker stack 에서 더 큰 R 을 가질 가능성 — Step C 영역.

### Per-stratum α 시도 (실패)

R_pred_raw quartile 별 α 를 별도 fit 했으나 train/test 간 R 분포 shift 때문에 overall MAPE 가 7.68% (global α 6.99% 보다 0.69pp 악화). Q1 short 은 +2.93% 으로 오히려 over-correct, Q4 long 은 +2.62% 으로 부분 개선. **stratum 분기보다는 via R refinement (Step C) 가 정공법.**

---

## 왜 이게 가능했는가

### 발견 1: SPEF *RES 가 곧 토폴로지

```
*RES
1 m1_n m1_n:14 1.94868   //  $l=0.1600 $w=0.0440 $lvl=7  ← m3 wire
2 m1_n m1_n:13 55.7085   //  $l=4.5650 $w=0.0440 $lvl=7  ← m3 wire
3 (..pin..)   m1_n:10 11.6141 // $vc=12 $lvl=10           ← via1
...
```

`$l, $w, $lvl` 이 모든 segment 에 명시. R = ρ × L / (W × T) 의 직접 계산. ML 은 이 분석식을 학습할 이유가 없음.

### 발견 2: 기존 compute_resistance.py 의 엉터리 default + 보정 스케일

```python
# 기존 (잘못)
DEFAULT_SHEET_R_INTEL22 = {"m1": 1.5, "m2": 0.42, ...}  # 골든과 2× 차이
"v0": 5.0, "v1": 5.0, ...                                  # 골든의 11.6/13.1 의 절반
R_CALIBRATION_SCALE = 3.5  # 위 두 오류를 brute-force 로 가리던 글로벌 보정
```

**이게 ML 모델이 11.92% 에서 정체된 진짜 이유.** 모델은 정확한 wirelength 와 잘못된 sheet_R 합 (잘못된 R_analytic) 의 비선형 관계를 학습했지만, 곱셈 상수 calibration 만 잘하면 닿을 수 있는 영역이었음.

### 발견 3: α 가 거의 디자인 무관

9개 train 디자인의 α 분포 [1.136, 1.222], σ=0.026. 이 변동은 (a) DEF parser 의 RECT 랜딩 누락 비율, (b) 디자인별 net 평균 길이 변동에서 기인. 1.16 의 글로벌 상수로 충분.

---

## <4% 목표 도달 경로 (v2 baseline 6.99%)

### ✅ Step A: RECT landing 처리 — DONE (v2)
PINNPEX `DefStreamParser` 가 이미 RECT 를 segment 로 emit. 본 정책은 RECT 의 R 기여를 **0** 으로 처리 (golden RES 도 landing 패치를 0.001Ω 로 표기). Q1 short bias -22% → -0.15% 해소.

### ❌ Step B: Per-stratum α — TRIED, FAILED
R_pred_raw quartile 별 α: 7.68% (global α 6.99% 보다 0.69pp 악화). train/test bucket shift 때문. **Step C 가 정공법.**

### Step C: Via R refinement (가장 큰 잔여 leverage, ~2h)
현재 v3/v4 via R = 13.07Ω 으로 단일값 (calibration 의 vc=8 클래스 median). 실제로는:
- thicker via stack (m4↔m5, m5↔m6) 의 등가 R 이 더 큼
- via 의 "vc class" (coverage code) 별 R 이 다름 — `*VIA_COVERAGE_CODES` 의 16개 분류
- 짧은 stub 위에 stacked via 의 boundary 효과

작업:
1. Golden RES 의 via segment 를 (lvl, vc) 두 변수로 다시 calibrate (현재는 lvl 만 활용).
2. PINNPEX parser 의 segments 에 vc class 정보 추가 (DEF VIA name 또는 LEF spec 으로 추론).
3. tv80s Q4 long 의 -5.8% bias 흡수.

기대: 6.99% → **~4.5%**.

### Step D: Pin landing stub (cell LEF 참조, ~2h)
LEF 의 cell pin geometry 에서 M1 pin stub 길이 추출. tv80s 의 Q1 short 잔여 9.6% MAPE (bias 거의 0) 의 magnitude 줄임. 기대: ~4.5% → **~3.5%**.

### Step E: SPECIALNETS power R (선택)
대부분 power net 은 lumped R 평가에서 빠지지만, 일부 시나리오에서 신뢰성 향상. 기대: marginal.

### 이론 ceiling

`analytic_r_feasibility_summary.json`: golden RES 의 ground-truth wirelen + via count 사용 시 분석식 MAPE = **2.61%**. 즉 via R 의 vc-class 분포 까지 정확히 모델링하면 ceiling 은 ~2.6% 이며, 우리 v2 의 6.99% → Step C → ~4.5% → Step D → ~3.5% 가 현실적 단계.

### 이론적 ceiling

`analytic_r_feasibility_summary.json`: golden RES 의 ground-truth wirelen + via count 사용 시 분석식 MAPE = **2.61%**. 즉 wirelen / via 추출이 완벽하면 분석식 정확도는 본질적으로 2.6% 가 한계 (sheet_R / via_R 의 vc 클래스 변동성 때문).

---

## 마이그레이션 계획

### Phase 1 (즉시) — v7 ML R 모델 deprecate
- `pex_pipeline/compute_resistance.py` 를 새 분석식 정책으로 교체
- 기존 `output/spef_e2e/total_r/{lgbm,cat}_seed*.{pkl,cbm}` (10개) 는 보관용으로 두되, predict_caps.py 에서 호출 제거
- `output/spef_e2e/total_r/stratum_weights.json` deprecate
- 디스크 절감: ~50 MB. inference latency 절감: ~1-2s
- v7 SPEF 에서 R MAPE 11.92% → 6.87% (predict 시점부터 적용)

### Phase 2 (이번 주) — Step A (RECT landing)
- `parse_def_via_counts.py` 에 RECT 파싱 추가
- 기대: 6.87% → ~4.5%
- α 재추정 (RECT 추가로 raw bias 가 줄어들고 α 가 1.16 → ~1.08 로 감소 예상)

### Phase 3 (다음 주) — Step B + C 로 <4% 달성
- Per-stratum α
- Pin stub 길이 (LEF parser)

### Phase 4 (장기) — 예측 영역에서 R 완전 제거
- 향후 추가 디자인 (다른 PDK) 도입 시 분석식 정책의 calibration 을 다시 한번 fit 하면 됨
- ML R 모델 코드/스크립트 (train_total_r*.py 등) 모두 archive 폴더로 이동

---

## 산출물

| 파일 | 역할 |
|---|---|
| `reports/sheet_r_calibration.json` | 9 train design 의 1.25M segments 에서 추출한 per-layer sheet R + per-via R |
| **`reports/alpha_global_v2.json`** | **PINNPEX parser 기반 global α + per-bucket α + tv80s 결과 (정책)** |
| `reports/alpha_global.json` | (v1, deprecated) ad-hoc parser 기반 |
| `reports/analytic_r_feasibility_summary.json` | analytic policy 의 이론 ceiling (2.61%) |
| `reports/analytic_r_v2_test_per_net.csv` | tv80s 의 per-net pred / gold / ape 상세 |
| `scripts/spef_e2e/calibrate_sheet_r_from_spef.py` | sheet/via R calibration 스크립트 (1회) |
| **`scripts/spef_e2e/analytic_r_v2_pinnpex_parser.py`** | **PINNPEX parser 기반 분석식 정책 — 채택** |
| `scripts/spef_e2e/parse_def_via_counts.py` | (v1, deprecated) ad-hoc DEF NETS 파서 |
| `scripts/spef_e2e/analytic_r_feasibility.py` | golden RES 기반 분석식 ceiling 측정 |
| `scripts/spef_e2e/analytic_r_full_pipeline.py` | (v1, deprecated) |
| `scripts/spef_e2e/fit_global_alpha.py` | (v1, deprecated) |

---

## 사용 예시 (proposed inference flow, v2)

```python
import json, sys
sys.path.insert(0, "/home/jslee/projects/PINNPEX")
from configs import config as cfg
from src.preprocessing.def_parser  import DefStreamParser
from src.preprocessing.layer_parser import LayerInfoParser
from src.preprocessing.lef_parser   import LefParser
from src.preprocessing.cell_parser  import CellLibParser

calib = json.load(open("reports/sheet_r_calibration.json"))
alpha = json.load(open("reports/alpha_global_v2.json"))["alpha_global_median"]
sheet_R = {r["name"].upper(): r["sheet_median"] for r in calib["metal_per_layer"]}
via_R   = {r["name"]: r["R_median"] for r in calib["via_per_layer"]}

layer_map = LayerInfoParser(cfg.LAYERS_INFO_PATH).parse()
tech_lef  = LefParser(cfg.TECH_LEF_PATH).parse()
cell_lib  = CellLibParser(cfg.CELL_LEF_PATH).parse()
parser = DefStreamParser(def_path, layer_map, tech_lef, cell_lib)

R_per_net = {}
for net, _, segs in parser.parse():
    R = 0.0
    for s in segs:
        if s["type"] == "WIRE":
            L = abs(s["end"][0]-s["start"][0]) + abs(s["end"][1]-s["start"][1])
            R += sheet_R[s["layer"].upper()] * L / max(s["width"], 1e-6)
        elif s["type"] == "VIA":
            # parse "VIA<n>_..." or fall back to bot/top metal layer numbers
            ...  # see compute_R_per_net in analytic_r_v2_pinnpex_parser.py
    R_per_net[net] = alpha * R
```

(Production-grade 구현은 `pex_pipeline/compute_resistance.py` 교체 — 다음 PR.)

---

_작성일: 2026-05-02 KST. v7 ML 정책 deprecate, analytic 정책 채택._
