# PINN-PEX Framework — LEF + DEF + LIB → SPEF

_작성일: 2026-05-03 KST. 본 문서는 **공유 프로젝트 계획서**로, 모든 PINN-PEX 세션이 참조 가능한 상위 framework 정의를 제공한다._

> **SCOPE**: 본 framework는 EDA PEX 도구 (StarRC / Cadence Quantus) 와 동등한 입출력으로 **routed layout (DEF) + 기술 파일 (LEF + LIB) → 전기적 파라시틱 (SPEF)** 을 생성한다. 출력 SPEF 는 IR drop 분석 (Voltus), STA (PrimeTime), Power 분석 등 downstream EDA flow 에서 변경 없이 consume 가능.

---

## 1. 입출력 계약 (StarRC 동등)

### 1.1 Inputs

| 파일 | 출처 | 역할 |
|---|---|---|
| `<design>.def` | placement-and-route tool (Innovus, ICC2 등) | routed wire/via geometry + cell instances + nets + pins |
| Tech LEF (e.g., `p1222_js.lef`) | foundry / PDK | metal layers, vias, design rules, sheet resistance |
| Cell LEF (e.g., `b15_nn.lef`) | foundry / std cell vendor | per-cell pin geometry, OBS (cell-internal routing) |
| `layers.info` (stack) | foundry | per-layer z position, thickness, ε (dielectric constants) |
| **`.lib` (LIBERTY)** | foundry / std cell vendor | **per-cell pin_capacitance** (transistor gate Cgg) |

> `.lib` 는 **선택적 입력** (degraded-mode 가능). intel22 등 일부 PDK에서는 `.lib` 미제공. 그 경우 cell intrinsic capacitance 추정 정확도가 낮아짐 (paradigm-independent 21% c_gnd ceiling 근거).

### 1.2 Output

```
*SPEF "IEEE 1481-1999"
*DESIGN "<top>"
*DATE ...
*VENDOR "PINN-PEX"
*PROGRAM "PINN-PEX"
*VERSION "v1"
*DESIGN_FLOW "PIN_CAP NONE" "NAME_SCOPE LOCAL"
*DIVIDER /
*DELIMITER :
*BUS_DELIMITER []
*T_UNIT 1.0 NS
*C_UNIT 1.0 FF
*R_UNIT 1.0 OHM
*L_UNIT 1.0 HENRY
...
*D_NET <net_name> <total_cap_fF>
  *CONN
    *P <port> O *C <x> <y>
    *I <inst_pin> I *C <x> <y>
    *N <node>:<i> *C <x> <y>
  *CAP
    1 <node> <c_gnd_fF>
    2 <node_a> <node_b>:<...> <c_cpl_fF>     (per-pair coupling)
    ...
  *RES
    1 <node>:1 <node>:2 <total_r_ohm>        (lumped 2-node R)
*END
```

### 1.3 Compatibility

| Downstream | Tool | 입력 |
|---|---|---|
| IR drop / EM / power integrity | Cadence **Voltus** | DEF + 우리 SPEF (signal nets) + power network from DEF SPECIALNETS |
| Static timing | Synopsys **PrimeTime**, Cadence Tempus | 우리 SPEF + library .lib + SDC |
| Crosstalk / glitch | PT-SI, Tempus-SI | 우리 SPEF (per-pair coupling 포함) |

---

## 2. Pipeline 7-Stage Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  DEF + LEF + layers.info ──► PINNPEX DefStreamParser                        │
│        + .lib (optional)                                                     │
│                                                                              │
│      Stage 1 ──► cuboid pkls (per-net 3D box decomposition)                 │
│                                                                              │
│      Stage 2 ──► 145-dim hand features (per-net wire/via stats)             │
│                                                                              │
│      Stage 3 ──► per-(target,aggressor) pair features (per coupling pair)   │
│                                                                              │
│      Stage 4 ──► 3-stream cuboid arrays (target / aggressor / power)        │
│                                                                              │
│      Stage 5 ──► ML inference                                                │
│                  ├─ total_R   (analytic + GBT residual cascade)              │
│                  ├─ total_cap (LGBM/CatBoost/MLP/DeepSet ensemble)           │
│                  ├─ c_gnd     (analytic + bounded MLP residual)              │
│                  └─ per-pair  (LGBM pair regressor + sum-rescale)            │
│                                                                              │
│      Stage 6 ──► decompose + distribute (split total → c_gnd + per-pair)    │
│                                                                              │
│      Stage 7 ──► IEEE 1481-1999 SPEF write (LumpedSPEFWriter)               │
│                                                                              │
│  Output: <design>.spef  (Voltus / PrimeTime compatible)                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Stage details

| Stage | Function | Code | Cost (tv80s 3.4K nets) |
|---|---|---|---|
| 1. DEF→cuboid | `scripts/build_dataset.py` (PINNPEX core) | DefStreamParser + LefParser + CellLibParser | 26.6 s |
| 2. features | `pex_pipeline/build_features_inference.py` | 145-dim hand features per net | 31.2 s |
| 3. pair feats | `pex_pipeline/build_pair_features_inference.py` | per (target, aggressor) features, multi-radius density | 110.9 s |
| 4. cuboid arr | `pex_pipeline/build_cuboid_arr_inference.py` | 3-stream npz for ML input | 25.2 s |
| 5. ML | `pex_pipeline/predict_caps.py` | LGBM + CatBoost + MLP + DeepSet ensemble (47 models) | 8.7 s |
| 6. decompose | `pex_pipeline/decompose_caps.py` + `distribute_pairs_lgbm.py` | total → c_gnd + per-pair | 38.3 s |
| 7. write | `pex_pipeline/write_spef.py` (LumpedSPEFWriter) | IEEE 1481-1999 streaming | 0.6 s |
| **Total** | — | — | **247.6 s (4.13 min)** |

### 2.2 Entry point

```bash
PYTHONPATH=. python3 scripts/predict_spef_e2e.py \
    --def_path /path/to/design.def \
    --out_spef /path/to/output.spef \
    --num_workers 16
# (LEF, layers.info, optional .lib loaded automatically from cfg)
```

---

## 3. Modeling layers (paradigm)

### 3.1 Physics-anchored hybrid (current)

Each per-net target (R, c_gnd, c_cpl) is modeled as:

```
prediction = analytic_base × residual_correction
```

where:
- `analytic_base` = closed-form physics formula (parallel-plate cap, sheet R × wirelength, via R count, ...)
- `residual_correction` = bounded multiplicative factor learned by ML

This ResCap-style architecture (ASPDAC 2025) provides:
- **Day-1 reasonable predictions** (ML can be zero, model defaults to physics)
- **Bounded deviation** (ML can't blow up beyond physical bounds)
- **Data efficiency** (analytic prior 만으로도 reasonable, residual은 fine-tune)

### 3.2 Per-target architecture (현재 status)

| Target | Architecture | OOD MAPE (canonical) | 한계 |
|---|---|---|---|
| **total_R** | calibrated sheet_R + R_via × n_via + NNLS-IRLS + 5-LGBM ensemble + Stage 3 stacking | **4.00% combined** | Stage 1 NNLS already optimal for combined OOD |
| **c_gnd** | Phase 1 hybrid (parallel-plate prior + NNLS calib + bounded MLP residual) | **23.92% combined** | Hand-feature ceiling ~21%; needs `.lib` Cgg or per-pattern Phase 1 |
| **total_cap** | LGBM/CatBoost/MLP/DeepSet 20-model ensemble + stratum blend | ~5-6% (pex_v3 cross-design) | gnd/cpl cancellation artifact |
| **c_cpl_total** | total - c_gnd derived | ~12-14% per-channel | follows c_gnd accuracy |
| **per-pair** | LGBM pair regressor + sum-rescale to c_cpl_total | 110% (small caps); 50% (>0.05fF) | lumped → per-pair distribution 한계 |

### 3.3 Future paradigm (Phase 1 + Phase 2, pex_v3 contribution)

```
DEF + cell shapes ──► canonical patterns (≤10 conductors, ≤10×10×20 μm)
                          │
                          ▼
                 Phase 1: per-pattern hybrid
                   - analytic layered Green's function (Mode A: stacked dielectric,
                                                         Mode B: image-charge)
                   - bounded multiplicative neural residual (RES_CLAMP=log(2))
                   - target: per-pattern MAPE < 4% (CNN-Cap territory)
                          │
                          ▼
                 Phase 2: pattern → full-net aggregation
                   - learned coupling/edge/shielding aggregator
                   - target: full-net MAPE < 4% (paper-grade)
```

본 framework는 Phase 1+2 에 대한 wrapping 으로서, 모델 결과가 들어올 때 SPEF 출력을 그대로 사용 가능.

---

## 4. Train / Test split (canonical)

intel22 PDK, 11 designs × 1.32M tiles × 257K nets:

| Split | Designs | Total nets |
|---|---|---|
| TRAIN (9) | aes_cipher_top, gcd, ibex_core, **ldpc_decoder_802_3an**, mc_top, spi_top, usbf_top, vga_enh_top, wb_conmax_top | ≈ 376K |
| **TEST (OOD, 2)** | **nova, tv80s** | **122K** |

**Hash-based net-level split**: 동일 net 이 train/test 에 동시 출현 불가능. pex_v3 의 H1 fix (2026-05-01) 와 일관.

> 사전 v3 작업 (이번 세션의 일부 초기 결과) 은 nova 를 TRAIN 에 잘못 포함시킨 leakage 가 있었음. 시정 후 canonical split 결과만 paper-grade 로 보고 (`r_analytic_v3/reports/PAPER_GRADE_FINAL.md`).

---

## 5. 주요 design decisions / lessons learned

### Decision 1 — Lumped per-net SPEF (per-segment 가 아닌)
- 이유: Voltus / PrimeTime 등 downstream tool 이 lumped 로 충분히 동작
- per-segment 는 더 정확하지만 시스템 복잡도 ↑↑, runtime ↑

### Decision 2 — Cell LEF OBS 활용
- DEF NETS section은 cell-external routing만 제공 → ~30 squares M1/net 누락
- Cell LEF OBS section 에 cell-internal routing 정보 보유
- VCC/VSS pin port 차감 후 signal-internal routing 분리 → R MAPE -0.25pp 개선

### Decision 3 — `.lib` (LIBERTY) 선택적 입력
- intel22 등 일부 PDK 에서는 `.lib` 미제공
- 미보유 시 c_gnd 정확도 ~21% (hand-feature ceiling)
- `.lib` 보유 시 cell intrinsic Cgg 직접 활용 → c_gnd ceiling 돌파 가능 (예측)

### Decision 4 — Production-ready pipeline 우선
- 실제 EDA flow 에서 사용 가능한 SPEF format 출력
- Voltus / PrimeTime / Tempus 등 변경 없이 consume
- 5-7× speedup vs StarRC (medium designs)

### Decision 5 — pex_v3 와의 분리
- pex_v3: per-pattern Phase 1 + Phase 2 aggregator paradigm 개발 (research)
- 본 세션: full-net pipeline + cell LEF OBS + canonical OOD measurement (production + audit)
- 두 세션 합치기 가능 시점: pex_v3 Phase 1 결과 확보 후

---

## 6. Paper composition outline

Top venue (DAC / ICCAD / DATE / TODAES) 제출 가능한 contribution:

### Title (proposed)
"PINN-PEX: Hybrid Analytic-ML Framework for Cross-Design SPEF Generation with Cell-Aware Feature Engineering"

### Abstract claims
1. End-to-end DEF+LEF→SPEF pipeline, StarRC-compatible I/O, 5-7× speedup
2. Cross-design OOD R MAPE **4.00%** (3-4× improvement over ML-only baselines)
3. Cell LEF OBS-aware feature engineering: signal vs power-rail separation
4. Paradigm-independent c_gnd ceiling **~21%** quantified across 5 ML methods (XGBoost, MLP, GBDT, NNLS+GBT, bounded-MLP residual)
5. Hybrid analytic + bounded residual paradigm transfers from per-pattern to full-net (24% c_gnd, -7pp from NNLS+GBT)

### Section structure
1. Introduction — problem statement, contributions
2. Related work — StarRC, ResCap, CNN-Cap, ParaGraph, pex_v3 Phase 1
3. Framework architecture — 7-stage pipeline, data representations
4. Methodology — analytic base + bounded residual cascade
5. Experiments — canonical split, per-design + per-channel + per-stage MAPE
6. Ablation — feature variants (v4 wire-only, v5 OBS raw, v6 signal OBS), clamp values, stage cascading
7. Runtime — tv80s benchmark + StarRC comparison
8. Limitations — c_gnd 21% ceiling (`.lib` integration future work), nova-scale algorithmic optimization
9. Conclusion

---

## 7. 산출물 위치 (cross-session shared)

### Code
- Production pipeline: `experiments/cross_design_tv80s_2026_05_02/scripts/predict_spef_e2e.py`
- Modeling research: `experiments/cross_design_tv80s_2026_05_02/r_analytic_v3/scripts/`
- pex_v3 Phase 1: `pex_v3/src/` (별도 세션, read-only)

### Reports
- Production: `experiments/cross_design_tv80s_2026_05_02/reports/SPEF_*.md`
- v3 research: `experiments/cross_design_tv80s_2026_05_02/r_analytic_v3/reports/`
  - `PAPER_GRADE_FINAL.md` — paper-grade results
  - `CURRENT_WORK_REPORT.md` — session timeline
  - `CGND_RESULTS.md` — c_gnd ceiling analysis
  - `R_ANALYTIC_POLICY_KO.md` — analytic R policy
- pex_v3 strategy: `pex_v3/PHASE_STATUS.md`, `pex_v3/docs/PHASE1_HYBRID_ARCH_SPEC.md`

### Data
- Training cache: `experiments/cross_design_tv80s_2026_05_02/cache/`
- Per-design features: `experiments/cross_design_tv80s_2026_05_02/r_analytic_v3/cache/feat_v*_<design>.parquet`
- Models: `experiments/cross_design_tv80s_2026_05_02/output/spef_e2e/{total_cap, total_r, c_gnd, pair_regressor}/`
- Golden SPEF: `/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22/`

### Memory (cross-session)
- `/home/jslee/.claude/projects/-home-jslee-projects-PINNPEX/memory/MEMORY.md` (index)

---

## 8. Quick reference

### How to generate SPEF for a new design

```bash
cd /home/jslee/projects/PINNPEX/experiments/cross_design_tv80s_2026_05_02
PYTHONPATH=.:/home/jslee/projects/PINNPEX python3 scripts/predict_spef_e2e.py \
    --def_path /path/to/your_design.def \
    --out_spef /path/to/predicted.spef \
    --num_workers 16
# Result: ~5 min for 3K-net design, ~3hr for 100K-net design
```

### How to validate against golden SPEF

```bash
python3 scripts/spef_e2e/validate_e2e.py \
    --predicted_spef /path/to/predicted.spef \
    --golden_spef    /path/to/golden.spef \
    --out_dir /path/to/report
```

### How to retrain

(documented in pex_pipeline/__init__.py and r_analytic_v3/scripts/fit_*.py)

---

_End of PEX_FRAMEWORK.md. 본 문서는 모든 PINN-PEX 세션에서 참조 가능._
