# Next Session Plan — Total Resistance MAPE < 4%

_Created 2026-05-02 KST. Continuation after `/compact`._

> **2026-05-02 갱신 — 정책 패러다임 전환 (R = analytic, NOT prediction)**
>
> v7 ML 의 R MAPE 11.92% 는 본질적 한계가 아니었다. SPEF *RES 가 segment 별 `length / width / layer` 를 명시적으로 가지고 있고, DEF NETS 의 `VIA{n}_*` 토큰으로 layer 변환을 결정론적으로 셀 수 있다. R 은 분석식의 합:
>
> &nbsp;&nbsp;&nbsp;&nbsp; `R = α × ( Σ sheet_R[layer] × wirelen / width  +  Σ R_via × n_via )`
>
> v2 정책 (PINNPEX `DefStreamParser` 활용, RECT/SPECIALNETS/per-segment width 정확) 의 tv80s test MAPE = **6.99%** ([CI 6.79, 7.19], 47-model ML 11.92% 대비 -4.94pp). v1 (ad-hoc parser) 6.87% 와 overall 은 비슷하지만 stratum 분포가 훨씬 건강 (Q1 short bias -22% → -0.15%). 정식 정책 문서: **`reports/R_ANALYTIC_POLICY_KO.md`**.
>
> 본 plan 의 Step 2-5 (via-count features → DeepSet for R → 15-mdl stratum) 는 **불필요**. <4% 경로 (R_ANALYTIC_POLICY_KO):
> - ✅ Step A (RECT landing) — done in v2
> - ❌ Step B (per-stratum α) — tried, failed (7.68% > 6.99% global α)
> - **Step C (via vc-class refinement)** — Q4 long -5.8% bias 의 정공법, 기대 ~4.5%
> - Step D (pin stub LEF) — Q1 short magnitude 줄임, 기대 ~3.5%

---

## Current state (v7 final)

- **total_R MAPE: 11.92%** (Path A — cached cuboids)
- **total_R MAPE: 18.58%** (Path B — TRUE e2e from raw DEF)
- R²(log): 0.888 (lowest among 4 metrics — most improvement headroom)
- Bias: -4.77% (systematic under-prediction)
- Q1 (short nets) 6.41% / Q4 (long nets) 11.01% — length-dependent error

## Target

**total_R MAPE < 4%** (3x reduction) — challenging but worth trying.

---

## 출발점 (resume after /compact)

### Saved models (already trained)
- `output/spef_e2e/total_r/`:
  - 5 LGBM (`lgbm_seed{0..4}.pkl`)
  - 5 CatBoost (`cat_seed{0..4}.cbm`)
  - stratum_weights.json (24 buckets, 10 models)
- Trained on 9 intel22 train designs + nova val + tv80s test
- Features: 145-dim v3 hand features (`output/spef_e2e/total_cap/fcols.json`)

### Key code references
- `scripts/spef_e2e/train_total_r_v2.py` — current training script
- `scripts/spef_e2e/stratum_generic.py` — stratum fitter
- `pex_pipeline/predict_caps.py:predict_total_r()` — inference path
- `pex_pipeline/compute_resistance.py` — analytic R fallback

### Compare to golden
```bash
python3 scripts/spef_e2e/validate_e2e.py \
  --predicted_spef output/spef_e2e/tv80s_FINAL.spef \
  --golden_spef /home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22/intel22_tv80s_f3_starrc.spef \
  --out_dir reports/spef_e2e_R_baseline \
  --design_name "tv80s R baseline (v7)"
```

---

## Strategy: 4-step plan

### Step 1: Diagnose where R errors come from (~30 min)
- Per-net length-stratified MAPE (already known: Q1 6.4 / Q4 11.0%)
- Per-design analysis: which nets have outliers?
- Compare predicted R to (sheet_R × wirelength) analytic
- Identify dominant error sources: (a) via R, (b) wire length accuracy, (c) sheet R variance

### Step 2: Add via-count features (~1 hour)
**Hypothesis**: Current features lack explicit via R contributions. Each via adds ~3-15 ohm.

Code change in `src/feat_extract_v3.py` (or new feature module):
```python
# Per-net per-(M_i → M_j) via transition count
# A "via" exists where target_layer changes between adjacent cuboids in routing topology
for cuboid in target_cuboids:
    layer = z_to_layer(cuboid_z)
    layer_count[layer] += 1
    if prev_layer != layer and prev_layer != -1:
        transition_count[prev_layer][layer] += 1
        n_total_vias += 1
```

New features (~40 dims):
- `tgt_n_vias_total`: total via count
- `tgt_n_vias_M{i}_to_M{j}`: per-layer-pair transition count (9×9 matrix triangular)
- `tgt_via_layer_diversity`: number of distinct (M_i, M_j) pairs

### Step 3: Retrain R models with new features (~30 min)
- LGBM × 5 + CatBoost × 5 (10 mdl) on 145+40 = 185 features
- Save val + test predictions
- Re-fit stratum_weights.json
- Expected: 11.92% → ~9-10%

### Step 4: Add DeepSet for R (~30 min GPU)
**Hypothesis**: DeepSet sees raw cuboid geometry — can learn via R implicitly from layer transitions in cuboid sequence.

```bash
# Adapt train_deepset_v2.py for total_res target
cp scripts/train_deepset_v2.py scripts/spef_e2e/train_deepset_total_r.py
sed -i 's|y = df\["total_cap_fF"\]|y = df["total_res_label"]|' \
    scripts/spef_e2e/train_deepset_total_r.py
# Adjust: total_res_label needs golden SPEF parsing (see train_total_r_v2.py)
```

Train 5 DeepSet seeds, save .pt weights, add to R ensemble.

### Step 5: 15-mdl stratum + iterate (~30 min)
- Combine 10 LGBM/CatBoost + 5 DeepSet for R
- Stratum sweep b ∈ {4, 8, 12, 16, 24, 40}
- Pick best, save, validate
- Expected: 9-10% → ~7-8%

### Step 6 (optional): Length-stratified specialty models
- Train separate R model for Q4 (long nets, R > 266 Ω)
- Q4 currently 11.01% MAPE (worst quartile)
- If Q4 specialty drops to ~7%, overall MAPE drops by ~1pp

---

## Realistic expectation

- v7 baseline: 11.92%
- With via features: ~9-10%
- With DeepSet: ~7-8%
- With Q4 specialty: ~6-7%
- **Realistic short-term ceiling: 5-7% MAPE**
- **<4% MAPE requires either**:
  - Synthetic data pretraining (analytic R augmentation)
  - In-design fine-tuning (some test nets in train)
  - Topology-aware multi-segment R network reconstruction

---

## Key files for resume

```
experiments/cross_design_tv80s_2026_05_02/
├── output/spef_e2e/
│   ├── total_r/                  ← R models live here
│   ├── total_cap/fcols.json      ← shared 145-dim features
│   └── tv80s_FINAL.spef          ← v7 canonical (37 MB)
├── pex_pipeline/
│   ├── predict_caps.py           ← predict_total_r() entry
│   └── compute_resistance.py     ← analytic fallback
├── scripts/spef_e2e/
│   ├── train_total_r_v2.py       ← current R trainer
│   ├── stratum_generic.py        ← stratum fitter
│   └── validate_e2e.py           ← R MAPE validation
├── src/feat_extract_v3.py        ← features V3 (no via count yet)
└── reports/
    ├── SPEF_E2E_SESSION_FULL_KO.md  ← v7 session summary
    └── NEXT_SESSION_TOTAL_R_PLAN.md ← this plan
```

---

## Resume checklist

After /compact:

1. [x] Read `reports/SPEF_E2E_SESSION_FULL_KO.md` for session context
2. [x] Read `reports/NEXT_SESSION_TOTAL_R_PLAN.md` (this file) for plan
3. [x] Verify v7 baseline R MAPE: run validate_e2e.py on tv80s_FINAL.spef → **11.925% confirmed** (CI [11.44, 12.41], bias -4.77%, R²(log) 0.888) → `reports/spef_e2e_R_baseline/`
4. [x] Step 1: diagnose R error sources → `reports/spef_e2e_R_diag/` + `scripts/spef_e2e/diag_total_r.py`. **Findings strongly support via-R hypothesis** (see below).
5. [ ] Step 2: implement per-via features in `src/feat_extract_v3_with_via.py`
6. [ ] Step 3: retrain LGBM+CatBoost R models
7. [ ] Step 4: train 5 DeepSet seeds for R
8. [ ] Step 5: 15-mdl stratum + finalize
9. [ ] Compare to v7 baseline, lock if improvement
10. [x] Update `SPEF_FLOW_PERFORMANCE_KO.md` with R-focused improvement (Step 1 diagnostic incorporated)

---

## Step 1 diagnostic findings (2026-05-02)

`scripts/spef_e2e/diag_total_r.py` 결과 (artifacts: `reports/spef_e2e_R_diag/`):

**Length-stratified (quartiles by wirelength)**:
- Q1 short: MAPE 8.6%, bias **+2.7%** (over-prediction)
- Q2: 7.2%, bias -1.1%
- Q3: 13.3%, bias -7.5%
- Q4 long (~17.5 μm median): MAPE **14.7%**, bias **-8.6%** (under-prediction)

**Layer-count (proxy for via count)** — *signature finding*:
| n_layers | n | MAPE | bias |
|---|---|---|---|
| 2 | 953 | 11.1% | +0.7% |
| 3 | 1515 | 9.3% | -4.2% |
| 4 | 469 | **14.8%** | **-8.9%** |
| 5 | 232 | 13.5% | -6.8% |

Layer-count 과 negative bias 의 **monotone 증가** (2→4층: +0.7% → -8.9%) 는 누락된 via R 이 만들어내는 정확한 패턴. 4층 이상 nets 에서 R 이 시스템적으로 부족.

**Pure analytic ceiling**: sheet_R × wirelen / width (단일 calibration scale, no model) → 39.2% MAPE. v7 ensemble (10.96% on 3169 joined nets) 이 28pp 를 회복했고, 남은 ~11pp 가 via/topology 가 설명할 영역.

**Top-20 outliers**: 19/20 이 -75 ~ -99% under-predicted. R_gold 140-777Ω 인데 R_pred 3-190Ω 으로 10-100× 부족. 짧은 net 에 via 많은 케이스 (metal stack jump) — via R 의 직접적 영향.

**결론**: Step 2 (via-count features) 진행 합당. 가설이 데이터로 강력하게 지지됨.
