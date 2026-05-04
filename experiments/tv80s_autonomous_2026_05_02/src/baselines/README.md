# pex_v3/src/baselines — Strong baseline implementations

Phase 0.5 deliverable. Owned by `classical-baseline-owner` agent.
Without these, Strategy v3 cannot claim improvement at top-tier review.

## Required baselines (4)

All trained + evaluated on the **same** v3 rebuilt manifest, **same**
clean split, **same** 5-seed protocol. Hyperparameter tuning budget
should be equal across baselines and the paradigm model.

### B1 — `xgboost_baseline.py` — XGBoost / CatBoost on hand features

**Inputs**: per-net hand-engineered physical features from `features.py`
- Geometric: total wire length, area, layer histogram (M1-M9), via count, BBOX
- Coupling-relevant: aggressor count, broadside overlap, lateral overlap, spacing histogram, neighbor density per layer
- Power-net context: VSS/VDD shielding presence, power-net edge density
- Topology: fanout, branch count, longest-path length, tree depth
- Layer-stack: ε(z), etch-stop presence
- Density: local routing density
- Compact-model intermediates: Sakurai-Tamaru per-edge cap (analytic)

**Output**: scalar `C_gnd`, per-aggressor `C_cpl[a]`
**Expected MAPE** (Codex round 2 hypothesis): 7-12%
**Role**: non-trivial non-neural baseline; reviewer's first sanity check

### B2 — `paragraph_baseline.py` — ParaGraph reproduction (DAC 2020, NVIDIA)

**Architecture**: heterogeneous graph (nets ↔ devices), 3-5 GNN layers
(GraphSAGE / GIN / R-GCN), per-net cap regression
**Reference**: arxiv 2007.00514
**Expected MAPE**: 6-10% (Codex hypothesis); ParaGraph reported 30%+ on hard cases
**Role**: published-architecture baseline — reviewers expect comparison

### B3 — `pinn_baseline.py` — Current PINN-PEX wrapper on rebuilt data

**Architecture**: legacy `DeepPEX_Model` (CuboidEncoder + NeuralFluxRouter)
on the v3 rebuilt manifest. No code change to legacy model — just point
it at v3 data.
**Role**: self-baseline, decomposes "what improved by data alone" vs
"what required Phase 1 paradigm".

### B4 — `gam_baseline.py` — Analytic compact-model + GAM/GBDT

**Architecture**: Sakurai-Tamaru per-edge formulas (or our existing
analytic base) + GAM / GBDT residual on per-net features
**Role**: physics-floor anchor; closest paradigm match to ResCap (ASPDAC 2025);
also a candidate Phase 1 architecture initialization

## Implementation order

1. `features.py` — hand-engineered feature extractor (used by B1, B4) — Phase 0.5 first task
2. `xgboost_baseline.py` (B1) — fastest to ship, anchors trees baseline
3. `pinn_baseline.py` (B3) — wraps legacy, easy
4. `gam_baseline.py` (B4) — compact + residual; reuses features.py
5. `paragraph_baseline.py` (B2) — most code; needs heterogeneous graph

## Output convention

```
pex_v3/output/baselines/
├── B1_xgboost/
│   ├── seed{0..4}/
│   │   ├── model.json
│   │   ├── eval_results.csv  (per-net cap MAPE + delay error + power error)
│   │   └── provenance.json
│   └── 5seed_summary.json
├── B2_paragraph/
├── B3_pinn_baseline/
└── B4_gam/
```

5-seed summary includes: median, mean, stdev, IQR, range, MWU vs paradigm,
Cohen's d, bootstrap 95% CI on median.

## Acceptance gate

Phase 0.5 → Phase 1 transition requires all 4 baselines:
- 5-seed reported on rebuilt v3 manifest
- Stratified error report (per-quartile, per-layer, per-design, per-class)
- Paper-grade comparison table (cap MAPE, delay error, power error, RC percentile)
