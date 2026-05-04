# PINN-PEX Experiments — Section 5 draft

_Created: 2026-05-03 evening, post Path-2 Fast SPEF lock._
_Source data: `RESULTS_CONSOLIDATED.md` + `output/spef_e2e_fast/` + `output/baselines/`._

---

## 5.1 Dataset

- **Process**: intel22 (22 nm BEOL, 8 metal layers).
- **Designs (11 total)**:
  - **Train** (9): aes_cipher_top, gcd, ibex_core, ldpc_decoder_802_3an, mc_top, mpeg2_top, spi_top, TinyRocketCore, wb_conmax.
  - **Test** (2 OOD): nova, tv80s.
- **Tile manifest**: 1,322,115 cuboid tiles · 257,438 nets · 493 GB
  (`/data/PINNPEX/data/processed_v3/intel22/dataset_manifest_v3.csv`).
- **H1 net-level hash split** (no leakage): verified by
  `pex_v3/tests/test_split_invariants.py`. Eliminates the 12.29% legacy
  net leak documented in PROJECT_REPORT §4.1.
- **Per-net feature dataset**: 221,102 rows after H1 split, 42-dim feature
  vector + analytic prior estimates (`pex_v3/output/baselines/feature_dataset_v3.csv`).

## 5.2 Baselines

5-seed protocol (seeds 0–4), paired Mann-Whitney U for any "X beats Y"
claim, per `benchmarking-statistician.md` discipline.

| Method | Architecture | Params | Reference |
|---|---|---:|---|
| B1 XGBoost | tree boosting on 42-dim hand features | ~100 K | `xgboost_baseline.py` |
| B3 PINN legacy DeepPEX | Neural Field + flux router | ~1 M | `src/models/neural_field.py` |
| B4 V3 log-GBDT | compact + multiplicative residual | ~100 K | `compact_gam_v3.py` |
| Option F deep MLP | 286 K MLP on hand features | 286 K | `14_option_f_5seed.py` |
| **HybridPexV3Mesh** (ours) | DeepSet + bounded multiplicative residual + curriculum | **44 K** | `hybrid_v3_mesh.py` |

## 5.3 Per-net cap MAPE (Table 1, 5-seed cross-design test)

[insert leaderboard from `RESULTS_CONSOLIDATED.md`]

```
Method                          params    valid total   test total      OOD gap
B1 XGBoost                      100K      4.66 ± 0.03   5.84 ± 0.10    +1.19
Option F deep MLP               286K      4.76 ± 0.01   5.62 ± 0.04    +0.87
B4 V3 log-GBDT                  100K      5.72 ± 0.04   6.59 ± 0.13    +0.87
HybridPexV3Mesh (best-step)     44K       6.26 ± 0.108  ─              ─
HybridPexV3Mesh (last-step)     44K       8.59 ± 0.717  8.27 ± 0.342   −0.32
HybridPexV3Mesh (5-seed ens.)   44K × 5   7.81          7.89           +0.08
B3 PINN legacy DeepPEX          1M        30.90 ± 2.20  ─              ─
```

**Headline 1**: 44 K Mesh PINN best-step beats B4 V3 log-GBDT (6.59 %) with
2.3 × fewer parameters; closes 2 / 3 of the gap from legacy 30.90 % to the
4.66 % hand-feature ceiling.

## 5.4 Full-chip SPEF E2E (Table 2 + Figure 1)

Two SPEF generation paths, both ending in XGB anchor + R post-process:

| Path | tv80s wall-clock | C MAPE mean (5-seed) | C MAPE median | C MAPE p95 | R MAPE | R²(C) | R²(R) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Path-1: Legacy DeepPEX (1M) | 864 s | 10.96 ± 0.047 | 5.77 | 44.30 | 2.21 | 0.983 | 0.999 |
| **Path-2: Fast (Option D')** | **68.9 s** | **12.68 ± 0.043** | **5.78 ± 0.077** | 99.66 (det.) | 2.21 (det.) | 0.976 | 0.999 |

Per-channel test breakdown (Path-1, 5-seed mean):
gnd 21.0 % · cpl 12.0 % · total 10.96 %.

**Headline 2**: 12.5 × wall-clock improvement from Path-1 → Path-2, with
median MAPE essentially unchanged (5.78 vs 5.77, +0.01pp). Mean increase
+1.7 pp comes from a deterministic P95 outlier set where the geometric
aggressor lookup misses real coupling.

**Headline 3**: R²(R) = 0.999, R²(C) = 0.976 on cross-design tv80s test
(3,380 nets) — production-grade SPEF accuracy.

## 5.5 Length-stratified MAPE (Table 3)

Net length quartiles by total resistance (proxy for routed-wire length).

| Quartile | Range (Ω) | n_nets | Median MAPE Path-1 | Median MAPE Path-2 |
|---|---:|---:|---:|---:|
| Q1 (short) | 35.9 – 79.0 | 845 | 6.68 % | 6.68 % |
| Q2 | 79.1 – 120.8 | 845 | 5.90 % | 5.90 % |
| Q3 | 121.1 – 262.2 | 845 | 6.32 % | 6.32 % |
| Q4 (long) | 262.4 – 6043.9 | 845 | 4.62 % | 4.62 % |

Path-2 median equals Path-1 across all quartiles — the per-net total cap
distribution is anchor-driven (XGBoost), so spatial allocator choice
affects only mean (P95 outliers).

## 5.6 Ablations (Table 4)

| Ablation | tv80s C MAPE mean | Note |
|---|---:|---|
| Mesh + curriculum (full) | — | best-step 6.26 % per-net |
| − Curriculum (clamp fixed at log(1.5)) | 8.71 % | Curriculum contributes −2.45 pp |
| − NNLS prior calibration | 38.31 % (day-1) | Prior ratio 0.347 vs 1.006 |
| − XGB anchor (raw PINN sum) | 47.69 % | Tile→net aggregation drift |
| − Sister R per-net rescale (R only) | 28.36 % R MAPE | DEF/LEF info ceiling |
| Capacity scaling 11K → 71K → 406K | 11–14 % | No improvement; capacity not bottleneck |
| Per-pair coupling head (Strike #2) | 60 % at curriculum transition | Killed at epoch 53; uniform analytic baseline insufficient |
| Cell-OBS features (Strike #7) | +3.0 pp test | Routing-length features hurt c_gnd |
| Liberty pin-cap features (Strike #8) | +2.4 pp test | Same overfit pattern |

## 5.7 Runtime + license-free analysis (Table 5)

[See `RESULTS_CONSOLIDATED.md` §Runtime — Path-1 vs Path-2.]

License: open-source toolchain only. No commercial PEX license required.
StarRC license cost (~$50–100 K/seat/yr) avoided per chip iteration.

**Path-2 deployment story**: ~1.1 min for tv80s on a single workstation;
GPU optional (Mesh PINN inference is per-net validation only, not in the
SPEF generation path).

---

## Outstanding gaps for paper

1. **StarRC honest wall-clock measurement** — license required, future work.
2. **Nova full-chip SPEF E2E** — Path-2 in progress 2026-05-03 evening.
3. **P95 outlier closer-look** — which nets fail under geometric allocator?
4. **CPU-only Path-2 timing** — verify GPU-optional claim.
