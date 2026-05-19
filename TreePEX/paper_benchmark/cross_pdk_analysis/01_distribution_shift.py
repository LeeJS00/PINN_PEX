#!/usr/bin/env python3
"""
Phase 1 — Cross-PDK feature distribution shift quantification.

For each of the 67 canonical features (41 V3 base + 26 V4 H3), compute on the
TRAIN split of each PDK:
- mean, std, median
- KS statistic + p-value (intel22-train vs ASAP7-train)
- relative-mean ratio (ASAP7/intel22)
- normalized Wasserstein-1 distance (W1 on z-scored)

Output: distribution_shift.csv ranked by KS desc.
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/home/jslee/projects/PINNPEX"
OUT  = "/data/PINNPEX/scratch/cross_pdk_analysis"
os.makedirs(OUT, exist_ok=True)

V3_INTEL = "/data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv"
V3_ASAP  = "/data/PINNPEX/data/processed_v3/asap7/features/all_designs.csv"
H3_INTEL = f"{ROOT}/archive/pex_v4/results/new_features_with_ids.csv"
H3_ASAP  = f"{ROOT}/TreePEX/inputs/asap7_new_features_with_ids.csv"

with open(f"{ROOT}/TreePEX/models/FEATURE_ORDER.txt") as f:
    FEATS = [l.strip() for l in f if l.strip()]
print(f"[feats] loaded {len(FEATS)} features from FEATURE_ORDER.txt")

print(f"[load] intel22 V3 ...")
iv3 = pd.read_csv(V3_INTEL)
print(f"[load] asap7 V3 ...")
av3 = pd.read_csv(V3_ASAP)
print(f"[load] intel22 H3 ...")
ih3 = pd.read_csv(H3_INTEL)
print(f"[load] asap7 H3 ...")
ah3 = pd.read_csv(H3_ASAP)

print(f"intel22 V3 rows={len(iv3)} H3 rows={len(ih3)}")
print(f"asap7   V3 rows={len(av3)} H3 rows={len(ah3)}")

# Merge V3 + H3 by (design_name, net_name)
def merge(v3, h3):
    h3_cols = [c for c in h3.columns if c in FEATS] + ['design_name', 'net_name']
    return v3.merge(h3[h3_cols], on=['design_name','net_name'], how='left',
                    suffixes=('', '_h3'))

mi = merge(iv3, ih3)
ma = merge(av3, ah3)
print(f"merged: intel22={len(mi)} asap7={len(ma)}")

# Use TRAIN split only for distribution comparison (avoid test contamination)
mi_tr = mi[mi.split == 'train'].copy()
ma_tr = ma[ma.split == 'train'].copy()
print(f"TRAIN: intel22={len(mi_tr)} asap7={len(ma_tr)}")

# Subsample to equal size for KS fairness (KS sensitive to N)
N = min(len(mi_tr), len(ma_tr), 100_000)
rng = np.random.default_rng(42)
mi_s = mi_tr.iloc[rng.choice(len(mi_tr), N, replace=False)].reset_index(drop=True)
ma_s = ma_tr.iloc[rng.choice(len(ma_tr), N, replace=False)].reset_index(drop=True)
print(f"subsampled (equal-N={N}) for KS")

rows = []
for f in FEATS:
    if f not in mi_s.columns or f not in ma_s.columns:
        print(f"  SKIP {f} (missing)")
        continue
    xi = mi_s[f].astype(float).fillna(0.0).values
    xa = ma_s[f].astype(float).fillna(0.0).values
    # Stats
    mu_i, sd_i, md_i = float(xi.mean()), float(xi.std()), float(np.median(xi))
    mu_a, sd_a, md_a = float(xa.mean()), float(xa.std()), float(np.median(xa))
    rmean = (mu_a / mu_i) if abs(mu_i) > 1e-12 else float('nan')
    # KS
    ks, p = stats.ks_2samp(xi, xa, mode='asymp')
    # Wasserstein on z-scored using POOLED std (robust to scale)
    pooled = float(np.std(np.concatenate([xi, xa])) + 1e-12)
    w1 = stats.wasserstein_distance(xi / pooled, xa / pooled)
    rows.append(dict(
        feature=f,
        intel_mean=mu_i, intel_std=sd_i, intel_med=md_i,
        asap_mean=mu_a, asap_std=sd_a, asap_med=md_a,
        mean_ratio_a_over_i=rmean,
        ks=float(ks), ks_p=float(p),
        w1_zscored=float(w1),
    ))

df = pd.DataFrame(rows).sort_values('ks', ascending=False).reset_index(drop=True)
df.to_csv(f"{OUT}/distribution_shift.csv", index=False)
print(f"\nwrote {OUT}/distribution_shift.csv ({len(df)} features)")

print("\nTop-20 most-shifted features (by KS):")
print(df[['feature','ks','ks_p','mean_ratio_a_over_i','w1_zscored',
         'intel_mean','asap_mean']].head(20).to_string(index=False))

print("\nBottom-10 most-stable features (by KS):")
print(df[['feature','ks','mean_ratio_a_over_i','intel_mean','asap_mean']]
      .tail(10).to_string(index=False))

# Summary
print("\n===== summary =====")
print(f"  median KS:  {df['ks'].median():.3f}")
print(f"  mean   KS:  {df['ks'].mean():.3f}")
print(f"  # KS>0.3:   {(df['ks']>0.3).sum()} / {len(df)}")
print(f"  # KS>0.1:   {(df['ks']>0.1).sum()} / {len(df)}")
print(f"  # KS<0.05:  {(df['ks']<0.05).sum()} / {len(df)}")
