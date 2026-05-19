#!/usr/bin/env python3
"""
Phase 2 — Per-PDK feature importance comparison.

For each PDK, load all 5 seeds × {gnd, cpl} = 10 XGBoost regressors and average
'gain' importance. Compare per-feature importance rank + magnitude between PDKs.
"""
import os, json, glob
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = "/home/jslee/projects/PINNPEX"
OUT  = "/data/PINNPEX/scratch/cross_pdk_analysis"

with open(f"{ROOT}/TreePEX/models/FEATURE_ORDER.txt") as f:
    FEATS = [l.strip() for l in f if l.strip()]

def load_importances(model_dir):
    """Average 'gain' importance over 5 seeds × {gnd, cpl}."""
    out = {f: 0.0 for f in FEATS}
    n = 0
    for ch in ['gnd', 'cpl']:
        for s in range(5):
            p = f"{model_dir}/tweedie_{ch}_seed{s}.json"
            if not os.path.exists(p):
                print(f"  missing {p}")
                continue
            booster = xgb.Booster()
            booster.load_model(p)
            # xgboost reports importance by f0/f1/... index; we need to map to FEATS.
            imp = booster.get_score(importance_type='gain')
            # imp keys are like 'f0', 'f1' ... or feature names if saved with names
            for k, v in imp.items():
                if k.startswith('f') and k[1:].isdigit():
                    idx = int(k[1:])
                    if idx < len(FEATS):
                        out[FEATS[idx]] += float(v)
                elif k in out:
                    out[k] += float(v)
            n += 1
    # average
    if n > 0:
        for f in out:
            out[f] /= n
    return out, n

print("[intel22] loading 5-seed × 2-channel models ...")
imp_i, n_i = load_importances(f"{ROOT}/TreePEX/models")
print(f"  loaded {n_i} models")

print("[asap7] loading 5-seed × 2-channel models ...")
imp_a, n_a = load_importances(f"{ROOT}/TreePEX/models_asap7")
print(f"  loaded {n_a} models")

df = pd.DataFrame({
    'feature': FEATS,
    'intel_gain': [imp_i[f] for f in FEATS],
    'asap_gain':  [imp_a[f] for f in FEATS],
})
# normalize each PDK so total=1 (rank by relative importance)
df['intel_norm'] = df['intel_gain'] / df['intel_gain'].sum()
df['asap_norm']  = df['asap_gain']  / df['asap_gain'].sum()
df['intel_rank'] = df['intel_norm'].rank(ascending=False).astype(int)
df['asap_rank']  = df['asap_norm'].rank(ascending=False).astype(int)
df['rank_diff']  = df['intel_rank'] - df['asap_rank']
df['abs_norm_diff'] = (df['intel_norm'] - df['asap_norm']).abs()

df = df.sort_values('abs_norm_diff', ascending=False).reset_index(drop=True)
df.to_csv(f"{OUT}/feature_importance_compare.csv", index=False)
print(f"\nwrote {OUT}/feature_importance_compare.csv")

# Rank correlation
from scipy.stats import spearmanr, pearsonr
rho_s, p_s = spearmanr(df['intel_rank'], df['asap_rank'])
rho_p, p_p = pearsonr(df['intel_norm'], df['asap_norm'])
print(f"\nSpearman rank corr: rho={rho_s:.4f} p={p_s:.3e}")
print(f"Pearson  norm corr: rho={rho_p:.4f} p={p_p:.3e}")

print("\n=== Top-15 most-divergent features (|Δ norm-importance|) ===")
print(df.head(15)[['feature','intel_norm','asap_norm','intel_rank','asap_rank','rank_diff']]
      .to_string(index=False))

print("\n=== intel22 top-10 ===")
print(df.sort_values('intel_norm', ascending=False).head(10)
      [['feature','intel_norm','asap_norm','rank_diff']].to_string(index=False))

print("\n=== asap7 top-10 ===")
print(df.sort_values('asap_norm', ascending=False).head(10)
      [['feature','intel_norm','asap_norm','rank_diff']].to_string(index=False))

# How concentrated is each PDK's importance?
print("\n=== concentration ===")
i_sorted = df['intel_norm'].sort_values(ascending=False).values
a_sorted = df['asap_norm'].sort_values(ascending=False).values
for k in [3, 5, 10, 20]:
    print(f"  top-{k} sum: intel={i_sorted[:k].sum():.3f}  asap={a_sorted[:k].sum():.3f}")
