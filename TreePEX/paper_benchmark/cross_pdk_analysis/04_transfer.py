#!/usr/bin/env python3
"""
Phase 4 — Cross-PDK transfer experiment.

Quantify: applying intel22-trained 5-seed ensemble to ASAP7 features (and vice
versa), without retraining. Reports MAPE per target design and per channel.

This is the RAW canonical inference path: no L5 calibration, no L11 specialist,
no fanout proxy — we use the gold fanout that's already in the feature CSV,
because we want to isolate "feature/label distribution mismatch", not
proxy-quality issues.
"""
import os, json
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = "/home/jslee/projects/PINNPEX"
OUT  = "/data/PINNPEX/scratch/cross_pdk_analysis"

V3_I = "/data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv"
V3_A = "/data/PINNPEX/data/processed_v3/asap7/features/all_designs.csv"
H3_I = f"{ROOT}/archive/pex_v4/results/new_features_with_ids.csv"
H3_A = f"{ROOT}/TreePEX/inputs/asap7_new_features_with_ids.csv"

MODELS_I = f"{ROOT}/TreePEX/models"
MODELS_A = f"{ROOT}/TreePEX/models_asap7"

with open(f"{MODELS_I}/FEATURE_ORDER.txt") as f:
    FEATS = [l.strip() for l in f if l.strip()]
print(f"FEATS: {len(FEATS)}")

# Designs to evaluate
TARGETS_I = ['intel22_tv80s_f3', 'intel22_nova_f3']
TARGETS_A = ['asap7_tv80s_x1']  # asap7 nova not in all_designs.csv (no train entry)

def merge_feats(v3, h3):
    h3_cols = [c for c in h3.columns if c in FEATS] + ['design_name', 'net_name']
    return v3.merge(h3[h3_cols], on=['design_name','net_name'], how='left')

print("[load] intel22 ..."); i_full = merge_feats(pd.read_csv(V3_I), pd.read_csv(H3_I))
print("[load] asap7 ...");   a_full = merge_feats(pd.read_csv(V3_A), pd.read_csv(H3_A))

def load_seeds(model_dir):
    """Load 5-seed × 2-channel XGBoost regressors."""
    out = {}
    for s in range(5):
        for ch in ['gnd','cpl']:
            p = f"{model_dir}/tweedie_{ch}_seed{s}.json"
            if not os.path.exists(p):
                continue
            m = xgb.XGBRegressor(); m.load_model(p)
            out[(s, ch)] = m
    return out

models_i = load_seeds(MODELS_I)
models_a = load_seeds(MODELS_A)
print(f"intel22 models: {len(models_i)} / 10")
print(f"asap7   models: {len(models_a)} / 10")

def predict_5seed(models, X):
    """5-seed prediction-mean per channel."""
    preds_g = [m.predict(X) for (s, ch), m in models.items() if ch == 'gnd']
    preds_c = [m.predict(X) for (s, ch), m in models.items() if ch == 'cpl']
    return np.mean(preds_g, axis=0), np.mean(preds_c, axis=0)

def mape_med(y_true, y_pred, eps=1e-12):
    e = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), eps)
    return float(np.median(e) * 100), float(np.mean(e) * 100)

def r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)

def evaluate(full_df, target_design, models, label):
    sub = full_df[full_df.design_name == target_design].copy()
    sub = sub[sub.split == 'test']  # use TEST split only
    if len(sub) == 0:
        print(f"  WARN: no test rows for {target_design}")
        return None
    # Ensure all FEATS exist; fill missing
    for f in FEATS:
        if f not in sub.columns:
            sub[f] = 0.0
    X = sub[FEATS].fillna(0.0).values
    g_pred, c_pred = predict_5seed(models, X)
    tot_pred = g_pred + c_pred
    g_true = sub['c_gnd_fF'].astype(float).values
    c_true = sub['c_cpl_total_fF'].astype(float).values
    tot_true = sub['total_cap_fF'].astype(float).values
    # MAPE
    g_med, g_mean = mape_med(g_true, g_pred)
    c_med, c_mean = mape_med(c_true, c_pred)
    t_med, t_mean = mape_med(tot_true, tot_pred)
    rt = r2(tot_true, tot_pred)
    print(f"  [{label}] n={len(sub):>6}  "
          f"tot MAPE_med={t_med:6.2f}% MAPE_mean={t_mean:6.2f}% R²={rt:.4f}  "
          f"| gnd_med={g_med:6.2f}% cpl_med={c_med:6.2f}%")
    return dict(
        target=target_design, source_model=label, n=len(sub),
        tot_mape_med=t_med, tot_mape_mean=t_mean, tot_r2=rt,
        gnd_mape_med=g_med, cpl_mape_med=c_med,
    )

rows = []
print("\n=== Target: intel22_tv80s_f3 ===")
for d in TARGETS_I:
    print(f"\n  -- {d} --")
    rows.append(evaluate(i_full, d, models_i, "intel22-model (same-PDK)"))
    rows.append(evaluate(i_full, d, models_a, "asap7-model (cross-PDK)"))

print("\n=== Target: asap7_tv80s_x1 ===")
for d in TARGETS_A:
    print(f"\n  -- {d} --")
    rows.append(evaluate(a_full, d, models_a, "asap7-model (same-PDK)"))
    rows.append(evaluate(a_full, d, models_i, "intel22-model (cross-PDK)"))

# Save
df = pd.DataFrame([r for r in rows if r is not None])
df.to_csv(f"{OUT}/transfer_matrix.csv", index=False)
print(f"\nwrote {OUT}/transfer_matrix.csv")

print("\n=== Summary table ===")
print(df.to_string(index=False))
