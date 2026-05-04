"""
Round 3 calibration experiments on Combined (InputSubset+ClampNorm) 5-seed:
  L1   CPL isotonic refit (parallel to gnd C1)
  L4'  LightGBM residual calibration (gnd, cpl) - more flexible than 1D iso
  L6   Stack: gnd_iso + cpl_iso (best from initial sweep)
  L4+6 Stack: gnd_lgbm + cpl_lgbm
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

variant_dir = Path('/home/jslee/projects/PINNPEX/pex_v3/output/ablation/HybridPexV3MeshInputSubsetClampNorm')


def fit_lgbm_residual(val_pred, val_gold, val_extra, tst_pred, tst_extra, n_estimators=200):
    msk = (val_pred > 0) & (val_gold > 0)
    X_val = np.column_stack([np.log(val_pred[msk] + 1e-9)] + [val_extra[c][msk] for c in val_extra])
    y_val = np.log(val_gold[msk] / np.maximum(val_pred[msk], 1e-9))
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators, learning_rate=0.03, max_depth=4,
        min_child_samples=80, reg_alpha=0.1, reg_lambda=0.1,
        random_state=0, verbose=-1, n_jobs=4, force_col_wise=True
    )
    model.fit(X_val, y_val)
    pos = tst_pred > 0
    X_tst = np.column_stack([np.log(tst_pred + 1e-9)] + [tst_extra[c] for c in tst_extra])
    correction = model.predict(X_tst)
    out = np.copy(tst_pred)
    out[pos] = tst_pred[pos] * np.exp(correction[pos])
    return out


def fit_apply_iso(val_pred, val_gold, tst_pred):
    msk = (val_pred > 0) & (val_gold > 0)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(np.log(val_pred[msk]), np.log(val_gold[msk]))
    pos = tst_pred > 0
    out = np.copy(tst_pred)
    out[pos] = np.exp(iso.predict(np.log(tst_pred[pos])))
    return out


def mape_median(p, g):
    return np.median(np.abs(p - g) / np.maximum(np.abs(g), 1e-9))


results = {}
for s in range(5):
    val = pd.read_parquet(variant_dir / f'seed{s}/eval_logger_valid.parquet')
    tst = pd.read_parquet(variant_dir / f'seed{s}/eval_logger_test.parquet')

    vp_g, vg_g = val['gnd_pred'].values, val['gnd_gold'].values
    vp_c, vg_c = val['cpl_pred'].values, val['cpl_gold'].values
    tp_g, tg_g = tst['gnd_pred'].values, tst['gnd_gold'].values
    tp_c, tg_c = tst['cpl_pred'].values, tst['cpl_gold'].values

    val_extra = {
        'fanout': np.log1p(val['fanout'].values),
        'bbox': np.log1p(val['bbox_xy_um2'].values),
        'compact_gnd_log': np.log1p(val['compact_gnd_estimate_fF'].values),
        'is_fanout1': (val['fanout'].values == 1).astype(float),
        'is_nova': (val['design'].values == 'intel22_nova_f3').astype(float),
        'layer_M3': (val['dominant_layer'].values == 'M3').astype(float),
        'cpl_pred_log': np.log1p(val['cpl_pred'].values),  # cross-channel feature
        'gnd_pred_log': np.log1p(val['gnd_pred'].values),  # cross-channel feature
    }
    tst_extra = {
        'fanout': np.log1p(tst['fanout'].values),
        'bbox': np.log1p(tst['bbox_xy_um2'].values),
        'compact_gnd_log': np.log1p(tst['compact_gnd_estimate_fF'].values),
        'is_fanout1': (tst['fanout'].values == 1).astype(float),
        'is_nova': (tst['design'].values == 'intel22_nova_f3').astype(float),
        'layer_M3': (tst['dominant_layer'].values == 'M3').astype(float),
        'cpl_pred_log': np.log1p(tst['cpl_pred'].values),
        'gnd_pred_log': np.log1p(tst['gnd_pred'].values),
    }

    g_lgbm = fit_lgbm_residual(vp_g, vg_g, val_extra, tp_g, tst_extra, n_estimators=400)
    c_lgbm = fit_lgbm_residual(vp_c, vg_c, val_extra, tp_c, tst_extra, n_estimators=400)
    g_iso = fit_apply_iso(vp_g, vg_g, tp_g)
    c_iso = fit_apply_iso(vp_c, vg_c, tp_c)

    variants = {
        'raw': (tp_g, tp_c),
        'gnd+cpl_iso (L6)': (g_iso, c_iso),
        'gnd_lgbm + cpl_iso': (g_lgbm, c_iso),
        'gnd_iso + cpl_lgbm': (g_iso, c_lgbm),
        'gnd_lgbm + cpl_lgbm (L4)': (g_lgbm, c_lgbm),
    }

    rel_raw = np.abs(tp_g - tg_g) / np.maximum(np.abs(tg_g), 1e-9)
    top50_idx = np.argsort(rel_raw)[-50:]

    for name, (gp, cp) in variants.items():
        rel_corr = np.abs(gp - tg_g) / np.maximum(np.abs(tg_g), 1e-9)
        results.setdefault(name, {'g': [], 'c': [], 't': [], 'p50': []})
        results[name]['g'].append(mape_median(gp, tg_g))
        results[name]['c'].append(mape_median(cp, tg_c))
        results[name]['t'].append(mape_median(gp + cp, tg_g + tg_c))
        results[name]['p50'].append(np.median(rel_corr[top50_idx]))

print(f"{'variant':<32} {'gnd':>8} {'cpl':>8} {'total':>8} {'top50':>8}")
print('-' * 72)
for name, vals in results.items():
    g = np.median(vals['g']) * 100
    c = np.median(vals['c']) * 100
    t = np.median(vals['t']) * 100
    p = np.median(vals['p50']) * 100
    print(f"{name:<32} {g:7.3f}% {c:7.3f}% {t:7.3f}% {p:7.1f}%")

print()
print('5-seed std for total:')
for name, vals in results.items():
    print(f"  {name:<30} std={np.std(vals['t'])*100:.3f}pp range=[{min(vals['t'])*100:.3f}, {max(vals['t'])*100:.3f}]")
