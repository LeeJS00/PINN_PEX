"""
Final L4 LGBM rich-feature evaluation — full anti-overclaim.

Best stack: HybridPexV3MeshInputSubsetClampNorm + per-seed LGBM-residual on (gnd,cpl).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
from scipy import stats
import json

variant_dir = Path('/home/jslee/projects/PINNPEX/pex_v3/output/ablation/HybridPexV3MeshInputSubsetClampNorm')
baseline_dir = Path('/home/jslee/projects/PINNPEX/pex_v3/output/phase1_mesh_5seed')
out_dir = Path('/home/jslee/projects/PINNPEX/pex_v3/experiments/auto_optimize_2026_05_03/outputs/final_hero')
out_dir.mkdir(parents=True, exist_ok=True)


def fit_lgbm_residual(val_pred, val_gold, val_extra, tst_pred, tst_extra, n_estimators=400):
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


def make_extras(df):
    return {
        'fanout': np.log1p(df['fanout'].values),
        'bbox': np.log1p(df['bbox_xy_um2'].values),
        'compact_gnd_log': np.log1p(df['compact_gnd_estimate_fF'].values),
        'is_fanout1': (df['fanout'].values == 1).astype(float),
        'is_nova': (df['design'].values == 'intel22_nova_f3').astype(float),
        'layer_M3': (df['dominant_layer'].values == 'M3').astype(float),
        'cpl_pred_log': np.log1p(df['cpl_pred'].values),
        'gnd_pred_log': np.log1p(df['gnd_pred'].values),
    }


def mape_median(p, g):
    return float(np.median(np.abs(p - g) / np.maximum(np.abs(g), 1e-9)))


def mape_mean(p, g):
    return float(np.mean(np.abs(p - g) / np.maximum(np.abs(g), 1e-9)))


per_seed = []
all_per_net_total = []   # for paired Wilcoxon
all_per_net_baseline = []
final_corrected = []     # save for reuse

for s in range(5):
    val = pd.read_parquet(variant_dir / f'seed{s}/eval_logger_valid.parquet')
    tst = pd.read_parquet(variant_dir / f'seed{s}/eval_logger_test.parquet')
    base_tst = pd.read_parquet(baseline_dir / f'seed{s}/eval_logger_test.parquet')

    vp_g, vg_g = val['gnd_pred'].values, val['gnd_gold'].values
    vp_c, vg_c = val['cpl_pred'].values, val['cpl_gold'].values
    tp_g, tg_g = tst['gnd_pred'].values, tst['gnd_gold'].values
    tp_c, tg_c = tst['cpl_pred'].values, tst['cpl_gold'].values

    val_extra = make_extras(val)
    tst_extra = make_extras(tst)

    g_corr = fit_lgbm_residual(vp_g, vg_g, val_extra, tp_g, tst_extra, n_estimators=400)
    c_corr = fit_lgbm_residual(vp_c, vg_c, val_extra, tp_c, tst_extra, n_estimators=400)

    # Per-seed metrics
    seed_metrics = {
        'seed': s,
        'gnd_median': mape_median(g_corr, tg_g),
        'gnd_mean':   mape_mean(g_corr, tg_g),
        'cpl_median': mape_median(c_corr, tg_c),
        'cpl_mean':   mape_mean(c_corr, tg_c),
        'total_median': mape_median(g_corr + c_corr, tg_g + tg_c),
        'total_mean':   mape_mean(g_corr + c_corr, tg_g + tg_c),
    }

    # Per-design
    for design in tst['design'].unique():
        m = tst['design'].values == design
        suffix = design.split('_')[1]  # nova / tv80s
        seed_metrics[f'{suffix}_total_median'] = mape_median((g_corr + c_corr)[m], (tg_g + tg_c)[m])
        seed_metrics[f'{suffix}_gnd_median']   = mape_median(g_corr[m], tg_g[m])
        seed_metrics[f'{suffix}_cpl_median']   = mape_median(c_corr[m], tg_c[m])

    # Per-fanout
    for fb_name, fb_mask in [('f1', tst['fanout'].values == 1),
                              ('f2-5', (tst['fanout'].values >= 2) & (tst['fanout'].values <= 5)),
                              ('f6-20', (tst['fanout'].values >= 6) & (tst['fanout'].values <= 20)),
                              ('f>20', tst['fanout'].values > 20)]:
        seed_metrics[f'{fb_name}_total_median'] = mape_median((g_corr + c_corr)[fb_mask], (tg_g + tg_c)[fb_mask]) if fb_mask.any() else float('nan')

    # Top-50 outliers (by raw gnd_rel_err)
    rel_raw = np.abs(tp_g - tg_g) / np.maximum(np.abs(tg_g), 1e-9)
    top50_idx = np.argsort(rel_raw)[-50:]
    rel_corr = np.abs(g_corr - tg_g) / np.maximum(np.abs(tg_g), 1e-9)
    seed_metrics['top50_gnd_median_raw'] = float(np.median(rel_raw[top50_idx]))
    seed_metrics['top50_gnd_median_corr'] = float(np.median(rel_corr[top50_idx]))

    per_seed.append(seed_metrics)

    # Per-net errors for paired Wilcoxon (use this seed's variant vs same seed's baseline)
    var_total_err = np.abs((g_corr + c_corr) - (tg_g + tg_c)) / np.maximum(np.abs(tg_g + tg_c), 1e-9)
    base_g, base_c = base_tst['gnd_pred'].values, base_tst['cpl_pred'].values
    base_total_err = np.abs((base_g + base_c) - (base_tst['gnd_gold'].values + base_tst['cpl_gold'].values)) / np.maximum(np.abs(base_tst['gnd_gold'].values + base_tst['cpl_gold'].values), 1e-9)
    all_per_net_total.append(var_total_err)
    all_per_net_baseline.append(base_total_err)

    # Save corrected predictions
    out_seed = out_dir / f'seed{s}'
    out_seed.mkdir(exist_ok=True)
    np.savez(out_seed / 'corrected_predictions.npz',
             gnd_pred=g_corr, cpl_pred=c_corr,
             gnd_gold=tg_g, cpl_gold=tg_c)

# Aggregate
totals = [m['total_median'] for m in per_seed]
gnds = [m['gnd_median'] for m in per_seed]
cpls = [m['cpl_median'] for m in per_seed]

# Baseline numbers (locked)
base_totals = [0.07720, 0.07914, 0.08272, 0.08527, 0.08764]
base_gnds = [0.19916, 0.20491, 0.20491, 0.20491, 0.20802]
base_cpls = [0.15174, 0.15528, 0.15528, 0.15528, 0.15803]

# Cohen's d
def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.sqrt(((len(a)-1)*np.var(a, ddof=1) + (len(b)-1)*np.var(b, ddof=1)) / (len(a)+len(b)-2))
    return float((np.mean(a) - np.mean(b)) / pooled)

d_total = cohens_d(totals, base_totals)
d_gnd = cohens_d(gnds, base_gnds)
d_cpl = cohens_d(cpls, base_cpls)
mwu_total = stats.mannwhitneyu(totals, base_totals, alternative='two-sided').pvalue

# Bootstrap 95% CI
rng = np.random.default_rng(42)
boot_totals = [np.median(rng.choice(totals, size=len(totals), replace=True)) for _ in range(2000)]
ci_lo, ci_hi = np.percentile(boot_totals, [2.5, 97.5])

# Paired Wilcoxon per-net (concatenate all seeds)
all_var = np.concatenate(all_per_net_total)
all_base = np.concatenate(all_per_net_baseline)
wilcox_p = stats.wilcoxon(all_var, all_base, alternative='less').pvalue
median_per_net_delta = float(np.median(all_var - all_base))

print(f"=== L4 LGBM rich features 5-seed FINAL ===")
print(f"Per-seed total: {[f'{x*100:.3f}%' for x in totals]}")
print(f"Median: {np.median(totals)*100:.3f}%")
print(f"Mean:   {np.mean(totals)*100:.3f}%")
print(f"Std:    {np.std(totals, ddof=1)*100:.3f}pp")
print(f"Range:  [{min(totals)*100:.3f}%, {max(totals)*100:.3f}%]")
print()
print(f"Per-seed gnd: {[f'{x*100:.3f}%' for x in gnds]}")
print(f"Median: {np.median(gnds)*100:.3f}% std={np.std(gnds, ddof=1)*100:.3f}pp")
print()
print(f"Per-seed cpl: {[f'{x*100:.3f}%' for x in cpls]}")
print(f"Median: {np.median(cpls)*100:.3f}% std={np.std(cpls, ddof=1)*100:.3f}pp")
print()
print(f"Cohen's d (total) vs baseline: {d_total:+.3f}")
print(f"Cohen's d (gnd):   {d_gnd:+.3f}")
print(f"Cohen's d (cpl):   {d_cpl:+.3f}")
print(f"MWU two-sided p (total): {mwu_total:.4f}")
print(f"Bootstrap 95% CI median total: [{ci_lo*100:.3f}%, {ci_hi*100:.3f}%]")
print(f"Paired per-net Wilcoxon (n={len(all_var)}): p={wilcox_p:.3e}")
print(f"  median per-net delta: {median_per_net_delta*100:+.3f}pp (negative = better)")

print()
print(f"=== Per-design (5-seed median) ===")
for design in ['nova', 'tv80s']:
    tots = [m[f'{design}_total_median'] for m in per_seed]
    gnd  = [m[f'{design}_gnd_median'] for m in per_seed]
    cpl  = [m[f'{design}_cpl_median'] for m in per_seed]
    print(f"  {design:>8}: total={np.median(tots)*100:.3f}% gnd={np.median(gnd)*100:.3f}% cpl={np.median(cpl)*100:.3f}%")

print()
print(f"=== Per-fanout (5-seed median total) ===")
for fb in ['f1', 'f2-5', 'f6-20', 'f>20']:
    vals = [m[f'{fb}_total_median'] for m in per_seed]
    print(f"  fanout {fb:>5}: {np.nanmedian(vals)*100:.3f}%")

print()
print(f"=== Top-50 outliers ===")
print(f"  raw  median: {np.median([m['top50_gnd_median_raw'] for m in per_seed])*100:.1f}%")
print(f"  corr median: {np.median([m['top50_gnd_median_corr'] for m in per_seed])*100:.1f}%")

# Save final report
report = {
    'method': 'HybridPexV3MeshInputSubsetClampNorm + per-seed LGBM residual calibration (8 features)',
    'per_seed': per_seed,
    'aggregate': {
        'total_median': float(np.median(totals)),
        'total_mean': float(np.mean(totals)),
        'total_std': float(np.std(totals, ddof=1)),
        'gnd_median': float(np.median(gnds)),
        'cpl_median': float(np.median(cpls)),
        'cohens_d_total': d_total,
        'mwu_p_total': mwu_total,
        'bootstrap_ci_total': [float(ci_lo), float(ci_hi)],
        'paired_per_net_wilcoxon_p': float(wilcox_p),
        'paired_per_net_median_delta': median_per_net_delta,
    },
}
with open(out_dir / 'final_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=float)
print(f"\nReport saved to: {out_dir / 'final_report.json'}")
