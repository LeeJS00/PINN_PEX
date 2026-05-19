#!/usr/bin/env python3
"""
Phase 3 — Label distribution / target shift.

Compare per-PDK distribution of (total_cap_fF, c_gnd_fF, c_cpl_total_fF) on
TRAIN + per-design (test). Show magnitude scale, dynamic range, ratio gnd/cpl.
"""
import numpy as np
import pandas as pd
from scipy import stats

V3_I = "/data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv"
V3_A = "/data/PINNPEX/data/processed_v3/asap7/features/all_designs.csv"
OUT  = "/data/PINNPEX/scratch/cross_pdk_analysis"

cols = ['design_name','split','total_cap_fF','c_gnd_fF','c_cpl_total_fF',
        'total_wire_length_um','fanout']
i = pd.read_csv(V3_I, usecols=cols)
a = pd.read_csv(V3_A, usecols=cols)
print(f"intel22={len(i)}  asap7={len(a)}")

def summary(df, label):
    rows = []
    if len(df) == 0:
        print(f"  WARN: {label} has 0 rows — skipping")
        return rows
    for ch in ['total_cap_fF','c_gnd_fF','c_cpl_total_fF']:
        x = df[ch].astype(float).values
        x = x[x > 0]
        if len(x) == 0:
            continue
        rows.append(dict(
            label=label, channel=ch,
            n=len(x), mean=x.mean(), std=x.std(),
            p10=np.percentile(x, 10), p50=np.median(x),
            p90=np.percentile(x, 90), p99=np.percentile(x, 99),
            max=x.max(),
        ))
    return rows

rows = []
rows += summary(i[i.split=='train'], 'intel22_train')
rows += summary(a[a.split=='train'], 'asap7_train')

# Per-design (held-out designs that matter for paper)
for d in ['intel22_tv80s_f3','intel22_nova_f3']:
    rows += summary(i[i.design_name==d], d)
for d in ['asap7_tv80s_x1','asap7_nova_x1']:
    rows += summary(a[a.design_name==d], d)

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/label_distribution.csv", index=False)
print(f"\nwrote {OUT}/label_distribution.csv")

# Pretty print
print("\n=== Per-label distribution summary (fF) ===")
for label in df['label'].unique():
    sub = df[df.label==label]
    print(f"\n--- {label} ---")
    print(sub[['channel','n','mean','p50','p90','p99','max']].to_string(index=False))

# Magnitude shift = asap_p50 / intel_p50 by channel
print("\n=== Magnitude shift (p50, ASAP7 / intel22 on train) ===")
for ch in ['total_cap_fF','c_gnd_fF','c_cpl_total_fF']:
    ip = df[(df.label=='intel22_train') & (df.channel==ch)].p50.iloc[0]
    ap = df[(df.label=='asap7_train')   & (df.channel==ch)].p50.iloc[0]
    print(f"  {ch}: intel p50={ip:.4f}  asap p50={ap:.4f}  ratio={ap/ip:.4f}")

# gnd/cpl ratio per PDK
print("\n=== gnd/cpl ratio (per net, p50 of ratio on positive-cpl) ===")
for label, df_ in [('intel22_train', i[i.split=='train']),
                   ('asap7_train',   a[a.split=='train'])]:
    sub = df_[(df_.c_cpl_total_fF > 0) & (df_.c_gnd_fF > 0)]
    r = (sub.c_gnd_fF / sub.c_cpl_total_fF).values
    print(f"  {label}: p10={np.percentile(r,10):.3f}  p50={np.median(r):.3f}  p90={np.percentile(r,90):.3f}  (n={len(r)})")

# KS on log10(total_cap)
print("\n=== KS on log10(total_cap_fF), train vs train ===")
xi = np.log10(i[(i.split=='train') & (i.total_cap_fF>0)].total_cap_fF.values)
xa = np.log10(a[(a.split=='train') & (a.total_cap_fF>0)].total_cap_fF.values)
ks, p = stats.ks_2samp(xi, xa, mode='asymp')
print(f"  KS={ks:.4f}  p={p:.3e}  (intel mean log={xi.mean():.3f}, asap mean log={xa.mean():.3f})")
