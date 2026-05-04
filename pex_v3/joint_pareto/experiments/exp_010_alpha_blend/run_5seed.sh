#!/bin/bash
# 5-seed v10 α-blend timing + accuracy (full pipeline)
set -e
REPO=/home/jslee/projects/PINNPEX
EXPDIR=$REPO/pex_v3/joint_pareto/experiments/exp_010_alpha_blend
mkdir -p $EXPDIR/runs
cd $REPO
GOLDEN=/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22/intel22_tv80s_f3_starrc.spef
RPARQ=experiments/cross_design_tv80s_2026_05_02/r_analytic_v3/outputs/test_predictions_v6_s3.parquet
MESH=pex_v3/output/phase1_mesh_5seed_ensemble/ensemble_predictions_test.csv
ALPHA=0.2

echo "{\"per_seed\": [" > $EXPDIR/timing_5seed.json
for seed in 0 1 2 3 4; do
    SD=$EXPDIR/runs/seed${seed}
    mkdir -p $SD
    XGB="pex_v3/output/baselines/B1_xgboost_real/seed${seed}/eval_predictions_test.csv"

    T0=$(date +%s.%N)
    python3 pex_v3/joint_pareto/experiments/exp_006_parallel_pass2/run_one_seed.py \
        --seed ${seed} --out-dir $SD > $SD/auto.log 2>&1 || true
    T1=$(date +%s.%N)

    AUTO_SPEF=$(find $SD -name "*autonomous_fast*.spef" | head -1)

    python3 pex_v3/joint_pareto/scripts/43_xgb_mesh_blend_calibrate_spef.py \
        --in-spef "$AUTO_SPEF" --xgb-csv "$XGB" --mesh-csv "$MESH" \
        --design intel22_tv80s_f3 --alpha $ALPHA \
        --out-spef $SD/blend.spef > $SD/blend.log 2>&1
    T2=$(date +%s.%N)

    python3 pex_v3/scripts/23_r_per_net_calibrate_spef.py \
        --in-spef $SD/blend.spef --out-spef $SD/HERO.spef \
        --r-pred-parquet "$RPARQ" --r-pred-col R_pred_v6_s3 > $SD/r.log 2>&1
    T3=$(date +%s.%N)

    python3 src/evaluation/compare_spef.py \
        --golden $GOLDEN --pred $SD/HERO.spef --out_dir $SD/compare > $SD/compare.log 2>&1

    AUTO=$(echo "$T1 - $T0" | bc)
    BLEND=$(echo "$T2 - $T1" | bc)
    R=$(echo "$T3 - $T2" | bc)
    TOT=$(echo "$T3 - $T0" | bc)
    MAPE_LINE=$(grep "Total Capacitance" $SD/compare.log | head -1)
    echo "seed ${seed}: auto=${AUTO}s · blend=${BLEND}s · r=${R}s · TOTAL=${TOT}s | $MAPE_LINE"

    if [ $seed -lt 4 ]; then SEP=","; else SEP=""; fi
    echo "  {\"seed\": ${seed}, \"auto_s\": ${AUTO}, \"blend_s\": ${BLEND}, \"r_s\": ${R}, \"total_s\": ${TOT}}${SEP}" >> $EXPDIR/timing_5seed.json
done
echo "]}" >> $EXPDIR/timing_5seed.json

python3 << EOF
import json, statistics
import pandas as pd
d = json.load(open('$EXPDIR/timing_5seed.json'))
totals = [s['total_s'] for s in d['per_seed']]
print(f'\n=== 5-seed v10 timing aggregate ===')
print(f'TOTAL end-to-end: {statistics.mean(totals):.2f} ± {statistics.stdev(totals):.2f} s')

# Aggregate accuracy
xgb_csv = pd.read_csv('pex_v3/output/baselines/B1_xgboost_real/seed0/eval_predictions_test.csv')
xgb_names = set(xgb_csv[xgb_csv['design_name']=='intel22_tv80s_f3']['net_name'].astype(str))
seeds = []
for s in range(5):
    df = pd.read_csv(f'$EXPDIR/runs/seed{s}/compare/spef_comparison_report.csv')
    df['matched'] = df['net'].isin(xgb_names)
    df['ape_gnd'] = (df['p_gnd']-df['g_gnd']).abs() / df['g_gnd'].clip(lower=1e-9) * 100
    df['ape_cpl'] = (df['p_cpl']-df['g_cpl']).abs() / df['g_cpl'].clip(lower=1e-9) * 100
    mat = df[df['matched']]
    seeds.append({'mean':df['mape'].mean(),'median':df['mape'].median(),'p95':df['mape'].quantile(0.95),
                  'gnd_m':mat['ape_gnd'].mean(),'cpl_m':mat['ape_cpl'].mean()})
agg = pd.DataFrame(seeds)
print(f'mean        : {agg["mean"].mean():.3f} ± {agg["mean"].std():.3f}')
print(f'median      : {agg["median"].mean():.3f} ± {agg["median"].std():.3f}')
print(f'p95         : {agg["p95"].mean():.3f} ± {agg["p95"].std():.3f}')
print(f'gnd matched : {agg["gnd_m"].mean():.3f} ± {agg["gnd_m"].std():.3f}')
print(f'cpl matched : {agg["cpl_m"].mean():.3f} ± {agg["cpl_m"].std():.3f}')
EOF
