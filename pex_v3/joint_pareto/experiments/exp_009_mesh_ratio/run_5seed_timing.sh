#!/bin/bash
# Clean 5-seed end-to-end wall-clock timing for v9.
# v7 engine (parallel pass-2) + XGB + Mesh-ratio + sister-R, top-to-bottom per seed.
set -e

REPO=/home/jslee/projects/PINNPEX
EXPDIR=$REPO/pex_v3/joint_pareto/experiments/exp_009_mesh_ratio
mkdir -p $EXPDIR/timing_runs

cd $REPO

GOLDEN=/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22/intel22_tv80s_f3_starrc.spef
RPARQ=experiments/cross_design_tv80s_2026_05_02/r_analytic_v3/outputs/test_predictions_v6_s3.parquet
MESH=pex_v3/output/phase1_mesh_5seed_ensemble/ensemble_predictions_test.csv

# Use the exp_006 run_one_seed which spawns a clean python; we wrap each seed in time
echo "{\"per_seed\": [" > $EXPDIR/timing_5seed.json
for seed in 0 1 2 3 4; do
    SEED_DIR=$EXPDIR/timing_runs/seed${seed}
    mkdir -p $SEED_DIR

    XGB="$REPO/pex_v3/output/baselines/B1_xgboost_real/seed${seed}/eval_predictions_test.csv"

    # --- Stage A: parallel autonomous SPEF (using exp_006 run_one_seed.py) ---
    T_AUTO_START=$(date +%s.%N)
    python3 $REPO/pex_v3/joint_pareto/experiments/exp_006_parallel_pass2/run_one_seed.py \
        --seed ${seed} --out-dir $SEED_DIR > $SEED_DIR/auto.log 2>&1 || true
    T_AUTO_END=$(date +%s.%N)

    # The autonomous SPEF lives in $SEED_DIR/intel22_tv80s_f3_autonomous_fast.spef per exp_006 layout
    AUTO_SPEF=$SEED_DIR/intel22_tv80s_f3_autonomous_fast.spef
    if [ ! -f "$AUTO_SPEF" ]; then
        AUTO_SPEF=$(find $SEED_DIR -name "*autonomous_fast*.spef" | head -1)
    fi

    # --- Stage B: XGB calibrate ---
    T_XGB_START=$(date +%s.%N)
    python3 $REPO/pex_v3/scripts/16_xgb_calibrate_spef.py \
        --in-spef "$AUTO_SPEF" --xgb-csv "$XGB" --design intel22_tv80s_f3 \
        --out-spef $SEED_DIR/xgb.spef > $SEED_DIR/xgb.log 2>&1
    T_XGB_END=$(date +%s.%N)

    # --- Stage C: Mesh-ratio calibrate ---
    T_MESH_START=$(date +%s.%N)
    python3 $REPO/pex_v3/joint_pareto/scripts/42_mesh_ratio_calibrate_spef.py \
        --in-spef $SEED_DIR/xgb.spef --xgb-csv "$XGB" --mesh-csv "$MESH" \
        --design intel22_tv80s_f3 --out-spef $SEED_DIR/meshratio.spef > $SEED_DIR/mesh.log 2>&1
    T_MESH_END=$(date +%s.%N)

    # --- Stage D: sister R rescale ---
    T_R_START=$(date +%s.%N)
    python3 $REPO/pex_v3/scripts/23_r_per_net_calibrate_spef.py \
        --in-spef $SEED_DIR/meshratio.spef --out-spef $SEED_DIR/HERO.spef \
        --r-pred-parquet "$RPARQ" --r-pred-col R_pred_v6_s3 > $SEED_DIR/r.log 2>&1
    T_R_END=$(date +%s.%N)

    # --- Stage E: compare (NOT counted as wall-clock; eval-only) ---
    python3 $REPO/src/evaluation/compare_spef.py \
        --golden $GOLDEN --pred $SEED_DIR/HERO.spef \
        --out_dir $SEED_DIR/compare > $SEED_DIR/compare.log 2>&1

    # Report
    AUTO_S=$(echo "$T_AUTO_END - $T_AUTO_START" | bc)
    XGB_S=$(echo "$T_XGB_END - $T_XGB_START" | bc)
    MESH_S=$(echo "$T_MESH_END - $T_MESH_START" | bc)
    R_S=$(echo "$T_R_END - $T_R_START" | bc)
    TOT_S=$(echo "$T_AUTO_END - $T_AUTO_START + $T_XGB_END - $T_XGB_START + $T_MESH_END - $T_MESH_START + $T_R_END - $T_R_START" | bc)
    MAPE_LINE=$(grep "Total Capacitance" $SEED_DIR/compare.log | head -1)

    echo "seed ${seed}: auto=${AUTO_S}s · xgb=${XGB_S}s · mesh=${MESH_S}s · r=${R_S}s · TOTAL=${TOT_S}s | $MAPE_LINE"
    if [ $seed -lt 4 ]; then SEP=","; else SEP=""; fi
    echo "  {\"seed\": ${seed}, \"auto_s\": ${AUTO_S}, \"xgb_s\": ${XGB_S}, \"mesh_s\": ${MESH_S}, \"r_s\": ${R_S}, \"total_s\": ${TOT_S}}${SEP}" >> $EXPDIR/timing_5seed.json
done
echo "]}" >> $EXPDIR/timing_5seed.json

# Aggregate
python3 -c "
import json, statistics
d = json.load(open('$EXPDIR/timing_5seed.json'))
totals = [s['total_s'] for s in d['per_seed']]
autos  = [s['auto_s']  for s in d['per_seed']]
print(f'\\n=== 5-seed timing aggregate ===')
print(f'autonomous (v7 parallel)  : {statistics.mean(autos):.2f} ± {statistics.stdev(autos):.2f} s')
print(f'TOTAL end-to-end          : {statistics.mean(totals):.2f} ± {statistics.stdev(totals):.2f} s')
"
