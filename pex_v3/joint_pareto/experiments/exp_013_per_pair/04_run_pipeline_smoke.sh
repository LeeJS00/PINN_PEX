#!/bin/bash
# 04_run_pipeline_smoke.sh — end-to-end pipeline for per-pair calibrated v10 SPEF.
#
# 1. Train residual model (depends on 01_extract_train_features.py output)
# 2. Apply 44_per_pair_calibrate_spef.py to v10 SPEF
# 3. Re-measure per-pair MAPE
#
set -e

ROOT=/home/jslee/projects/PINNPEX
EXP=$ROOT/pex_v3/joint_pareto/experiments/exp_013_per_pair
DESIGN=intel22_tv80s_f3
TOPO_DIR=/data/PINNPEX/data/processed_v3/intel22/$DESIGN/topology
V10_IN=$ROOT/pex_v3/output/spef_e2e_fast_v10/${DESIGN}_HERO_v10.spef
OUT_DIR=$EXP/results
OUT_SPEF=$OUT_DIR/${DESIGN}_v11_per_pair.spef
GOLDEN_SPEF=/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22/${DESIGN}_starrc.spef

mkdir -p $OUT_DIR

echo ""
echo "=== STEP 1: Apply per-pair calibration ==="
python3 $ROOT/pex_v3/joint_pareto/scripts/44_per_pair_calibrate_spef.py \
    --in-spef $V10_IN \
    --design $DESIGN \
    --topo-dir $TOPO_DIR \
    --model $OUT_DIR/residual_model.lgb \
    --out-spef $OUT_SPEF

echo ""
echo "=== STEP 2: Re-measure per-pair MAPE ==="
python3 - <<EOF
import sys
sys.path.insert(0, '$EXP')
import diag_v10_per_pair as diag
from pathlib import Path
diag.PRED_SPEF = Path('$OUT_SPEF')
diag.OUT_DIR = Path('$OUT_DIR/v11_per_pair_eval')
diag.OUT_DIR.mkdir(parents=True, exist_ok=True)
diag.main()
EOF

echo ""
echo "=== STEP 3: Total cap MAPE vs golden ==="
python3 $ROOT/src/evaluation/compare_spef.py \
    --golden $GOLDEN_SPEF \
    --pred $OUT_SPEF \
    --out_dir $OUT_DIR/v11_total_eval || true
