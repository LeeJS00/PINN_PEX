#!/bin/bash
# Nova post-process pipeline (Path-2):
#   autonomous_fast.spef → XGB cap calibration → R-α global rescale → compare
# Note: sister R per-net parquet has tv80s only; nova uses cross-codebase R-α=1.4777.
set -e
SEED=${1:-0}
RALPHA=${2:-1.4777}
AUTO=/home/jslee/projects/PINNPEX/pex_v3/output/spef_fast_parallel/intel22_nova_f3_autonomous_fast.spef
XGBCSV=/home/jslee/projects/PINNPEX/pex_v3/output/baselines/B1_xgboost_real/seed${SEED}/eval_predictions_test.csv
OUTDIR=/home/jslee/projects/PINNPEX/pex_v3/output/spef_fast_parallel
GOLDEN=/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22/intel22_nova_f3_starrc.spef

cd /home/jslee/projects/PINNPEX

echo ">>> XGB cap calibration (seed=${SEED})"
python3 pex_v3/scripts/16_xgb_calibrate_spef.py \
    --in-spef "${AUTO}" --xgb-csv "${XGBCSV}" --design intel22_nova_f3 \
    --out-spef "${OUTDIR}/intel22_nova_f3_xgb_seed${SEED}.spef"

echo ">>> R-α global rescale (α=${RALPHA})"
python3 pex_v3/scripts/20_r_alpha_calibrate_spef.py \
    --in-spef "${OUTDIR}/intel22_nova_f3_xgb_seed${SEED}.spef" \
    --out-spef "${OUTDIR}/intel22_nova_f3_HERO_seed${SEED}.spef" \
    --alpha "${RALPHA}"

echo ">>> Compare vs golden"
python3 src/evaluation/compare_spef.py \
    --golden "${GOLDEN}" \
    --pred "${OUTDIR}/intel22_nova_f3_HERO_seed${SEED}.spef" \
    --out_dir "${OUTDIR}/compare_HERO_seed${SEED}" 2>&1 | grep -E "Total|MAPE" | head -8
