#!/bin/bash
# Orchestrate parallel training of all models on the v2 features (full).
# Usage: bash scripts/orchestrate_final.sh

set -e

PYBIN="/tool/etc/python/install/3.11.9/bin/python3"
WS="/home/jslee/projects/PINNPEX/experiments/cross_design_tv80s_2026_05_02"
cd "$WS"
mkdir -p logs

# 1. LGBM × 5 seeds × 2 strategies (direct + residual)
$PYBIN -u scripts/run_final_pipeline.py \
  --cache features_v2 --out final_pipe \
  --models lgbm \
  --strategies direct residual \
  --seeds 0 1 2 3 4 > logs/final_lgbm.log 2>&1 &
echo "LGBM PID: $!"

# 2. XGB × 5 seeds × 2 strategies
$PYBIN -u scripts/run_final_pipeline.py \
  --cache features_v2 --out final_pipe \
  --models xgb \
  --strategies direct residual \
  --seeds 0 1 2 3 4 > logs/final_xgb.log 2>&1 &
echo "XGB PID: $!"

# 3. CatBoost × 5 seeds × 1 strategy (slower)
$PYBIN -u scripts/run_final_pipeline.py \
  --cache features_v2 --out final_pipe \
  --models cat \
  --strategies direct \
  --seeds 0 1 2 3 4 > logs/final_cat.log 2>&1 &
echo "CAT PID: $!"

# 4. ResMLP × 5 seeds (GPU 2)
for s in 0 1 2 3 4; do
  $PYBIN -u scripts/train_mlp_v2.py \
    --cache features_v2 --seed $s --epochs 300 --gpu 2 \
    --out resmlp_v2 > logs/resmlp_s${s}.log 2>&1 &
  echo "ResMLP seed $s PID: $!"
  sleep 1
done

echo ""
echo "All jobs launched. Monitor with:  tail -f logs/final_*.log logs/resmlp_*.log"
