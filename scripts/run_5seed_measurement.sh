#!/bin/bash
# 5-seed measurement protocol launcher.
#
# Variants:
#   A. v3_baseline   (no calibration JSON, hardcoded ζ)
#   B. v4_full_calib (data-driven ρ + CPL via calibration_init.json)
#   C. v5_gnd_only   (data-driven ρ + hardcoded CPL via calibration_init_gnd_only.json)
#
# Per variant: 5 seeds (0, 1, 2, 3, 4). Each run --max_iters 1 --steps_per_iter 5000.
# Best model + per-iter MAPE saved per (variant, seed) under
#   output_intel22/active_learning/m5_<variant>_seed<N>/
#
# Usage:
#   bash scripts/run_5seed_measurement.sh
#
# Runs 4 in parallel across GPUs 1-4, cycles through the queue until all
# 15 jobs complete. Logs to output_intel22/al_5seed_<variant>_seed<N>.log.

set -u

PYTHON=/tool/etc/python/install/3.11.9/bin/python3
ROOT=/home/jslee/projects/PINNPEX
OUT_DIR="$ROOT/output_intel22"
CAL_FULL="/data/PINNPEX/data/processed/intel22/calibration_init.json"
CAL_GND_ONLY="/data/PINNPEX/data/processed/intel22/calibration_init_gnd_only.json"

cd "$ROOT" || exit 1

# (variant, calib_arg, label_prefix)
declare -A VARIANT_CALIB
VARIANT_CALIB[v3_baseline]="none"
VARIANT_CALIB[v4_full_calib]="$CAL_FULL"
VARIANT_CALIB[v5_gnd_only]="$CAL_GND_ONLY"

VARIANTS=(v3_baseline v4_full_calib v5_gnd_only)
SEEDS=(0 1 2 3 4)
GPUS=(1 2 3 4)

# Build job queue: (variant seed) tuples
declare -a QUEUE
for variant in "${VARIANTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        QUEUE+=("${variant}:${seed}")
    done
done
echo ">>> Job queue: ${#QUEUE[@]} jobs"
printf '  %s\n' "${QUEUE[@]}"

run_one() {
    local variant="$1" seed="$2" gpu="$3"
    local calib="${VARIANT_CALIB[$variant]}"
    local model_name="m5_${variant}_seed${seed}"
    local log="$OUT_DIR/al_5seed_${variant}_seed${seed}.log"
    # Status to stderr so it doesn't pollute the captured PID.
    echo "  [LAUNCH] ${variant} seed=${seed} on GPU ${gpu} → log=${log##*/}" >&2
    nohup $PYTHON -u run_active_learning.py \
        --model_name "$model_name" \
        --gpu "$gpu" \
        --use_dspinn \
        --calib_path "$calib" \
        --seed "$seed" \
        --max_iters 1 \
        --steps_per_iter 5000 \
        > "$log" 2>&1 &
    printf '%d\n' "$!"
}

# Schedule jobs across GPUs; at most one job per GPU at a time.
declare -A GPU_PID  # gpu_id -> pid (0 if free)
for g in "${GPUS[@]}"; do GPU_PID[$g]=0; done

NEXT_JOB=0
ACTIVE=0

while [ "$NEXT_JOB" -lt "${#QUEUE[@]}" ] || [ "$ACTIVE" -gt 0 ]; do
    # Try to fill empty GPUs
    for g in "${GPUS[@]}"; do
        if [ "${GPU_PID[$g]}" -eq 0 ] && [ "$NEXT_JOB" -lt "${#QUEUE[@]}" ]; then
            entry="${QUEUE[$NEXT_JOB]}"
            variant="${entry%%:*}"
            seed="${entry##*:}"
            pid=$(run_one "$variant" "$seed" "$g")
            GPU_PID[$g]="$pid"
            ACTIVE=$((ACTIVE + 1))
            NEXT_JOB=$((NEXT_JOB + 1))
        fi
    done
    # Wait a bit, then reap finished
    sleep 60
    for g in "${GPUS[@]}"; do
        pid="${GPU_PID[$g]}"
        if [ "$pid" -ne 0 ] && ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null
            ACTIVE=$((ACTIVE - 1))
            GPU_PID[$g]=0
            echo "  [DONE] GPU ${g} freed (pid=${pid}) — ${NEXT_JOB}/${#QUEUE[@]} dispatched, ${ACTIVE} active"
        fi
    done
done

echo ">>> All ${#QUEUE[@]} jobs complete."
