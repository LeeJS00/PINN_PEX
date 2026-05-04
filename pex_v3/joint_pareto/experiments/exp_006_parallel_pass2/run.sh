#!/bin/bash
# run.sh — exp_006 parallel-pass-2 5-seed driver.
#
# For each XGB seed N in {0..4}:
#   1. Generate autonomous_fast SPEF via the parallel engine.
#   2. XGB calibrate (per-net rescale) using seed-N CSV.
#   3. Sister-R per-net rescale.
#   4. Compare vs golden.
#   5. Persist per-channel metrics + runtime breakdown.
#
# Then aggregate to measurement.json and run admit_to_frontier (dry-run).
#
# All artefacts land under <EXP_DIR>/runs/.

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${EXP_DIR}/runs"
WORKERS="${WORKERS:-16}"

mkdir -p "${RUN_DIR}"

cd /home/jslee/projects/PINNPEX

echo ">>> exp_006 parallel pass-2 — 5-seed driver"
echo "    workers per seed: ${WORKERS}"
echo "    run dir:          ${RUN_DIR}"
echo

for SEED in 0 1 2 3 4; do
    echo "===================================================================="
    echo ">>> seed ${SEED} starting"
    echo "===================================================================="
    python3 "${EXP_DIR}/run_one_seed.py" \
        --seed "${SEED}" \
        --out-dir "${RUN_DIR}" \
        --workers "${WORKERS}"
    echo "<<< seed ${SEED} done"
    echo
done

echo ">>> Aggregating 5-seed measurement"
python3 "${EXP_DIR}/aggregate_5seed.py" \
    --in-dir "${RUN_DIR}" \
    --out "${EXP_DIR}/measurement.json" \
    --n-seeds 5

echo
echo ">>> admit_to_frontier (dry run)"
python3 /home/jslee/projects/PINNPEX/pex_v3/joint_pareto/scripts/admit_to_frontier.py \
    --measurement "${EXP_DIR}/measurement.json" \
    --variant-id v7_parallel \
    --label "Path-2 v7 parallel pass-2" \
    --dry-run
