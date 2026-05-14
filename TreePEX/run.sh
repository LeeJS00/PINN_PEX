#!/bin/bash
# run.sh — end-to-end TreePEX PEX tool for a NEW design.
#
# Pipeline (each stage timed):
#   1. Build cuboid tiles      (scripts/build_dataset.py)
#   2. Extract V3 41-D features (TreePEX/scripts/extract_v3_features.py)
#   3. Extract V4 H3 26-D features (TreePEX/scripts/extract_v4_h3.py)
#   4. Run TreePEX XGBoost 5-seed inference (TreePEX/scripts/02_inference.py)
#   5. Write predicted SPEF (TreePEX/scripts/03_write_spef.py)
#   6. Compare to golden if provided (TreePEX/scripts/04_compare_golden.py)
#
# Usage:
#   ./TreePEX/run.sh \
#       --def     /path/to/intel22_mydesign.def \
#       --design  intel22_mydesign_f3 \
#       [--spef   /path/to/intel22_mydesign_starrc.spef] \
#       [--workers 16] \
#       [--skip-tile]   (skip stage 1 if tiles already built)
#       [--skip-features] (skip stages 2-3 if features already extracted)
#
# Required: DEF file. Optional: golden SPEF (for compare).
# Outputs:
#   TreePEX/outputs/predictions/<design>_pred.csv
#   TreePEX/outputs/spef/<design>_pred.spef
#   TreePEX/outputs/reports/<design>_compare.csv  (only if --spef provided)

set -euo pipefail

# ─── default args ────────────────────────────────────────────────────────────
DEF_PATH=""
DESIGN=""
SPEF_PATH=""
WORKERS=16
SKIP_TILE=0
SKIP_FEATURES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --def)            DEF_PATH="$2"; shift 2 ;;
    --design)         DESIGN="$2"; shift 2 ;;
    --spef)           SPEF_PATH="$2"; shift 2 ;;
    --workers)        WORKERS="$2"; shift 2 ;;
    --skip-tile)      SKIP_TILE=1; shift ;;
    --skip-features)  SKIP_FEATURES=1; shift ;;
    -h|--help)
      sed -n '1,30p' "$0"; exit 0 ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$DEF_PATH" || -z "$DESIGN" ]]; then
  echo "ERROR: --def and --design required" >&2
  echo "Usage: $0 --def <DEF> --design <name> [--spef <SPEF>]" >&2
  exit 1
fi
if [[ ! -f "$DEF_PATH" ]]; then
  echo "ERROR: DEF not found: $DEF_PATH" >&2
  exit 1
fi

# ─── paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PEX_V6="$PROJECT_ROOT/TreePEX"
PY="${PYTHON:-python3}"

V3_DATA_ROOT="/data/PINNPEX/data/processed_v3/intel22"
V3_TILE_DIR="$V3_DATA_ROOT/per_net_cuboids"
V3_FEATURES_CSV="$V3_DATA_ROOT/features/all_designs.csv"
V3_MANIFEST="$V3_DATA_ROOT/dataset_manifest_v3.csv"
V4_FEATURES_CSV="$PROJECT_ROOT/archive/pex_v4/results/new_features_with_ids.csv"

OUT_PRED="$PEX_V6/outputs/predictions"
OUT_SPEF="$PEX_V6/outputs/spef"
OUT_REPORT="$PEX_V6/outputs/reports"
mkdir -p "$OUT_PRED" "$OUT_SPEF" "$OUT_REPORT"

# ─── stage timer ─────────────────────────────────────────────────────────────
_t_start_total=$(date +%s.%N)
declare -A STAGE_TIMES
_stage() {
  local name="$1"; shift
  echo "════════════════════════════════════════════════════"
  echo "  $name"
  echo "════════════════════════════════════════════════════"
  local t0=$(date +%s.%N)
  "$@"
  local rc=$?
  local t1=$(date +%s.%N)
  local elapsed=$(awk "BEGIN {printf \"%.2f\", $t1 - $t0}")
  STAGE_TIMES["$name"]=$elapsed
  echo "[$name] ${elapsed}s (rc=$rc)"
  return $rc
}

# ─── stage 1: cuboid tile build ──────────────────────────────────────────────
stage_tile() {
  $PY "$PROJECT_ROOT/scripts/build_dataset.py" \
      --def_path "$DEF_PATH" \
      --out_dir "$V3_TILE_DIR" \
      --pt_out_dir "$V3_DATA_ROOT/pt" \
      --num_workers "$WORKERS"
}
if [[ $SKIP_TILE -eq 0 ]]; then
  _stage "stage 1 ▸ cuboid tiling" stage_tile
else
  echo "[stage 1] SKIPPED (--skip-tile)"
  STAGE_TIMES["stage 1 ▸ cuboid tiling"]="skipped"
fi

# ─── stage 2: V3 41-D base features ──────────────────────────────────────────
stage_v3_feat() {
  local args=(
    --def-path "$DEF_PATH"
    --design "$DESIGN"
    --manifest "$V3_MANIFEST"
    --features-csv "$V3_FEATURES_CSV"
  )
  if [[ -n "$SPEF_PATH" ]]; then
    args+=(--spef-path "$SPEF_PATH")
  fi
  $PY "$PEX_V6/scripts/extract_v3_features.py" "${args[@]}"
}
if [[ $SKIP_FEATURES -eq 0 ]]; then
  _stage "stage 2 ▸ V3 41-D base features" stage_v3_feat
else
  echo "[stage 2] SKIPPED (--skip-features)"
  STAGE_TIMES["stage 2 ▸ V3 41-D base features"]="skipped"
fi

# ─── stage 3: V4 H3 26-D pair features ───────────────────────────────────────
stage_v4_h3() {
  $PY "$PEX_V6/scripts/extract_v4_h3.py" \
      --design "$DESIGN" \
      --manifest "$V3_MANIFEST" \
      --data-root "$V3_DATA_ROOT" \
      --out-csv "$V4_FEATURES_CSV" \
      --n-workers "$WORKERS"
}
if [[ $SKIP_FEATURES -eq 0 ]]; then
  _stage "stage 3 ▸ V4 H3 26-D features" stage_v4_h3
else
  echo "[stage 3] SKIPPED (--skip-features)"
  STAGE_TIMES["stage 3 ▸ V4 H3 26-D features"]="skipped"
fi

# ─── stage 4: TreePEX XGBoost 5-seed inference ────────────────────────────────
stage_infer() {
  $PY "$PEX_V6/scripts/02_inference.py" --design "$DESIGN"
}
_stage "stage 4 ▸ XGBoost 5-seed inference" stage_infer

# ─── stage 5: SPEF write ─────────────────────────────────────────────────────
stage_spef() {
  $PY "$PEX_V6/scripts/03_write_spef.py" --design "$DESIGN"
}
_stage "stage 5 ▸ SPEF write" stage_spef

# ─── stage 6: compare to golden (optional) ───────────────────────────────────
if [[ -n "$SPEF_PATH" ]]; then
  stage_compare() {
    $PY "$PEX_V6/scripts/04_compare_golden.py" --design "$DESIGN" --golden "$SPEF_PATH"
  }
  _stage "stage 6 ▸ compare to golden" stage_compare
else
  echo "[stage 6] SKIPPED (no --spef provided)"
  STAGE_TIMES["stage 6 ▸ compare to golden"]="skipped"
fi

# ─── summary ─────────────────────────────────────────────────────────────────
_t_end_total=$(date +%s.%N)
total=$(awk "BEGIN {printf \"%.2f\", $_t_end_total - $_t_start_total}")
echo ""
echo "════════════════════════════════════════════════════"
echo "  SUMMARY — design=$DESIGN"
echo "════════════════════════════════════════════════════"
for k in \
    "stage 1 ▸ cuboid tiling" \
    "stage 2 ▸ V3 41-D base features" \
    "stage 3 ▸ V4 H3 26-D features" \
    "stage 4 ▸ XGBoost 5-seed inference" \
    "stage 5 ▸ SPEF write" \
    "stage 6 ▸ compare to golden"; do
  val="${STAGE_TIMES[$k]}"
  if [[ "$val" == "skipped" ]]; then
    printf "  %-45s skipped\n" "$k"
  else
    printf "  %-45s %ss\n" "$k" "$val"
  fi
done
echo "  ─────────────────────────────────────────────"
printf "  %-45s %ss\n" "TOTAL e2e wall" "$total"
echo ""
echo "Artifacts:"
echo "  predictions:  $OUT_PRED/${DESIGN}_pred.csv"
echo "  SPEF:         $OUT_SPEF/${DESIGN}_pred.spef"
if [[ -n "$SPEF_PATH" ]]; then
  echo "  comparison:   $OUT_REPORT/${DESIGN}_compare.csv"
fi
