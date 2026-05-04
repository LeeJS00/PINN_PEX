"""
config_v3.py — Strategy v3 configuration.

Imports legacy `configs.config` and overrides only what Strategy v3 changes.
Phase 0 changes:
  - PROCESSED_DIR_V3: new data root (separate from legacy)
  - MANIFEST_PATH_V3: H1 net-level split manifest
  - CONTEXT_MARGIN_V3: H3 rebuild margin (2 → 6 μm); takes effect when
    `02_rebuild_dataset_h3.py` runs
  - NF_PAD_TO_CUBOIDS_V3: kept at 1024 for Phase 0; H2 priority truncation
    is a runtime sort, not a pad change
  - VALID_RATIO_V3: kept at 0.10 (legacy)
  - SCHEMA_VERSION: bumped from "v9" to "v3" — loaders error on mismatch

Phase 0+ changes will append below.
"""
from pathlib import Path
import sys

# Allow `from configs.config import ...` to resolve to the legacy module.
# We *import* from legacy (read-only), but we never write to legacy paths.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import config as _legacy  # noqa: E402

# ----------------------------------------------------------------------
# Re-export legacy keys that Strategy v3 does NOT change
# ----------------------------------------------------------------------
TRAIN_DEFS = _legacy.TRAIN_DEFS
TEST_DEFS = _legacy.TEST_DEFS
TRAIN_SPEFS = _legacy.TRAIN_SPEFS
TEST_SPEFS = _legacy.TEST_SPEFS
LAYERS_INFO_PATH = _legacy.LAYERS_INFO_PATH
TECH_LEF_PATH = _legacy.TECH_LEF_PATH
CELL_LEF_PATH = _legacy.CELL_LEF_PATH
PEX_TEMPLATE_PATH = _legacy.PEX_TEMPLATE_PATH
PEX_BIN = _legacy.PEX_BIN
PEX_LICENCES = _legacy.PEX_LICENCES
RAW_DIR = _legacy.RAW_DIR
SPEF_DIR = _legacy.SPEF_DIR
PDK_DIR = _legacy.PDK_DIR
WINDOW_SIZE = _legacy.WINDOW_SIZE
TILING_OVERLAP = _legacy.TILING_OVERLAP
DEFAULT_DBU = _legacy.DEFAULT_DBU
USE_VSS_AGGRESSORS = _legacy.USE_VSS_AGGRESSORS
INPUT_DIM = _legacy.INPUT_DIM
CUTOFF_RADIUS = _legacy.CUTOFF_RADIUS
USE_RAIL_COUPLING = _legacy.USE_RAIL_COUPLING

# ----------------------------------------------------------------------
# Strategy v3 OVERRIDES — Phase 0
# ----------------------------------------------------------------------

# Schema version. Loaders MUST refuse to load a manifest with a different
# SCHEMA_VERSION value; this is the leak-prevention discipline (Codex round 1
# experiment-systems-engineer mandate).
SCHEMA_VERSION = "v3"

# v3 data root — fully separate from legacy /data/PEX_SSL/data/processed/intel22.
PROCESSED_DIR_V3 = Path("/data/PINNPEX/data/processed_v3/intel22")
PT_DIR_V3 = PROCESSED_DIR_V3.parent / "intel22_pt_v3"

# v3 manifest path. Phase 0 H1 step writes this; legacy manifest at
# /data/PEX_SSL/data/processed/intel22/dataset_manifest.csv is never touched.
MANIFEST_PATH_V3 = PROCESSED_DIR_V3 / "dataset_manifest_v3.csv"

# Legacy manifest source (read-only). Used by 01_resplit_manifest.py to copy
# rows + recompute split column with hash-based H1 fix.
LEGACY_PROCESSED_DIR = Path("/data/PEX_SSL/data/processed/intel22")
LEGACY_MANIFEST_PATH = LEGACY_PROCESSED_DIR / "dataset_manifest.csv"

# H1: net-level hash-based split.
# (design_name, net_name) → SHA256 → uniform [0, 1) → bucket
VALID_RATIO_V3 = 0.10
H1_HASH_SEED = "pinn_pex_v3_2026_05_01"  # any string; locks the split deterministically

# H3: context margin for rebuild. Phase 0 step 02_rebuild_dataset_h3.py
# uses this to extend stored context window from 8×8 μm (margin 2) to
# 14×14 μm (margin 5) so cutoff_r=4μm always has 1μm slack.
CONTEXT_MARGIN_V3 = 5.0  # μm; window becomes (4+10, 4+10, 20) μm

# H2: priority truncation runtime sort (no padding change at Phase 0).
NF_PAD_TO_CUBOIDS_V3 = 1024  # same as legacy; H2 sorts before truncating

# H4: edge enumeration mode for CPL search.
# "closest_dist"  — legacy (collapses long parallel runs)
# "pairwise"      — H4 fix (enumerate pairs up to cutoff)
H4_EDGE_MODE = "pairwise"
# Edge count grows ~2.25× → bump aggressor budget so memory stays bounded.
MAX_AGGR_BUDGET_V3 = 768

# Output root for v3 experiments — separate from legacy output_intel22/.
OUTPUT_DIR_V3 = PROJECT_ROOT / "pex_v3" / "output"
OUTPUT_DIR_V3.mkdir(parents=True, exist_ok=True)

# Run name for v3 SSL basis. Independent from legacy ssl_basis_dspinn_v1.
RUN_NAME_V3 = "ssl_basis_v3_clean"

# Single-GPU policy (unchanged from legacy).
GPU_ID = _legacy.GPU_ID

# ----------------------------------------------------------------------
# Active-Learning settings (Phase 0 baseline runs use these)
# ----------------------------------------------------------------------
AL_BATCH_SIZE = _legacy.AL_BATCH_SIZE
AL_LR = _legacy.AL_LR
AL_TRAIN_STEPS_PER_ITER = _legacy.AL_TRAIN_STEPS_PER_ITER
AL_FINE_ITERS = _legacy.AL_FINE_ITERS
AL_BATCH_NETS = _legacy.AL_BATCH_NETS
AL_MAX_TILES_PER_BATCH = _legacy.AL_MAX_TILES_PER_BATCH
AL_NUM_WORKERS = _legacy.AL_NUM_WORKERS
AL_MAX_BUDGET_RATIO = _legacy.AL_MAX_BUDGET_RATIO
AL_MIN_ENTROPY_THRESHOLD = _legacy.AL_MIN_ENTROPY_THRESHOLD
AL_PREDEFINED_DESIGNS = _legacy.AL_PREDEFINED_DESIGNS
AL_SAMPLING_METHOD = _legacy.AL_SAMPLING_METHOD

SSL_BATCH_SIZE = _legacy.SSL_BATCH_SIZE
SSL_LR = _legacy.SSL_LR
SSL_EPOCHS = _legacy.SSL_EPOCHS
SSL_W_BC = _legacy.SSL_W_BC
SSL_W_ENERGY = _legacy.SSL_W_ENERGY
SSL_W_FAR = _legacy.SSL_W_FAR
SSL_NUM_WORKERS = _legacy.SSL_NUM_WORKERS

# ----------------------------------------------------------------------
# Convenience: snapshot of all v3-specific keys for manifest hash logging.
# ----------------------------------------------------------------------
def v3_snapshot() -> dict:
    """Return a JSON-serializable dict of all Strategy v3 overrides.

    `experiment-systems-engineer` writes this into every run dir as
    `config_snapshot.json` for reproducibility.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "processed_dir_v3": str(PROCESSED_DIR_V3),
        "manifest_path_v3": str(MANIFEST_PATH_V3),
        "valid_ratio_v3": VALID_RATIO_V3,
        "h1_hash_seed": H1_HASH_SEED,
        "context_margin_v3": CONTEXT_MARGIN_V3,
        "nf_pad_to_cuboids_v3": NF_PAD_TO_CUBOIDS_V3,
        "h4_edge_mode": H4_EDGE_MODE,
        "max_aggr_budget_v3": MAX_AGGR_BUDGET_V3,
        "run_name_v3": RUN_NAME_V3,
        "cutoff_radius": CUTOFF_RADIUS,
        "input_dim": INPUT_DIM,
        "use_vss_aggressors": USE_VSS_AGGRESSORS,
    }
