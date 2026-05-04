"""
Workspace config for cross-design tv80s test (2026-05-02).

Strategy:
    Train on 9 small designs, validate on nova, test on tv80s as full-chip PEX.
    Goal: per-net total_cap MAPE < 4% on tv80s.

This file only contains the workspace-local paths. Do not write to other
experiments' directories.
"""
from pathlib import Path

WORKSPACE = Path("/home/jslee/projects/PINNPEX/experiments/cross_design_tv80s_2026_05_02")
CACHE_DIR = WORKSPACE / "cache"
OUTPUT_DIR = WORKSPACE / "output"
LOGS_DIR   = WORKSPACE / "logs"
REPORTS_DIR = WORKSPACE / "reports"

# Read-only data sources (shared with other sessions; do not modify).
DATA_ROOT     = Path("/data/PINNPEX/data/processed_v3/intel22")
MANIFEST_PATH = DATA_ROOT / "dataset_manifest_v3.csv"
SPEF_DIR      = Path("/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22")
PDK_DIR       = Path("/home/jslee/projects/PINNPEX/tool/pdk/22nm")
LAYERS_INFO   = PDK_DIR / "layers" / "layers.info"

# Design split — cross-design generalization
TRAIN_DESIGNS = [
    "intel22_aes_cipher_top_f3",
    "intel22_gcd_f3",
    "intel22_ibex_core_f3",
    "intel22_ldpc_decoder_802_3an_f3",
    "intel22_mc_top_f3",
    "intel22_spi_top_f3",
    "intel22_usbf_top_f3",
    "intel22_vga_enh_top_f3",
    "intel22_wb_conmax_top_f3",
]
VAL_DESIGNS  = ["intel22_nova_f3"]
TEST_DESIGNS = ["intel22_tv80s_f3"]

ALL_DESIGNS = TRAIN_DESIGNS + VAL_DESIGNS + TEST_DESIGNS

# Targets and metric
PRIMARY_TARGET = "total_cap_fF"
TARGETS = ["total_cap_fF", "c_gnd_fF", "c_cpl_total_fF"]

# 5-seed protocol
SEEDS = [0, 1, 2, 3, 4]

# GPU policy: avoid GPU 0 (in use 7%) and GPU 1 (running PINN baseline).
PREFERRED_GPUS = [2, 3, 4, 7]

# Coupling cutoff (μm) for geometric coupling search inside a tile.
CPL_CUTOFF_UM = 4.0

# Layer count for histograms
N_METAL_LAYERS = 9
