# DeepPEX_v2/configs/config.py
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "def" / "intel22"
PDK_DIR = PROJECT_ROOT / "tool" / "pdk" / "22nm"
PROCESSED_DIR = DATA_DIR / "processed" / "intel22__0327"
OUTPUT_DIR = PROJECT_ROOT / "output_intel22__0327"
ORACLE_PATH = "/home/jslee/projects/PEX_SSL/data/STARRC_LOG/master_starrc_history.csv"

# Files (User should update these paths)

TRAIN_DEFS = [
    RAW_DIR / "intel22_aes_cipher_top_t1.def",
    RAW_DIR / "intel22_gcd_t1.def",
    RAW_DIR / "intel22_ibex_core_t1.def",
    RAW_DIR / "intel22_ldpc_decoder_t1.def",
    RAW_DIR / "intel22_mc_top_t1.def",
    RAW_DIR / "intel22_vga_enh_top_t1.def",
    RAW_DIR / "intel22_usbf_top_t1.def",
]

TEST_DEFS = [    
    RAW_DIR / "intel22_spi_top_t1.def",
    RAW_DIR / "intel22_tv80s_t1.def",
    RAW_DIR / "intel22_wb_conmax_top_t1.def",
]
VALID_RATIO = 0.1  # 10% for Validation

LAYERS_INFO_PATH = PDK_DIR / "layers" / "layers.info"
TECH_LEF_PATH = PDK_DIR / "tech_lef" / "p1222_js.lef" # VIA 정보 확인용
CELL_LEF_PATH = PDK_DIR / "cell_lef" / "b15_nn.lef" # Cell LEF 파일 경로
# Geometry
# Window size in microns
WINDOW_SIZE = (4.0, 4.0, 20.0) 
# Overlap for tiling (microns)
TILING_OVERLAP = 0.5
CONTEXT_MARGIN = 1.0 
# DBU default (will be overridden by DEF header)
DEFAULT_DBU = 2000
AL_BUDGET_RATIO = 0.05
# SCALE_FACTOR = 2.4
# Parallel Processing
NUM_JOBS = 16

# StarRC Configuration
# PEX_BIN = 'quantus' # or "StarXtract"
PEX_BIN = 'StarXtract'
PEX_TEMPLATE_PATH = PROJECT_ROOT / "templates" / "run_starrc_template.tcl"
# PEX_TEMPLATE_PATH = PROJECT_ROOT / "templates" / "run_quantus_template.ccl"
# PEX_LICENCES = 48
PEX_LICENCES = 14
KEEP_PEX_FILES = False
GPU_ID = 3

# MODEL
NUM_HEADS = 4
MODEL_DIM = 256
BASIS_LAYERS = 4
CORRECTION_LAYERS = 3

# SSL
SSL_BATCH_SIZE = 128 # 4 - 650MiB
SSL_LR = 5e-4
SSL_EPOCHS = 100
SSL_W_BC = 10.0
SSL_W_ENERGY = 0.1
SSL_W_FAR = 1.0

# AL
AL_BUDGET_RATIO = 0.01
AL_ITERS = 10
AL_FINE_ITERS = 30

# Output CSV Columns
CSV_COLS = ['sample_filename', 'net_name', 'def_name', 'tile_idx', 'capacitance']
