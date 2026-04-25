# configs/config.py
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "def" / "intel22"
PDK_DIR = PROJECT_ROOT / "tool" / "pdk" / "22nm"
SPEF_DIR = Path("/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22")
PROCESSED_DIR = Path("/data/PEX_SSL/data/processed/intel22")
PT_DIR = PROCESSED_DIR.parent / "intel22_pt"
OUTPUT_DIR = PROJECT_ROOT / "output_intel22"
ORACLE_PATH = Path("/home/jslee/projects/PEX_SSL/data/STARRC_LOG/master_starrc_history.db")
# Files (User should update these paths)

TRAIN_DEFS = [
    RAW_DIR / "intel22_aes_cipher_top_f3.def",
    RAW_DIR / "intel22_gcd_f3.def",
    RAW_DIR / "intel22_ibex_core_f3.def",
    RAW_DIR / "intel22_ldpc_decoder_802_3an_f3.def",
    RAW_DIR / "intel22_mc_top_f3.def",
    RAW_DIR / "intel22_spi_top_f3.def",
    RAW_DIR / "intel22_usbf_top_f3.def",
    RAW_DIR / "intel22_vga_enh_top_f3.def",
    RAW_DIR / "intel22_wb_conmax_top_f3.def",
]

TEST_DEFS = [    
    RAW_DIR / "intel22_nova_f3.def",
    RAW_DIR / "intel22_tv80s_f3.def",
    # RAW_DIR / "intel22_mpeg2_top_f3.def",
    #RAW_DIR / "intel22_TinyRocketCore_f3.def",
]

TRAIN_SPEFS = [
    SPEF_DIR / "intel22_aes_cipher_top_f3_starrc.spef",
    SPEF_DIR / "intel22_gcd_f3_starrc.spef",
    SPEF_DIR / "intel22_ibex_core_f3_starrc.spef",
    SPEF_DIR / "intel22_ldpc_decoder_802_3an_f3_starrc.spef",
    SPEF_DIR / "intel22_mc_top_f3_starrc.spef",
    SPEF_DIR / "intel22_spi_top_f3_starrc.spef",
    SPEF_DIR / "intel22_usbf_top_f3_starrc.spef",
    SPEF_DIR / "intel22_vga_enh_top_f3_starrc.spef",
    SPEF_DIR / "intel22_wb_conmax_top_f3_starrc.spef",
]

TEST_SPEFS = [
    SPEF_DIR / "intel22_nova_f3_starrc.spef",
    SPEF_DIR / "intel22_tv80s_f3_starrc.spef",
    # SPEF_DIR / "intel22_mpeg2_top_f3_starrc.spef",
    #PEF_DIR / "intel22_TinyRocketCore_f3_starrc.spef",
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
NUM_JOBS = 8

RUN_NAME = "ssl_basis_v13_bem"

# StarRC Configuration
# PEX_BIN = 'quantus' # or "StarXtract"
PEX_BIN = 'StarXtract'
PEX_TEMPLATE_PATH = PROJECT_ROOT / "templates" / "run_starrc_template.tcl"
# PEX_TEMPLATE_PATH = PROJECT_ROOT / "templates" / "run_quantus_template.ccl"
# PEX_LICENCES = 48
PEX_LICENCES = 13
KEEP_PEX_FILES = True
GPU_ID = 1

# MODEL
NUM_HEADS = 4
MODEL_DIM = 256
BASIS_LAYERS = 4
CORRECTION_LAYERS = 3

# SSL
SSL_BATCH_SIZE = 2048 # 4 - 650MiB
SSL_LR = 1e-4
SSL_EPOCHS = 500
SSL_W_BC = 10.0
SSL_W_ENERGY = 0.1
SSL_W_FAR = 1.0
SSL_NUM_WORKERS = 64
SSL_RESUME_CHECKPOINT = None # Path to checkpoint to resume from (e.g. "/path/to/bem_ssl_ep10.pth")

# AL
AL_BATCH_SIZE = 4
AL_LR = 1e-4
AL_MAX_BUDGET_RATIO = 0.5
# AL_MIN_ENTROPY_THRESHOLD = 0.0015
AL_MIN_ENTROPY_THRESHOLD = -float('inf')
AL_TRAIN_STEPS_PER_ITER = 5000
AL_FINE_ITERS = 5
AL_BATCH_NETS = 2          # multi-net batches: reduces gradient variance (Codex rec.)
AL_MAX_TILES_PER_BATCH = 256  # 2 nets × 128 tiles max each
AL_NUM_WORKERS = 32
NF_PAD_TO_CUBOIDS = 768  # 768 - N² 메모리 44% 감소 (~186s runtime), 1024 - 650MiB
AL_SAMPLING_METHOD = "Predefined"
AL_PREDEFINED_DESIGNS = [
    'intel22_gcd_f3',
    'intel22_spi_top_f3',
    'intel22_aes_cipher_top_f3',
]
# AL_SAMPLING_METHOD = "SSL"
GNN_TRAIN_STEPS_PER_ITER = 10000

# Output CSV Columns
CSV_COLS = ['sample_filename', 'net_name', 'def_name', 'tile_idx', 'capacitance']

# Maximum unique aggressors tracked per training batch.
# Dense nets can have 1000+ aggressors; this caps label matrix size.
# Memory: A_aggr (B, MAX_AGGR_BUDGET, PAD) @ B=2,PAD=1024,budget=512 -> ~4MB
MAX_AGGR_BUDGET = 512
