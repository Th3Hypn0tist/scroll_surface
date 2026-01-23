# config.py  (project root)
# Minimal locked config for the 3-stage pipeline.

# --- dataset ---
# Set this to your dataset root; defaults to current directory for portability.

DATA_DIR = "d://chal-data///vesuvius"
PLY_OUT_DIR = "ply"        
PLY_CLEAN_DIR = "ply_clean"
SURF_OUT_DIR = "surf"     
OUT_ENTRY = "out_entry"


SPLIT = "test"            # "train" | "test"

FLIP_Y = True

# --- thresholds / sampling ---
VMIN = 74                 # legacy scanner FG threshold (kept for compatibility)
STEP = 3                   # scanner subsample step (1=all, 2/3 recommended)

# --- scanner background removal ---
BG_ENABLE = True
BG_MODE = "tile_mode"
BG_TILE = 96
BG_SAMPLE_STRIDE = 4
BG_BINS = 256
BG_SMOOTH_HIST = 5
BG_PEAK_RANGE_QLOW = 0.10
BG_PEAK_RANGE_QHIGH = 0.90
BG_FALLBACK = "median"
BG_FIELD_SMOOTH = 2
BG_INTERP = "bilinear"
BG_CLAMP_NEGATIVE = True
BG_DEBUG = False

# --- scanner debug ply ---
DEBUG_PLY_ENABLE = True
DEBUG_PLY_MODE = "slice"          # "slice" | "full"
DEBUG_PLY_Z = 0
DEBUG_PLY_EXPORT = "final"         # "high" | "final" | "both"
DEBUG_PLY_APPLY_SUPPORT_GATE = False
DEBUG_PLY_SUFFIX = "_debug"
DEBUG_PLY_USE_EMITTED = True
DEBUG_PLY_WRITE_META = True
DEBUG_PLY_Z_SNAP_TO_STEP = True
DEBUG_PLY_RAW_SLICE = True
DEBUG_PLY_RAW_STEP  = 2  # 1 = every pixel (big), 2/3 usually enough

# --- scanner thresholding ---
MASK_MODE = "hysteresis"               # "single" | "hysteresis"
THRESH_MODE = "fixed"              # "fixed" | "quantile" | "mad"
VMIN_Q = 0.98 # Optional quantile threshold (if THRESH_MODE="quantile")
THRESH_FIXED = VMIN
THRESH_FALLBACK = VMIN
THRESH_TMIN = 1
THRESH_KSIGMA = 8.0
THRESH_SAMPLE_STRIDE = 8
THRESH_MIN_SAMPLES = 10000

# Hysteresis-specific thresholding
HYST_HIGH_MODE = "fixed"           # "fixed" | "auto"
HYST_HIGH_FIXED = VMIN
HYST_LOW_FIXED = None
HYST_LOW_RATIO = 0.25
HYST_CONNECTIVITY = 1
HYST_MAX_ITERS = 2000000
HYST_SLICE_ONLY = True
# hysteresis: seed capture for thin filaments
HYST_SEED_RATIO = 0.95   # 1.0 = legacy; try 0.97..0.90 to catch ultra-thin strands
HYST_MICRO_RESCUE_ENABLE = True 
HYST_MICRO_RESCUE_EPS = 5.0
HYST_MICRO_RESCUE_NEI_MIN= 1


FG_SUPPORT_N = 0
FG_SUPPORT_APPLY_AFTER_STEP = True








VCOUNT = 2



BAND = 0                   # optional dilation radius around outline (0..2)

# --- misc ---
ADD_MISSING_CORNERS = True # add bbox corners only if missing (rgba=0000, intensity=0)
CORNER_ALPHA0 = True       # corner anchors have a=0 (so they can be ignored later if needed)
DTYPE_MASK = "uint8"       # output label dtype (competition accepts uint8)
