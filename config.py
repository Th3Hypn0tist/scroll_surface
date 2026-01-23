# config.py  (project root)
# Minimal locked config for the 3-stage pipeline.

# --- dataset ---
# Set this to your dataset root; defaults to current directory for portability.
DATA_DIR = "d://chal-data///vesuvius"
SPLIT = "train"            # "train" | "test"

# --- thresholds / sampling ---
VMIN = 50                  # legacy scanner FG threshold (kept for compatibility)
STEP = 3                   # scanner subsample step (1=all, 2/3 recommended)

# --- scanner background removal ---
BG_ENABLE = False
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
DEBUG_PLY_ENABLE = False
DEBUG_PLY_MODE = "slice"
DEBUG_PLY_Z = 0
DEBUG_PLY_APPLY_SUPPORT_GATE = False
DEBUG_PLY_SUFFIX = "_debug"

# --- scanner thresholding ---
THRESH_MODE = "fixed"
THRESH_FIXED = VMIN
THRESH_FALLBACK = VMIN
THRESH_TMIN = 1
THRESH_KSIGMA = 8.0
THRESH_SAMPLE_STRIDE = 8
THRESH_MIN_SAMPLES = 10000
FG_SUPPORT_N = 0
FG_SUPPORT_APPLY_AFTER_STEP = True

# Optional quantile threshold (if THRESH_MODE="quantile")
VMIN_Q = 0.98

# --- scanner export ---
FLIP_Y = False


# --- stage1 output ---
PLY_OUT_DIR = "ply"        # scanner writes ply/<id>.ply (no subdirs)

# cleaner.py
PLY_CLEAN_DIR = "ply_clean"

# Remove voxels with this many or fewer direct (6-neighbor) connections.
VCOUNT = 2

# --- stage2 params/output ---
SURF_OUT_DIR = "surf"      # find_outlines writes surf/<id>.ply


# --- stage3 output/params ---
OUT_ENTRY = "out_entry"    # label_output writes out_entry/<id>.tif
BAND = 0                   # optional dilation radius around outline (0..2)

# --- misc ---
ADD_MISSING_CORNERS = True # add bbox corners only if missing (rgba=0000, intensity=0)
CORNER_ALPHA0 = True       # corner anchors have a=0 (so they can be ignored later if needed)
DTYPE_MASK = "uint8"       # output label dtype (competition accepts uint8)
