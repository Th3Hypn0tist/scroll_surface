# config.py  (project root)
# Minimal locked config for the 3-stage pipeline.

# --- dataset ---
# Set this to your dataset root; defaults to current directory for portability.
DATA_DIR = "."
SPLIT = "train"            # "train" | "test"

# --- thresholds / sampling ---
VMIN = 74                  # scanner FG threshold: orig >= VMIN
STEP = 3                   # scanner subsample step (1=all, 2/3 recommended)


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
