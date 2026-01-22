# config.py  (project root)
# Minimal locked config for the 3-stage pipeline.

# --- dataset ---
DATA_DIR = r"D:\chal-data\vesuvius"
SPLIT = "train"            # "train" | "test"

# --- thresholds / sampling ---
VMIN = 74                  # scanner FG threshold: orig >= VMIN
STEP = 3                   # scanner subsample step (1=all, 2/3 recommended)


# --- stage1 output ---
PLY_OUT_DIR = "ply"        # scanner writes ply/<id>.ply (no subdirs)

#cleaner.py
PLY_CLEAN_DIR = "ply_clean"

DMAX   = 1
VCOUNT = 2000

BAND = 0


# --- stage2 params/output ---
SURF_OUT_DIR = "surf"      # find_outlines writes surf/<id>.npz


# --- stage3 output/params ---
OUT_ENTRY = "out_entry"    # label_output writes out_entry/<id>.tif
BAND = 0                   # optional dilation radius around outline (0..2)

# --- misc ---
ADD_MISSING_CORNERS = True # add bbox corners only if missing (rgba=0000, intensity=0)
CORNER_ALPHA0 = True       # corner anchors have a=0 (so they can be ignored later if needed)
DTYPE_MASK = "uint8"       # output label dtype (competition accepts uint8)
# config.py — add/ensure these exist

