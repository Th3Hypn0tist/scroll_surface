# modules/label_output.py  (UPDATED: reads surf/<id>.ply)
import argparse
import sys
from pathlib import Path
import numpy as np
import tifffile as tiff

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def cfg(name, fallback):
    return getattr(config, name, fallback)


def find_image_path(data_dir: Path, split: str, image_id: str) -> Path:
    p = data_dir / f"{split}_images" / f"{image_id}.tif"
    if p.exists():
        return p
    raise FileNotFoundError(f"Image not found: {p}")


def read_ascii_ply_xyz(ply_path: Path):
    with open(ply_path, "r", encoding="utf-8", errors="ignore") as f:
        n_verts = None
        props = []
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Bad PLY header: {ply_path}")
            s = line.strip()
            if s.startswith("element vertex"):
                n_verts = int(s.split()[-1])
                in_vertex = True
            elif s.startswith("element ") and not s.startswith("element vertex"):
                in_vertex = False
            elif s.startswith("property") and in_vertex:
                props.append(s.split()[-1])
            elif s == "end_header":
                break

        rows = []
        for _ in range(n_verts):
            rows.append(f.readline().strip().split())

    arr = np.array(rows, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(0, len(props))
    col = {p: i for i, p in enumerate(props) if i < arr.shape[1]}
    for k in ("x", "y", "z"):
        if k not in col:
            raise ValueError(f"PLY missing {k}: props={props}")

    if arr.shape[0] == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    xx = np.rint(arr[:, col["x"]]).astype(np.int32)
    yy = np.rint(arr[:, col["y"]]).astype(np.int32)
    zz = np.rint(arr[:, col["z"]]).astype(np.int32)
    return xx, yy, zz


def dilate_cube(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask
    out = mask.copy()
    for dz in range(-r, r + 1):
        tmpz = np.roll(mask, dz, axis=0)
        for dy in range(-r, r + 1):
            tmpy = np.roll(tmpz, dy, axis=1)
            for dx in range(-r, r + 1):
                out |= np.roll(tmpy, dx, axis=2)
    return out


def run(image_id: str):
    data_dir = Path(cfg("DATA_DIR", "."))
    split = cfg("SPLIT", "train")
    surf_dir = Path(cfg("SURF_OUT_DIR", "surf"))
    out_entry = Path(cfg("OUT_ENTRY", "out_entry"))
    band = int(cfg("BAND", 0))
    dtype = np.dtype(cfg("DTYPE_MASK", "uint8"))

    img_path = find_image_path(data_dir, split, image_id)
    vol = tiff.imread(str(img_path))
    Z, Y, X = vol.shape

    ply_path = surf_dir / f"{image_id}.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"Missing {ply_path}. Run modules/find_outlines.py first.")

    xx, yy, zz = read_ascii_ply_xyz(ply_path)

    mask = np.zeros((Z, Y, X), dtype=bool)
    m = (zz >= 0) & (zz < Z) & (yy >= 0) & (yy < Y) & (xx >= 0) & (xx < X)
    mask[zz[m], yy[m], xx[m]] = True

    if band > 0:
        mask = dilate_cube(mask, band)

    out_entry.mkdir(parents=True, exist_ok=True)
    out_tif = out_entry / f"{image_id}.tif"
    tiff.imwrite(str(out_tif), mask.astype(dtype))

    fg = int(mask.sum())
    print("img:", str(img_path))
    print("shape:", vol.shape, "dtype_out:", str(dtype), "band:", band)
    print("fg_voxels:", fg, "ratio:", fg / mask.size)
    print("wrote:", str(out_tif))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    args = ap.parse_args()
    run(str(args.id))


if __name__ == "__main__":
    main()
