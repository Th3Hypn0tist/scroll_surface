# modules/scanner.py
import argparse
from pathlib import Path
import numpy as np
import tifffile as tiff

try:
    import config as CFG
except Exception:
    CFG = None


def cfg(name, fallback):
    return getattr(CFG, name, fallback) if CFG is not None else fallback


def find_image_path(data_dir: Path, split: str, image_id: str) -> Path:
    p = data_dir / f"{split}_images" / f"{image_id}.tif"
    if p.exists():
        return p
    raise FileNotFoundError(f"Image not found: {p}")


def add_missing_corners_if_needed(shape_zyx, xx, yy, zz, r, g, b, a, intensity):
    if not cfg("ADD_MISSING_CORNERS", True):
        return xx, yy, zz, r, g, b, a, intensity

    Z, Y, X = shape_zyx
    corners = [
        (0,   0,   0),
        (X-1, 0,   0),
        (0,   Y-1, 0),
        (X-1, Y-1, 0),
        (0,   0,   Z-1),
        (X-1, 0,   Z-1),
        (0,   Y-1, Z-1),
        (X-1, Y-1, Z-1),
    ]

    have = set(zip(xx.tolist(), yy.tolist(), zz.tolist()))
    add = [(cx, cy, cz) for (cx, cy, cz) in corners if (cx, cy, cz) not in have]
    if not add:
        return xx, yy, zz, r, g, b, a, intensity

    k = len(add)
    ax = np.array([p[0] for p in add], dtype=xx.dtype)
    ay = np.array([p[1] for p in add], dtype=yy.dtype)
    az = np.array([p[2] for p in add], dtype=zz.dtype)

    xx = np.concatenate([xx, ax])
    yy = np.concatenate([yy, ay])
    zz = np.concatenate([zz, az])

    r = np.concatenate([r, np.zeros(k, dtype=np.uint8)])
    g = np.concatenate([g, np.zeros(k, dtype=np.uint8)])
    b = np.concatenate([b, np.zeros(k, dtype=np.uint8)])
    intensity = np.concatenate([intensity, np.zeros(k, dtype=np.uint8)])

    if cfg("CORNER_ALPHA0", True):
        a = np.concatenate([a, np.zeros(k, dtype=np.uint8)])
    else:
        a = np.concatenate([a, np.full(k, 255, dtype=np.uint8)])

    return xx, yy, zz, r, g, b, a, intensity


def write_ply_xyz_rgba_intensity(ply_path: Path, xx, yy, zz, r, g, b, a, intensity):
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(xx)
    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar r\n")
        f.write("property uchar g\n")
        f.write("property uchar b\n")
        f.write("property uchar a\n")
        f.write("property uchar intensity\n")
        f.write("end_header\n")
        # XYZ standard: x=xx, y=yy, z=zz
        for i in range(n):
            f.write(f"{float(xx[i])} {float(yy[i])} {float(zz[i])} {int(r[i])} {int(g[i])} {int(b[i])} {int(a[i])} {int(intensity[i])}\n")


def run(image_id: str):
    data_dir = Path(cfg("DATA_DIR", "."))
    split = cfg("SPLIT", "train")
    vmin = int(cfg("VMIN", 74))
    step = int(cfg("STEP", 3))
    out_dir = Path(cfg("PLY_OUT_DIR", "ply"))
    img_path = find_image_path(data_dir, split, image_id)

    vol = tiff.imread(str(img_path))  # shape (Z,Y,X)
    fg = (vol >= vmin)

    zz, yy, xx = np.where(fg)
    if step > 1:
        # deterministic subsample: keep every step-th point by index
        keep = (np.arange(len(xx)) % step) == 0
        zz, yy, xx = zz[keep], yy[keep], xx[keep]

    # color/intensity: simple default
    intensity = vol[zz, yy, xx].astype(np.uint8, copy=False)
    r = intensity.copy()
    g = intensity.copy()
    b = intensity.copy()
    a = np.full_like(intensity, 255, dtype=np.uint8)

    xx, yy, zz, r, g, b, a, intensity = add_missing_corners_if_needed(vol.shape, xx.astype(np.int32), yy.astype(np.int32), zz.astype(np.int32), r, g, b, a, intensity)

    ply_path = out_dir / f"{image_id}.ply"
    write_ply_xyz_rgba_intensity(ply_path, xx, yy, zz, r, g, b, a, intensity)

    # quick sanity
    z_unique = len(np.unique(zz))
    print("img:", str(img_path))
    print("shape(Z,Y,X):", vol.shape, "dtype:", vol.dtype)
    print("FG points:", len(zz), "| z_unique:", z_unique, "| z_minmax:", int(zz.min()), int(zz.max()))
    print("wrote:", str(ply_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    args = ap.parse_args()
    run(str(args.id))


if __name__ == "__main__":
    main()
