# modules/scanner.py
import argparse
import json
from pathlib import Path
import numpy as np
import tifffile as tiff

import config as CFG


def cfg(name, fallback):
    return getattr(CFG, name, fallback)


def smooth_histogram(counts: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return counts.astype(np.float32, copy=False)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(counts.astype(np.float32, copy=False), kernel, mode="same")


def bilinear_upsample(grid: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    gh, gw = grid.shape
    h, w = out_shape
    if gh == 1 and gw == 1:
        return np.full((h, w), grid[0, 0], dtype=np.float32)

    ys = np.linspace(0, gh - 1, h, dtype=np.float32)
    xs = np.linspace(0, gw - 1, w, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, gh - 1)
    x1 = np.clip(x0 + 1, 0, gw - 1)
    wy = ys - y0
    wx = xs - x0

    g00 = grid[y0[:, None], x0[None, :]]
    g01 = grid[y0[:, None], x1[None, :]]
    g10 = grid[y1[:, None], x0[None, :]]
    g11 = grid[y1[:, None], x1[None, :]]

    wy = wy[:, None]
    wx = wx[None, :]
    return (
        g00 * (1 - wy) * (1 - wx)
        + g01 * (1 - wy) * wx
        + g10 * wy * (1 - wx)
        + g11 * wy * wx
    ).astype(np.float32, copy=False)


def compute_bg_field(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tile = int(cfg("BG_TILE", 96))
    stride = int(cfg("BG_SAMPLE_STRIDE", 4))
    bins = int(cfg("BG_BINS", 256))
    smooth = int(cfg("BG_SMOOTH_HIST", 5))
    qlow = float(cfg("BG_PEAK_RANGE_QLOW", 0.10))
    qhigh = float(cfg("BG_PEAK_RANGE_QHIGH", 0.90))
    fallback = str(cfg("BG_FALLBACK", "median"))
    field_smooth = int(cfg("BG_FIELD_SMOOTH", 2))

    h, w = img.shape
    grid_h = (h + tile - 1) // tile
    grid_w = (w + tile - 1) // tile
    bg_grid = np.zeros((grid_h, grid_w), dtype=np.float32)

    for gy in range(grid_h):
        for gx in range(grid_w):
            y0 = gy * tile
            x0 = gx * tile
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)
            tile_vals = img[y0:y1, x0:x1][::stride, ::stride].astype(np.float32, copy=False)
            flat = tile_vals.ravel()
            if flat.size == 0:
                bg_grid[gy, gx] = 0.0
                continue

            ql = np.quantile(flat, qlow)
            qh = np.quantile(flat, qhigh)
            if not np.isfinite(ql) or not np.isfinite(qh) or qh <= ql:
                ql = None
                qh = None

            bg_value = None
            if ql is not None:
                clipped = np.clip(flat, ql, qh)
                if clipped.size > 0 and qh > ql:
                    counts, edges = np.histogram(clipped, bins=bins, range=(ql, qh))
                    smooth_counts = smooth_histogram(counts, smooth)
                    if smooth_counts.size > 0 and np.nanmax(smooth_counts) > 0:
                        peak_idx = int(np.nanargmax(smooth_counts))
                        bg_value = 0.5 * (edges[peak_idx] + edges[peak_idx + 1])

            if bg_value is None or not np.isfinite(bg_value):
                if fallback == "p30":
                    bg_value = float(np.quantile(flat, 0.30))
                else:
                    bg_value = float(np.median(flat))

            bg_grid[gy, gx] = bg_value

    for _ in range(max(field_smooth, 0)):
        padded = np.pad(bg_grid, ((1, 1), (1, 1)), mode="edge")
        bg_grid = (
            padded[0:-2, 0:-2]
            + padded[0:-2, 1:-1]
            + padded[0:-2, 2:]
            + padded[1:-1, 0:-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, 0:-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 9.0

    bg_img = bilinear_upsample(bg_grid.astype(np.float32, copy=False), (h, w))
    return bg_img, bg_grid


def compute_threshold(vol_corr: np.ndarray) -> float:
    mode = str(cfg("VMIN_MODE", "fixed"))
    if mode == "fixed":
        return float(cfg("VMIN_FIXED", cfg("VMIN", 74)))

    stride = int(cfg("VMIN_SAMPLE_STRIDE", 6))
    q = float(cfg("VMIN_Q", 0.98))
    fallback = float(cfg("VMIN_FALLBACK", 50))
    sample = vol_corr[::stride, ::stride, ::stride].ravel()
    sample = sample[sample > 0]
    if sample.size == 0:
        return fallback
    return float(np.quantile(sample, q))

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


def write_ply_xyz_rgba_intensity(ply_path: Path, xx, yy, zz, r, g, b, a, intensity, intensity_type: str):
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
        f.write(f"property {intensity_type} intensity\n")
        f.write("end_header\n")
        # XYZ standard: x=xx, y=yy, z=zz
        for i in range(n):
            f.write(f"{float(xx[i])} {float(yy[i])} {float(zz[i])} {int(r[i])} {int(g[i])} {int(b[i])} {int(a[i])} {int(intensity[i])}\n")


def run(image_id: str):
    data_dir = Path(cfg("DATA_DIR", "."))
    split = cfg("SPLIT", "train")
    step = int(cfg("STEP", 3))
    out_dir = Path(cfg("PLY_OUT_DIR", "ply"))
    img_path = find_image_path(data_dir, split, image_id)

    vol = tiff.imread(str(img_path))  # shape (Z,Y,X)
    bg_enable = bool(cfg("BG_ENABLE", False))
    clamp_negative = bool(cfg("BG_CLAMP_NEGATIVE", True))
    bg_mode = str(cfg("BG_MODE", "tile_mode"))
    bg_interp = str(cfg("BG_INTERP", "bilinear"))
    if bg_enable and bg_mode != "tile_mode":
        raise ValueError(f"Unsupported BG_MODE: {bg_mode}")
    if bg_enable and bg_interp != "bilinear":
        raise ValueError(f"Unsupported BG_INTERP: {bg_interp}")

    if bg_enable:
        vol_corr = np.empty_like(vol, dtype=np.float32)
        bg_grid_stats = {"min": np.inf, "max": -np.inf, "sum": 0.0, "count": 0}
        for z in range(vol.shape[0]):
            bg_img, bg_grid = compute_bg_field(vol[z])
            corrected = vol[z].astype(np.float32) - bg_img
            if clamp_negative:
                corrected = np.maximum(corrected, 0.0)
            vol_corr[z] = corrected
            bg_grid_stats["min"] = float(min(bg_grid_stats["min"], float(bg_grid.min())))
            bg_grid_stats["max"] = float(max(bg_grid_stats["max"], float(bg_grid.max())))
            bg_grid_stats["sum"] += float(bg_grid.sum())
            bg_grid_stats["count"] += int(bg_grid.size)
    else:
        vol_corr = vol
        bg_grid_stats = None

    vmin_used = compute_threshold(vol_corr)
    fg = (vol_corr >= vmin_used)

    zz, yy, xx = np.where(fg)
    if step > 1:
        # deterministic subsample: keep every step-th point by index
        keep = (np.arange(len(xx)) % step) == 0
        zz, yy, xx = zz[keep], yy[keep], xx[keep]

    # color/intensity: simple default
    intensity_source = vol_corr if bg_enable else vol
    intensity = intensity_source[zz, yy, xx]
    if np.issubdtype(vol.dtype, np.integer) and vol.max(initial=0) > 255:
        intensity = np.clip(intensity, 0, 65535).astype(np.uint16, copy=False)
        intensity_type = "ushort"
    else:
        intensity = np.clip(intensity, 0, 255).astype(np.uint8, copy=False)
        intensity_type = "uchar"

    rgb = np.clip(intensity, 0, 255).astype(np.uint8, copy=False)
    r = rgb.copy()
    g = rgb.copy()
    b = rgb.copy()
    a = np.full_like(rgb, 255, dtype=np.uint8)

    xx, yy, zz, r, g, b, a, intensity = add_missing_corners_if_needed(vol.shape, xx.astype(np.int32), yy.astype(np.int32), zz.astype(np.int32), r, g, b, a, intensity)

    ply_path = out_dir / f"{image_id}.ply"
    write_ply_xyz_rgba_intensity(ply_path, xx, yy, zz, r, g, b, a, intensity, intensity_type)

    if cfg("BG_DEBUG", False):
        debug_path = out_dir / f"{image_id}.scanner_bg.json"
        stride = int(cfg("VMIN_SAMPLE_STRIDE", 6))
        try:
            sample = vol_corr[::stride, ::stride, ::stride].ravel()
            pos = sample[sample > 0]
            stats = {
                "p50": float(np.quantile(pos, 0.50)) if pos.size else None,
                "p90": float(np.quantile(pos, 0.90)) if pos.size else None,
                "p99": float(np.quantile(pos, 0.99)) if pos.size else None,
                "max": float(pos.max()) if pos.size else None,
                "fg_fraction": float(np.mean(sample >= vmin_used)) if sample.size else 0.0,
            }
            bg_stats = None
            if bg_grid_stats is not None and bg_grid_stats["count"] > 0:
                bg_stats = {
                    "min": bg_grid_stats["min"],
                    "max": bg_grid_stats["max"],
                    "mean": bg_grid_stats["sum"] / bg_grid_stats["count"],
                }
            payload = {
                "image_id": image_id,
                "bg_enable": bg_enable,
                "bg_mode": cfg("BG_MODE", "tile_mode"),
                "bg_params": {
                    "tile": cfg("BG_TILE", 96),
                    "sample_stride": cfg("BG_SAMPLE_STRIDE", 4),
                    "bins": cfg("BG_BINS", 256),
                    "smooth_hist": cfg("BG_SMOOTH_HIST", 5),
                    "peak_range_qlow": cfg("BG_PEAK_RANGE_QLOW", 0.10),
                    "peak_range_qhigh": cfg("BG_PEAK_RANGE_QHIGH", 0.90),
                    "fallback": cfg("BG_FALLBACK", "median"),
                    "field_smooth": cfg("BG_FIELD_SMOOTH", 2),
                    "interp": cfg("BG_INTERP", "bilinear"),
                    "clamp_negative": cfg("BG_CLAMP_NEGATIVE", True),
                },
                "vmin_used": vmin_used,
                "stats": stats,
                "bg_grid": bg_stats,
            }
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"bg debug skipped: {exc}")

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
