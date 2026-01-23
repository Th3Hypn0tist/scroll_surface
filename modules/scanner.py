# modules/scanner.py
import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import tifffile as tiff

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def cfg(name, fallback):
    return getattr(config, name, fallback)


def hysteresis_settings_modified() -> bool:
    default_vmin = float(cfg("VMIN", 74))
    defaults = {
        "HYST_HIGH_MODE": "fixed",
        "HYST_HIGH_FIXED": default_vmin,
        "HYST_LOW_FIXED": None,
        "HYST_LOW_RATIO": 0.75,
        "HYST_CONNECTIVITY": 4,
        "HYST_MAX_ITERS": 2000000,
        "HYST_SLICE_ONLY": True,
    }
    for key, default in defaults.items():
        current = cfg(key, default)
        if isinstance(default, float):
            if current != default:
                return True
        else:
            if current != default:
                return True
    return False


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


def compute_bg_field_2d_with_grid(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def compute_bg_field_2d(img: np.ndarray) -> np.ndarray:
    bg_img, _ = compute_bg_field_2d_with_grid(img)
    return bg_img


def apply_bg_subtract(img: np.ndarray) -> np.ndarray:
    bg_img = compute_bg_field_2d(img)
    corrected = img.astype(np.float32) - bg_img
    if bool(cfg("BG_CLAMP_NEGATIVE", True)):
        corrected = np.maximum(corrected, 0.0)
    return corrected


def correct_slice(img: np.ndarray) -> np.ndarray:
    if bool(cfg("BG_ENABLE", False)):
        return apply_bg_subtract(img)
    return img


def collect_threshold_samples(vol_corr: np.ndarray) -> np.ndarray:
    stride = int(cfg("THRESH_SAMPLE_STRIDE", 8))
    sample = vol_corr[::stride, ::stride, ::stride].ravel()
    return sample[sample > 0]


def compute_threshold_info(samples: np.ndarray) -> tuple[float, int, float | None]:
    mode = str(cfg("THRESH_MODE", "fixed"))
    sample_count = int(samples.size)
    if mode == "fixed":
        return float(cfg("THRESH_FIXED", cfg("VMIN", 74))), sample_count, None

    min_samples = int(cfg("THRESH_MIN_SAMPLES", 10000))
    fallback = float(cfg("THRESH_FALLBACK", cfg("VMIN", 74)))
    if sample_count < min_samples:
        return fallback, sample_count, None

    if mode == "quantile":
        q = cfg("VMIN_Q", None)
        if q is None:
            return fallback, sample_count, None
        return float(np.quantile(samples, float(q))), sample_count, None

    if mode == "mad":
        median = float(np.median(samples))
        mad = float(np.median(np.abs(samples - median)))
        robust_sigma = 1.4826 * mad
        tmin = float(cfg("THRESH_TMIN", 1))
        ksigma = float(cfg("THRESH_KSIGMA", 8.0))
        thresh = max(tmin, median + ksigma * robust_sigma)
        if not np.isfinite(thresh) or robust_sigma == 0.0:
            return fallback, sample_count, robust_sigma
        return float(thresh), sample_count, robust_sigma

    return fallback, sample_count, None


def compute_threshold_from_samples(samples: np.ndarray) -> float:
    return compute_threshold_info(samples)[0]


def hysteresis_mask_2d(
    img_corr: np.ndarray,
    t_high: float,
    t_low: float,
    conn: int,
) -> tuple[np.ndarray, bool, int]:
    m_high = img_corr >= t_high
    if not np.any(m_high):
        return np.zeros_like(m_high, dtype=bool), False, 0
    if t_low >= t_high:
        return m_high, False, 0

    m_low = img_corr >= t_low
    fg = m_high.copy()
    visited = fg.copy()
    q = deque(zip(*np.where(m_high)))
    max_iters = int(cfg("HYST_MAX_ITERS", 2000000))
    if conn == 8:
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    h, w = m_high.shape
    iters = 0
    while q:
        if iters >= max_iters:
            break
        y, x = q.popleft()
        for dy, dx in neighbors:
            ny = y + dy
            nx = x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if visited[ny, nx] or not m_low[ny, nx]:
                continue
            visited[ny, nx] = True
            fg[ny, nx] = True
            q.append((ny, nx))
        iters += 1
    cap_hit = iters >= max_iters and len(q) > 0
    return fg, cap_hit, iters


def apply_support_gate(mask: np.ndarray, required: int) -> np.ndarray:
    if required <= 0:
        return mask
    padded = np.pad(mask, ((1, 1), (1, 1), (1, 1)), mode="constant", constant_values=False)
    count = (
        padded[:-2, 1:-1, 1:-1].astype(np.uint8)
        + padded[2:, 1:-1, 1:-1].astype(np.uint8)
        + padded[1:-1, :-2, 1:-1].astype(np.uint8)
        + padded[1:-1, 2:, 1:-1].astype(np.uint8)
        + padded[1:-1, 1:-1, :-2].astype(np.uint8)
        + padded[1:-1, 1:-1, 2:].astype(np.uint8)
    )
    return mask & (count >= required)


def build_fg_volume(
    fg_volume: np.ndarray,
    step: int,
    fg_support_n: int,
    apply_after_step: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    fg_before_gate = float(fg_volume.mean()) if fg_volume.size else 0.0

    if fg_support_n > 0 and apply_after_step:
        z_idx = np.arange(0, fg_volume.shape[0], step, dtype=np.int32)
        y_idx = np.arange(0, fg_volume.shape[1], step, dtype=np.int32)
        x_idx = np.arange(0, fg_volume.shape[2], step, dtype=np.int32)
        fg_grid = fg_volume[::step, ::step, ::step]
        keep_mask = apply_support_gate(fg_grid, fg_support_n)
        fg_after_gate = float(keep_mask.mean()) if keep_mask.size else 0.0
        zz, yy, xx = np.where(keep_mask)
        zz = z_idx[zz]
        yy = y_idx[yy]
        xx = x_idx[xx]
        return zz, yy, xx, fg_before_gate, fg_after_gate

    if fg_support_n > 0:
        fg_volume = apply_support_gate(fg_volume, fg_support_n)
    fg_after_gate = float(fg_volume.mean()) if fg_volume.size else 0.0
    zz, yy, xx = np.where(fg_volume)
    if step > 1:
        keep = (np.arange(len(xx)) % step) == 0
        zz, yy, xx = zz[keep], yy[keep], xx[keep]
    return zz, yy, xx, fg_before_gate, fg_after_gate


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
            f.write(
                f"{float(xx[i])} {float(yy[i])} {float(zz[i])} "
                f"{int(r[i])} {int(g[i])} {int(b[i])} {int(a[i])} {int(intensity[i])}\n"
            )


def apply_y_flip(yy: np.ndarray, y_size: int) -> np.ndarray:
    if not cfg("FLIP_Y", False):
        return yy
    return (y_size - 1) - yy


def build_intensity(vol: np.ndarray, zz, yy, xx) -> tuple[np.ndarray, str]:
    intensity = vol[zz, yy, xx]
    if np.issubdtype(vol.dtype, np.integer) and vol.max(initial=0) > 255:
        intensity = np.clip(intensity, 0, 65535).astype(np.uint16, copy=False)
        return intensity, "ushort"
    intensity = np.clip(intensity, 0, 255).astype(np.uint8, copy=False)
    return intensity, "uchar"


def build_rgb(intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.clip(intensity, 0, 255).astype(np.uint8, copy=False)
    r = rgb.copy()
    g = rgb.copy()
    b = rgb.copy()
    a = np.full_like(rgb, 255, dtype=np.uint8)
    return r, g, b, a


def write_ply(points: tuple[np.ndarray, np.ndarray, np.ndarray], ply_path: Path, vol: np.ndarray) -> None:
    zz, yy, xx = points
    intensity, intensity_type = build_intensity(vol, zz, yy, xx)
    r, g, b, a = build_rgb(intensity)

    yy = apply_y_flip(yy, vol.shape[1])
    xx, yy, zz, r, g, b, a, intensity = add_missing_corners_if_needed(
        vol.shape,
        xx.astype(np.int32),
        yy.astype(np.int32),
        zz.astype(np.int32),
        r,
        g,
        b,
        a,
        intensity,
    )

    write_ply_xyz_rgba_intensity(ply_path, xx, yy, zz, r, g, b, a, intensity, intensity_type)


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

    bg_grid_stats = None
    if bg_enable:
        vol_corr = np.empty_like(vol, dtype=np.float32)
        if cfg("BG_DEBUG", False):
            bg_grid_stats = {"min": np.inf, "max": -np.inf, "sum": 0.0, "count": 0}
        for z in range(vol.shape[0]):
            if bg_grid_stats is not None:
                bg_img, bg_grid = compute_bg_field_2d_with_grid(vol[z])
            else:
                bg_img = compute_bg_field_2d(vol[z])
                bg_grid = None
            corrected = vol[z].astype(np.float32) - bg_img
            if clamp_negative:
                corrected = np.maximum(corrected, 0.0)
            vol_corr[z] = corrected
            if bg_grid_stats is not None and bg_grid is not None:
                bg_grid_stats["min"] = float(min(bg_grid_stats["min"], float(bg_grid.min())))
                bg_grid_stats["max"] = float(max(bg_grid_stats["max"], float(bg_grid.max())))
                bg_grid_stats["sum"] += float(bg_grid.sum())
                bg_grid_stats["count"] += int(bg_grid.size)
    else:
        vol_corr = vol

    samples = collect_threshold_samples(vol_corr)
    thresh_used, sample_count, robust_sigma = compute_threshold_info(samples)

    mask_mode = str(cfg("MASK_MODE", "single"))
    fg_volume = np.zeros(vol_corr.shape, dtype=bool)

    debug_ply_enable = bool(cfg("DEBUG_PLY_ENABLE", False))
    debug_mode = str(cfg("DEBUG_PLY_MODE", "slice"))
    debug_export = str(cfg("DEBUG_PLY_EXPORT", "final"))
    debug_z = int(cfg("DEBUG_PLY_Z", 0))
    debug_z = max(0, min(debug_z, vol_corr.shape[0] - 1))
    debug_use_emitted = bool(cfg("DEBUG_PLY_USE_EMITTED", True))
    debug_write_meta = bool(cfg("DEBUG_PLY_WRITE_META", True))
    debug_snap_z = bool(cfg("DEBUG_PLY_Z_SNAP_TO_STEP", True))

    need_high_volume = (
        debug_ply_enable
        and mask_mode == "hysteresis"
        and debug_export in {"high", "both"}
        and debug_mode == "full"
    )
    need_high_slice = (
        debug_ply_enable
        and mask_mode == "hysteresis"
        and debug_export in {"high", "both"}
        and debug_mode == "slice"
    )
    high_volume = np.zeros_like(fg_volume) if need_high_volume else None
    high_slice = None
    cap_hit_debug_slice = None
    iters_debug_slice = None

    t_high = None
    t_low = None
    if mask_mode == "hysteresis":
        if str(cfg("HYST_HIGH_MODE", "fixed")) == "fixed":
            t_high = float(cfg("HYST_HIGH_FIXED", thresh_used))
        else:
            t_high = compute_threshold_from_samples(samples)
        t_low_fixed = cfg("HYST_LOW_FIXED", None)
        if t_low_fixed is not None:
            t_low = float(t_low_fixed)
        else:
            t_low = float(cfg("HYST_LOW_RATIO", 0.75)) * t_high
        tmin = float(cfg("THRESH_TMIN", 1))
        t_high = max(tmin, t_high)
        t_low = max(tmin, min(t_low, t_high))
        conn = int(cfg("HYST_CONNECTIVITY", 4))

    debug_enabled = debug_ply_enable or bool(cfg("BG_DEBUG", False))
    if mask_mode != "hysteresis" and debug_enabled and hysteresis_settings_modified():
        print(f"[scanner] NOTE: MASK_MODE='{mask_mode}' => hysteresis settings ignored")

    for z in range(vol_corr.shape[0]):
        img_corr = vol_corr[z]
        if mask_mode == "hysteresis":
            fg_slice, cap_hit, iters = hysteresis_mask_2d(img_corr, t_high, t_low, conn)
            if need_high_volume:
                high_volume[z] = img_corr >= t_high
            if need_high_slice and z == debug_z:
                high_slice = img_corr >= t_high
            if z == debug_z:
                cap_hit_debug_slice = cap_hit
                iters_debug_slice = iters
        else:
            fg_slice = img_corr >= thresh_used
        fg_volume[z] = fg_slice

    fg_support_n = int(cfg("FG_SUPPORT_N", 0))
    apply_after_step = bool(cfg("FG_SUPPORT_APPLY_AFTER_STEP", True))
    zz, yy, xx, fg_before_gate, fg_after_gate = build_fg_volume(
        fg_volume, step, fg_support_n, apply_after_step
    )

    intensity_source = vol_corr if bg_enable else vol
    ply_path = out_dir / f"{image_id}.ply"
    write_ply((zz, yy, xx), ply_path, intensity_source)

    if debug_ply_enable:
        debug_suffix = str(cfg("DEBUG_PLY_SUFFIX", "_debug"))
        debug_paths = []
        debug_z_effective = debug_z
        points_in_debug_slice = None

        def write_debug(points, extra_suffix: str | None):
            if extra_suffix:
                debug_path = out_dir / f"{image_id}{debug_suffix}_{extra_suffix}.ply"
            else:
                debug_path = out_dir / f"{image_id}{debug_suffix}.ply"
            write_ply(points, debug_path, intensity_source)
            debug_paths.append(debug_path)

        def emitted_slice_points(points, requested_z: int) -> tuple[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
            ez, ey, ex = points
            if ez.size == 0:
                return requested_z, (ez, ey, ex)
            emitted_z = np.unique(ez)
            if debug_snap_z and emitted_z.size > 0:
                if apply_after_step and step > 1:
                    target = int(round(requested_z / step) * step)
                else:
                    target = requested_z
                target = max(0, min(target, int(emitted_z.max())))
                idx = int(np.argmin(np.abs(emitted_z - target)))
                effective_z = int(emitted_z[idx])
            else:
                effective_z = requested_z
            mask = ez == effective_z
            return effective_z, (ez[mask], ey[mask], ex[mask])

        if debug_mode == "slice":
            if debug_export in {"high", "both"}:
                if mask_mode == "hysteresis":
                    if high_slice is None:
                        high_slice = vol_corr[debug_z] >= t_high
                    if debug_use_emitted:
                        temp_volume = np.zeros_like(fg_volume)
                        temp_volume[debug_z] = high_slice
                        high_zz, high_yy, high_xx, _, _ = build_fg_volume(
                            temp_volume, step, fg_support_n, apply_after_step
                        )
                        debug_z_effective, (high_zz, high_yy, high_xx) = emitted_slice_points(
                            (high_zz, high_yy, high_xx), debug_z
                        )
                        write_debug(
                            (high_zz, high_yy, high_xx),
                            "high" if debug_export == "both" else None,
                        )
                        if points_in_debug_slice is None:
                            points_in_debug_slice = len(high_zz)
                    else:
                        debug_yy, debug_xx = np.where(high_slice)
                        debug_zz = np.full_like(debug_yy, debug_z)
                        write_debug(
                            (debug_zz, debug_yy, debug_xx),
                            "high" if debug_export == "both" else None,
                        )
                elif debug_export == "high":
                    if debug_use_emitted:
                        debug_z_effective, (debug_zz, debug_yy, debug_xx) = emitted_slice_points(
                            (zz, yy, xx), debug_z
                        )
                        write_debug((debug_zz, debug_yy, debug_xx), None)
                        points_in_debug_slice = len(debug_zz)
                    else:
                        debug_yy, debug_xx = np.where(fg_volume[debug_z])
                        debug_zz = np.full_like(debug_yy, debug_z)
                        write_debug((debug_zz, debug_yy, debug_xx), None)
                        points_in_debug_slice = len(debug_zz)
            if debug_export in {"final", "both"}:
                if debug_use_emitted:
                    debug_z_effective, (debug_zz, debug_yy, debug_xx) = emitted_slice_points(
                        (zz, yy, xx), debug_z
                    )
                    write_debug(
                        (debug_zz, debug_yy, debug_xx),
                        "final" if debug_export == "both" else None,
                    )
                    points_in_debug_slice = len(debug_zz)
                else:
                    debug_yy, debug_xx = np.where(fg_volume[debug_z])
                    debug_zz = np.full_like(debug_yy, debug_z)
                    write_debug(
                        (debug_zz, debug_yy, debug_xx),
                        "final" if debug_export == "both" else None,
                    )
                    points_in_debug_slice = len(debug_zz)
        elif debug_mode == "full":
            if debug_export in {"high", "both"}:
                if mask_mode == "hysteresis":
                    if high_volume is None:
                        high_volume = vol_corr >= t_high
                    if debug_use_emitted:
                        debug_zz, debug_yy, debug_xx, _, _ = build_fg_volume(
                            high_volume, step, fg_support_n, apply_after_step
                        )
                    else:
                        debug_zz, debug_yy, debug_xx = np.where(high_volume)
                    write_debug(
                        (debug_zz, debug_yy, debug_xx),
                        "high" if debug_export == "both" else None,
                    )
                elif debug_export == "high":
                    if debug_use_emitted:
                        write_debug((zz, yy, xx), None)
                    else:
                        debug_zz, debug_yy, debug_xx = np.where(fg_volume)
                        write_debug((debug_zz, debug_yy, debug_xx), None)
            if debug_export in {"final", "both"}:
                if debug_use_emitted:
                    write_debug((zz, yy, xx), "final" if debug_export == "both" else None)
                else:
                    debug_zz, debug_yy, debug_xx = np.where(fg_volume)
                    write_debug((debug_zz, debug_yy, debug_xx), "final" if debug_export == "both" else None)

        if debug_write_meta:
            meta_path = out_dir / f"{image_id}{debug_suffix}.json"
            try:
                meta = {
                    "mask_mode": mask_mode,
                    "bg_enable": bg_enable,
                    "step": step,
                    "fg_support_n": fg_support_n,
                    "apply_after_step": apply_after_step,
                    "thresh_mode": cfg("THRESH_MODE", "fixed"),
                    "thresh_used": thresh_used,
                    "sample_count": sample_count,
                    "robust_sigma": robust_sigma,
                    "debug_mode": debug_mode,
                    "debug_export": debug_export,
                    "debug_z_requested": debug_z,
                    "debug_z_effective": debug_z_effective,
                    "points_emitted_main": len(zz),
                    "points_in_debug_slice": points_in_debug_slice,
                    "fg_before_gate": fg_before_gate,
                    "fg_after_gate": fg_after_gate,
                    "intensity_source_dtype": str(intensity_source.dtype),
                    "vol_corr_dtype": str(vol_corr.dtype),
                }
                if mask_mode == "hysteresis":
                    meta["hysteresis"] = {
                        "t_high": t_high,
                        "t_low": t_low,
                        "conn": cfg("HYST_CONNECTIVITY", 4),
                        "HYST_MAX_ITERS": cfg("HYST_MAX_ITERS", 2000000),
                        "cap_hit_debug_slice": cap_hit_debug_slice,
                        "iters_debug_slice": iters_debug_slice,
                    }
                meta_path.parent.mkdir(parents=True, exist_ok=True)
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            except Exception as exc:
                print(f"debug meta skipped: {exc}")

    if cfg("BG_DEBUG", False):
        debug_path = out_dir / f"{image_id}.scanner_bg.json"
        try:
            pos = vol_corr[vol_corr > 0]
            stats = {
                "p50": float(np.quantile(pos, 0.50)) if pos.size else None,
                "p90": float(np.quantile(pos, 0.90)) if pos.size else None,
                "p99": float(np.quantile(pos, 0.99)) if pos.size else None,
                "max": float(pos.max()) if pos.size else None,
                "fg_fraction": float(np.mean(pos >= thresh_used)) if pos.size else 0.0,
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
                "threshold_mode": cfg("THRESH_MODE", "fixed"),
                "threshold_used": thresh_used,
                "robust_sigma": robust_sigma,
                "sample_count": sample_count,
                "mask_mode": mask_mode,
                "hysteresis": {
                    "t_high": t_high,
                    "t_low": t_low,
                    "connectivity": cfg("HYST_CONNECTIVITY", 4),
                }
                if mask_mode == "hysteresis"
                else None,
                "fg_fraction_before_gate": fg_before_gate,
                "fg_fraction_after_gate": fg_after_gate,
                "fg_support_n": fg_support_n,
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
    if len(zz) > 0:
        print("FG points:", len(zz), "| z_unique:", z_unique, "| z_minmax:", int(zz.min()), int(zz.max()))
    else:
        print("FG points:", len(zz), "| z_unique:", z_unique)
    print("wrote:", str(ply_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    args = ap.parse_args()
    run(str(args.id))


if __name__ == "__main__":
    main()
