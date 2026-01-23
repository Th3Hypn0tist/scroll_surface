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
    # Best-effort heuristic: if user set any common hysteresis knobs away from typical defaults
    # while MASK_MODE isn't hysteresis, warn (only under debug).
    defaults = {
        "HYST_HIGH_MODE": "fixed",
        "HYST_HIGH_FIXED": None,
        "HYST_LOW_FIXED": None,
        "HYST_LOW_RATIO": 0.75,
        "HYST_CONNECTIVITY": 4,
        "HYST_MAX_ITERS": 2000000,
        "HYST_SEED_RATIO": 1.0,
    }
    for k, v in defaults.items():
        cur = getattr(config, k, v)
        if v is None:
            if cur is not None:
                return True
        else:
            if cur != v:
                return True
    return False


def find_image_path(data_dir: Path, split: str, image_id: str) -> Path:
    # Tries common layouts; do not delete or modify input dirs.
    # Expected: data/<split>/<id>.tif OR data/<split>/images/<id>.tif OR data/<id>.tif
    candidates = [
	    data_dir / f"{split}_images" / f"{image_id}.tif",  # <-- fixed join
        data_dir / split / f"{image_id}.tif",
        data_dir / f"{image_id}.tif",
        data_dir / "images" / f"{image_id}.tif",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find TIFF for id='{image_id}' under data_dir='{data_dir}' split='{split}'")


def compute_bg_field_2d(img: np.ndarray) -> np.ndarray:
    """
    Robust 2D background estimate via tile-mode histogram peak.
    Returns float32 bg field same shape as img.
    """
    tile = int(cfg("BG_TILE", 64))
    tile = max(8, tile)
    stride = int(cfg("BG_SAMPLE_STRIDE", 2))
    stride = max(1, stride)
    bins = int(cfg("BG_BINS", 256))
    bins = max(32, bins)

    smooth_hist = int(cfg("BG_SMOOTH_HIST", 3))
    smooth_hist = max(0, smooth_hist)

    qlow = float(cfg("BG_PEAK_RANGE_QLOW", 0.02))
    qhigh = float(cfg("BG_PEAK_RANGE_QHIGH", 0.60))
    qlow = max(0.0, min(qlow, 1.0))
    qhigh = max(0.0, min(qhigh, 1.0))
    if qhigh < qlow:
        qlow, qhigh = qhigh, qlow

    fallback = str(cfg("BG_FALLBACK", "median"))

    h, w = img.shape
    img_f = img.astype(np.float32, copy=False)

    # Tile grid
    th = (h + tile - 1) // tile
    tw = (w + tile - 1) // tile
    bg_tiles = np.zeros((th, tw), dtype=np.float32)

    for ty in range(th):
        y0 = ty * tile
        y1 = min((ty + 1) * tile, h)
        for tx in range(tw):
            x0 = tx * tile
            x1 = min((tx + 1) * tile, w)
            patch = img_f[y0:y1:stride, x0:x1:stride]
            if patch.size < 64:
                # too small -> fallback
                if fallback == "p30":
                    bg_tiles[ty, tx] = float(np.percentile(patch, 30)) if patch.size else 0.0
                else:
                    bg_tiles[ty, tx] = float(np.median(patch)) if patch.size else 0.0
                continue

            lo = float(np.quantile(patch, qlow))
            hi = float(np.quantile(patch, qhigh))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                if fallback == "p30":
                    bg_tiles[ty, tx] = float(np.percentile(patch, 30))
                else:
                    bg_tiles[ty, tx] = float(np.median(patch))
                continue

            hist, edges = np.histogram(patch, bins=bins, range=(lo, hi))
            if smooth_hist > 0 and hist.size > 3:
                k = smooth_hist
                ker = np.ones((2 * k + 1,), dtype=np.float32) / float(2 * k + 1)
                hist = np.convolve(hist.astype(np.float32), ker, mode="same")

            peak = int(np.argmax(hist))
            # Bin center
            bg_tiles[ty, tx] = float((edges[peak] + edges[peak + 1]) * 0.5)

    # Smooth tile field
    field_smooth = int(cfg("BG_FIELD_SMOOTH", 1))
    field_smooth = max(0, field_smooth)
    if field_smooth > 0:
        k = field_smooth
        # simple separable box blur
        tmp = bg_tiles.copy()
        # horizontal
        for ty in range(th):
            row = tmp[ty]
            pad = np.pad(row, (k, k), mode="edge")
            out = np.convolve(pad, np.ones(2 * k + 1, dtype=np.float32), mode="valid") / float(2 * k + 1)
            bg_tiles[ty] = out.astype(np.float32, copy=False)
        # vertical
        tmp = bg_tiles.copy()
        for tx in range(tw):
            col = tmp[:, tx]
            pad = np.pad(col, (k, k), mode="edge")
            out = np.convolve(pad, np.ones(2 * k + 1, dtype=np.float32), mode="valid") / float(2 * k + 1)
            bg_tiles[:, tx] = out.astype(np.float32, copy=False)

    # Upsample to full res (nearest + bilinear-ish via simple repeat then smooth)
    bg = np.repeat(np.repeat(bg_tiles, tile, axis=0), tile, axis=1)[:h, :w].astype(np.float32, copy=False)
    return bg


def apply_bg_subtract(img: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32, copy=False)
    bg = compute_bg_field_2d(img)
    corr = img_f - bg
    if bool(cfg("BG_CLAMP_NEGATIVE", True)):
        corr = np.maximum(corr, 0.0, out=corr)
    return corr.astype(np.float32, copy=False)


def compute_threshold_info(samples: np.ndarray) -> tuple[float, int, float | None]:
    mode = str(cfg("THRESH_MODE", "fixed"))
    sample_count = int(samples.size)
    if mode == "fixed":
        ret = float(cfg("THRESH_FIXED", cfg("VMIN", 74)))
        return ret, sample_count, None

    fallback = float(cfg("THRESH_FALLBACK", cfg("THRESH_FIXED", cfg("VMIN", 74))))
    min_samples = int(cfg("THRESH_MIN_SAMPLES", 5000))
    if sample_count < min_samples:
        return fallback, sample_count, None

    samples = samples.astype(np.float32, copy=False)
    samples = samples[np.isfinite(samples)]
    if samples.size < min_samples:
        return fallback, int(samples.size), None

    tmin = float(cfg("THRESH_TMIN", 1))
    if mode == "quantile":
        q = float(cfg("THRESH_Q", 0.995))
        q = max(0.0, min(q, 1.0))
        thresh = float(np.quantile(samples, q))
        if not np.isfinite(thresh):
            return fallback, int(samples.size), None
        return max(tmin, thresh), int(samples.size), None

    if mode == "mad":
        median = float(np.median(samples))
        mad = float(np.median(np.abs(samples - median)))
        robust_sigma = 1.4826 * mad
        ksigma = float(cfg("THRESH_KSIGMA", 8.0))
        thresh = max(tmin, median + ksigma * robust_sigma)
        if not np.isfinite(thresh) or robust_sigma == 0.0:
            return fallback, int(samples.size), robust_sigma
        return float(thresh), int(samples.size), robust_sigma

    # unknown mode
    return fallback, int(samples.size), None


def hysteresis_mask_2d(
    img_corr: np.ndarray,
    t_high: float,
    t_low: float,
    conn: int,
) -> tuple[np.ndarray, bool, int]:
    # Seed threshold can be slightly lower than t_high to catch ultra-thin filaments.
    seed_ratio = float(cfg("HYST_SEED_RATIO", 1.0))
    seed_ratio = 1.0 if (not np.isfinite(seed_ratio)) else seed_ratio
    seed_ratio = max(0.0, min(seed_ratio, 1.0))
    t_seed = t_high * seed_ratio
    m_high = img_corr >= t_seed
    if not np.any(m_high):
        return np.zeros_like(m_high, dtype=bool), False, 0
    if t_low >= t_high:
        return m_high, False, 0

    m_low = img_corr >= t_low
    fg = m_high.copy()
    visited = fg.copy()
    q = deque(zip(*np.where(m_high)))

    max_iters = int(cfg("HYST_MAX_ITERS", 2000000))
    if conn in (8, 26, 2):
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

    while q and iters < max_iters:
        y, x = q.popleft()
        iters += 1
        for dy, dx in neighbors:
            ny = y + dy
            nx = x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if visited[ny, nx]:
                continue
            visited[ny, nx] = True
            if not m_low[ny, nx]:
                continue
            fg[ny, nx] = True
            q.append((ny, nx))

    cap_hit = bool(q) and iters >= max_iters
    return fg, cap_hit, iters


def build_fg_volume(
    fg_volume: np.ndarray,
    step: int,
    fg_support_n: int,
    apply_after_step: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Builds emitted coordinates from FG volume. Optionally apply 6-neighbor support gate.
    Returns (zz,yy,xx, fg_before_gate, fg_after_gate)
    """
    step = max(1, int(step))
    fg_before_gate = int(np.count_nonzero(fg_volume))

    if fg_support_n > 0:
        if apply_after_step and step > 1:
            # gate after stepping
            fg_grid = fg_volume[::step, ::step, ::step].copy()
        else:
            fg_grid = fg_volume.copy()

        # 6-neighbor count
        c = np.zeros_like(fg_grid, dtype=np.uint8)
        c[1:, :, :] += fg_grid[:-1, :, :]
        c[:-1, :, :] += fg_grid[1:, :, :]
        c[:, 1:, :] += fg_grid[:, :-1, :]
        c[:, :-1, :] += fg_grid[:, 1:, :]
        c[:, :, 1:] += fg_grid[:, :, :-1]
        c[:, :, :-1] += fg_grid[:, :, 1:]
        fg_grid = fg_grid & (c >= fg_support_n)

        if apply_after_step and step > 1:
            # expand back to original grid for coordinate extraction
            fg_volume = np.zeros_like(fg_volume, dtype=bool)
            fg_volume[::step, ::step, ::step] = fg_grid
        else:
            fg_volume = fg_grid

    fg_after_gate = int(np.count_nonzero(fg_volume))

    zz, yy, xx = np.where(fg_volume)
    if step > 1:
        zz = (zz * 1).astype(np.int32)
        yy = (yy * 1).astype(np.int32)
        xx = (xx * 1).astype(np.int32)

    return zz.astype(np.int32), yy.astype(np.int32), xx.astype(np.int32), fg_before_gate, fg_after_gate


def write_ply(points: tuple[np.ndarray, np.ndarray, np.ndarray], path: Path, intensity_source: np.ndarray):
    zz, yy, xx = points
    zdim, ydim, xdim = intensity_source.shape
    flip_y = bool(cfg("FLIP_Y", False))

    # Build colors from intensity (uchar)
    if zz.size:
        intens = intensity_source[zz, yy, xx].astype(np.float32, copy=False)
        intens = np.clip(intens, 0, 255).astype(np.uint8, copy=False)
    else:
        intens = np.zeros((0,), dtype=np.uint8)

    if flip_y and yy.size:
        yy_out = (ydim - 1) - yy
    else:
        yy_out = yy

    # ASCII PLY
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {int(xx.size)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar r\n")
        f.write("property uchar g\n")
        f.write("property uchar b\n")
        f.write("property uchar a\n")
        f.write("property uchar intensity\n")
        f.write("end_header\n")
        for i in range(xx.size):
            v = int(intens[i])
            f.write(f"{float(xx[i])} {float(yy_out[i])} {float(zz[i])} {v} {v} {v} 255 {v}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="Image/stack id (filename stem)")
    args = ap.parse_args()

    image_id = str(args.id)

    data_dir = Path(cfg("DATA_DIR", "data"))
    split = str(cfg("SPLIT", "train"))
    step = int(cfg("STEP", 3))
    out_dir = Path(cfg("PLY_OUT_DIR", "ply"))
    img_path = find_image_path(data_dir, split, image_id)

    out_dir.mkdir(parents=True, exist_ok=True)

    vol = tiff.imread(str(img_path))
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D TIFF stack, got shape={vol.shape}")

    bg_enable = bool(cfg("BG_ENABLE", False))
    vol_corr = np.empty_like(vol, dtype=np.float32)

    if bg_enable:
        for z in range(vol.shape[0]):
            vol_corr[z] = apply_bg_subtract(vol[z])
    else:
        vol_corr = vol.astype(np.float32, copy=False)

    # Threshold sampling
    thresh_stride = int(cfg("THRESH_SAMPLE_STRIDE", 2))
    thresh_stride = max(1, thresh_stride)
    samples = vol_corr[::thresh_stride, ::thresh_stride, ::thresh_stride].reshape(-1)
    thresh_used, sample_count, robust_sigma = compute_threshold_info(samples)

    # Mask settings
    mask_mode = str(cfg("MASK_MODE", "single"))
    debug_ply_enable = bool(cfg("DEBUG_PLY_ENABLE", False))
    bg_debug = bool(cfg("BG_DEBUG", False))
    if (debug_ply_enable or bg_debug) and mask_mode != "hysteresis" and hysteresis_settings_modified():
        print(f"[scanner] NOTE: MASK_MODE='{mask_mode}' => hysteresis settings ignored")

    fg_volume = np.zeros_like(vol_corr, dtype=bool)
    high_volume = None
    high_slice = None

    debug_mode = str(cfg("DEBUG_PLY_MODE", "slice"))
    debug_export = str(cfg("DEBUG_PLY_EXPORT", "final"))
    debug_z = int(cfg("DEBUG_PLY_Z", 0))
    debug_z = max(0, min(debug_z, vol_corr.shape[0] - 1))

    debug_use_emitted = bool(cfg("DEBUG_PLY_USE_EMITTED", True))
    debug_write_meta = bool(cfg("DEBUG_PLY_WRITE_META", True))
    debug_snap_z = bool(cfg("DEBUG_PLY_Z_SNAP_TO_STEP", True))
    debug_raw_slice = bool(cfg("DEBUG_PLY_RAW_SLICE", False))
    debug_raw_step = int(cfg("DEBUG_PLY_RAW_STEP", 2))
    debug_raw_step = max(1, debug_raw_step)

    # Hysteresis thresholds
    t_high = None
    t_low = None
    conn = 4
    need_high_volume = bool(debug_ply_enable and debug_mode == "full" and debug_export in {"high", "both"})
    need_high_slice = bool(debug_ply_enable and debug_mode == "slice" and debug_export in {"high", "both"})

    cap_hit_debug = False
    iters_debug = 0

    if mask_mode == "hysteresis":
        high_mode = str(cfg("HYST_HIGH_MODE", "fixed"))
        if high_mode == "auto":
            t_high = float(thresh_used)
        else:
            t_high = float(cfg("HYST_HIGH_FIXED", cfg("VMIN", 74)))

        low_fixed = cfg("HYST_LOW_FIXED", None)
        if low_fixed is not None:
            t_low = float(low_fixed)
        else:
            low_ratio = float(cfg("HYST_LOW_RATIO", 0.75))
            low_ratio = 0.75 if not np.isfinite(low_ratio) else low_ratio
            t_low = float(max(float(cfg("THRESH_TMIN", 1)), low_ratio * t_high))

        conn = int(cfg("HYST_CONNECTIVITY", 4))

        if need_high_volume:
            high_volume = np.zeros_like(fg_volume, dtype=bool)
    else:
        # single threshold
        t = float(thresh_used)
        for z in range(vol_corr.shape[0]):
            fg_volume[z] = vol_corr[z] >= t

    if mask_mode == "hysteresis":
        for z in range(vol_corr.shape[0]):
            img_corr = vol_corr[z]
            fg_slice, cap_hit, iters = hysteresis_mask_2d(img_corr, t_high, t_low, conn)

            # Micro-rescue: add ultra-thin bright pixels adjacent to FG (helps tiny strands without bringing bg back).
            if bool(cfg("HYST_MICRO_RESCUE_ENABLE", False)):
                eps = float(cfg("HYST_MICRO_RESCUE_EPS", 5.0))
                eps = 0.0 if (not np.isfinite(eps)) else eps
                t_rescue = float(t_high) - eps
                cand = (img_corr >= t_rescue) & (~fg_slice)
                if np.any(cand):
                    neigh = np.zeros_like(fg_slice, dtype=np.uint8)
                    fg = fg_slice
                    neigh[1:, 1:] += fg[:-1, :-1]
                    neigh[1:, :] += fg[:-1, :]
                    neigh[1:, :-1] += fg[:-1, 1:]
                    neigh[:, 1:] += fg[:, :-1]
                    neigh[:, :-1] += fg[:, 1:]
                    neigh[:-1, 1:] += fg[1:, :-1]
                    neigh[:-1, :] += fg[1:, :]
                    neigh[:-1, :-1] += fg[1:, 1:]
                    nei_min = int(cfg("HYST_MICRO_RESCUE_NEI_MIN", 1))
                    nei_min = max(1, nei_min)
                    fg_slice = fg_slice | (cand & (neigh >= nei_min))

            if need_high_volume:
                high_volume[z] = img_corr >= t_high
            if need_high_slice and z == debug_z:
                high_slice = img_corr >= t_high

            if z == debug_z:
                cap_hit_debug = cap_hit
                iters_debug = iters

            fg_volume[z] = fg_slice

    # Optional support gate on 3D emitted voxels
    fg_support_n = int(cfg("FG_SUPPORT_N", 0))
    apply_after_step = bool(cfg("FG_SUPPORT_APPLY_AFTER_STEP", True))
    zz, yy, xx, fg_before_gate, fg_after_gate = build_fg_volume(fg_volume, step, fg_support_n, apply_after_step)

    intensity_source = vol_corr if bg_enable else vol
    ply_path = out_dir / f"{image_id}.ply"
    write_ply((zz, yy, xx), ply_path, intensity_source)

    # Debug PLY(s)
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

        def write_raw_slice_ply(z_idx: int):
            # Writes ALL pixels of one slice as points (subsampled) to visualize true intensity field.
            z_idx = int(max(0, min(z_idx, intensity_source.shape[0] - 1)))
            img2d = intensity_source[z_idx]
            yy2, xx2 = np.mgrid[0:img2d.shape[0]:debug_raw_step, 0:img2d.shape[1]:debug_raw_step]
            yy2 = yy2.reshape(-1).astype(np.int32)
            xx2 = xx2.reshape(-1).astype(np.int32)
            zz2 = np.full_like(yy2, z_idx, dtype=np.int32)
            raw_path = out_dir / f"{image_id}{debug_suffix}_raw.ply"
            write_ply((zz2, yy2, xx2), raw_path, intensity_source)
            debug_paths.append(raw_path)

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
            if debug_raw_slice:
                write_raw_slice_ply(debug_z)

            if debug_export in {"high", "both"}:
                if mask_mode == "hysteresis":
                    if high_slice is None:
                        high_slice = vol_corr[debug_z] >= float(t_high)
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
                            points_in_debug_slice = int(high_zz.size)
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
                        points_in_debug_slice = int(debug_zz.size)
                    else:
                        debug_yy, debug_xx = np.where(fg_volume[debug_z])
                        debug_zz = np.full_like(debug_yy, debug_z)
                        write_debug((debug_zz, debug_yy, debug_xx), None)
                        points_in_debug_slice = int(debug_zz.size)

            if debug_export in {"final", "both"}:
                if debug_use_emitted:
                    debug_z_effective, (debug_zz, debug_yy, debug_xx) = emitted_slice_points(
                        (zz, yy, xx), debug_z
                    )
                    write_debug(
                        (debug_zz, debug_yy, debug_xx),
                        "final" if debug_export == "both" else None,
                    )
                    points_in_debug_slice = int(debug_zz.size)
                else:
                    debug_yy, debug_xx = np.where(fg_volume[debug_z])
                    debug_zz = np.full_like(debug_yy, debug_z)
                    write_debug(
                        (debug_zz, debug_yy, debug_xx),
                        "final" if debug_export == "both" else None,
                    )
                    points_in_debug_slice = int(debug_zz.size)

        elif debug_mode == "full":
            if debug_export in {"high", "both"}:
                if mask_mode == "hysteresis":
                    if high_volume is None:
                        high_volume = vol_corr >= float(t_high)
                    if debug_use_emitted:
                        debug_zz, debug_yy, debug_xx, _, _ = build_fg_volume(
                            high_volume, step, fg_support_n, apply_after_step
                        )
                        write_debug(
                            (debug_zz, debug_yy, debug_xx),
                            "high" if debug_export == "both" else None,
                        )
                    else:
                        debug_zz, debug_yy, debug_xx = np.where(high_volume)
                        write_debug(
                            (debug_zz.astype(np.int32), debug_yy.astype(np.int32), debug_xx.astype(np.int32)),
                            "high" if debug_export == "both" else None,
                        )
            if debug_export in {"final", "both"}:
                write_debug(
                    (zz, yy, xx),
                    "final" if debug_export == "both" else None,
                )

        # Meta JSON
        if debug_write_meta:
            try:
                meta = {
                    "image_id": image_id,
                    "mask_mode": mask_mode,
                    "bg_enable": bg_enable,
                    "step": int(step),
                    "fg_support_n": int(fg_support_n),
                    "apply_after_step": bool(apply_after_step),
                    "thresh_mode": str(cfg("THRESH_MODE", "fixed")),
                    "thresh_used": float(thresh_used),
                    "sample_count": int(sample_count),
                    "robust_sigma": None if robust_sigma is None else float(robust_sigma),
                    "debug_mode": debug_mode,
                    "debug_export": debug_export,
                    "debug_z_requested": int(debug_z),
                    "debug_z_effective": int(debug_z_effective),
                    "points_emitted_main": int(zz.size),
                    "points_in_debug_slice": None if points_in_debug_slice is None else int(points_in_debug_slice),
                    "fg_before_gate": int(fg_before_gate),
                    "fg_after_gate": int(fg_after_gate),
                    "debug_paths": [str(p) for p in debug_paths],
                }
                if mask_mode == "hysteresis":
                    meta.update(
                        {
                            "t_high": float(t_high),
                            "t_low": float(t_low),
                            "hyst_conn": int(conn),
                            "hyst_max_iters": int(cfg("HYST_MAX_ITERS", 2000000)),
                            "cap_hit_debug_slice": bool(cap_hit_debug),
                            "iters_debug_slice": int(iters_debug),
                            "hyst_seed_ratio": float(cfg("HYST_SEED_RATIO", 1.0)),
                            "micro_rescue": {
                                "enable": bool(cfg("HYST_MICRO_RESCUE_ENABLE", False)),
                                "eps": float(cfg("HYST_MICRO_RESCUE_EPS", 5.0)),
                                "nei_min": int(cfg("HYST_MICRO_RESCUE_NEI_MIN", 1)),
                            },
                        }
                    )
                meta_path = out_dir / f"{image_id}{debug_suffix}.json"
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, sort_keys=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
