# modules/cleaner.py
import argparse
from pathlib import Path
import numpy as np

try:
    import config as CFG
except ImportError:
    CFG = None


def cfg(name, fallback):
    return getattr(CFG, name, fallback) if CFG is not None else fallback


def read_ascii_ply(ply_path: Path):
    with open(ply_path, "r", encoding="utf-8", errors="ignore") as f:
        n_verts = None
        props = []
        in_vertex = False
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Bad PLY header: {ply_path}")
            header_lines.append(line)
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

        if n_verts is None:
            raise ValueError(f"No vertex element: {ply_path}")

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

    # Keep full row for rewriting (preserve all columns)
    if arr.shape[0] == 0:
        xyz = np.empty((0, 3), dtype=np.int32)
    else:
        xyz = np.stack([
            np.rint(arr[:, col["x"]]).astype(np.int32),
            np.rint(arr[:, col["y"]]).astype(np.int32),
            np.rint(arr[:, col["z"]]).astype(np.int32),
        ], axis=1)

    alpha = None
    if "a" in col:
        alpha = np.rint(arr[:, col["a"]]).astype(np.int32)

    intensity = None
    if "intensity" in col:
        intensity = np.rint(arr[:, col["intensity"]]).astype(np.int32)

    return props, arr, xyz, alpha, intensity


def write_ascii_ply(ply_path: Path, props, arr_rows):
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    n = arr_rows.shape[0]
    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        # types: keep as float for xyz + uchar-ish for the rest; TD is fine with this
        # we will write everything as numbers.
        for p in props:
            if p in ("x", "y", "z"):
                f.write(f"property float {p}\n")
            else:
                f.write(f"property float {p}\n")
        f.write("end_header\n")
        for row in arr_rows:
            f.write(" ".join(str(float(v)) for v in row.tolist()) + "\n")


def count_direct_neighbors(points_xyz: np.ndarray) -> np.ndarray:
    idx = {tuple(p): i for i, p in enumerate(points_xyz)}
    neigh6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    counts = np.zeros(len(points_xyz), dtype=np.int32)
    for i, (x, y, z) in enumerate(points_xyz):
        c = 0
        for dx, dy, dz in neigh6:
            if (x + dx, y + dy, z + dz) in idx:
                c += 1
        counts[i] = c
    return counts


def run(image_id: str):
    ply_dir = Path(cfg("PLY_OUT_DIR", "ply"))
    out_dir = Path(cfg("PLY_CLEAN_DIR", "ply_clean"))
    vcount = int(cfg("VCOUNT", 2))

    in_ply = ply_dir / f"{image_id}.ply"
    if not in_ply.exists():
        raise FileNotFoundError(f"Missing {in_ply}. Run modules/scanner.py first.")

    props, arr, xyz, a, intensity = read_ascii_ply(in_ply)

    keep = np.ones(len(xyz), dtype=bool)
    if a is not None:
        keep &= (a > 0)
    elif intensity is not None:
        keep &= (intensity > 0)

    arr2 = arr[keep]
    xyz2 = xyz[keep]

    neighbor_counts = count_direct_neighbors(xyz2)
    keep = neighbor_counts > vcount
    arr3 = arr2[keep]

    out_ply = out_dir / f"{image_id}.ply"
    write_ascii_ply(out_ply, props, arr3)

    print("in:", str(in_ply))
    print("points:", len(xyz2), "kept_points:", int(keep.sum()), "vcount>", vcount)
    print("wrote:", str(out_ply))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    args = ap.parse_args()
    run(str(args.id))


if __name__ == "__main__":
    main()
