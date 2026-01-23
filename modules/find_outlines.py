# modules/find_outlines.py  (UPDATED: outputs surf/<id>.ply)
import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def cfg(name, fallback):
    return getattr(config, name, fallback)


def read_ascii_ply(ply_path: Path):
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
        xyz = np.empty((0, 3), dtype=np.int32)
    else:
        xyz = np.stack([
            np.rint(arr[:, col["x"]]).astype(np.int32),
            np.rint(arr[:, col["y"]]).astype(np.int32),
            np.rint(arr[:, col["z"]]).astype(np.int32),
        ], axis=1)

    return props, arr, xyz


def write_ascii_ply(ply_path: Path, props, arr_rows):
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    n = arr_rows.shape[0]
    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        for p in props:
            if p in ("x", "y", "z"):
                f.write(f"property float {p}\n")
            else:
                f.write(f"property float {p}\n")
        f.write("end_header\n")
        for row in arr_rows:
            f.write(" ".join(str(float(v)) for v in row.tolist()) + "\n")


def outline_mask(points_xyz):
    idx = {tuple(p): i for i, p in enumerate(points_xyz)}
    keep = np.zeros(len(points_xyz), dtype=bool)
    neigh6 = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    for i, (x, y, z) in enumerate(points_xyz):
        for dx, dy, dz in neigh6:
            if (x + dx, y + dy, z + dz) not in idx:
                keep[i] = True
                break
    return keep


def run(image_id: str):
    clean_dir = Path(cfg("PLY_CLEAN_DIR", "ply_clean"))
    surf_dir = Path(cfg("SURF_OUT_DIR", "surf"))

    in_ply = clean_dir / f"{image_id}.ply"
    if not in_ply.exists():
        raise FileNotFoundError(f"Missing {in_ply}. Run modules/cleaner.py first.")

    props, arr, xyz = read_ascii_ply(in_ply)
    keep = outline_mask(xyz)

    out_arr = arr[keep]
    out_ply = surf_dir / f"{image_id}.ply"
    write_ascii_ply(out_ply, props, out_arr)

    print("in:", str(in_ply))
    print("points:", len(xyz), "outline:", int(keep.sum()))
    print("wrote:", str(out_ply))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    args = ap.parse_args()
    run(str(args.id))


if __name__ == "__main__":
    main()
