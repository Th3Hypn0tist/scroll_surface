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
    col = {p: i for i, p in enumerate(props) if i < arr.shape[1]}
    for k in ("x", "y", "z"):
        if k not in col:
            raise ValueError(f"PLY missing {k}: props={props}")

    # Keep full row for rewriting (preserve all columns)
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


def union_find_components(points_xyz, dmax: int):
    idx = {tuple(p): i for i, p in enumerate(points_xyz)}
    parent = np.arange(len(points_xyz), dtype=np.int32)
    size = np.ones(len(points_xyz), dtype=np.int32)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    offs = []
    r2 = dmax * dmax
    for dz in range(-dmax, dmax + 1):
        for dy in range(-dmax, dmax + 1):
            for dx in range(-dmax, dmax + 1):
                if dx == dy == dz == 0:
                    continue
                if (dx*dx + dy*dy + dz*dz) <= r2:
                    offs.append((dx, dy, dz))

    for i, (x, y, z) in enumerate(points_xyz):
        for dx, dy, dz in offs:
            j = idx.get((x + dx, y + dy, z + dz))
            if j is not None:
                union(i, j)

    roots = np.array([find(i) for i in range(len(points_xyz))], dtype=np.int32)
    uniq, inv = np.unique(roots, return_inverse=True)
    comp_sizes = np.bincount(inv)
    return inv.astype(np.int32), comp_sizes.astype(np.int32)


def run(image_id: str):
    ply_dir = Path(cfg("PLY_OUT_DIR", "ply"))
    out_dir = Path(cfg("PLY_CLEAN_DIR", "ply_clean"))
    dmax = int(cfg("DMAX", 1))
    vcount = int(cfg("VCOUNT", 2000))

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

    comp, sizes = union_find_components(xyz2, dmax=dmax)
    keep_big = sizes[comp] >= vcount

    arr3 = arr2[keep_big]

    out_ply = out_dir / f"{image_id}.ply"
    write_ascii_ply(out_ply, props, arr3)

    print("in:", str(in_ply))
    print("points:", len(xyz2), "components:", len(sizes), "dmax:", dmax)
    print("kept_points:", int(keep_big.sum()), "vcount>=", vcount)
    print("wrote:", str(out_ply))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    args = ap.parse_args()
    run(str(args.id))


if __name__ == "__main__":
    main()
