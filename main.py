# main.py  (UPDATED: scanner -> cleaner -> outlines -> label)
import argparse
from pathlib import Path

try:
    import config as CFG
except Exception:
    CFG = None

from modules.scanner import run as run_scanner
from modules.cleaner import run as run_cleaner
from modules.find_outlines import run as run_outlines
from modules.label_output import run as run_label


def cfg(name, fallback):
    return getattr(CFG, name, fallback) if CFG is not None else fallback


def list_ids(data_dir: Path, split: str):
    p = data_dir / f"{split}_images"
    if not p.exists():
        raise FileNotFoundError(f"Missing images folder: {p}")
    ids = [f.stem for f in p.glob("*.tif")]
    ids.sort()
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--id", default="")
    args = ap.parse_args()

    data_dir = Path(cfg("DATA_DIR", "."))
    split = cfg("SPLIT", "train")

    if args.all:
        ids = list_ids(data_dir, split)
    elif args.id:
        ids = [args.id]
    else:
        raise SystemExit("Give --id <id> OR --all")

    for image_id in ids:
        print("\n=== ID:", image_id, "===")
        run_scanner(image_id)
        run_cleaner(image_id)
        run_outlines(image_id)
        run_label(image_id)

    print("\nDONE. ids:", len(ids))


if __name__ == "__main__":
    main()
