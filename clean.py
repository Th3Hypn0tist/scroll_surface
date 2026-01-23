import argparse
from pathlib import Path
import shutil

TARGET_DIRS = [Path("ply"), Path("ply_clean"), Path("surf"), Path("out_entry")]


def gather_entries(target: Path) -> list[Path]:
    if not target.exists():
        return []
    return [p for p in target.iterdir() if p.name != ".gitkeep"]


def delete_entry(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean generated output directories.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    args = parser.parse_args()

    if not args.yes:
        print("Targets:")
        for target in TARGET_DIRS:
            print(f"  - {target}")
        confirm = input("Type YES to continue: ")
        if confirm != "YES":
            print("Aborted.")
            return

    for target in TARGET_DIRS:
        if not target.exists():
            print(f"{target}: missing")
            ensure_dir(target)
            continue

        entries = gather_entries(target)
        if not entries:
            print(f"{target}: empty (gitkeep only or empty)")
            continue

        for entry in entries:
            if args.dry_run:
                print(f"would delete: {entry}")
            else:
                delete_entry(entry)
                print(f"deleted: {entry}")

    for target in TARGET_DIRS:
        ensure_dir(target)


if __name__ == "__main__":
    main()
