"""Locate the submitted task-OOD split and copy it under
``runs/corl2026_rebuttal/splits/`` with a SHA-256 stored alongside.

Usage:
  python scripts/rebuttal/hash_split.py <path-to-split.json>

If invoked with no argument, scans a few well-known locations. Never
overwrites an existing hash file — call with ``--force`` to replace.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "runs/corl2026_rebuttal/splits"

SEARCH_GLOBS = (
    "runs/**/*split*.json",
    "src/**/*split*.json",
    "runs/full_results/**/split_*.json",
)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default=None,
                    help="path to a split JSON; if omitted, search well-known locations")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    candidates: list[Path] = []
    if args.source:
        candidates = [Path(args.source).resolve()]
    else:
        for g in SEARCH_GLOBS:
            candidates.extend(REPO_ROOT.glob(g))

    if not candidates:
        sys.stderr.write(
            "no split JSON found. locate on the compute cluster and re-run:\n"
            "  python scripts/rebuttal/hash_split.py "
            "<path>/task_ood_split.json\n"
        )
        return 1

    for src in candidates:
        if not src.is_file():
            continue
        dst = SPLITS_DIR / src.name
        h = sha256(src)
        h_file = SPLITS_DIR / (src.name + ".sha256")
        if h_file.exists() and not args.force:
            existing = h_file.read_text().split()[0]
            if existing != h:
                sys.stderr.write(
                    f"REFUSING to overwrite {h_file} (sha mismatch): "
                    f"existing={existing[:12]}, new={h[:12]}. "
                    "use --force to replace.\n")
                return 2
        shutil.copy2(src, dst)
        h_file.write_text(f"{h}  {src.name}\n")
        print(f"copied {src} -> {dst}\n  sha256={h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
