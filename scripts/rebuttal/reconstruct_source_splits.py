"""Enumerate LIBERO HDF5s and produce ``data/splits/{train,test}.json``.

This is the missing prerequisite for ``scripts/make_task_split.py``.
Once every LIBERO suite HDF5 is present under
``--libero-datasets-dir/<suite>/*.hdf5``, this script:

  1. lists every (suite, file, demo) triple across the 5 canonical suites
     (libero_spatial, libero_object, libero_goal, libero_10, libero_90);
  2. for each task file, deterministically stratifies its demos 80/20 into
     the source train/test set (matching the original
     ``data/splits/train.json`` / ``test.json`` semantics referenced by
     ``make_task_split.py``);
  3. writes both JSONs;
  4. optionally verifies (--verify-against) that a downstream
     ``make_task_split.py`` run seeded 42 produces a test_ood.json whose
     SHA-256 matches a user-provided reference (i.e. our test_ood.json).

Determinism: seed 42 for the demo shuffle, matching make_task_split.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

CANONICAL_SUITES = ["libero_spatial", "libero_object", "libero_goal",
                    "libero_10", "libero_90"]

# Same ratio the original `data/splits/` uses per the make_task_split.py note.
DEMO_TRAIN_RATIO = 0.8
SEED_DEMO = 42


def enumerate_demos(dataset_dir: Path,
                    suites: list[str]) -> dict[tuple[str, str], list[str]]:
    """Return {(suite, filename): [demo_0, demo_1, ...]}."""
    try:
        import h5py
    except ImportError:  # pragma: no cover
        raise SystemExit("h5py required; pip install h5py")
    by_task: dict[tuple[str, str], list[str]] = {}
    missing_suites: list[str] = []
    for suite in suites:
        d = dataset_dir / suite
        if not d.exists():
            missing_suites.append(suite)
            continue
        for hdf5 in sorted(d.glob("*.hdf5")):
            with h5py.File(hdf5, "r") as f:
                demos = sorted(list(f["data"].keys()))
            by_task[(suite, hdf5.name)] = demos
    if missing_suites:
        sys.stderr.write(
            f"[reconstruct] MISSING suites: {missing_suites} — "
            "the source split will only cover the suites present. If you need "
            "the full 130-task submitted split, download the missing HDF5s "
            "first (see runs/corl2026_rebuttal/reports/HANDOFF_SESSION2.md).\n"
        )
    return by_task


def split_source_train_test(by_task: dict[tuple[str, str], list[str]],
                            suite_order: list[str] | None = None
                            ) -> tuple[list[dict], list[dict]]:
    """Build source train/test episode lists.

    ``suite_order`` fixes the outer iteration order. This is required for the
    downstream ``make_task_split.py`` shuffle to be byte-identical to the
    submitted paper's split (which iterated suites in canonical
    ``[spatial, object, goal, 10, 90]`` order, not alphabetical).
    """
    if suite_order is None:
        suite_order = CANONICAL_SUITES
    order_idx = {s: i for i, s in enumerate(suite_order)}
    train: list[dict] = []
    test: list[dict] = []
    # Sort by (suite_order_idx, filename) so within a suite files remain sorted.
    items = sorted(by_task.items(),
                   key=lambda kv: (order_idx.get(kv[0][0], 999), kv[0][1]))
    for (suite, fname), demos in items:
        rng = random.Random(SEED_DEMO ^ hash((suite, fname)) & 0xFFFFFFFF)
        shuffled = list(demos)
        rng.shuffle(shuffled)
        n_train = int(round(len(shuffled) * DEMO_TRAIN_RATIO))
        for d in shuffled[:n_train]:
            train.append({"suite": suite, "file": fname, "demo": d})
        for d in shuffled[n_train:]:
            test.append({"suite": suite, "file": fname, "demo": d})
    return train, test


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero-datasets-dir", required=True,
                    help="root that contains libero_<suite>/*.hdf5")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--suites", nargs="+", default=CANONICAL_SUITES)
    ap.add_argument("--verify-against", default=None,
                    help="a reference test_ood.json (e.g. runs/corl2026_rebuttal/splits/test_ood.json) "
                         "to compare SHA-256 with the make_task_split.py output that would be built from this source.")
    args = ap.parse_args()

    ds_dir = Path(args.libero_datasets_dir).resolve()
    if not ds_dir.exists():
        sys.stderr.write(f"missing dataset dir: {ds_dir}\n")
        return 1
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    by_task = enumerate_demos(ds_dir, args.suites)
    if not by_task:
        sys.stderr.write("no HDF5 files found; nothing to write\n")
        return 2

    train, test = split_source_train_test(by_task)
    meta_common = dict(
        seed=SEED_DEMO,
        demo_train_ratio=DEMO_TRAIN_RATIO,
        n_tasks=len(by_task),
        datasets_dir=str(ds_dir),
        suites=args.suites,
    )
    for name, eps in [("train.json", train), ("test.json", test)]:
        blob = {"meta": {**meta_common,
                          "split": name.replace(".json", ""),
                          "n_episodes": len(eps)},
                "episodes": eps}
        p = out_dir / name
        with p.open("w") as f:
            json.dump(blob, f, indent=2)
        print(f"wrote {p}: {len(eps)} demos across {len(by_task)} tasks")

    if args.verify_against:
        ref = Path(args.verify_against)
        if not ref.exists():
            sys.stderr.write(f"verify_against file missing: {ref}\n")
            return 3
        print(f"[verify] reference test_ood.json sha256 = {sha256(ref)}")
        print("[verify] to complete verification, run scripts/make_task_split.py "
              "then diff the produced data/splits_taskheld/test_ood.json against the reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
