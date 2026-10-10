"""Build demo-level held-out splits for Robomimic (lift/square/transport).

Robomimic only has 3 tasks → task-level OOD doesn't apply. Instead we
do demo-level split: 80% of each task's demos to train, 20% to test_indist.
No test_ood (Stage B (c) option per user).

Usage:
    uv run python scripts/build_robomimic_splits.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import h5py


REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "datasets" / "robomimic"
SPLIT_DIR = REPO / "data" / "splits_robomimic"
TASKS = ["lift", "square", "transport"]
TEST_FRAC = 0.20
SEED = 42


def main() -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    train_eps, test_eps = [], []
    for task in TASKS:
        path = DATA_DIR / f"{task}_image.hdf5"
        if not path.exists():
            print(f"[skip] {task}: {path} missing")
            continue
        try:
            with h5py.File(path, "r") as f:
                demos = sorted(f["data"].keys(),
                                key=lambda d: int(d.split("_")[-1]))
        except OSError as e:
            print(f"[skip] {task}: HDF5 invalid ({e})")
            continue
        # shuffle and split
        idx = list(range(len(demos)))
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * TEST_FRAC)))
        test_idx = set(idx[:n_test])
        for i, demo in enumerate(demos):
            ep = {"task": task, "demo": demo}
            if i in test_idx:
                test_eps.append(ep)
            else:
                train_eps.append(ep)
        print(f"{task}: {len(demos)} demos → "
              f"{len(demos)-len(test_idx)} train + {len(test_idx)} test")

    train_path = SPLIT_DIR / "train.json"
    test_path = SPLIT_DIR / "test_indist.json"
    train_path.write_text(json.dumps({"episodes": train_eps}, indent=2))
    test_path.write_text(json.dumps({"episodes": test_eps}, indent=2))
    print(f"\nwrote {train_path} ({len(train_eps)} eps)")
    print(f"wrote {test_path} ({len(test_eps)} eps)")


if __name__ == "__main__":
    main()
