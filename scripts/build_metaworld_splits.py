"""Build task-level held-out splits for MetaWorld MT50.

Holds out 10 of 49 tasks for OOD evaluation. Within remaining 39 tasks,
uses 80/20 of episodes for train/test_indist.

Usage:
    uv run python scripts/build_metaworld_splits.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = Path("external_models/metaworld_data/data")
SPLIT_DIR = REPO / "data" / "splits_metaworld"
OOD_TASKS = 10
TEST_INDIST_FRAC = 0.20
SEED = 42


def main() -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # scan all available parquet files; build (file_idx, episode_index, task_index) index
    parquets = sorted(DATA.glob("file-*.parquet"))
    print(f"scanning {len(parquets)} parquet files ...")
    index = []  # (file_idx, episode_index, task_index)
    for p in parquets:
        file_idx = int(p.stem.split("-")[-1])
        try:
            df = pd.read_parquet(p, columns=["episode_index", "task_index"])
        except Exception as e:
            print(f"  skip {p.name}: {e}")
            continue
        groups = df.groupby("episode_index")["task_index"].first()
        for ep_idx, task_idx in groups.items():
            index.append((file_idx, int(ep_idx), int(task_idx)))
    print(f"total episodes: {len(index)}")

    all_tasks = sorted({t for _, _, t in index})
    print(f"total tasks: {len(all_tasks)}")

    # held-out tasks
    rng.shuffle(all_tasks)
    ood_set = set(all_tasks[:OOD_TASKS])
    train_tasks = sorted(t for t in all_tasks if t not in ood_set)
    print(f"OOD tasks: {sorted(ood_set)}")
    print(f"train tasks: {len(train_tasks)}")

    # split: held-out tasks → test_ood. remaining tasks: 80/20 demo split per task.
    train_eps, test_indist_eps, test_ood_eps = [], [], []
    for task in all_tasks:
        eps = [(f, e) for f, e, t in index if t == task]
        rng.shuffle(eps)
        if task in ood_set:
            for f, e in eps:
                test_ood_eps.append({"file_idx": f, "episode_index": e, "task_index": task})
        else:
            n_test = max(1, int(round(len(eps) * TEST_INDIST_FRAC)))
            for i, (f, e) in enumerate(eps):
                rec = {"file_idx": f, "episode_index": e, "task_index": task}
                if i < n_test:
                    test_indist_eps.append(rec)
                else:
                    train_eps.append(rec)

    for name, eps in [("train", train_eps), ("test_indist", test_indist_eps),
                      ("test_ood", test_ood_eps)]:
        path = SPLIT_DIR / f"{name}.json"
        path.write_text(json.dumps({"episodes": eps}, indent=2))
        print(f"wrote {path}: {len(eps)} eps")


if __name__ == "__main__":
    main()
