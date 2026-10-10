"""Per-task stratified 80:20 train/test split over LIBERO demos.

Each .hdf5 in datasets/<suite>/ is one task containing N demos
(typically 50). We shuffle the demo list per task with a fixed seed
and write episode-level metadata to data/splits/{train,test}.json.

The hdf5 files are not modified; downstream loaders read each
episode by (suite, file, demo_key).
"""
import argparse
import json
import random
from pathlib import Path

import h5py

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]


def collect_demos(datasets_dir: Path) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for suite in SUITES:
        suite_dir = datasets_dir / suite
        if not suite_dir.exists():
            continue
        out[suite] = {}
        for hdf in sorted(suite_dir.glob("*.hdf5")):
            with h5py.File(hdf, "r") as f:
                demos = sorted(f["data"].keys())
            out[suite][hdf.name] = demos
    return out


def split_per_task(
    demos_by_task: dict[str, dict[str, list[str]]],
    train_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    train, test = [], []
    for suite, tasks in demos_by_task.items():
        for task_file, demos in tasks.items():
            shuffled = list(demos)
            rng.shuffle(shuffled)
            n_train = int(round(len(shuffled) * train_ratio))
            for d in shuffled[:n_train]:
                train.append({"suite": suite, "file": task_file, "demo": d})
            for d in shuffled[n_train:]:
                test.append({"suite": suite, "file": task_file, "demo": d})
    return train, test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets-dir",
        default="datasets",
        help="dir containing libero_* suite subdirs",
    )
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    datasets_dir = Path(args.datasets_dir).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    demos = collect_demos(datasets_dir)
    if not demos:
        raise SystemExit(f"no LIBERO suite dirs found under {datasets_dir}")

    train, test = split_per_task(demos, args.train_ratio, args.seed)

    meta = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "datasets_dir": str(datasets_dir),
        "suites": list(demos.keys()),
        "n_train": len(train),
        "n_test": len(test),
    }

    with (out_dir / "train.json").open("w") as f:
        json.dump({"meta": meta, "episodes": train}, f, indent=2)
    with (out_dir / "test.json").open("w") as f:
        json.dump({"meta": meta, "episodes": test}, f, indent=2)

    print(f"seed={args.seed}, train_ratio={args.train_ratio}")
    for suite, tasks in demos.items():
        st = sum(1 for e in train if e["suite"] == suite)
        ss = sum(1 for e in test if e["suite"] == suite)
        print(f"  {suite}: {len(tasks)} tasks -> train={st}, test={ss}")
    print(f"  TOTAL: train={len(train)}, test={len(test)}")
    print(f"wrote {out_dir/'train.json'} and {out_dir/'test.json'}")


if __name__ == "__main__":
    main()
