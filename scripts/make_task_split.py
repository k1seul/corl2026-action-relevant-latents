"""Build a task-level held-out split on top of LIBERO.

Original ``data/splits/{train,test}.json`` is per-task stratified
80:20 *across demos*: every one of the 130 tasks appears in both train
and test, only different demonstrations of the same task. That gives
"new-execution-of-known-task" generalization, not "new-task"
generalization.

This script produces a stricter task-level split:

  * ``data/splits_taskheld/train.json``       : 80% of tasks × 40 demos
  * ``data/splits_taskheld/test_indist.json`` : same 80% tasks × 10 demos
  * ``data/splits_taskheld/test_ood.json``    : remaining 20% tasks × 50 demos

Stratified by suite: each LIBERO suite (libero_spatial / object / goal /
10 / 90) keeps the same 80/20 task ratio in train/ood. Demos within
train tasks get a 40/10 split for the in-distribution test set so
probes can be evaluated under both regimes.

Reproducibility: seed 42 for both task and demo shuffles.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_TRAIN = REPO_ROOT / "data/splits/train.json"
SRC_TEST = REPO_ROOT / "data/splits/test.json"
OUT_DIR = REPO_ROOT / "data/splits_taskheld"

TASK_RATIO = 0.8        # fraction of tasks used for training
DEMO_RATIO = 0.8        # fraction of demos within each train task
SEED = 42


def main() -> None:
    with SRC_TRAIN.open() as f:
        src_train = json.load(f)
    with SRC_TEST.open() as f:
        src_test = json.load(f)

    # All (suite, file, demo) triples from the original 80:20 demo split.
    all_eps = src_train["episodes"] + src_test["episodes"]
    by_task: dict[tuple[str, str], list[str]] = defaultdict(list)
    for ep in all_eps:
        by_task[(ep["suite"], ep["file"])].append(ep["demo"])
    for k, demos in by_task.items():
        # all demos per task should be unique
        assert len(set(demos)) == len(demos), f"duplicate demo in {k}"

    by_suite: dict[str, list[str]] = defaultdict(list)
    for (suite, fname) in by_task.keys():
        by_suite[suite].append(fname)
    for suite in by_suite:
        by_suite[suite].sort()

    rng = random.Random(SEED)
    task_train: list[tuple[str, str]] = []
    task_ood: list[tuple[str, str]] = []
    for suite, files in by_suite.items():
        n = len(files)
        n_train = int(round(n * TASK_RATIO))
        shuffled = files[:]
        rng.shuffle(shuffled)
        for f in shuffled[:n_train]:
            task_train.append((suite, f))
        for f in shuffled[n_train:]:
            task_ood.append((suite, f))

    print(f"task split: {len(task_train)} train / {len(task_ood)} ood (of {len(by_task)})")
    train_per_suite = defaultdict(int)
    ood_per_suite = defaultdict(int)
    for (s, _) in task_train:
        train_per_suite[s] += 1
    for (s, _) in task_ood:
        ood_per_suite[s] += 1
    for s in by_suite:
        print(f"  {s}: train={train_per_suite[s]} ood={ood_per_suite[s]}")

    train_eps: list[dict] = []
    indist_eps: list[dict] = []
    for (suite, fname) in task_train:
        demos = sorted(by_task[(suite, fname)])
        rng2 = random.Random(SEED ^ hash((suite, fname)) & 0xFFFFFFFF)
        shuffled = demos[:]
        rng2.shuffle(shuffled)
        n_t = int(round(len(shuffled) * DEMO_RATIO))
        for d in shuffled[:n_t]:
            train_eps.append({"suite": suite, "file": fname, "demo": d})
        for d in shuffled[n_t:]:
            indist_eps.append({"suite": suite, "file": fname, "demo": d})

    ood_eps: list[dict] = []
    for (suite, fname) in task_ood:
        for d in sorted(by_task[(suite, fname)]):
            ood_eps.append({"suite": suite, "file": fname, "demo": d})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_meta = {
        "seed": SEED,
        "task_ratio": TASK_RATIO,
        "demo_ratio_within_train_tasks": DEMO_RATIO,
        "n_train_tasks": len(task_train),
        "n_ood_tasks": len(task_ood),
        "datasets_dir": src_train["meta"].get("datasets_dir"),
        "suites": list(by_suite.keys()),
    }
    splits = {
        "train.json": train_eps,
        "test_indist.json": indist_eps,
        "test_ood.json": ood_eps,
    }
    for name, eps in splits.items():
        out_path = OUT_DIR / name
        meta = {**base_meta, "split": name.replace(".json", ""), "n_episodes": len(eps)}
        with out_path.open("w") as f:
            json.dump({"meta": meta, "episodes": eps}, f, indent=2)
        print(f"  wrote {out_path}: {len(eps)} demos")

    # quick sanity: no demo overlap between any pair
    s_train = {(e["suite"], e["file"], e["demo"]) for e in train_eps}
    s_indist = {(e["suite"], e["file"], e["demo"]) for e in indist_eps}
    s_ood = {(e["suite"], e["file"], e["demo"]) for e in ood_eps}
    print(f"  demo overlap train∩indist={len(s_train & s_indist)} (expect 0)")
    print(f"  demo overlap train∩ood   ={len(s_train & s_ood)} (expect 0)")
    print(f"  demo overlap indist∩ood  ={len(s_indist & s_ood)} (expect 0)")
    # task overlap: train and indist share tasks; train and ood do not
    t_train = {(e["suite"], e["file"]) for e in train_eps}
    t_ood = {(e["suite"], e["file"]) for e in ood_eps}
    print(f"  task overlap train∩ood = {len(t_train & t_ood)} (expect 0)")


if __name__ == "__main__":
    main()
