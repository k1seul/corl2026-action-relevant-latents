"""Build CALVIN-D task-level held-out splits using language annotations.

CALVIN-D's `auto_lang_ann.npy` provides 5124 train + 1011 val annotations,
each tagged with one of 34 unique task classes. We use language
annotations as our 'demos' (each = one ~64-frame window).

Strategy:
  - Hold out 7 of 34 tasks for OOD
  - Train = training/ subset's 27 train-task annotations (~4050)
  - Test_indist = validation/ subset's 27 train-task annotations (~810)
  - Test_ood = validation/ subset's 7 held-out-task annotations (~210)

Each split JSON has `data_subdir` + episodes with (start, end, task, task_index, caption).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np

CALVIN_ROOT = Path(
    "external_models/calvin_data/task_D_D")
SPLIT_DIR = (Path(__file__).resolve().parent.parent /
             "data" / "splits_calvin_taskheld")
OOD_TASKS = 7
SEED = 42


def load_lang(subdir: str) -> dict:
    return np.load(CALVIN_ROOT / subdir / "lang_annotations" / "auto_lang_ann.npy",
                    allow_pickle=True).item()


def annotations(lang: dict) -> list[dict]:
    out = []
    anns = lang["language"]["ann"]
    tasks = lang["language"]["task"]
    indx = lang["info"]["indx"]
    for ann, task, (s, e) in zip(anns, tasks, indx):
        out.append({"caption": str(ann), "task": str(task),
                    "start": int(s), "end": int(e)})
    return out


def main() -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    train_lang = load_lang("training")
    val_lang = load_lang("validation")
    train_anns = annotations(train_lang)
    val_anns = annotations(val_lang)

    # collect all unique tasks (train and val should have same set)
    all_tasks = sorted(set(a["task"] for a in train_anns) |
                       set(a["task"] for a in val_anns))
    print(f"total unique tasks: {len(all_tasks)}")

    # hold out OOD_TASKS
    rng.shuffle(all_tasks)
    ood_tasks = sorted(all_tasks[:OOD_TASKS])
    train_task_set = sorted(set(all_tasks) - set(ood_tasks))
    print(f"OOD tasks ({OOD_TASKS}): {ood_tasks}")
    print(f"train tasks: {len(train_task_set)}")

    # task → int id (stable across splits)
    task_index = {t: i for i, t in enumerate(sorted(train_task_set + ood_tasks))}

    def filter_eps(anns: Iterable[dict], task_set: set) -> list[dict]:
        return [{**a, "task_index": task_index[a["task"]]}
                for a in anns if a["task"] in task_set]

    train_eps = filter_eps(train_anns, set(train_task_set))
    test_indist_eps = filter_eps(val_anns, set(train_task_set))
    test_ood_eps = filter_eps(val_anns, set(ood_tasks))

    # also include OOD tasks from training subset for richer ood eval
    test_ood_eps_train_subset = filter_eps(train_anns, set(ood_tasks))
    # we only use validation OOD for the test_ood split (cleaner train/val isolation)
    print(f"\nsplit sizes:")
    print(f"  train      ({len(train_task_set)} tasks): {len(train_eps)} eps")
    print(f"  test_indist({len(train_task_set)} tasks): {len(test_indist_eps)} eps")
    print(f"  test_ood   ({OOD_TASKS} tasks):  {len(test_ood_eps)} eps")
    print(f"  (extra: test_ood_train_subset = {len(test_ood_eps_train_subset)} train-subset OOD)")

    # write splits
    for name, eps, subdir in [
        ("train", train_eps, "training"),
        ("test_indist", test_indist_eps, "validation"),
        ("test_ood", test_ood_eps, "validation"),
    ]:
        path = SPLIT_DIR / f"{name}.json"
        path.write_text(json.dumps({
            "data_subdir": subdir,
            "data_root": str(CALVIN_ROOT),
            "task_index": task_index,
            "episodes": eps,
        }, indent=2))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
