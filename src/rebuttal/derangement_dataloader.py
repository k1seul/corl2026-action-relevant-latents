"""Glue between ``derangement.py`` (pure permutation logic) and
``LiberoVideoClips`` (raw data access).

Session-2 STEP 3: the primary shuffled-ID control replaces each clip's
action sequence with the action sequence of a DIFFERENT episode of the
SAME task, via a deterministic derangement seeded by ``(run_seed, task_id)``.

Two helpers:

  * ``preload_action_bank`` — walks a LIBERO split and pre-loads every
    episode's full action sequence (small: ~4 kB per episode × ≤6500
    episodes ≈ 26 MB total for the full LIBERO train split).

  * ``build_derangement_action_source`` — returns a callable that
    ``LiberoVideoClips.action_source_override=`` can consume, mapping
    ``(task_key, own_episode_id, read_indices, pad_after)`` to the
    corresponding action window of the derangement-selected source
    episode.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from rebuttal.derangement import DerangementPlan, build_derangement_plan


TaskKey = tuple[str, str]


def enumerate_episodes_per_task(split_path: str | Path) -> dict[TaskKey, list[str]]:
    """Return {(suite, file): [demo_ids...]} from a LIBERO split JSON."""
    with open(split_path) as f:
        blob = json.load(f)
    per_task: dict[TaskKey, list[str]] = {}
    for ep in blob["episodes"]:
        key = (ep["suite"], ep["file"])
        per_task.setdefault(key, []).append(ep["demo"])
    return per_task


def preload_action_bank(split_path: str | Path,
                        datasets_dir: str | Path,
                        action_dim: int = 7) -> dict[TaskKey, dict[str, "np.ndarray"]]:
    """Load actions[T,action_dim] for every episode in the split.

    Returns nested dict ``bank[(suite, file)][demo_id] = np.ndarray``.

    Requires h5py; kept out of module-import so pure-stdlib code paths
    still work.
    """
    import h5py
    import numpy as np

    per_task = enumerate_episodes_per_task(split_path)
    bank: dict[TaskKey, dict[str, "np.ndarray"]] = {}
    file_cache: dict[TaskKey, h5py.File] = {}
    for (suite, fname), demos in per_task.items():
        p = Path(datasets_dir) / suite / fname
        if not p.exists():
            raise FileNotFoundError(f"missing LIBERO hdf5: {p}")
        f = h5py.File(str(p), "r")
        file_cache[(suite, fname)] = f
        bank[(suite, fname)] = {}
        for demo in demos:
            arr = np.asarray(f["data"][demo]["actions"][:], dtype=np.float32)
            if arr.shape[-1] > action_dim:
                arr = arr[..., :action_dim]
            bank[(suite, fname)][demo] = arr
    # do NOT close the h5 files — the caller's LiberoVideoClips workers
    # will open their own. We close ours after loading to free file
    # descriptors.
    for f in file_cache.values():
        f.close()
    return bank


def build_derangement_action_source(
    plan: DerangementPlan,
    action_bank: dict[TaskKey, dict[str, "np.ndarray"]],
    *,
    strict: bool = True,
) -> Callable:
    """Return a callable suited for ``LiberoVideoClips.action_source_override=``.

    Signature: ``fn(task_key, own_episode_id, read_indices, pad_after) -> np.ndarray``

    * Looks up ``plan.per_task[task_key][own_episode_id]`` → the SOURCE
      episode whose actions we substitute in.
    * Slices the source episode's actions by ``read_indices``. If the
      source episode is shorter than the requested window, we pad by
      repeating the last available action (deterministic; documented).
    * ``strict=True`` raises if either the task_key was skipped (singleton
      task) or the episode is not in the plan — this guarantees the
      shuffled-ID control never silently falls back to true-ID actions.
    """
    import numpy as np

    def fn(task_key, own_episode_id, read_indices, pad_after):
        if task_key not in plan.per_task:
            if strict:
                raise KeyError(f"derangement plan skipped task {task_key!r} "
                               "(likely singleton). Exclude these episodes "
                               "from the shuffled-ID split.")
            # non-strict: fall through to own actions
            src_ep = own_episode_id
        else:
            src_ep = plan.per_task[task_key].get(own_episode_id)
            if src_ep is None:
                if strict:
                    raise KeyError(f"episode {own_episode_id!r} of task "
                                    f"{task_key!r} missing from derangement plan.")
                src_ep = own_episode_id
        src_actions = action_bank[task_key][src_ep]

        # slice by read_indices, with clamping if any index exceeds source length
        max_i = src_actions.shape[0]
        clamped = [min(i, max_i - 1) for i in read_indices]
        arr = src_actions[clamped]
        if pad_after > 0:
            arr = np.concatenate([arr, np.repeat(arr[-1:], pad_after, axis=0)], axis=0)
        return arr.astype("float32", copy=False)

    return fn


def build_plan_and_action_source(
    run_seed: int,
    split_path: str | Path,
    datasets_dir: str | Path,
    action_dim: int = 7,
    strict: bool = True,
) -> tuple[DerangementPlan, Callable, dict[TaskKey, dict[str, "np.ndarray"]]]:
    """One-shot helper: enumerate episodes → build plan → preload bank →
    build action-source callable. Returns ``(plan, action_source_fn, bank)``.

    The bank is returned so callers can persist it or pass it to a
    ``__reduce__`` hook when using multi-process dataloaders (torch's
    DataLoader forks the parent, so the bank is inherited copy-on-write
    on POSIX and does not multiply memory).
    """
    per_task = enumerate_episodes_per_task(split_path)
    plan = build_derangement_plan(run_seed, per_task)
    bank = preload_action_bank(split_path, datasets_dir, action_dim=action_dim)
    src = build_derangement_action_source(plan, bank, strict=strict)
    return plan, src, bank
