"""Single helper that builds the appropriate train ClipDataset per benchmark.

Used by all backbone aux-finetune scripts so the per-dataset glue lives in one
place. Mirrors the inline switch in scripts/train_vjepa2_aux.py.

Session-2 patches:

* paths are env-overridable so the rebuttal launcher can point
  at ``datasets/libero_downloaded`` (rebuttal cluster) instead of the
  original ``datasets`` root — without editing this file per launch.

    LIBERO_SPLIT_PATH        default data/splits_taskheld/train.json
    LIBERO_DATASETS_DIR      default datasets

* if ``REBUTTAL_CONDITION == "shuffled_id"``, the LIBERO dataset's action
  channel is silently swapped for actions drawn from a DIFFERENT episode
  of the SAME task via a deterministic within-task cross-episode
  derangement seeded by ``REBUTTAL_RUN_SEED``. See
  ``src/rebuttal/derangement_dataloader.py``. Every other backbone code
  path is unchanged.

Returns:
    (train_ds, action_dim)
"""
from __future__ import annotations

import os


def _maybe_build_derangement_source(split_path: str, datasets_dir: str,
                                     action_dim: int = 7):
    """Return an action_source_override callable or None depending on
    ``REBUTTAL_CONDITION``.

    Only the ``shuffled_id`` primary condition triggers this path; every
    other condition (true_id, forward_dynamics, action_free_adaptation)
    keeps the original per-clip actions untouched.
    """
    cond = os.environ.get("REBUTTAL_CONDITION", "true_id")
    if cond != "shuffled_id":
        return None
    run_seed = int(os.environ.get("REBUTTAL_RUN_SEED", "0"))
    # lazy import so non-rebuttal code paths still work without the
    # rebuttal package on PYTHONPATH.
    from rebuttal.derangement_dataloader import build_plan_and_action_source

    plan, action_source, _ = build_plan_and_action_source(
        run_seed=run_seed, split_path=split_path,
        datasets_dir=datasets_dir, action_dim=action_dim, strict=True,
    )
    # print for the training log: every rebuttal run must be able to
    # confirm the derangement was actually applied.
    print(f"[rebuttal] REBUTTAL_CONDITION=shuffled_id  "
          f"derangement plan hash={plan.global_hash()}  "
          f"tasks={len(plan.per_task)}  "
          f"skipped_singletons={plan.skipped_singleton_tasks}",
          flush=True)
    return action_source


def build_train_dataset(dataset: str, *, clip_len: int, seed: int,
                          resolution: int = 128, return_actions: bool = True):
    if dataset == "libero":
        from data.libero_video import LiberoVideoClips
        split_path = os.environ.get(
            "LIBERO_SPLIT_PATH", "data/splits_taskheld/train.json")
        datasets_dir = os.environ.get("LIBERO_DATASETS_DIR", "datasets")
        action_source = _maybe_build_derangement_source(
            split_path, datasets_dir, action_dim=7)
        ds = LiberoVideoClips(
            split_path=split_path,
            datasets_dir=datasets_dir,
            clip_len=clip_len, cameras=["agentview_rgb"], resolution=resolution,
            seed=seed, return_actions=return_actions,
            action_source_override=action_source,
        )
        return ds, 7

    if dataset == "robomimic":
        from comparison.datasets.robomimic import RobomimicVideoClips
        ds = RobomimicVideoClips(
            split_path="data/splits_robomimic/train.json",
            datasets_dir="datasets/robomimic",
            clip_len=clip_len, cameras=["agentview_image"], resolution=resolution,
            seed=seed, return_actions=return_actions,
        )
        return ds, 7

    if dataset == "calvin":
        from comparison.datasets.calvin import CalvinClips
        ds = CalvinClips(
            split_path="data/splits_calvin_taskheld/train.json",
            datasets_dir=None,
            clip_len=clip_len, cameras=["rgb_static"], resolution=resolution,
            seed=seed, return_actions=return_actions,
        )
        return ds, 7

    if dataset == "metaworld":
        from comparison.datasets.metaworld_lerobot import MetaWorldClips
        ds = MetaWorldClips(
            split_path="data/splits_metaworld/train.json",
            datasets_dir="external_models/metaworld_data/data",
            clip_len=clip_len, cameras=["observation.image"], resolution=resolution,
            seed=seed, return_actions=return_actions, action_dim=4,
        )
        return ds, 4

    raise ValueError(f"unknown dataset: {dataset!r}")


DATASET_CHOICES = ("libero", "robomimic", "calvin", "metaworld")
