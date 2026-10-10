"""Per-benchmark eval dataset factory. Mirrors build_train_dataset but for
the probe pipeline, which needs the same interface across benchmarks.

Returns a torch Dataset where __getitem__ yields (clip, actions) tensors.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple


def build_eval_dataset(benchmark: str, split_path: str, *, clip_len: int,
                         resolution: int, seed: int):
    """Build the appropriate ClipDataset for the requested benchmark + split.

    Returns (dataset, cameras_used, action_dim).
    """
    if benchmark == "libero":
        from data.libero_video import LiberoVideoClips
        cameras = ["agentview_rgb"]
        ds = LiberoVideoClips(
            split_path=split_path, datasets_dir="datasets",
            clip_len=clip_len, cameras=cameras, resolution=resolution,
            seed=seed, return_actions=True,
        )
        return ds, cameras, 7

    if benchmark == "calvin":
        from comparison.datasets.calvin import CalvinClips
        cameras = ["rgb_static"]
        ds = CalvinClips(
            split_path=split_path, datasets_dir=None,
            clip_len=clip_len, cameras=cameras, resolution=resolution,
            seed=seed, return_actions=True,
        )
        return ds, cameras, 7

    if benchmark == "metaworld":
        from comparison.datasets.metaworld_lerobot import MetaWorldClips
        cameras = ["observation.image"]
        ds = MetaWorldClips(
            split_path=split_path,
            datasets_dir="external_models/metaworld_data/data",
            clip_len=clip_len, cameras=cameras, resolution=resolution,
            seed=seed, return_actions=True, action_dim=4,
        )
        return ds, cameras, 4

    raise ValueError(f"unknown benchmark: {benchmark!r}")


def benchmark_splits(benchmark: str) -> Tuple[str, str]:
    """Return (train_split_path, test_ood_split_path) for the benchmark."""
    if benchmark == "libero":
        base = "data/splits_taskheld"
    elif benchmark == "calvin":
        base = "data/splits_calvin_taskheld"
    elif benchmark == "metaworld":
        base = "data/splits_metaworld"
    else:
        raise ValueError(f"unknown benchmark: {benchmark!r}")
    return f"{base}/train.json", f"{base}/test_ood.json"


BENCHMARK_CHOICES = ("libero", "calvin", "metaworld")
