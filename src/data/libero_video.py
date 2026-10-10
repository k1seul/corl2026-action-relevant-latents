"""LIBERO video clip dataset.

Reads a train/test split JSON of `(suite, file, demo)` entries and
samples a fixed-length clip from each episode's RGB stream(s). Episodes
shorter than `clip_len` are padded by repeating the last frame.

Frames are returned as float32 tensors in `[0, 1]` with shape
`(L, C, H, W)`, where `C = 3 * len(cameras)` (cameras stacked along the
channel axis).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class LiberoVideoClips(Dataset):
    """LIBERO clip dataset.

    Optional rebuttal extensions:

      * ``return_ids=True`` — additionally yield ``(task_key, episode_id)``
        so the training loop can build a within-task cross-episode
        derangement plan. ``task_key = (suite, file)`` as a tuple of str;
        ``episode_id = demo`` string.
      * ``action_source_override`` — a callable ``(task_key, episode_id,
        read_indices) -> np.ndarray[T, action_dim]``. When set, the actions
        returned for a clip are drawn via this callable INSTEAD of from the
        clip's own demo. Used for the within-task cross-episode derangement
        (session-2 STEP 3). Only the ACTIONS are swapped; the RGB clip and
        (task_key, episode_id) reported in the yielded tuple are always the
        clip's OWN identity — the training loop must know it is training on
        (clip_of_episode_A, action_of_episode_B_of_same_task).
    """

    def __init__(
        self,
        split_path: str | Path,
        datasets_dir: str | Path,
        clip_len: int = 16,
        cameras: Sequence[str] = ("agentview_rgb",),
        resolution: int = 128,
        seed: int = 0,
        return_actions: bool = False,
        return_ids: bool = False,
        action_source_override=None,
    ) -> None:
        with open(split_path) as f:
            blob = json.load(f)
        self.episodes: list[dict] = blob["episodes"]
        self.datasets_dir = Path(datasets_dir)
        self.clip_len = int(clip_len)
        self.cameras = list(cameras)
        self.resolution = int(resolution)
        self.return_actions = bool(return_actions)
        self.return_ids = bool(return_ids)
        self.action_source_override = action_source_override
        self._rng = random.Random(seed)
        # h5py file cache; reset per-worker via worker_init_fn
        self._cache: dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.episodes)

    def _file(self, suite: str, name: str) -> h5py.File:
        key = f"{suite}/{name}"
        if key not in self._cache:
            self._cache[key] = h5py.File(self.datasets_dir / suite / name, "r")
        return self._cache[key]

    def __getitem__(self, idx: int):
        ep = self.episodes[idx]
        f = self._file(ep["suite"], ep["file"])
        demo = f["data"][ep["demo"]]
        L = self.clip_len

        # uniform start within episode; pad if too short
        T = demo["obs"][self.cameras[0]].shape[0]
        if T <= L:
            # h5py rejects duplicate / non-increasing indices, so read T
            # unique frames first and then repeat the last frame to fill.
            read_indices = list(range(T))
            pad_after = L - T
        else:
            start = self._rng.randint(0, T - L)
            read_indices = list(range(start, start + L))
            pad_after = 0

        per_cam = []
        for cam in self.cameras:
            arr = np.asarray(demo["obs"][cam][read_indices])  # (≤L, H, W, 3)
            if pad_after:
                arr = np.concatenate([arr, np.repeat(arr[-1:], pad_after, axis=0)], axis=0)
            per_cam.append(arr)
        clip = np.concatenate(per_cam, axis=-1) if len(per_cam) > 1 else per_cam[0]
        # (L, H, W, 3*Ncam) -> (L, 3*Ncam, H, W) in [0, 1]
        clip_t = torch.from_numpy(clip).permute(0, 3, 1, 2).contiguous().float() / 255.0
        if clip_t.shape[-1] != self.resolution:
            clip_t = F.interpolate(
                clip_t, size=self.resolution, mode="bilinear", align_corners=False
            )
        if not self.return_actions and not self.return_ids:
            return clip_t

        # actions (either own or overridden)
        if self.action_source_override is not None:
            actions = self.action_source_override(
                (ep["suite"], ep["file"]), ep["demo"], read_indices, pad_after,
            )
        else:
            actions = np.asarray(demo["actions"][read_indices], dtype=np.float32)
            if pad_after:
                actions = np.concatenate(
                    [actions, np.repeat(actions[-1:], pad_after, axis=0)], axis=0
                )
        actions_t = torch.from_numpy(actions).contiguous()

        if self.return_ids:
            # collate default stacks tensors; strings/tuples become lists.
            return clip_t, actions_t, ep["suite"], ep["file"], ep["demo"]
        return clip_t, actions_t


def worker_init_fn(worker_id: int) -> None:
    """Reset h5py file cache per worker so each process owns its handles."""
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = info.dataset
    if isinstance(ds, LiberoVideoClips):
        ds._cache = {}
        ds._rng = random.Random(info.seed % (2**32))
