"""LIBERO demonstrations formatted for pi0.5 fine-tuning with a future-target loss.

Each sample is one random clip start t of one demonstration:

    image_main       (3, H, W)     agentview at t, float in [0, 1]
    image_wrist      (3, H, W)     eye-in-hand at t
    image_main_aux   (K, 3, H, W)  agentview at t + h for each h in aux_horizons
    image_wrist_aux  (K, 3, H, W)  eye-in-hand at t + h
    state            (15,)         ee_pos, ee_ori (axis-angle), gripper_states, joint_states at t
    actions          (16, 7)       a_t .. a_{t+15}
    language         str           task instruction, from the HDF5 file name

Images are flipped vertically, which matches the evaluation (`--flip vertical`).
Demonstrations are split 90/10 at the demo level with a fixed seed.
"""
from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

DEFAULT_SUITES: Tuple[str, ...] = ("libero_10", "libero_90", "libero_goal", "libero_object", "libero_spatial")
STATE_KEYS: Tuple[str, ...] = ("ee_pos", "ee_ori", "gripper_states", "joint_states")


@dataclass(frozen=True)
class _DemoMeta:
    file_path: str
    demo: str
    length: int
    task: str


def _filename_to_language(stem: str) -> str:
    """'KITCHEN_SCENE3_turn_on_the_stove_demo' -> 'kitchen scene3 turn on the stove'."""
    s = stem.lower()
    if s.endswith("_demo"):
        s = s[: -len("_demo")]
    return s.replace("_", " ").strip()


class LiberoVLADataset(Dataset):
    def __init__(
        self,
        root: str | os.PathLike,
        *,
        suites: Sequence[str] = DEFAULT_SUITES,
        action_horizon: int = 16,
        aux_horizons: Sequence[int] = (16,),
        camera_main: str = "agentview_rgb",
        camera_wrist: str = "eye_in_hand_rgb",
        split: str = "train",
        split_ratio: float = 0.9,
        split_seed: int = 0,
        random_clip: bool = True,
    ):
        super().__init__()
        if split not in ("all", "train", "test"):
            raise ValueError(f"split must be all/train/test, got {split!r}")
        self.root = Path(root)
        self.action_horizon = int(action_horizon)
        self.aux_horizons = tuple(int(h) for h in aux_horizons)
        if not self.aux_horizons or any(h <= 0 for h in self.aux_horizons):
            raise ValueError(f"aux_horizons must be positive ints, got {aux_horizons!r}")
        self.camera_main = camera_main
        self.camera_wrist = camera_wrist
        self.random_clip = bool(random_clip)
        # a clip needs the action chunk and every future frame inside the demo
        self.window = max(self.action_horizon, max(self.aux_horizons)) + 1

        files: List[str] = []
        for s in suites:
            files.extend(sorted(glob.glob(str(self.root / s / "*.hdf5"))))
        if not files:
            raise FileNotFoundError(f"No HDF5 files under {self.root} for suites={tuple(suites)}")

        full_index: List[_DemoMeta] = []
        for fp in files:
            task = _filename_to_language(Path(fp).stem)
            with h5py.File(fp, "r") as f:
                for demo in f["data"].keys():
                    T = int(f[f"data/{demo}/obs/{self.camera_main}"].shape[0])
                    if T >= self.window:
                        full_index.append(_DemoMeta(fp, demo, T, task))
        if not full_index:
            raise RuntimeError(f"No demos with >= {self.window} frames")

        if split == "all":
            self.index = full_index
        else:
            order = list(range(len(full_index)))
            random.Random(split_seed).shuffle(order)
            n_train = int(round(len(order) * split_ratio))
            chosen = order[:n_train] if split == "train" else order[n_train:]
            self.index = [full_index[i] for i in chosen]
        self._handles: Dict[str, h5py.File] = {}

    # h5py handles are not fork-safe: open lazily in each worker
    def _file(self, path: str) -> h5py.File:
        h = self._handles.get(path)
        if h is None:
            h = h5py.File(path, "r")
            self._handles[path] = h
        return h

    def __getstate__(self):
        st = self.__dict__.copy()
        st["_handles"] = {}
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        self._handles = {}

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _read_image(grp: h5py.Group, key: str, t: int) -> torch.Tensor:
        arr = grp[f"obs/{key}"][t][::-1, :, :]                         # vertical flip
        x = torch.from_numpy(np.ascontiguousarray(arr))
        return x.permute(2, 0, 1).contiguous().float().div_(255.0)

    def __getitem__(self, idx: int):
        meta = self.index[idx]
        grp = self._file(meta.file_path)[f"data/{meta.demo}"]
        max_start = meta.length - self.window
        t = random.randint(0, max_start) if (self.random_clip and max_start > 0) else 0

        img_main = self._read_image(grp, self.camera_main, t)
        img_wrist = self._read_image(grp, self.camera_wrist, t)
        img_main_aux = torch.stack([self._read_image(grp, self.camera_main, t + h) for h in self.aux_horizons])
        img_wrist_aux = torch.stack([self._read_image(grp, self.camera_wrist, t + h) for h in self.aux_horizons])

        state = np.concatenate([np.asarray(grp[f"obs/{k}"][t], dtype=np.float32) for k in STATE_KEYS])
        actions = np.asarray(grp["actions"][t: t + self.action_horizon], dtype=np.float32)
        return {
            "image_main": img_main,
            "image_wrist": img_wrist,
            "image_main_aux": img_main_aux,
            "image_wrist_aux": img_wrist_aux,
            "state": torch.from_numpy(state),
            "actions": torch.from_numpy(actions),
            "language": meta.task,
        }


def collate(batch: List[Dict]) -> Dict:
    return {k: (torch.stack([b[k] for b in batch]) if isinstance(batch[0][k], torch.Tensor)
                else [b[k] for b in batch]) for k in batch[0]}


def build_loader(root, *, suites, action_horizon, aux_horizons, batch_size, num_workers,
                 split, split_ratio, split_seed, shuffle, drop_last, random_clip) -> DataLoader:
    ds = LiberoVLADataset(root, suites=suites, action_horizon=action_horizon, aux_horizons=aux_horizons,
                          split=split, split_ratio=split_ratio, split_seed=split_seed, random_clip=random_clip)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=False, drop_last=drop_last, persistent_workers=num_workers > 0,
                      prefetch_factor=2 if num_workers > 0 else None, collate_fn=collate)
