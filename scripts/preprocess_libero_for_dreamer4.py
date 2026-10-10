"""Convert LIBERO HDF5 episodes → nicklashansen/dreamer4 sharded format.

Produces two parallel output trees consumable by:
  - `ShardedFrameDataset` (used by train_tokenizer.py)
  - `WMDataset` (used by train_dynamics.py)

Layout written by this script:

  <out_root>/libero_dreamer4_frames/<task>/<task>_shard{NNNN}.pt
      each shard: {"frames": uint8 tensor (≤2048, 3, 128, 128)}

  <out_root>/libero_dreamer4_demos/<task>.pt
      single per-task file: {
        "episode": int64  (N,)   episode id per frame, monotone, unique-per-demo
        "action":  float32 (N, 16) LIBERO 7-D padded to 16-D with zeros
        "reward":  float32 (N,)  zeros (LIBERO has no native reward; not used
                                  by the tokenizer; only fed to dynamics —
                                  see README for caveat)
      }

  <out_root>/libero_tasks.json
      {<task>: {"action_dim": 7,
                 "text_embedding": [0]*512,
                 "max_episode_steps": <max len>,
                 "instruction": "<file>"}}

Task naming: ``<suite>__<file_stem>`` (e.g.
"libero_spatial__pick_up_the_black_bowl_on_the_ramekin_..."). The
file_stem is the .hdf5 filename without extension. This guarantees
LIBERO's 130 (or 104 train) tasks each become one Dreamer-4 "task".

Two camera modes:
  --cameras agentview      (default) 1-cam, 3-channel, compatible with
                              nicklashansen's HF checkpoint architecture
  --cameras agentview,eye_in_hand    2-cam, 6-channel; requires changes to
                              nicklashansen's ShardedFrameDataset (drops the
                              frames.shape[1] != 3 filter) and train_tokenizer
                              (set --C 6). The shards we write are still
                              uint8 but with 6 channels. See README.

Usage:
    uv run python scripts/preprocess_libero_for_dreamer4.py \\
        --split data/splits_taskheld/train.json \\
        --out-root data/libero_dreamer4 \\
        --cameras agentview \\
        --resolution 128

By default we read every demo listed in the split JSON. Add
``--smoke`` to process only 2 tasks for a quick sanity check.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

import h5py
import numpy as np
import torch
import torch.nn.functional as F

SHARD_SIZE = 2048
A_PAD = 16          # WMDataset.A default
LANG_DIM = 512


def _task_name(suite: str, file_name: str) -> str:
    stem = Path(file_name).stem        # strip .hdf5
    return f"{suite}__{stem}"


def _resize(frames_uint8: np.ndarray, size: int) -> np.ndarray:
    """frames_uint8: (T, H, W, 3) uint8 → (T, 3, size, size) uint8."""
    if frames_uint8.shape[1] == size and frames_uint8.shape[2] == size:
        return frames_uint8.transpose(0, 3, 1, 2).copy()
    t = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).contiguous().float() / 255.0
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return (t.clamp(0, 1) * 255.0).to(torch.uint8).numpy()


def _save_shards(frames_chw_uint8: torch.Tensor, out_dir: Path, prefix: str):
    """frames_chw_uint8: (N, C, H, W) uint8. Splits into SHARD_SIZE-frame shards."""
    out_dir.mkdir(parents=True, exist_ok=True)
    N = frames_chw_uint8.shape[0]
    for i, lo in enumerate(range(0, N, SHARD_SIZE)):
        hi = min(lo + SHARD_SIZE, N)
        shard = frames_chw_uint8[lo:hi].contiguous().clone()
        torch.save({"frames": shard}, out_dir / f"{prefix}_shard{i:04d}.pt")


def _process_task(
    task: str,
    demos: list[tuple[str, str, str]],     # list of (suite, file, demo_key)
    datasets_dir: Path,
    cameras: list[str],
    resolution: int,
    out_frames_root: Path,
    out_demos_root: Path,
) -> dict:
    """Read every demo for one LIBERO task, write shards + demo .pt."""
    eps_list, acts_list, rews_list = [], [], []
    all_frames: list[np.ndarray] = []     # list of (T, C, H, W) uint8
    ep_offset = 0
    max_ep_len = 0
    total_frames = 0

    # group demos by file to avoid reopening HDF5 per demo
    by_file: dict[tuple[str, str], list[str]] = defaultdict(list)
    for suite, fname, dkey in demos:
        by_file[(suite, fname)].append(dkey)

    for (suite, fname), demo_keys in by_file.items():
        hdf5_path = datasets_dir / suite / fname
        with h5py.File(hdf5_path, "r") as f:
            for dkey in demo_keys:
                grp = f["data"][dkey]
                per_cam = []
                for cam in cameras:
                    arr = np.asarray(grp["obs"][cam])      # (T, H, W, 3) uint8
                    per_cam.append(_resize(arr, resolution))   # (T, 3, S, S)
                # channel-concat across cams: (T, 3*ncam, S, S)
                clip = np.concatenate(per_cam, axis=1)
                T = clip.shape[0]
                acts = np.asarray(grp["actions"]).astype(np.float32)   # (T, 7)
                if acts.shape[0] != T:
                    # rare: actions logged off by one; truncate to min
                    T_min = min(acts.shape[0], T)
                    clip = clip[:T_min]
                    acts = acts[:T_min]
                    T = T_min
                # pad action to A_PAD (16)
                if acts.shape[-1] < A_PAD:
                    pad = np.zeros((T, A_PAD - acts.shape[-1]), dtype=np.float32)
                    acts = np.concatenate([acts, pad], axis=-1)
                else:
                    acts = acts[:, :A_PAD]
                ep_ids = np.full((T,), ep_offset, dtype=np.int64)
                rews = np.zeros((T,), dtype=np.float32)

                all_frames.append(clip)
                eps_list.append(ep_ids)
                acts_list.append(acts)
                rews_list.append(rews)

                ep_offset += 1
                max_ep_len = max(max_ep_len, T)
                total_frames += T

    if not all_frames:
        return {"task": task, "frames": 0, "demos": 0}

    frames_np = np.concatenate(all_frames, axis=0)        # (N, C, H, W)
    frames_t = torch.from_numpy(frames_np).contiguous()   # uint8
    ep_t = torch.from_numpy(np.concatenate(eps_list, axis=0))
    act_t = torch.from_numpy(np.concatenate(acts_list, axis=0))
    rew_t = torch.from_numpy(np.concatenate(rews_list, axis=0))

    # write shards
    _save_shards(frames_t, out_frames_root / task, prefix=task)
    # write demo metadata
    out_demos_root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"episode": ep_t, "action": act_t, "reward": rew_t},
        out_demos_root / f"{task}.pt",
    )
    return {"task": task, "frames": total_frames, "demos": len(demos),
            "max_ep_len": max_ep_len}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/splits_taskheld/train.json")
    ap.add_argument("--out-root", default="data/libero_dreamer4")
    ap.add_argument("--cameras", default="agentview_rgb",
                    help="comma-separated; e.g. agentview_rgb or "
                          "agentview_rgb,eye_in_hand_rgb")
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--smoke", action="store_true",
                    help="process only 2 tasks (for quick sanity test)")
    ap.add_argument("--limit-tasks", type=int, default=None,
                    help="process at most N tasks (for partial runs)")
    args = ap.parse_args()

    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    assert cameras, "at least one camera required"
    print(f"cameras={cameras} resolution={args.resolution}")

    split_path = REPO_ROOT / args.split
    blob = json.loads(split_path.read_text())
    datasets_dir = Path(blob["meta"]["datasets_dir"])
    eps = blob["episodes"]

    # group episodes by task
    by_task: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for e in eps:
        key = _task_name(e["suite"], e["file"])
        by_task[key].append((e["suite"], e["file"], e["demo"]))

    tasks = sorted(by_task.keys())
    if args.smoke:
        tasks = tasks[:2]
    if args.limit_tasks:
        tasks = tasks[:args.limit_tasks]

    out_root = REPO_ROOT / args.out_root
    out_frames = out_root / "frames"
    out_demos = out_root / "demos"
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    tasks_meta: dict[str, dict] = {}
    for i, t in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {t}  ({len(by_task[t])} demos)")
        info = _process_task(
            t, by_task[t], datasets_dir,
            cameras=cameras, resolution=args.resolution,
            out_frames_root=out_frames, out_demos_root=out_demos,
        )
        summary.append(info)
        print(f"  -> {info['frames']} frames in {info['demos']} demos "
              f"(max_ep_len={info.get('max_ep_len', 0)})")
        tasks_meta[t] = {
            "action_dim": 7,
            "max_episode_steps": int(info.get("max_ep_len", 0)),
            "instruction": Path(by_task[t][0][1]).stem.replace("_", " "),
            "text_embedding": [0.0] * LANG_DIM,
        }

    # write tasks.json compatible with WMDataset
    (out_root / "libero_tasks.json").write_text(
        json.dumps(tasks_meta, indent=2))

    total_frames = sum(s["frames"] for s in summary)
    total_demos = sum(s["demos"] for s in summary)
    print(f"\nDone. wrote {len(summary)} tasks, {total_demos} demos, {total_frames} frames.")
    print(f"  frames root: {out_frames}")
    print(f"  demos  root: {out_demos}")
    print(f"  tasks meta:  {out_root / 'libero_tasks.json'}")


if __name__ == "__main__":
    main()
