"""Counterfactual rollouts for an action-conditioned ckpt.

Given the same prefix observations, sweep four different action
sequences and write the resulting predicted clips side by side as a
single mp4 per test episode. Useful for verifying that the model is
genuinely action-conditioned rather than ignoring the conditioning.

Action mutations:
  gt        — the ground-truth action sequence (baseline)
  zero      — all-zero actions (no-op control)
  reversed  — same sequence reversed in time
  random    — actions sampled from another test episode

Output mp4 layout (vertical stack of 4 rows, gt frames overlaid as the
top reference):
  row 0:  ground-truth full clip
  row 1:  predicted with gt actions
  row 2:  predicted with zero actions
  row 3:  predicted with reversed actions
  row 4:  predicted with shuffled actions

Run:
    uv run python scripts/counterfactual_rollout.py \\
        --ckpt runs/sweep/variants/A_diff_v5_pc_cos_ema_action_100k/ckpt_last.pt \\
        --out-dir runs/counterfactual/v5 --num-samples 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import imageio.v3 as iio
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.libero_video import LiberoVideoClips, worker_init_fn
from models.block_causal_transformer import BlockCausalVideoTransformer
from training.diffusion_objective import (
    ddim_rollout_prefix_conditional,
    make_schedule,
)


def build_model(cfg, in_channels: int) -> BlockCausalVideoTransformer:
    return BlockCausalVideoTransformer(
        in_channels=in_channels,
        patch_size=cfg.model.patch_size,
        num_frames=cfg.data.clip_len,
        image_size=cfg.data.resolution,
        dim=cfg.model.dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=0.0,
        enable_timestep=True,
        action_dim=int(cfg.model.get("action_dim", 0)),
    )


def to_uint8(t: torch.Tensor) -> np.ndarray:
    t = t.clamp(0, 1).float().cpu().numpy()
    t = np.transpose(t, (0, 2, 3, 1))
    return (t * 255.0).round().astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-samples", type=int, default=6)
    ap.add_argument("--rollout-prefix", type=int, default=None)
    ap.add_argument("--diff-sample-steps", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["config"])
    assert int(cfg.model.get("action_dim", 0)) > 0, "ckpt is not action-conditioned"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_channels = 3 * len(cfg.data.cameras)
    test_ds = LiberoVideoClips(
        split_path=cfg.data.test_split,
        datasets_dir=cfg.data.datasets_dir,
        clip_len=cfg.data.clip_len,
        cameras=cfg.data.cameras,
        resolution=cfg.data.resolution,
        seed=cfg.train.seed + 1,
        return_actions=True,
    )
    loader = DataLoader(
        test_ds, batch_size=cfg.eval.batch_size, shuffle=False,
        num_workers=2, worker_init_fn=worker_init_fn,
    )

    model = build_model(cfg, in_channels).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    P = cfg.model.patch_size
    H = cfg.data.resolution
    T = cfg.data.clip_len
    L_pre = args.rollout_prefix or int(cfg.eval.get("rollout_prefix", T // 4))
    L_post = T - L_pre
    sched = make_schedule(
        num_steps=int(cfg.train.get("diff_num_steps", 1000)),
        schedule=cfg.train.get("diff_schedule", "linear"),
        device=device,
    )

    @torch.no_grad()
    def roll(prefix_clip, actions):
        return ddim_rollout_prefix_conditional(
            model, prefix_clip, L_post, sched,
            num_sampling_steps=args.diff_sample_steps,
            autocast_dtype=torch.bfloat16 if device.type == "cuda" else None,
            actions=actions,
        )

    sample_idx = 0
    cached_random_action = None
    for clip, actions in loader:
        if sample_idx >= args.num_samples:
            break
        clip = clip.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        B = clip.shape[0]
        prefix = clip[:, :L_pre]

        # action mutations -- all of shape (B, T, 7)
        a_gt = actions
        a_zero = torch.zeros_like(actions)
        a_reversed = torch.flip(actions, dims=(1,))
        if cached_random_action is None or cached_random_action.shape[0] < B:
            cached_random_action = actions.flip(0).clone()
        a_random = cached_random_action[:B]

        rollouts = {}
        for name, a in [("gt", a_gt), ("zero", a_zero), ("reversed", a_reversed), ("random", a_random)]:
            rollouts[name] = roll(prefix, a).float().clamp(0, 1)

        for b in range(min(B, args.num_samples - sample_idx)):
            rows = [clip[b, :, :3]]
            rows.extend([rollouts[k][b, :, :3] for k in ("gt", "zero", "reversed", "random")])
            stacked = torch.cat(rows, dim=-2)  # (T, 3, 5*H, W)
            iio.imwrite(
                out_dir / f"sample_{sample_idx:03d}.mp4",
                to_uint8(stacked), fps=10, codec="libx264",
            )
            sample_idx += 1

    print(f"wrote {sample_idx} counterfactual mp4(s) to {out_dir}")
    print("rows top->bottom: gt clip | rollout(gt actions) | rollout(zero) | rollout(reversed) | rollout(random)")


if __name__ == "__main__":
    main()
