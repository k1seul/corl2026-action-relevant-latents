"""Evaluate a trained model on the test split.

Two evaluation modes (both run by default):

* **inpainting** — random per-frame masking at `cfg.train.mask_ratio`;
  report masked-patch MSE and PSNR. Sample mp4 is
  `[masked_input | reconstruction | gt]`.
* **rollout** — feed the first `cfg.eval.rollout_prefix` frames as
  prefix and predict the rest in a single block-causal forward pass
  (variant A behaviour). Report per-frame PSNR/SSIM on the predicted
  span and overall means. Sample mp4 is `[predicted_clip | gt]`.

Run:
    uv run python scripts/eval.py --ckpt path/to/ckpt_last.pt
"""
from __future__ import annotations

import argparse
import math
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
from models.token_video_transformer import TokenVideoTransformer
from models.tokenizer import FSQTokenizer
from training.diffusion_objective import (
    ddim_rollout_prefix_conditional,
    ddpm_rollout_with_prefix,
    make_schedule,
)
from training.flow_matching import flow_rollout_prefix_conditional
from training.mask_objective import make_mask, masked_pixel_loss, patchify, unpatchify
from training.metrics import psnr_from_mse, ssim


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
        enable_timestep=(cfg.train.get("objective", "mae") in ("diffusion", "flow_matching")),
        action_dim=int(cfg.model.get("action_dim", 0)),
    )


def load_tokenizer(ckpt_path: str, in_channels: int, device: torch.device) -> FSQTokenizer:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    tcfg = state["config"]
    tok = FSQTokenizer(
        in_channels=in_channels,
        levels=list(tcfg["model"]["levels"]),
        base_ch=tcfg["model"]["base_ch"],
    ).to(device)
    tok.load_state_dict(state["model"])
    tok.eval()
    return tok


def build_token_model(cfg, codebook_size: int, num_tokens_per_frame: int) -> TokenVideoTransformer:
    return TokenVideoTransformer(
        codebook_size=codebook_size,
        num_tokens_per_frame=num_tokens_per_frame,
        num_frames=cfg.data.clip_len,
        dim=cfg.model.dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=0.0,
        action_dim=int(cfg.model.get("action_dim", 0)),
    )


@torch.no_grad()
def encode_clip_to_tokens(tok: FSQTokenizer, clip: torch.Tensor) -> torch.Tensor:
    B, T, C, H, W = clip.shape
    flat = clip.reshape(B * T, C, H, W)
    _, idx_grid = tok.encode(flat)
    return idx_grid.reshape(B, T, -1)


@torch.no_grad()
def decode_tokens_to_clip(tok: FSQTokenizer, token_ids: torch.Tensor, hw: int) -> torch.Tensor:
    """`(B, T, N) -> (B, T, C, H, W)`."""
    B, T, N = token_ids.shape
    h = w = hw
    flat = token_ids.reshape(B * T, h, w)
    img = tok.decode_indices(flat)  # (B*T, C, H, W)
    C, H, W = img.shape[1:]
    return img.reshape(B, T, C, H, W)


@torch.no_grad()
def rollout_token(
    model: TokenVideoTransformer,
    tokenizer: FSQTokenizer,
    prefix_clip: torch.Tensor,
    num_post: int,
    sample_temperature: float = 1.0,
    autoregressive: bool = True,
    actions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Variant-(B) rollout via categorical sampling over the codebook.

    With ``autoregressive=True`` (default), generate one frame at a time:
    each new frame is sampled given the prefix and all previously sampled
    frames as clean context. This matches how the model would dream a
    rollout in a planning setting and substantially beats single-shot
    prediction at long horizons.
    """
    B, L_pre = prefix_clip.shape[:2]
    device = prefix_clip.device
    prefix_tokens = encode_clip_to_tokens(tokenizer, prefix_clip)  # (B, L_pre, N)
    N = prefix_tokens.shape[-1]
    L = L_pre + num_post
    hw = int(round(N ** 0.5))

    out_ids = torch.zeros(B, L, N, dtype=torch.long, device=device)
    out_ids[:, :L_pre] = prefix_tokens

    if not autoregressive:
        mask = torch.zeros(B, L, N, dtype=torch.bool, device=device)
        mask[:, L_pre:] = True
        logits = model(out_ids, mask, actions=actions)
        if sample_temperature <= 0:
            sampled = logits.argmax(dim=-1)
        else:
            probs = torch.softmax(logits / sample_temperature, dim=-1)
            flat = probs.reshape(-1, probs.shape[-1])
            sampled = torch.multinomial(flat, 1).reshape(B, L, N)
        out_ids[:, L_pre:] = sampled[:, L_pre:]
        return decode_tokens_to_clip(tokenizer, out_ids, hw)

    # frame-by-frame autoregressive
    for f in range(L_pre, L):
        cur_ids = out_ids[:, : f + 1]  # (B, f+1, N)
        cur_mask = torch.zeros(B, f + 1, N, dtype=torch.bool, device=device)
        cur_mask[:, f] = True
        cur_actions = actions[:, : f + 1] if actions is not None else None
        logits = model(cur_ids, cur_mask, actions=cur_actions)  # (B, f+1, N, V)
        logits_f = logits[:, f]  # (B, N, V)
        if sample_temperature <= 0:
            sampled = logits_f.argmax(dim=-1)
        else:
            probs = torch.softmax(logits_f / sample_temperature, dim=-1)
            sampled = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(B, N)
        out_ids[:, f] = sampled

    return decode_tokens_to_clip(tokenizer, out_ids, hw)


def to_uint8_video(t: torch.Tensor) -> np.ndarray:
    t = t.clamp(0, 1).float().cpu().numpy()
    t = np.transpose(t, (0, 2, 3, 1))
    return (t * 255.0).round().astype(np.uint8)


# ---- variant-(A) prediction interfaces ----


@torch.no_grad()
def reconstruct_inpainting(
    model: BlockCausalVideoTransformer,
    clip: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int,
    image_size: int,
    channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_patches = model(clip, mask)
    gt_patches = patchify(clip, patch_size)
    composite_patches = torch.where(mask.unsqueeze(-1), pred_patches, gt_patches)
    composite = unpatchify(composite_patches, patch_size, image_size, channels)
    return composite, pred_patches


@torch.no_grad()
def rollout_a(
    model: BlockCausalVideoTransformer,
    prefix_clip: torch.Tensor,
    num_post: int,
    patch_size: int,
    image_size: int,
    channels: int,
) -> torch.Tensor:
    """Variant-A rollout: single-shot block-causal prediction of post frames.

    Concatenates `prefix_clip` with zero-padded post frames, masks all
    post-frame patches, runs one forward pass, and substitutes model
    predictions for the post frames.
    """
    B, L_pre, C, H, W = prefix_clip.shape
    L = L_pre + num_post
    N = (image_size // patch_size) ** 2
    pad = torch.zeros(B, num_post, C, H, W, device=prefix_clip.device, dtype=prefix_clip.dtype)
    full = torch.cat([prefix_clip, pad], dim=1)
    mask = torch.zeros(B, L, N, dtype=torch.bool, device=prefix_clip.device)
    mask[:, L_pre:] = True
    pred_patches = model(full, mask)
    pred_clip = unpatchify(pred_patches, patch_size, image_size, channels)
    out = full.clone()
    out[:, L_pre:] = pred_clip[:, L_pre:]
    return out


# ---- mp4 writers ----


def write_inpainting_mp4(
    out_path: Path,
    masked_input: torch.Tensor,
    composite: torch.Tensor,
    gt: torch.Tensor,
    fps: int = 10,
) -> None:
    cols = [t[:, :3] for t in (masked_input, composite, gt)]
    iio.imwrite(out_path, to_uint8_video(torch.cat(cols, dim=-1)), fps=fps, codec="libx264")


def write_rollout_mp4(
    out_path: Path,
    predicted_full: torch.Tensor,
    gt: torch.Tensor,
    fps: int = 10,
) -> None:
    cols = [t[:, :3] for t in (predicted_full, gt)]
    iio.imwrite(out_path, to_uint8_video(torch.cat(cols, dim=-1)), fps=fps, codec="libx264")


# ---- evaluation pipelines ----


def run_inpainting_eval(
    model, loader, cfg, in_channels: int, out_dir: Path,
    num_clips: int, sample_count: int, device: torch.device,
) -> dict:
    P = cfg.model.patch_size
    H = cfg.data.resolution
    N = (H // P) ** 2

    total_mse, total_n, seen, sample_idx = 0.0, 0, 0, 0
    sample_dir = out_dir / "samples_inpainting"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for clip in loader:
        clip = clip.to(device, non_blocking=True)
        B, T = clip.shape[:2]
        torch.manual_seed(cfg.train.seed + 100 + seen)
        mask = make_mask(cfg.train.mask_type, B, T, N, cfg.train.mask_ratio, device=device)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
            composite, pred_patches = reconstruct_inpainting(model, clip, mask, P, H, in_channels)
            mse = masked_pixel_loss(pred_patches, clip, mask, P)

        m = mask.float().sum().item()
        total_mse += mse.item() * m
        total_n += m

        if sample_idx < sample_count:
            keep = (~mask).float().unsqueeze(-1)
            kept_patches = patchify(clip, P) * keep
            masked_in = unpatchify(kept_patches, P, H, in_channels)
            for b in range(min(B, sample_count - sample_idx)):
                write_inpainting_mp4(
                    sample_dir / f"sample_{sample_idx:03d}.mp4",
                    masked_in[b].float(), composite[b].float(), clip[b],
                )
                sample_idx += 1

        seen += B
        if seen >= num_clips:
            break

    mean_mse = total_mse / max(1.0, total_n)
    psnr = 10.0 * math.log10(1.0 / max(1e-12, mean_mse))
    return {
        "inpainting/clips": seen,
        "inpainting/masked_mse": mean_mse,
        "inpainting/masked_psnr_db": psnr,
        "inpainting/sample_count": sample_idx,
    }


def run_rollout_eval(
    model, loader, cfg, in_channels: int, out_dir: Path,
    num_clips: int, sample_count: int, device: torch.device,
    prefix_frames: int,
) -> dict:
    P = cfg.model.patch_size
    H = cfg.data.resolution
    T = cfg.data.clip_len
    L_pre = max(1, min(prefix_frames, T - 1))
    L_post = T - L_pre
    objective = cfg.train.get("objective", "mae")

    sample_dir = out_dir / "samples_rollout"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # diffusion sampler setup (only used by variant A'')
    if objective == "diffusion":
        diff_T = int(cfg.train.get("diff_num_steps", 1000))
        sample_steps = int(cfg.eval.get("diff_sample_steps", 50))
        diff_schedule = cfg.train.get("diff_schedule", "linear")
        diff_prefix_cond = bool(cfg.train.get("diff_prefix_conditional", False))
        sched = make_schedule(num_steps=diff_T, schedule=diff_schedule, device=device)
    else:
        sched = None
        sample_steps = 0
        diff_prefix_cond = False

    # accumulators per post-frame index
    sum_mse = torch.zeros(L_post, device=device)
    sum_ssim = torch.zeros(L_post, device=device)
    n_seen = 0
    sample_idx = 0

    for batch in loader:
        if isinstance(batch, (tuple, list)):
            clip, actions = batch
            actions = actions.to(device, non_blocking=True)
        else:
            clip = batch
            actions = None
        clip = clip.to(device, non_blocking=True)
        B = clip.shape[0]
        prefix = clip[:, :L_pre]
        gt_post = clip[:, L_pre:]

        if objective == "mae":
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                pred_full = rollout_a(model, prefix, L_post, P, H, in_channels)
        elif objective == "diffusion":
            if diff_prefix_cond:
                pred_full = ddim_rollout_prefix_conditional(
                    model, prefix, L_post, sched,
                    num_sampling_steps=sample_steps,
                    autocast_dtype=torch.bfloat16 if device.type == "cuda" else None,
                    actions=actions,
                )
            else:
                pred_full = ddpm_rollout_with_prefix(
                    model, prefix, L_post, sched,
                    num_sampling_steps=sample_steps,
                    autocast_dtype=torch.bfloat16 if device.type == "cuda" else None,
                )
        elif objective == "flow_matching":
            fm_steps = int(cfg.eval.get("flow_sample_steps", 50))
            pred_full = flow_rollout_prefix_conditional(
                model, prefix, L_post,
                num_sampling_steps=fm_steps,
                autocast_dtype=torch.bfloat16 if device.type == "cuda" else None,
                actions=actions,
            )
        elif objective == "token_ce":
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                pred_full = rollout_token(
                    model, getattr(model, "_tokenizer"), prefix, L_post,
                    sample_temperature=cfg.eval.get("token_temperature", 1.0),
                    actions=actions,
                )
        else:
            raise ValueError(f"unknown objective: {objective!r}")
        pred_post = pred_full[:, L_pre:].float().clamp(0, 1)
        gt_post_f = gt_post.float()

        # per-frame MSE and SSIM, averaged across batch+channel+space
        diff2 = (pred_post - gt_post_f).pow(2).mean(dim=(0, 2, 3, 4))  # (L_post,)
        sum_mse += diff2 * B
        for f in range(L_post):
            sum_ssim[f] += ssim(pred_post[:, f, :3], gt_post_f[:, f, :3]) * B

        if sample_idx < sample_count:
            for b in range(min(B, sample_count - sample_idx)):
                write_rollout_mp4(
                    sample_dir / f"sample_{sample_idx:03d}.mp4",
                    pred_full[b].float(), clip[b],
                )
                sample_idx += 1

        n_seen += B
        if n_seen >= num_clips:
            break

    mean_mse_per_frame = (sum_mse / max(1, n_seen)).cpu().tolist()
    mean_ssim_per_frame = (sum_ssim / max(1, n_seen)).cpu().tolist()
    mean_psnr_per_frame = [10.0 * math.log10(1.0 / max(1e-12, m)) for m in mean_mse_per_frame]
    mean_mse = sum(mean_mse_per_frame) / max(1, len(mean_mse_per_frame))
    mean_psnr = 10.0 * math.log10(1.0 / max(1e-12, mean_mse))
    mean_ssim = sum(mean_ssim_per_frame) / max(1, len(mean_ssim_per_frame))

    return {
        "rollout/prefix_frames": L_pre,
        "rollout/predicted_frames": L_post,
        "rollout/clips": n_seen,
        "rollout/mse_mean": mean_mse,
        "rollout/psnr_db_mean": mean_psnr,
        "rollout/ssim_mean": mean_ssim,
        "rollout/psnr_db_per_frame": ",".join(f"{p:.3f}" for p in mean_psnr_per_frame),
        "rollout/ssim_per_frame": ",".join(f"{s:.4f}" for s in mean_ssim_per_frame),
        "rollout/sample_count": sample_idx,
    }


# ---- driver ----


def write_metrics(out_dir: Path, all_metrics: dict) -> None:
    # also keep flat backward-compatible keys for existing summary.md parsers
    flat = dict(all_metrics)
    if "inpainting/masked_psnr_db" in flat:
        flat["clips_evaluated"] = flat["inpainting/clips"]
        flat["masked_patch_mse"] = f"{flat['inpainting/masked_mse']:.6f}"
        flat["masked_patch_psnr_db"] = f"{flat['inpainting/masked_psnr_db']:.4f}"
    with (out_dir / "metrics.txt").open("w") as f:
        for k, v in flat.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.6f}\n")
            else:
                f.write(f"{k}: {v}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--config", type=str, default=None,
                    help="override config; default uses ckpt's saved config")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--num-clips", type=int, default=None)
    ap.add_argument("--sample-count", type=int, default=None)
    ap.add_argument("--mode", choices=["inpainting", "rollout", "both"], default="both")
    ap.add_argument("--rollout-prefix", type=int, default=None,
                    help="number of prefix frames for rollout eval")
    ap.add_argument("--diff-sample-steps", type=int, default=None,
                    help="DDIM sampling steps for the diffusion variant")
    ap.add_argument("--flow-sample-steps", type=int, default=None,
                    help="Euler sampling steps for the flow matching variant")
    ap.add_argument("--test-split", type=str, default=None,
                    help="override cfg.data.test_split (e.g. for OOD eval)")
    args = ap.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["config"]) if args.config is None else OmegaConf.load(args.config)
    if args.test_split is not None:
        cfg.data.test_split = args.test_split

    out_dir = Path(args.out_dir or cfg.eval.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_clips = args.num_clips or cfg.eval.num_clips
    sample_count = args.sample_count or cfg.eval.sample_count
    prefix = args.rollout_prefix or int(cfg.eval.get("rollout_prefix", max(1, cfg.data.clip_len // 4)))
    if args.diff_sample_steps is not None:
        cfg.eval.diff_sample_steps = args.diff_sample_steps
    if args.flow_sample_steps is not None:
        cfg.eval.flow_sample_steps = args.flow_sample_steps

    in_channels = 3 * len(cfg.data.cameras)
    action_dim = int(cfg.model.get("action_dim", 0))
    use_actions = action_dim > 0
    test_ds = LiberoVideoClips(
        split_path=cfg.data.test_split,
        datasets_dir=cfg.data.datasets_dir,
        clip_len=cfg.data.clip_len,
        cameras=cfg.data.cameras,
        resolution=cfg.data.resolution,
        seed=cfg.train.seed + 1,
        return_actions=use_actions,
    )

    def make_loader():
        return DataLoader(
            test_ds, batch_size=cfg.eval.batch_size, shuffle=False,
            num_workers=2, worker_init_fn=worker_init_fn,
        )

    objective = cfg.train.get("objective", "mae")
    if objective == "token_ce":
        tokenizer = load_tokenizer(cfg.train.tokenizer_ckpt, in_channels, device)
        # determine token grid size from a sample
        peek_batch = next(iter(make_loader()))
        peek_clip = peek_batch[0] if isinstance(peek_batch, (tuple, list)) else peek_batch
        peek = peek_clip[:1, :1].to(device)
        with torch.no_grad():
            sample_ids = encode_clip_to_tokens(tokenizer, peek)
        N_per_frame = sample_ids.shape[-1]
        model = build_token_model(cfg, tokenizer.codebook_size, N_per_frame).to(device)
        model.load_state_dict(state["model"])
        # stash tokenizer on the model for the rollout function
        model._tokenizer = tokenizer
    else:
        model = build_model(cfg, in_channels).to(device)
        model.load_state_dict(state["model"])
    model.eval()

    metrics: dict = {"objective": objective}

    # inpainting eval is only meaningful for the MAE variant
    if args.mode in ("inpainting", "both") and objective == "mae":
        metrics.update(run_inpainting_eval(
            model, make_loader(), cfg, in_channels, out_dir,
            num_clips, sample_count, device,
        ))
    if args.mode in ("rollout", "both"):
        metrics.update(run_rollout_eval(
            model, make_loader(), cfg, in_channels, out_dir,
            num_clips, sample_count, device, prefix,
        ))

    write_metrics(out_dir, metrics)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print(f"output dir: {out_dir}")


if __name__ == "__main__":
    main()
