"""Random per-token masking and patch-pixel reconstruction loss."""
from __future__ import annotations

import torch
from einops import rearrange


def random_token_mask(
    batch: int, num_frames: int, num_patches: int, mask_ratio: float, device=None
) -> torch.Tensor:
    """Random per-frame mask of shape `(B, T, N)`, True = masked.

    Exactly `floor(N * mask_ratio)` patches are masked per frame, drawn
    independently per (batch, time).
    """
    n_mask = int(num_patches * mask_ratio)
    if n_mask <= 0:
        return torch.zeros(batch, num_frames, num_patches, dtype=torch.bool, device=device)
    noise = torch.rand(batch, num_frames, num_patches, device=device)
    rank = noise.argsort(dim=-1).argsort(dim=-1)
    return rank < n_mask


def tube_token_mask(
    batch: int, num_frames: int, num_patches: int, mask_ratio: float, device=None
) -> torch.Tensor:
    """Tube mask of shape `(B, T, N)`, True = masked.

    A single random spatial pattern per batch element is shared across
    all `T` frames (a.k.a. VideoMAE-style tube masking). Encourages
    temporal-prediction signal because masked locations cannot be
    inferred from any frame in the clip.
    """
    n_mask = int(num_patches * mask_ratio)
    if n_mask <= 0:
        return torch.zeros(batch, num_frames, num_patches, dtype=torch.bool, device=device)
    noise = torch.rand(batch, num_patches, device=device)
    rank = noise.argsort(dim=-1).argsort(dim=-1)
    spatial = rank < n_mask  # (B, N)
    return spatial.unsqueeze(1).expand(batch, num_frames, num_patches).contiguous()


def rollout_block_mask(
    batch: int, num_frames: int, num_patches: int, mask_ratio: float, device=None
) -> torch.Tensor:
    """Block-future mask: pick a random split per sample and mask all
    frames after it (including their patches). The ``mask_ratio`` arg is
    unused for this mode; the split point is sampled uniformly.

    Trains the model on the same protocol used at rollout-eval time
    (clean prefix -> predict the rest), making MAE rollout PSNR a fair
    comparison with diffusion variants.
    """
    # split point in [1, T - 1] so at least one frame is given and at least one is masked
    if num_frames < 2:
        return torch.zeros(batch, num_frames, num_patches, dtype=torch.bool, device=device)
    splits = torch.randint(1, num_frames, (batch,), device=device)
    frame_idx = torch.arange(num_frames, device=device).unsqueeze(0)  # (1, T)
    mask_per_frame = frame_idx >= splits.unsqueeze(1)  # (B, T)
    return mask_per_frame.unsqueeze(-1).expand(batch, num_frames, num_patches).contiguous()


def make_mask(
    mask_type: str,
    batch: int,
    num_frames: int,
    num_patches: int,
    mask_ratio: float,
    device=None,
) -> torch.Tensor:
    """Dispatch to a masking strategy by name (`random` | `tube` | `rollout_block`)."""
    if mask_type == "random":
        return random_token_mask(batch, num_frames, num_patches, mask_ratio, device)
    if mask_type == "tube":
        return tube_token_mask(batch, num_frames, num_patches, mask_ratio, device)
    if mask_type == "rollout_block":
        return rollout_block_mask(batch, num_frames, num_patches, mask_ratio, device)
    raise ValueError(f"unknown mask_type: {mask_type!r}")


def patchify(clip: torch.Tensor, patch_size: int) -> torch.Tensor:
    """`(B,T,C,H,W) -> (B,T,N,P*P*C)`."""
    return rearrange(
        clip,
        "b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)",
        p1=patch_size, p2=patch_size,
    )


def unpatchify(
    x: torch.Tensor, patch_size: int, image_size: int, channels: int
) -> torch.Tensor:
    """`(B,T,N,P*P*C) -> (B,T,C,H,W)`."""
    h = w = image_size // patch_size
    return rearrange(
        x,
        "b t (h w) (p1 p2 c) -> b t c (h p1) (w p2)",
        h=h, w=w, p1=patch_size, p2=patch_size, c=channels,
    )


def masked_pixel_loss(
    pred_patches: torch.Tensor,
    gt_clip: torch.Tensor,
    mask_indicator: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """MSE on masked patches only (mean over masked elements)."""
    gt_patches = patchify(gt_clip, patch_size)
    err = (pred_patches - gt_patches).pow(2).mean(dim=-1)  # (B, T, N)
    m = mask_indicator.float()
    return (err * m).sum() / m.sum().clamp_min(1.0)
