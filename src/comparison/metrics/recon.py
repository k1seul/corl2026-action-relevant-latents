"""Reconstruction-quality metrics for the Full Results table.

Three metrics, all averaged over (clips × predicted frames):
  - recon_mse : pixel MSE in [0, 1]
  - psnr      : -10 log10(mse + eps)
  - lpips     : perceptual distance (AlexNet backbone, official lpips lib)

LPIPS expects inputs in [-1, 1]; we convert internally. Encoder-only models
(V-JEPA, LAPA) cannot produce pixel rollouts — call sites should mark
NA / NaN rather than invoking this module.

Rollout protocol (matches scripts/eval.py):
  - Use the first ``rollout_prefix`` frames as clean context
  - Predict the remaining (clip_len - rollout_prefix) frames
  - Compute metrics only on the predicted span
"""
from __future__ import annotations

import math
from typing import Optional

import torch


def _ensure_lpips(device: torch.device):
    """Lazy import + cache. LPIPS prints a banner on first init that we
    suppress (verbose=False)."""
    import lpips as _lpips
    if not hasattr(_ensure_lpips, "_cache"):
        _ensure_lpips._cache = {}
    cache = _ensure_lpips._cache
    key = str(device)
    if key not in cache:
        net = _lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        cache[key] = net
    return cache[key]


def _split_cams(clip_3xC: torch.Tensor) -> torch.Tensor:
    """clip is (B, T, C, H, W) with C = 3 * n_cams. Reshape into
    (B, n_cams, T, 3, H, W) for metric computation cam-by-cam, then
    average — every cam is weighted equally.
    """
    B, T, C, H, W = clip_3xC.shape
    n_cams = C // 3
    return clip_3xC.reshape(B, T, n_cams, 3, H, W).permute(0, 2, 1, 3, 4, 5)


def _per_cam_mean(metric_fn, pred: torch.Tensor, gt: torch.Tensor) -> float:
    """metric_fn(pred_cam, gt_cam) returns a scalar tensor or float per cam.
    Inputs are (B, n_cams, T, 3, H, W). Average over cams.
    """
    n_cams = pred.shape[1]
    vals = []
    for c in range(n_cams):
        vals.append(float(metric_fn(pred[:, c], gt[:, c])))
    return sum(vals) / max(1, n_cams)


def recon_psnr_lpips(
    pred_clip: torch.Tensor,
    gt_clip: torch.Tensor,
    rollout_prefix: int,
    device: torch.device,
    compute_lpips: bool = True,
) -> dict[str, float]:
    """Compute MSE / PSNR / LPIPS on the predicted span of one rollout.

    Inputs:
      pred_clip: (B, T, C, H, W) full clip with predictions in
        ``frames[rollout_prefix:]`` and identical-to-gt prefix.
      gt_clip:   (B, T, C, H, W) ground truth, same shape.

    Outputs (dict, MSE/PSNR in pixel space [0,1]):
      {recon_mse, psnr, lpips}
    """
    assert pred_clip.shape == gt_clip.shape
    pred = pred_clip[:, rollout_prefix:].clamp(0, 1).to(device).float()
    gt = gt_clip[:, rollout_prefix:].clamp(0, 1).to(device).float()
    # MSE / PSNR over all pixels of the predicted span
    diff = (pred - gt)
    mse_val = diff.pow(2).mean().item()
    psnr_val = -10.0 * math.log10(max(1e-12, mse_val))

    out = {"recon_mse": float(mse_val), "psnr": float(psnr_val)}

    if not compute_lpips:
        out["lpips"] = float("nan")
        return out

    # LPIPS per (B*T*n_cams) image pair, then mean
    pred_cams = _split_cams(pred)  # (B, n_cams, T, 3, H, W)
    gt_cams = _split_cams(gt)
    B, n_cams, T = pred_cams.shape[:3]
    flat_pred = pred_cams.reshape(-1, 3, *pred_cams.shape[-2:])
    flat_gt = gt_cams.reshape(-1, 3, *gt_cams.shape[-2:])
    # LPIPS expects inputs in [-1, 1]
    fp = flat_pred * 2.0 - 1.0
    fg = flat_gt * 2.0 - 1.0
    lpips_net = _ensure_lpips(device)
    # batch over a chunk to keep memory bounded
    vals = []
    chunk = 64
    for i in range(0, fp.shape[0], chunk):
        v = lpips_net(fp[i:i + chunk], fg[i:i + chunk]).flatten()
        vals.append(v)
    lpips_val = torch.cat(vals).mean().item()
    out["lpips"] = float(lpips_val)
    return out


def rollout_diff_or_fm(
    model,
    cfg,
    prefix_clip: torch.Tensor,
    num_post: int,
    device: torch.device,
    actions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dispatch DIFF / FM / MAE rollout. Returns (B, T, C, H, W) full clip."""
    from training.diffusion_objective import (
        ddim_rollout_prefix_conditional,
        ddpm_rollout_with_prefix,
        make_schedule,
    )
    from training.flow_matching import flow_rollout_prefix_conditional
    from training.mask_objective import unpatchify  # noqa: F401  (rollout funcs use it)

    objective = cfg.train.get("objective", "mae")
    n_steps = int(cfg.eval.get("diff_sample_steps", 50))
    if objective == "diffusion":
        sched = make_schedule(
            num_steps=int(cfg.train.get("diff_num_steps", 1000)),
            schedule=cfg.train.get("diff_schedule", "linear"),
            device=device,
        )
        if cfg.train.get("diff_prefix_conditional", False):
            return ddim_rollout_prefix_conditional(
                model, prefix_clip, num_post, sched,
                num_sampling_steps=n_steps,
                autocast_dtype=torch.bfloat16,
                actions=actions,
            )
        return ddpm_rollout_with_prefix(
            model, prefix_clip, num_post, sched,
            num_sampling_steps=n_steps,
            autocast_dtype=torch.bfloat16,
        )
    if objective == "flow_matching":
        return flow_rollout_prefix_conditional(
            model, prefix_clip, num_post,
            num_sampling_steps=n_steps,
            autocast_dtype=torch.bfloat16,
            actions=actions,
        )
    # mae: single-shot block-causal rollout
    import importlib.util
    from pathlib import Path as _Path
    eval_py = _Path(__file__).resolve().parents[3] / "scripts" / "eval.py"
    spec = importlib.util.spec_from_file_location("_eval_mod_mae", eval_py)
    eval_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_mod)
    return eval_mod.rollout_a(
        model, prefix_clip, num_post,
        patch_size=cfg.model.patch_size,
        image_size=cfg.data.resolution,
        channels=3 * len(cfg.data.cameras),
    )
