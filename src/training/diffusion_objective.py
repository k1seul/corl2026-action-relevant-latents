"""DDPM utilities for the variant-(A'') diffusion training/sampling.

Training objective is x0-prediction: at each step, sample a per-sample
timestep `t` uniformly in `[0, T_diff)`, noise the entire clean clip at
that level, and train the model to predict the clean clip via MSE.

Sampling for rollout uses the standard "Repaint"-style imputation
trick: at each denoising step we replace the prefix's currently-noisy
state with a freshly q_sampled noised version of the *known* clean
prefix. This conditions the post frames on the prefix without any
training-time modification.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DDPMSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor


def make_schedule(
    num_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    schedule: str = "linear",
    device=None,
) -> DDPMSchedule:
    """Build a DDPM schedule. ``schedule`` is ``linear`` or ``cosine``."""
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
    elif schedule == "cosine":
        # Nichol & Dhariwal 2021
        s = 0.008
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps, dtype=torch.float64, device=device)
        ac = torch.cos(((x / num_steps) + s) / (1 + s) * (torch.pi / 2)).pow(2)
        ac = ac / ac[0]
        betas = (1.0 - ac[1:] / ac[:-1]).clamp(max=0.999)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        betas = betas.float()
        alphas = alphas.float()
        alphas_cumprod = alphas_cumprod.float()
    else:
        raise ValueError(f"unknown schedule: {schedule!r}")
    return DDPMSchedule(betas=betas, alphas=alphas, alphas_cumprod=alphas_cumprod)


def _broadcast(scalar_per_sample: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    while scalar_per_sample.dim() < target.dim():
        scalar_per_sample = scalar_per_sample.unsqueeze(-1)
    return scalar_per_sample


def q_sample(
    clean: torch.Tensor,
    t: torch.Tensor,
    sched: DDPMSchedule,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward diffusion: noise `clean` at integer timesteps `t`.

    ``clean``: any shape with leading batch dim. ``t``: ``(B,)``.
    Returns ``(noisy, noise)``.
    """
    if noise is None:
        noise = torch.randn_like(clean)
    a_bar = _broadcast(sched.alphas_cumprod[t], clean)
    return torch.sqrt(a_bar) * clean + torch.sqrt(1.0 - a_bar) * noise, noise


@torch.no_grad()
def ddim_step(
    x_t: torch.Tensor,
    x0_pred: torch.Tensor,
    t_cur: torch.Tensor,
    t_next: torch.Tensor,
    sched: DDPMSchedule,
) -> torch.Tensor:
    """Deterministic DDIM update (eta=0) from `t_cur` to `t_next < t_cur`.

    Tolerates sparse `t_next` (skipping arbitrarily many DDPM steps).
    When `t_next < 0`, returns `x0_pred` directly (final clean step).
    """
    a_bar_cur = _broadcast(sched.alphas_cumprod[t_cur], x_t)
    eps_pred = (x_t - a_bar_cur.sqrt() * x0_pred) / (1.0 - a_bar_cur).clamp_min(1e-12).sqrt()
    if (t_next < 0).all():
        return x0_pred
    a_bar_next = _broadcast(sched.alphas_cumprod[t_next.clamp(min=0)], x_t)
    return a_bar_next.sqrt() * x0_pred + (1.0 - a_bar_next).clamp_min(0.0).sqrt() * eps_pred


@torch.no_grad()
def ddpm_posterior_sample(
    x_t: torch.Tensor,
    x0_pred: torch.Tensor,
    t: torch.Tensor,
    sched: DDPMSchedule,
) -> torch.Tensor:
    """One step of ancestral DDPM sampling using x0 prediction.

    Returns ``x_{t-1}``. When `t == 0` for some batch elements, the
    deterministic posterior mean is returned for those (no extra noise).
    """
    a_bar_t = _broadcast(sched.alphas_cumprod[t], x_t)
    alpha_t = _broadcast(sched.alphas[t], x_t)
    beta_t = 1.0 - alpha_t

    t_prev = (t - 1).clamp(min=0)
    a_bar_prev = _broadcast(sched.alphas_cumprod[t_prev], x_t)

    coef_x0 = (a_bar_prev.sqrt() * beta_t) / (1.0 - a_bar_t).clamp_min(1e-12)
    coef_xt = ((1.0 - a_bar_prev) * alpha_t.sqrt()) / (1.0 - a_bar_t).clamp_min(1e-12)
    mean = coef_x0 * x0_pred + coef_xt * x_t
    var = ((1.0 - a_bar_prev) / (1.0 - a_bar_t).clamp_min(1e-12)) * beta_t
    noise = torch.randn_like(x_t)
    nonzero = (t > 0).float().reshape((-1,) + (1,) * (x_t.dim() - 1))
    return mean + nonzero * var.sqrt() * noise


@torch.no_grad()
def ddim_rollout_prefix_conditional(
    model,
    prefix_clip: torch.Tensor,
    num_post_frames: int,
    sched: DDPMSchedule,
    num_sampling_steps: int = 50,
    autocast_dtype: torch.dtype | None = None,
    actions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rollout for a model trained with explicit prefix-conditional diffusion.

    The model receives per-frame timesteps (0 for the clean prefix,
    sampling-step `t` for post frames). At each step, only the post-
    frame slice is updated; prefix is held clean throughout.
    """
    device = prefix_clip.device
    B, L_pre, C, H, W = prefix_clip.shape
    L = L_pre + num_post_frames
    T_diff = sched.alphas_cumprod.shape[0]

    if num_sampling_steps >= T_diff:
        t_grid = torch.arange(T_diff - 1, -1, -1, device=device)
    else:
        t_grid = torch.linspace(T_diff - 1, 0, num_sampling_steps, device=device).round().long()
    grid_list = t_grid.tolist()

    x = torch.randn(B, L, C, H, W, device=device, dtype=prefix_clip.dtype)
    x[:, :L_pre] = prefix_clip
    use_amp = autocast_dtype is not None and device.type == "cuda"
    from training.mask_objective import unpatchify

    for i, step_t in enumerate(t_grid):
        # per-frame timestep: 0 for clean prefix, step_t for noisy post
        t_per_frame = torch.zeros(B, L, device=device, dtype=torch.long)
        t_per_frame[:, L_pre:] = int(step_t.item())
        # ensure prefix is clean before forward (ddim_step on post may have
        # leaked into prefix in earlier code paths; this is idempotent here)
        x[:, :L_pre] = prefix_clip
        if use_amp:
            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                x0_pred = model(x, mask_indicator=None, t=t_per_frame, actions=actions).float()
        else:
            x0_pred = model(x, mask_indicator=None, t=t_per_frame, actions=actions)
        x0_clip = unpatchify(x0_pred, model.patch_size, x.shape[-1], C)

        next_t = grid_list[i + 1] if i + 1 < len(grid_list) else -1
        t_cur_b = torch.full((B,), int(step_t.item()), device=device, dtype=torch.long)
        t_next_b = torch.full((B,), int(next_t), device=device, dtype=torch.long)
        post = ddim_step(x[:, L_pre:], x0_clip[:, L_pre:], t_cur_b, t_next_b, sched)
        x[:, L_pre:] = post

    x[:, :L_pre] = prefix_clip
    return x.clamp(0.0, 1.0)


@torch.no_grad()
def ddpm_rollout_with_prefix(
    model,
    prefix_clip: torch.Tensor,
    num_post_frames: int,
    sched: DDPMSchedule,
    num_sampling_steps: int = 50,
    autocast_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Generate the `num_post_frames` frames after `prefix_clip` via DDPM.

    Uses Repaint-style conditioning: the prefix portion is overwritten
    at every step with a fresh q_sample of the clean prefix at the
    current timestep.

    Returns the concatenated full clip ``(B, L_pre + num_post_frames, C, H, W)``.
    """
    device = prefix_clip.device
    B, L_pre, C, H, W = prefix_clip.shape
    L = L_pre + num_post_frames
    T_diff = sched.alphas_cumprod.shape[0]

    # uniformly-spaced subset of training timesteps (low resolution sampler)
    if num_sampling_steps >= T_diff:
        t_grid = torch.arange(T_diff - 1, -1, -1, device=device)
    else:
        t_grid = torch.linspace(T_diff - 1, 0, num_sampling_steps, device=device).round().long()

    # init: pure noise everywhere
    x = torch.randn(B, L, C, H, W, device=device, dtype=prefix_clip.dtype)

    use_amp = autocast_dtype is not None and device.type == "cuda"
    from training.mask_objective import unpatchify  # local import to avoid cycles
    grid_list = t_grid.tolist()
    for i, step_t in enumerate(t_grid):
        t_b = step_t.expand(B)
        # overwrite prefix slice with q_sample of clean prefix at this t
        x_pre, _ = q_sample(prefix_clip, t_b, sched)
        x[:, :L_pre] = x_pre
        if use_amp:
            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                x0_pred = model(x, mask_indicator=None, t=t_b).float()
        else:
            x0_pred = model(x, mask_indicator=None, t=t_b)
        x0_clip = unpatchify(x0_pred, model.patch_size, x.shape[-1], C)
        # DDIM step from current t to next t in the grid (or below 0 for the
        # final iteration). Deterministic (eta=0) and tolerates sparse grids.
        next_t = grid_list[i + 1] if i + 1 < len(grid_list) else -1
        t_next_b = torch.full_like(t_b, next_t)
        x = ddim_step(x, x0_clip, t_b, t_next_b, sched)

    # final: clean prefix + denoised post
    x[:, :L_pre] = prefix_clip
    return x.clamp(0.0, 1.0)
