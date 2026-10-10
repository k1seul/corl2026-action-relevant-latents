"""Conditional flow matching for the variant-(A''') training/sampling.

Following the rectified-flow / linear-interpolation parameterisation
(Liu et al. 2022, Lipman et al. 2022): for each clean clip ``x_0`` we
sample noise ``x_1 ~ N(0, I)`` and a continuous ``t ~ U[0, 1]`` and
train the model to predict the constant velocity field
``v_target = x_1 - x_0`` from the linearly-interpolated state
``x_t = (1 - t) * x_0 + t * x_1``.

Sampling is a deterministic Euler ODE solve from ``t = 1`` (pure noise)
back to ``t = 0`` (clean) using the predicted velocity. With
prefix-conditional training, we hold the prefix clean throughout and
update only the post-frame slice each step.

The model uses the same timestep-embedded BlockCausalVideoTransformer
as the diffusion variants; we scale the continuous t into the model's
integer embedding range by multiplying by ``time_scale - 1``.
"""
from __future__ import annotations

import torch

from training.mask_objective import unpatchify

DEFAULT_TIME_SCALE = 1000  # quantisation grid for the timestep embedding


def _t_to_embed_idx(t: torch.Tensor, time_scale: int = DEFAULT_TIME_SCALE) -> torch.Tensor:
    """Map continuous ``t in [0, 1]`` to the model's integer embedding index."""
    return (t.clamp(0.0, 1.0) * (time_scale - 1)).round().long()


def sample_t(
    batch: int,
    distribution: str = "uniform",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    device=None,
) -> torch.Tensor:
    """Sample per-sample t in `(0, 1)`.

    ``logit_normal`` (SD3-style) samples ``u ~ N(mean, std)`` and returns
    ``sigmoid(u)``. With mean=0, std=1 most mass lands in [0.27, 0.73],
    avoiding the noisy extremes where flow matching signal is weakest.
    """
    if distribution == "uniform":
        return torch.rand(batch, device=device).clamp(min=1e-4, max=1.0 - 1e-4)
    if distribution == "logit_normal":
        u = logit_mean + logit_std * torch.randn(batch, device=device)
        return torch.sigmoid(u).clamp(min=1e-4, max=1.0 - 1e-4)
    raise ValueError(f"unknown t distribution: {distribution!r}")


def flow_matching_loss(
    model,
    clip: torch.Tensor,
    in_channels: int,
    patch_size: int,
    image_size: int,
    prefix_conditional: bool = True,
    time_scale: int = DEFAULT_TIME_SCALE,
    t_distribution: str = "uniform",
    t_logit_mean: float = 0.0,
    t_logit_std: float = 1.0,
    actions: torch.Tensor | None = None,
) -> torch.Tensor:
    """One training step's loss. Returns the (scalar) MSE on noised frames."""
    B, T = clip.shape[:2]
    device = clip.device

    if prefix_conditional:
        L_pre = torch.randint(0, T, (B,), device=device)
        is_noised = (
            torch.arange(T, device=device)[None, :] >= L_pre[:, None]
        )  # (B, T)
    else:
        is_noised = torch.ones(B, T, dtype=torch.bool, device=device)

    t = sample_t(
        B,
        distribution=t_distribution,
        logit_mean=t_logit_mean,
        logit_std=t_logit_std,
        device=device,
    )
    eps = torch.randn_like(clip)
    t_b = t.view(-1, 1, 1, 1, 1)
    x_t_full = (1.0 - t_b) * clip + t_b * eps
    x_t = torch.where(is_noised[:, :, None, None, None], x_t_full, clip)
    v_target = eps - clip  # constant along the linear path

    t_per_frame_idx = _t_to_embed_idx(t, time_scale)[:, None].expand(B, T)
    t_per_frame = torch.where(
        is_noised, t_per_frame_idx, torch.zeros(B, T, dtype=torch.long, device=device)
    )

    pred_patches = model(x_t, mask_indicator=None, t=t_per_frame, actions=actions)
    pred_v = unpatchify(pred_patches, patch_size, image_size, in_channels)

    sq = (pred_v - v_target).pow(2).mean(dim=(2, 3, 4))  # (B, T)
    denom = is_noised.float().sum().clamp_min(1.0)
    return (sq * is_noised.float()).sum() / denom


@torch.no_grad()
def flow_rollout_prefix_conditional(
    model,
    prefix_clip: torch.Tensor,
    num_post_frames: int,
    num_sampling_steps: int = 50,
    autocast_dtype: torch.dtype | None = None,
    time_scale: int = DEFAULT_TIME_SCALE,
    actions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Euler ODE solve from t=1 (noise) to t=0 (clean), with prefix held clean."""
    device = prefix_clip.device
    B, L_pre, C, H, W = prefix_clip.shape
    L = L_pre + num_post_frames

    x = torch.randn(B, L, C, H, W, device=device, dtype=prefix_clip.dtype)
    x[:, :L_pre] = prefix_clip
    use_amp = autocast_dtype is not None and device.type == "cuda"

    dt = 1.0 / num_sampling_steps
    for step in range(num_sampling_steps):
        # discretize the t grid as the segment midpoints (Heun-like) is fine,
        # but for plain Euler we use the left endpoint.
        t_val = 1.0 - step * dt
        t_per_frame = torch.zeros(B, L, device=device, dtype=torch.long)
        t_per_frame[:, L_pre:] = int(round(t_val * (time_scale - 1)))
        # ensure prefix is clean at every step
        x[:, :L_pre] = prefix_clip
        if use_amp:
            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                v_patches = model(x, mask_indicator=None, t=t_per_frame, actions=actions).float()
        else:
            v_patches = model(x, mask_indicator=None, t=t_per_frame, actions=actions)
        v_clip = unpatchify(v_patches, model.patch_size, x.shape[-1], C)
        # Euler step on post frames only; we integrate dx/dt = -v from t=1 -> 0
        x[:, L_pre:] = x[:, L_pre:] - dt * v_clip[:, L_pre:]

    x[:, :L_pre] = prefix_clip
    return x.clamp(0.0, 1.0)
