"""PSNR and SSIM utilities (pure torch, no extra dependencies).

`ssim` follows the standard 11x11 Gaussian-window definition (Wang et
al. 2004) and is computed channel-wise, which means it works for any
channel count -- including the 6-channel agentview+eye_in_hand case.
Inputs are expected to be in [0, 1].
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr_from_mse(mse: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return 10.0 * torch.log10(1.0 / mse.clamp_min(eps))


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-coords.pow(2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    g2 = g.unsqueeze(0) * g.unsqueeze(1)
    return g2.unsqueeze(0).unsqueeze(0).expand(channels, 1, window_size, window_size).contiguous()


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Mean SSIM across batch / channels / spatial. `x`, `y`: `(B, C, H, W)` in `[0, 1]`."""
    assert x.shape == y.shape
    B, C, H, W = x.shape
    K1, K2, L = 0.01, 0.03, 1.0
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    pad = window_size // 2
    win = _gaussian_window(window_size, sigma, C, x.device, x.dtype)
    mu_x = F.conv2d(x, win, padding=pad, groups=C)
    mu_y = F.conv2d(y, win, padding=pad, groups=C)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, win, padding=pad, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, win, padding=pad, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, win, padding=pad, groups=C) - mu_xy
    num = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return (num / den).mean()
