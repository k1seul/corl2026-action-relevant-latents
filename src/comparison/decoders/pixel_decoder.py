"""Lightweight pixel decoder for encoder-only backbones.

Takes a per-frame latent vector (B, D) and decodes to a 128x128 RGB image.
Architecture: linear projection to a (256, 8, 8) tile, then 4x ConvTranspose2d
upsamples (8 -> 16 -> 32 -> 64 -> 128). Output uses Tanh then is rescaled to
[0, 1] to match LIBERO frame range.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PixelDecoder(nn.Module):
    def __init__(self, latent_dim: int, base_channels: int = 256, grid: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.grid = grid
        self.proj = nn.Linear(latent_dim, base_channels * grid * grid)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(base_channels, 128, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, D) or (B, T, D) -> (B, 3, 128, 128) or (B, T, 3, 128, 128) in [0, 1]."""
        if z.ndim == 3:
            B, T, D = z.shape
            flat = self.proj(z.reshape(B * T, D))
            tile = flat.reshape(B * T, self.base_channels, self.grid, self.grid)
            img = self.upsample(tile)
            img = (img + 1.0) * 0.5
            return img.reshape(B, T, 3, img.shape[-2], img.shape[-1])
        flat = self.proj(z)
        tile = flat.reshape(z.shape[0], self.base_channels, self.grid, self.grid)
        img = self.upsample(tile)
        return (img + 1.0) * 0.5
