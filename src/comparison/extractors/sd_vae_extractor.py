"""SD VAE encoder feature extractor (SDXL VAE used; SD3 VAE is gated).

Stable Diffusion VAE encoder is a conv-based encoder trained for pixel
reconstruction within a Latent Diffusion Model. Distinct from the
ViT-based encoders in dreamer-vla — tests the "reconstruction-aligned
encoder" axis (paper Section 3.9).

Encoder output at 128×128 input: (B, 4, 16, 16). We flatten the spatial
+ channel dims to produce a (B, T, 1024) per-frame feature comparable
to V-JEPA ViT-L (also 1024 dim).

SD3 VAE primary choice is gated on HF; we use the open `stabilityai/sdxl-vae`
which shares architecture and pretraining lineage. Paper should document
this substitution as a faithful proxy for "reconstruction-aligned VAE".
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from comparison.extractors.base import ExtractorMeta, RepresentationExtractor


DEFAULT_CKPT = "stabilityai/sdxl-vae"


class SDVAEExtractor(RepresentationExtractor):
    """SDXL VAE encoder used as a frozen feature extractor.

    Encoder output: (B, 4, H/8, W/8). At 128px input → (B, 4, 16, 16)
    → flatten 1024-dim per frame.
    """

    def __init__(
        self,
        device: torch.device,
        ckpt_path: Optional[str] = None,
        native_image_size: int = 128,
        finetuned_state_dict: Optional[dict] = None,
        variant_name: Optional[str] = None,
    ) -> None:
        from diffusers import AutoencoderKL

        repo = ckpt_path or DEFAULT_CKPT
        vae = AutoencoderKL.from_pretrained(repo)
        vae = vae.to(device=device, dtype=torch.bfloat16)

        # We only use the encoder + quant_conv. Discard decoder.
        if finetuned_state_dict is not None:
            msg = vae.load_state_dict(finetuned_state_dict, strict=False)
            if msg.missing_keys or msg.unexpected_keys:
                print(f"[sd_vae] load missing={len(msg.missing_keys)} "
                       f"unexpected={len(msg.unexpected_keys)}")

        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False

        self._vae = vae
        self._device = device
        self._native_size = int(native_image_size)
        # latent shape at native: (4, native/8, native/8)
        latent_h = self._native_size // 8
        feat_dim = 4 * latent_h * latent_h
        n_params = sum(p.numel() for p in vae.parameters())
        self._meta = ExtractorMeta(
            family="sd3vae",
            variant=variant_name or "sdxl_vae",
            feat_dim=feat_dim,
            n_params=n_params,
            native_resolution=self._native_size,
            native_clip_len=1,
            has_native_recon=True,
        )

    @property
    def meta(self) -> ExtractorMeta:
        return self._meta

    @torch.no_grad()
    def extract(self, clip: torch.Tensor,
                task_caption: Optional[str] = None) -> torch.Tensor:
        """clip: (B, T, C, H, W) in [0, 1] → (B, T, D=feat_dim).

        VAE expects inputs in [-1, 1]; we convert internally."""
        clip = clip.to(self._device)
        B, T, C, H, W = clip.shape
        if C > 3:
            clip = clip[:, :, :3]
            C = 3
        x = clip.reshape(B * T, C, H, W)
        if (H, W) != (self._native_size, self._native_size):
            x = F.interpolate(x, size=(self._native_size, self._native_size),
                                mode="bilinear", align_corners=False)
        x = x * 2.0 - 1.0                                   # [-1, 1]
        x = x.to(torch.bfloat16)
        out = self._vae.encode(x)
        # use latent distribution mean (deterministic)
        latent = out.latent_dist.mean                       # (BT, 4, H/8, W/8)
        latent = latent.flatten(1)                          # (BT, D)
        return latent.float().reshape(B, T, -1)
