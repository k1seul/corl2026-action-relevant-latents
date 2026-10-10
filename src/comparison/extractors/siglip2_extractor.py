"""SigLIP 2 ViT-L vision tower feature extractor.

Uses google/siglip2-large-patch16-256: a ~316M-param vision tower from
the SigLIP 2 vision-language model (Tschannen et al. 2026). Pretrained
with contrastive image-text alignment on web-scale data.

Tests the "vision-language pretraining" axis in the dreamer-vla
cross-architecture comparison (paper Section 3.9), distinct from
V-JEPA's video masked latent prediction and Web-DINO's image-only
contrastive masked prediction.

We load only the vision tower (316M); discard the text tower (548M).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from comparison.extractors.base import ExtractorMeta, RepresentationExtractor


DEFAULT_CKPT = "google/siglip2-large-patch16-256"


class SigLIP2Extractor(RepresentationExtractor):
    """SigLIP 2 vision tower (ViT-L, 316M) frozen extractor."""

    def __init__(
        self,
        device: torch.device,
        ckpt_path: Optional[str] = None,
        native_image_size: int = 256,
        finetuned_state_dict: Optional[dict] = None,
        variant_name: Optional[str] = None,
    ) -> None:
        from transformers import AutoModel

        repo = ckpt_path or DEFAULT_CKPT
        try:
            full_model = AutoModel.from_pretrained(repo, dtype=torch.bfloat16)
        except TypeError:
            full_model = AutoModel.from_pretrained(repo, torch_dtype=torch.bfloat16)

        # We only need the vision tower; release text tower to save memory.
        vision = full_model.vision_model
        del full_model

        if finetuned_state_dict is not None:
            msg = vision.load_state_dict(finetuned_state_dict, strict=False)
            if msg.missing_keys or msg.unexpected_keys:
                print(f"[siglip2] load missing={len(msg.missing_keys)} "
                       f"unexpected={len(msg.unexpected_keys)}")

        vision = vision.to(device).eval()
        for p in vision.parameters():
            p.requires_grad = False

        self._vision = vision
        self._device = device
        self._native_size = int(native_image_size)
        hidden = int(vision.config.hidden_size)
        n_params = sum(p.numel() for p in vision.parameters())
        self._meta = ExtractorMeta(
            family="siglip2",
            variant=variant_name or "siglip2_large_p16_256",
            feat_dim=hidden,
            n_params=n_params,
            native_resolution=self._native_size,
            native_clip_len=1,        # per-frame
            has_native_recon=False,
        )

    @property
    def meta(self) -> ExtractorMeta:
        return self._meta

    @torch.no_grad()
    def extract(self, clip: torch.Tensor,
                task_caption: Optional[str] = None) -> torch.Tensor:
        """clip: (B, T, C, H, W) in [0, 1] → (B, T, D=1024)."""
        clip = clip.to(self._device)
        B, T, C, H, W = clip.shape
        if C > 3:
            clip = clip[:, :, :3]
            C = 3
        x = clip.reshape(B * T, C, H, W)
        if (H, W) != (self._native_size, self._native_size):
            x = F.interpolate(x, size=(self._native_size, self._native_size),
                                mode="bilinear", align_corners=False)
        x = x.to(torch.bfloat16)
        out = self._vision(x)
        # SigLIP has no CLS token; last_hidden_state is all patches
        tokens = out.last_hidden_state              # (BT, N, D)
        pooled = tokens.mean(dim=1)                  # (BT, D)
        return pooled.float().reshape(B, T, -1)
