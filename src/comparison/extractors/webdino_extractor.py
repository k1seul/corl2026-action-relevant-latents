"""Web-DINO (facebook/webssl-dino300m-full2b-224) feature extractor.

Web-DINO is DINOv2-architecture (ViT-L, 304M params) pretrained with
contrastive masked image modeling on a 2B-image web corpus (Fang et al.,
"Web-DINO" 2026). Native input is 224×224 with patch_size=14, so we
resize LIBERO's 128×128 clips up to 224×224 before forward pass.

Features: mean-pool over patch tokens (excluding CLS) → (B, T, 1024).

Used in dreamer-vla paper Section 3.9 as a "semantic / masked-prediction"
backbone test, complementing V-JEPA's masked-latent-prediction objective.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from comparison.extractors.base import ExtractorMeta, RepresentationExtractor


DEFAULT_CKPT = "facebook/webssl-dino300m-full2b-224"


class WebDINOExtractor(RepresentationExtractor):
    """Web-DINO 300M (DINOv2 ViT-L architecture) frozen extractor."""

    def __init__(
        self,
        device: torch.device,
        ckpt_path: Optional[str] = None,
        native_image_size: int = 224,
        finetuned_state_dict: Optional[dict] = None,
        variant_name: Optional[str] = None,
    ) -> None:
        from transformers import AutoModel  # lazy import

        repo = ckpt_path or DEFAULT_CKPT
        # torch_dtype renamed to dtype in transformers 5.x, support both
        try:
            model = AutoModel.from_pretrained(repo, dtype=torch.bfloat16)
        except TypeError:
            model = AutoModel.from_pretrained(repo, torch_dtype=torch.bfloat16)

        if finetuned_state_dict is not None:
            msg = model.load_state_dict(finetuned_state_dict, strict=False)
            if msg.missing_keys or msg.unexpected_keys:
                print(f"[webdino] state_dict load missing={len(msg.missing_keys)} "
                       f"unexpected={len(msg.unexpected_keys)}")

        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad = False

        self._model = model
        self._device = device
        self._native_size = int(native_image_size)
        n_params = sum(p.numel() for p in model.parameters())
        # Hidden size from config
        hidden = int(model.config.hidden_size)
        self._meta = ExtractorMeta(
            family="webdino",
            variant=variant_name or "webssl_dino300m",
            feat_dim=hidden,
            n_params=n_params,
            native_resolution=self._native_size,
            native_clip_len=1,           # DINOv2 is per-frame
            has_native_recon=False,
        )

    @property
    def meta(self) -> ExtractorMeta:
        return self._meta

    @torch.no_grad()
    def extract(self, clip: torch.Tensor,
                task_caption: Optional[str] = None) -> torch.Tensor:
        """clip: (B, T, C, H, W) in [0, 1] → (B, T, D=1024).

        DINOv2 is a per-frame model; we flatten the time dim, forward each
        frame independently, then unflatten.
        """
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
        out = self._model(x)
        # Tokens: [CLS, patch_1, ..., patch_N]. Mean-pool over patches only.
        tokens = out.last_hidden_state                            # (BT, 1+N, D)
        patch_tokens = tokens[:, 1:]                              # (BT, N, D)
        pooled = patch_tokens.mean(dim=1)                         # (BT, D)
        pooled = pooled.float().reshape(B, T, -1)                 # (B, T, D)
        return pooled
