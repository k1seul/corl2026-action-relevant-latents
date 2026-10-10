"""VideoMAE V1 large feature extractor.

MCG-NJU/videomae-large: 340M params, ViT-L architecture, pretrained on
Kinetics-400 via video masked autoencoding (pixel reconstruction).
Tests the "video pretrain alone, without JEPA-style predictor" axis vs.
V-JEPA's masked latent prediction.

Native input: 16 frames × 224×224, tubelet_size=2, patch=16.
Output tokens: 8 temporal × 14*14 spatial = 1568 tokens of 1024-d.
We mean-pool spatial → (B, T_tubes=8, D=1024), then repeat each tubelet
feature to span the 2 frames it covers → (B, 16, D). For 32-frame
clips (our pipeline default), we process two 16-frame halves and concat.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from comparison.extractors.base import ExtractorMeta, RepresentationExtractor


DEFAULT_REPO = "MCG-NJU/videomae-large"


class VideoMAEExtractor(RepresentationExtractor):
    """VideoMAE V1 ViT-L (340M) frozen extractor for action probe."""

    def __init__(
        self,
        device: torch.device,
        ckpt_path: Optional[str] = None,
        native_image_size: int = 224,
        finetuned_state_dict: Optional[dict] = None,
        variant_name: Optional[str] = None,
    ) -> None:
        from transformers import AutoModel

        repo = ckpt_path or DEFAULT_REPO
        try:
            model = AutoModel.from_pretrained(repo, dtype=torch.bfloat16)
        except TypeError:
            model = AutoModel.from_pretrained(repo, torch_dtype=torch.bfloat16)

        if finetuned_state_dict is not None:
            msg = model.load_state_dict(finetuned_state_dict, strict=False)
            if msg.missing_keys or msg.unexpected_keys:
                print(f"[videomae] load missing={len(msg.missing_keys)} "
                       f"unexpected={len(msg.unexpected_keys)}")

        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad = False

        self._model = model
        self._device = device
        self._native_size = int(native_image_size)
        self._chunk_frames = int(model.config.num_frames)  # 16
        self._tubelet = int(model.config.tubelet_size)      # 2
        self._patch = int(model.config.patch_size)          # 16
        self._hidden = int(model.config.hidden_size)        # 1024
        n_params = sum(p.numel() for p in model.parameters())
        self._meta = ExtractorMeta(
            family="videomae",
            variant=variant_name or "videomae_v1_large",
            feat_dim=self._hidden,
            n_params=n_params,
            native_resolution=self._native_size,
            native_clip_len=self._chunk_frames,
            has_native_recon=False,
        )

    @property
    def meta(self) -> ExtractorMeta:
        return self._meta

    @torch.no_grad()
    def extract(self, clip: torch.Tensor,
                task_caption: Optional[str] = None) -> torch.Tensor:
        """clip: (B, T, C, H, W) in [0, 1] → (B, T, D=1024) per-frame feature.

        VideoMAE consumes 16-frame chunks; we split T into chunks of size
        ``chunk_frames``, forward each, then concat along time."""
        clip = clip.to(self._device)
        B, T, C, H, W = clip.shape
        if C > 3:
            clip = clip[:, :, :3]
            C = 3
        if (H, W) != (self._native_size, self._native_size):
            x = clip.reshape(B * T, C, H, W)
            x = F.interpolate(x, size=(self._native_size, self._native_size),
                                mode="bilinear", align_corners=False)
            clip = x.reshape(B, T, C, self._native_size, self._native_size)

        # Pad T to multiple of chunk_frames so we can split into chunks
        pad = (-T) % self._chunk_frames
        if pad:
            last = clip[:, -1:].repeat(1, pad, 1, 1, 1)
            clip = torch.cat([clip, last], dim=1)
        T_padded = clip.shape[1]
        n_chunks = T_padded // self._chunk_frames

        clip = clip.to(torch.bfloat16)
        chunk_feats = []
        for i in range(n_chunks):
            chunk = clip[:, i * self._chunk_frames:(i + 1) * self._chunk_frames]
            # (B, F, C, H, W); VideoMAE expects (B, F, C, H, W) for HF AutoModel
            out = self._model(chunk)
            tokens = out.last_hidden_state              # (B, F/tubelet * H_p * W_p, D)
            n_spatial = (self._native_size // self._patch) ** 2
            n_temp = self._chunk_frames // self._tubelet
            tokens = tokens.reshape(B, n_temp, n_spatial, self._hidden)
            pooled = tokens.mean(dim=2)                  # (B, n_temp=F/tubelet, D)
            # repeat each tubelet feat to span its tubelet_size frames
            per_frame = pooled.repeat_interleave(self._tubelet, dim=1)   # (B, F, D)
            chunk_feats.append(per_frame.float())
        feats = torch.cat(chunk_feats, dim=1)             # (B, T_padded, D)
        return feats[:, :T]                                # crop pad
