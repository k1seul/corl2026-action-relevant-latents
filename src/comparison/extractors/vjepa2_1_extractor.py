"""V-JEPA 2.1 ViT-base @ 384 extractor (param-matched comparison).

V-JEPA 2 only has ViT-L/H/G — V-JEPA 2.1 is the closest available
ViT-base option. ViT-base @ 384 is distilled from ViT-G's pretraining,
so this is "ViT-G knowledge in ViT-base size" (~86M params, much closer
to our DIFF's 91M than V-JEPA 2 ViT-L's 304M).

Caveat: this is V-JEPA 2.1, not V-JEPA 2 — different SSL recipe.

Usage in eval pipeline config:
    - name: V-JEPA2.1-vit_base_384
      family: vjepa2_1
      ckpt: <path to vjepa2_1_vitb_dist_vitG_384.pt>
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from comparison.extractors.base import ExtractorMeta, RepresentationExtractor
from comparison.extractors.vjepa2_extractor import (
    EXTERNAL_DIR, VJEPA2_REPO, _IMAGENET_MEAN, _IMAGENET_STD,
)


def _ensure_vjepa2_1_on_path() -> None:
    p = str(VJEPA2_REPO)
    if p not in sys.path:
        sys.path.insert(0, p)


class VJEPA21Extractor(RepresentationExtractor):
    def __init__(self, device: torch.device,
                 ckpt_path: Optional[str] = None,
                 model_arch: str = "vit_base",
                 num_frames: int = 32,
                 img_size: int = 384,
                 ckpt_key: str = "ema_encoder") -> None:
        _ensure_vjepa2_1_on_path()
        from app.vjepa_2_1.models import vision_transformer as vit_encoder

        # match official V-JEPA 2.1 vit_base config (from src/hub/backbones.py)
        vit_kwargs = dict(
            patch_size=16,
            img_size=(img_size, img_size),
            num_frames=num_frames,
            tubelet_size=2,
            use_sdpa=True,
            use_silu=False,
            wide_silu=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )
        arch_fn = getattr(vit_encoder, model_arch)
        encoder = arch_fn(**vit_kwargs)

        ckpt = ckpt_path or str(EXTERNAL_DIR / "vjepa2_ckpts" /
                                  "vjepa2_1_vitb_dist_vitG_384.pt")
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"V-JEPA 2.1 ckpt not at {ckpt}")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = state.get(ckpt_key, state.get("encoder", state))
        sd = {k.replace("module.", "").replace("backbone.", ""): v
              for k, v in sd.items()}
        msg = encoder.load_state_dict(sd, strict=False)
        if msg.missing_keys or msg.unexpected_keys:
            sys.stderr.write(
                f"[vjepa2_1] load_state_dict missing={len(msg.missing_keys)} "
                f"unexpected={len(msg.unexpected_keys)}\n")

        encoder = encoder.to(device).eval()
        for p in encoder.parameters():
            p.requires_grad = False

        self._encoder = encoder
        self._device = device
        self._img_size = img_size
        self._num_frames = num_frames
        self._tubelet = 2
        n_params = sum(p.numel() for p in encoder.parameters())
        self._meta = ExtractorMeta(
            family="vjepa2_1",
            variant=f"{model_arch}_{img_size}",
            feat_dim=int(encoder.embed_dim),
            n_params=n_params,
            native_resolution=img_size,
            native_clip_len=num_frames,
            has_native_recon=False,
        )

    @property
    def meta(self) -> ExtractorMeta:
        return self._meta

    def _preprocess(self, clip: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = clip.shape
        if C > 3:
            clip = clip[:, :, :3]; C = 3
        clip = clip.reshape(B * T, C, H, W)
        clip = F.interpolate(clip, size=(self._img_size, self._img_size),
                             mode="bilinear", align_corners=False)
        mean = torch.tensor(_IMAGENET_MEAN, device=clip.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=clip.device).view(1, 3, 1, 1)
        clip = (clip - mean) / std
        clip = clip.reshape(B, T, C, self._img_size, self._img_size)
        return clip.permute(0, 2, 1, 3, 4).contiguous()

    @torch.no_grad()
    def extract(self, clip: torch.Tensor,
                task_caption: Optional[str] = None) -> torch.Tensor:
        clip = clip.to(self._device)
        B, T = clip.shape[:2]
        x = self._preprocess(clip)
        feats = self._encoder(x)
        # V-JEPA 2.1 forward returns (B, num_patches, D) by default
        # (return_all_tokens=True per the factory). Reshape to per-frame.
        D = feats.shape[-1]
        H_p = self._img_size // 16
        T_tubes = T // self._tubelet
        feats = feats.reshape(B, T_tubes, H_p * H_p, D)
        feats = feats.mean(dim=2)
        feats = feats.repeat_interleave(self._tubelet, dim=1)
        return feats[:, :T].float()
