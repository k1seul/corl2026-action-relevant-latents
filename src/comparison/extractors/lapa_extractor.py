"""LAPA Latent Action Quantizer (LAQ) feature extractor.

LAPA's LAQ component is implemented in PyTorch (the JAX dependency is
only for the LLaMA-based VLA decoder, which we don't use). We can
load laq_openx.pt directly and extract per-frame features from the
encoder *before* quantization.

Per-frame features come from the encoder output: the spatial+temporal
transformer's pre-VQ tensor of shape (B, T, H/p, W/p, D). We mean-pool
over the spatial token grid to get (B, T, D).

Caveat: LAPA's LAQ was trained on **frame pairs** (window_size=k); the
temporal transformer has only seen T=2 inputs. Passing T=32 here
extrapolates beyond training distribution. If results are bad, fall
back to per-pair extraction (slower but faithful).

External setup required:
  1. Repo: ``$EXTERNAL_MODELS_DIR/LAPA``
  2. Checkpoint: ``$EXTERNAL_MODELS_DIR/lapa_ckpts/laq_openx.pt``
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from comparison.extractors.base import ExtractorMeta, RepresentationExtractor

EXTERNAL_DIR = Path(os.environ.get(
    "EXTERNAL_MODELS_DIR",
    "external_models",
))
LAPA_REPO = EXTERNAL_DIR / "LAPA"
LAPA_CKPT = EXTERNAL_DIR / "lapa_ckpts" / "laq_openx.pt"


def _ensure_lapa_on_path() -> None:
    if not LAPA_REPO.exists():
        raise FileNotFoundError(
            f"LAPA repo not at {LAPA_REPO}. Clone it: git clone "
            f"https://github.com/LatentActionPretraining/LAPA {LAPA_REPO}"
        )
    p = str(LAPA_REPO / "laq")
    if p not in sys.path:
        sys.path.insert(0, p)


class LapaExtractor(RepresentationExtractor):
    """LAPA LAQ encoder, frozen, used as a per-frame feature extractor."""

    def __init__(self, device: torch.device,
                 ckpt_path: Optional[str] = None,
                 image_size: int = 256,
                 patch_size: int = 32,
                 dim: int = 1024,
                 quant_dim: int = 32,
                 codebook_size: int = 8,
                 spatial_depth: int = 8,
                 temporal_depth: int = 8,
                 dim_head: int = 64,
                 heads: int = 16,
                 code_seq_len: int = 4) -> None:
        _ensure_lapa_on_path()
        # laq_model is on path now
        from laq_model.latent_action_quantization import (
            LatentActionQuantization,
        )

        ckpt = ckpt_path or str(LAPA_CKPT)
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"LAPA ckpt not at {ckpt}")

        model = LatentActionQuantization(
            dim=dim, quant_dim=quant_dim,
            codebook_size=codebook_size,
            image_size=image_size, patch_size=patch_size,
            spatial_depth=spatial_depth, temporal_depth=temporal_depth,
            dim_head=dim_head, heads=heads,
            code_seq_len=code_seq_len,
        )
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        # Accept both pretrained LAQ ckpt (raw state_dict) and our finetune
        # ckpt format (dict with "encoder" key).
        if isinstance(state, dict) and "encoder" in state:
            state = state["encoder"]
        # LAPA's custom load_state_dict already forces strict=False
        msg = model.load_state_dict(state)
        if msg.missing_keys or msg.unexpected_keys:
            sys.stderr.write(
                f"[lapa] load_state_dict missing={len(msg.missing_keys)} "
                f"unexpected={len(msg.unexpected_keys)}\n")

        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad = False

        self._model = model
        self._device = device
        self._image_size = image_size
        self._patch_size = patch_size
        n_params = sum(p.numel() for p in model.parameters())
        self._meta = ExtractorMeta(
            family="lapa",
            variant="laq_openx",
            feat_dim=dim,
            n_params=n_params,
            native_resolution=image_size,
            native_clip_len=2,  # LAQ trained on frame pairs
            has_native_recon=False,
        )

    @property
    def meta(self) -> ExtractorMeta:
        return self._meta

    @torch.no_grad()
    def extract(self, clip: torch.Tensor,
                task_caption: Optional[str] = None) -> torch.Tensor:
        """clip: (B, T, C, H, W) [0, 1] → (B, T, D=1024).

        LAQ's patch embedding hardcodes T=1 per call (it was trained on frame
        pairs). We process all T-1 adjacent pairs in one batched forward to
        get per-frame features. Frame t's feature comes from pair (t, t+1)'s
        ``first_tokens``; the last frame's feature comes from the final
        pair's ``last_tokens``.
        """
        clip = clip.to(self._device)
        B, T, C, H, W = clip.shape
        if C > 3:
            clip = clip[:, :, :3]
            C = 3
        # resize to LAPA's native 256×256
        x = clip.reshape(B * T, C, H, W)
        x = F.interpolate(x, size=(self._image_size, self._image_size),
                           mode="bilinear", align_corners=False)
        x = x.reshape(B, T, C, self._image_size, self._image_size)

        # build (T-1) adjacent pairs, each shape (B, 3, 2, H, W)
        # then flatten to (B*(T-1), 3, 2, H, W) for one big batched forward
        pair_first = x[:, :-1]   # (B, T-1, 3, H, W) — frame t
        pair_last = x[:, 1:]     # (B, T-1, 3, H, W) — frame t+1
        pairs = torch.stack([pair_first, pair_last], dim=3)  # (B, T-1, 3, 2, H, W)
        pairs = pairs.reshape(B * (T - 1), C, 2, self._image_size, self._image_size)

        # patch-embed each side; LAQ's patch_emb_first_frame handles T=1 only
        first_frames = pairs[:, :, :1]                                         # (BN, 3, 1, H, W)
        last_frames  = pairs[:, :, 1:]                                          # (BN, 3, 1, H, W)
        first_emb = self._model.to_patch_emb_first_frame(first_frames)         # (BN, 1, h, w, D)
        last_emb  = self._model.to_patch_emb_first_frame(last_frames)          # (BN, 1, h, w, D)
        tokens = torch.cat([first_emb, last_emb], dim=1)                       # (BN, 2, h, w, D)

        first_tokens, last_tokens = self._model.encode(tokens)
        # first_tokens: (BN, 1, h, w, D) → frame t feature
        # last_tokens : (BN, 1, h, w, D) → frame t+1 feature
        first_pooled = first_tokens.mean(dim=(2, 3))[:, 0]   # (BN, D)
        last_pooled  = last_tokens.mean(dim=(2, 3))[:, 0]    # (BN, D)

        # reshape back to (B, T-1, D)
        first_pooled = first_pooled.reshape(B, T - 1, -1)
        last_pooled  = last_pooled.reshape(B, T - 1, -1)

        # frame t (t in 0..T-2) feature = first_pooled[:, t]
        # frame T-1 feature             = last_pooled[:, -1]
        feats = torch.cat([first_pooled, last_pooled[:, -1:]], dim=1)         # (B, T, D)
        return feats.float()
