"""DIFF (block-causal video diffusion) extractor.

Wraps our trained ckpt (e.g. aux λ=0.05 best). Loads EMA weights from
ckpt_last.pt; runs ``forward_features`` with t=0 (clean input) and
mean-pools over the spatial token grid.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf

from comparison.extractors.base import ExtractorMeta, RepresentationExtractor
from models.block_causal_transformer import BlockCausalVideoTransformer


def _build_model(cfg, in_channels: int) -> BlockCausalVideoTransformer:
    objective = cfg.train.get("objective", "mae")
    return BlockCausalVideoTransformer(
        in_channels=in_channels,
        patch_size=cfg.model.patch_size,
        num_frames=cfg.data.clip_len,
        image_size=cfg.data.resolution,
        dim=cfg.model.dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=0.0,
        enable_timestep=(objective in ("diffusion", "flow_matching")),
        action_dim=int(cfg.model.get("action_dim", 0)),
    )


class DiffExtractor(RepresentationExtractor):
    def __init__(self, ckpt_path: str, device: torch.device,
                 variant_name: Optional[str] = None) -> None:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = OmegaConf.create(state["config"])
        in_channels = 3 * len(cfg.data.cameras)
        model = _build_model(cfg, in_channels).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        self._model = model
        self._cfg = cfg
        self._device = device
        self._n_cams = len(cfg.data.cameras)
        n_params = sum(p.numel() for p in model.parameters())
        self._meta = ExtractorMeta(
            family="diff",
            variant=variant_name or Path(ckpt_path).parent.name,
            feat_dim=int(cfg.model.dim),
            n_params=n_params,
            native_resolution=int(cfg.data.resolution),
            native_clip_len=int(cfg.data.clip_len),
            has_native_recon=True,
        )

    @property
    def meta(self) -> ExtractorMeta:
        return self._meta

    @torch.no_grad()
    def extract(self, clip: torch.Tensor,
                task_caption: Optional[str] = None) -> torch.Tensor:
        # clip canonical: (B, T, C, H, W) at 128×128, C = 3*n_cams
        B, T = clip.shape[:2]
        clip = clip.to(self._device)
        t = torch.zeros(B, T, device=self._device, dtype=torch.long)
        feats = self._model.forward_features(clip, mask_indicator=None, t=t)
        return feats.float().mean(dim=2)  # (B, T, N, D) → (B, T, D)

    @torch.no_grad()
    def native_recon_loss(self, clip: torch.Tensor) -> Optional[float]:
        # MAE-style indistinct masked-patch loss on a single random mask;
        # not the full diffusion rollout (that's expensive). For headline
        # recon use the existing eval.py pipeline instead.
        return None
