"""TOKEN (FSQ-tokenized causal transformer) extractor.

Wraps our trained ckpt + the FSQ tokenizer ckpt. Tokenizes pixels then
runs ``forward_features`` and mean-pools over the spatial token grid.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf

from comparison.extractors.base import ExtractorMeta, RepresentationExtractor
from models.token_video_transformer import TokenVideoTransformer
from models.tokenizer import FSQTokenizer


def _load_tokenizer(ckpt_path: str, in_channels: int,
                    device: torch.device) -> FSQTokenizer:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    tcfg = state["config"]
    tok = FSQTokenizer(
        in_channels=in_channels,
        levels=list(tcfg["model"]["levels"]),
        base_ch=tcfg["model"]["base_ch"],
    ).to(device)
    tok.load_state_dict(state["model"])
    tok.eval()
    return tok


@torch.no_grad()
def _encode(tok: FSQTokenizer, clip: torch.Tensor) -> torch.Tensor:
    B, T, C, H, W = clip.shape
    flat = clip.reshape(B * T, C, H, W)
    _, idx_grid = tok.encode(flat)
    return idx_grid.reshape(B, T, -1)


class TokenExtractor(RepresentationExtractor):
    def __init__(self, ckpt_path: str, device: torch.device,
                 variant_name: Optional[str] = None) -> None:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = OmegaConf.create(state["config"])
        in_channels = 3 * len(cfg.data.cameras)
        tokenizer = _load_tokenizer(cfg.train.tokenizer_ckpt, in_channels, device)

        # peek a sample shape — we need num_tokens_per_frame
        peek = torch.zeros(1, 1, in_channels, cfg.data.resolution,
                            cfg.data.resolution, device=device)
        idx = _encode(tokenizer, peek)
        N_per_frame = idx.shape[-1]

        model = TokenVideoTransformer(
            codebook_size=tokenizer.codebook_size,
            num_tokens_per_frame=N_per_frame,
            num_frames=cfg.data.clip_len,
            dim=cfg.model.dim,
            depth=cfg.model.depth,
            num_heads=cfg.model.num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            dropout=0.0,
            action_dim=int(cfg.model.get("action_dim", 0)),
        ).to(device)
        model.load_state_dict(state["model"])
        model.eval()

        self._tok = tokenizer
        self._model = model
        self._cfg = cfg
        self._device = device
        self._tok_in_channels = in_channels  # what the tokenizer was trained on
        n_params = sum(p.numel() for p in model.parameters())
        self._meta = ExtractorMeta(
            family="token",
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
        clip = clip.to(self._device)
        # tokenizer was trained on a specific camera count; drop extras
        # if the dataset feeds more cams than the tokenizer expects (e.g.
        # 2-cam DIFF dataset → 1-cam TOKEN model: keep agentview only).
        if clip.shape[2] > self._tok_in_channels:
            clip = clip[:, :, : self._tok_in_channels]
        token_ids = _encode(self._tok, clip)
        feats = self._model.forward_features(token_ids)  # (B, T, N, D)
        return feats.float().mean(dim=2)

    @torch.no_grad()
    def native_recon_loss(self, clip: torch.Tensor) -> Optional[float]:
        # token-CE on a single masked-token batch — use eval.py for headline.
        return None
