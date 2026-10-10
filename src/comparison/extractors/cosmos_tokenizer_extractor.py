"""Cosmos-1 Continuous Image Tokenizer (CI16x16) frozen feature extractor.

NVIDIA's reconstruction-aligned autoencoder used in Cosmos Foundation Models.
Same encoder family as SDXL VAE in our cross-architecture comparison; pair
together as "reconstruction-aligned" representatives.

The HF repo `nvidia/Cosmos-Tokenizer-CI16x16` ships TorchScript JIT files
(encoder.jit, decoder.jit) — no Python class. We use the JIT encoder
directly, which works on (B, C, H, W) RGB inputs.

At 128×128 input (LIBERO native): latent shape (B, 16, 8, 8) → flatten
1024 dim per frame, matching V-JEPA ViT-L and SDXL VAE feat_dim.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from comparison.extractors.base import ExtractorMeta, RepresentationExtractor


DEFAULT_REPO = "nvidia/Cosmos-Tokenizer-CI16x16"


def _download_encoder(repo: str, device: torch.device):
    """Download encoder.jit from HF and load as TorchScript module."""
    from huggingface_hub import hf_hub_download
    enc_path = hf_hub_download(repo_id=repo, filename="encoder.jit")
    encoder = torch.jit.load(enc_path, map_location=device)
    return encoder, enc_path


class CosmosTokenizerExtractor(RepresentationExtractor):
    """Cosmos-1 CI16x16 image tokenizer encoder, frozen.

    Encoder output at 128×128 input: (B, 16, 8, 8). Flatten to 1024-d
    per frame (matching ViT-L family dimensionality for paper parity).
    """

    def __init__(
        self,
        device: torch.device,
        ckpt_path: Optional[str] = None,
        native_image_size: int = 128,
        finetuned_state_dict: Optional[dict] = None,
        variant_name: Optional[str] = None,
    ) -> None:
        repo = ckpt_path or DEFAULT_REPO

        # TorchScript JIT modules don't accept finetuned state dict via
        # load_state_dict — finetune protocol saves the JIT module pickled
        # whole (see training script). Detect format by inspecting input.
        if finetuned_state_dict is not None and isinstance(finetuned_state_dict, torch.nn.Module):
            encoder = finetuned_state_dict.to(device)
        else:
            encoder, _ = _download_encoder(repo, device)
            if finetuned_state_dict is not None:
                # JIT modules don't typically accept state_dict reload; warn
                try:
                    encoder.load_state_dict(finetuned_state_dict, strict=False)
                except Exception as e:
                    print(f"[cosmos] WARN: state_dict load failed ({e}); "
                          f"falling back to pretrained encoder.")

        encoder = encoder.to(device).eval()
        for p in encoder.parameters():
            p.requires_grad = False

        self._encoder = encoder
        self._device = device
        self._native_size = int(native_image_size)

        # latent shape (16, native/16, native/16)
        latent_h = self._native_size // 16
        feat_dim = 16 * latent_h * latent_h
        n_params = sum(p.numel() for p in encoder.parameters())
        self._meta = ExtractorMeta(
            family="cosmos_tokenizer",
            variant=variant_name or "cosmos_ci16x16",
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

        Cosmos tokenizer expects [-1, 1] inputs (typical VAE convention)."""
        clip = clip.to(self._device)
        B, T, C, H, W = clip.shape
        if C > 3:
            clip = clip[:, :, :3]
            C = 3
        x = clip.reshape(B * T, C, H, W)
        if (H, W) != (self._native_size, self._native_size):
            x = F.interpolate(x, size=(self._native_size, self._native_size),
                                mode="bilinear", align_corners=False)
        x = (x * 2.0 - 1.0).to(torch.bfloat16)
        out = self._encoder(x)
        # CI16x16 returns (latent, extra) tuple; first element is the latent
        if isinstance(out, (tuple, list)):
            latent = out[0]
        else:
            latent = out
        latent = latent.flatten(1)                          # (BT, 16*h*h)
        return latent.float().reshape(B, T, -1)
