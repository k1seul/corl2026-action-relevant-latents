"""Common feature-extractor interface for cross-architecture comparison.

All wrappers (DIFF, TOKEN, V-JEPA2, LAPA) implement ``extract(clip)`` to
return per-frame features ``(B, T, D)`` already mean-pooled over the
spatial token grid. The wrapper handles model-specific preprocessing
(e.g. V-JEPA2 wants 224×224 at ImageNet stats).

Inputs to ``extract``:
  clip : (B, T, C, H, W) float in [0, 1]
         our canonical format: 2 cams stacked on C dim → C=6
         resolution 128, clip_len 32
  task_caption : optional string, used by language-conditioned extractors

Outputs:
  features : (B, T, D) — D varies per family (768 ours, 1024 V-JEPA2, ...)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ExtractorMeta:
    """Static info about an extractor — for logging / table headers."""
    family: str          # "diff" | "token" | "vjepa2" | "lapa"
    variant: str         # specific ckpt name
    feat_dim: int
    n_params: int        # frozen-eval params (encoder only)
    native_resolution: int
    native_clip_len: Optional[int] = None
    has_native_recon: bool = True


class RepresentationExtractor(ABC):
    """Abstract feature extractor. Subclasses adapt one model family."""

    @property
    @abstractmethod
    def meta(self) -> ExtractorMeta:
        ...

    @abstractmethod
    @torch.no_grad()
    def extract(
        self,
        clip: torch.Tensor,
        task_caption: Optional[str] = None,
    ) -> torch.Tensor:
        """Extract per-frame features.

        Args:
            clip: (B, T, C, H, W) float in [0, 1] at canonical 128×128
            task_caption: optional caption (ignored by non-language models)

        Returns:
            (B, T, D) per-frame features, mean-pooled over spatial tokens.
        """
        ...

    def native_recon_loss(
        self, clip: torch.Tensor
    ) -> Optional[float]:
        """Compute the model's native reconstruction loss on a batch.

        Returns a scalar (mean over batch) or ``None`` if the model has
        no meaningful pixel-recon analogue (V-JEPA2 latent prediction,
        LAPA quantized targets).

        Sub-classes override; default returns None.
        """
        return None
