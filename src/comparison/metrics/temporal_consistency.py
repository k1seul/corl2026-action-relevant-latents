"""Temporal consistency metric on per-frame features.

For features ``(B, T, D)``:
  cos_adjacent : mean cosine similarity between consecutive frames
                 (high = smooth trajectory; very high might indicate
                 the encoder is not changing meaningfully across time)
  cos_curve(k) : cos(f_t, f_{t+k}) decay over k ∈ {1, 2, 4, 8, 16}

A "good" representation has high cos_adjacent (smoothness) but a
non-trivial decay curve (frames remain distinct over longer gaps —
no collapse).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F


@dataclass
class TemporalConsistencyResult:
    cos_adjacent: float          # = cos_curve[0]
    cos_curve_ks: List[int]      # ks tested
    cos_curve_vals: List[float]  # mean cos sim at each k
    n_clips: int


def temporal_consistency(
    features: torch.Tensor,
    ks: List[int] = (1, 2, 4, 8, 16),
) -> TemporalConsistencyResult:
    """features: (B, T, D)."""
    if features.ndim != 3:
        raise ValueError(f"expected (B, T, D), got {tuple(features.shape)}")
    B, T, _ = features.shape
    vals: List[float] = []
    used_ks: List[int] = []
    for k in ks:
        if k >= T:
            continue
        a = features[:, :-k]
        b = features[:, k:]
        cos = F.cosine_similarity(a, b, dim=-1)  # (B, T-k)
        vals.append(cos.mean().item())
        used_ks.append(k)
    return TemporalConsistencyResult(
        cos_adjacent=vals[0] if vals else float("nan"),
        cos_curve_ks=used_ks,
        cos_curve_vals=vals,
        n_clips=B,
    )
