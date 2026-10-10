"""Past-only ("causal") feature extraction for video encoders.

The instruction (§5.1, §8) requires that closed-loop feature extraction and
one of the two probe-context sensitivity cells use a strictly past-only clip
context: at timestep t, the encoder receives frames [t-clip_len+1, ..., t]
with the beginning padded by repeating the first observation. The encoder
must never see future frames.

We wrap any ``RepresentationExtractor`` (see src/comparison/extractors/base.py)
with ``CausalContextExtractor`` and produce per-step ``z_t`` where each
``z_t`` was computed from a strictly past-only window.

Two performance notes:

* The naive implementation re-encodes an entire clip per timestep, i.e. O(T²)
  frames per episode. For short clips (T=32 or so) on frozen encoders this is
  acceptable at rollout evaluation time. Set ``strategy="per_timestep"`` to
  get this.
* For faster offline extraction where causality only matters relative to
  clip boundaries, set ``strategy="chunk_causal"``: split the episode into
  non-overlapping windows and only expose the final feature of each window
  as ``z_t``. This is a coarser approximation and must be labeled as such in
  the report.

We do *not* replace the original offline extractor. Every result computed
with this wrapper is written to a JSONL field ``extractor_mode:
"causal_context"`` so it can never be silently confused with the
bidirectionally-contextualized native features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CausalContextConfig:
    clip_len: int
    strategy: Literal["per_timestep", "chunk_causal"] = "per_timestep"
    # If True, verify at extraction time that no future frame ever enters the
    # window. This is a runtime assertion, not a training-time check.
    verify_no_future: bool = True


def build_causal_windows(num_frames: int, clip_len: int) -> list[list[int]]:
    """Return, for each t in ``range(num_frames)``, the list of frame indices
    that form the past-only window ending at t.

    The window is left-padded by repeating index 0 whenever t + 1 < clip_len.
    """
    if clip_len <= 0:
        raise ValueError(f"clip_len must be positive, got {clip_len}")
    windows: list[list[int]] = []
    for t in range(num_frames):
        start = t - clip_len + 1
        if start >= 0:
            windows.append(list(range(start, t + 1)))
        else:
            pad = [0] * (-start)
            windows.append(pad + list(range(0, t + 1)))
    return windows


def build_chunk_causal_windows(num_frames: int, clip_len: int) -> list[list[int]]:
    """Non-overlapping windows of size ``clip_len``; the final frame of each
    window becomes the ``z_t`` for the *last* timestep in that window. Earlier
    timesteps within the window are given the same window's final feature —
    consumers should be aware this coarsens per-step resolution.
    """
    windows: list[list[int]] = []
    t = 0
    while t < num_frames:
        end = min(t + clip_len, num_frames)
        # left-pad the final chunk if it's shorter than clip_len
        idx = list(range(t, end))
        if len(idx) < clip_len:
            idx = [idx[0]] * (clip_len - len(idx)) + idx
        windows.append(idx)
        t = end
    return windows


def assert_no_future(windows: list[list[int]]) -> None:
    for t, w in enumerate(windows):
        for f in w:
            if f > t:
                raise AssertionError(
                    f"causal window for t={t} contains future frame {f}; "
                    "this indicates a bug in window construction."
                )


# ── torch adaptor ───────────────────────────────────────────────────────────


class CausalContextExtractor:
    """Adaptor around a ``RepresentationExtractor`` producing past-only ``z_t``.

    Instantiate as::

        base = VJEPA2Extractor(device)
        causal = CausalContextExtractor(base, CausalContextConfig(clip_len=32))
        z = causal.extract(episode)   # episode: (T, C, H, W) or (B, T, C, H, W)
    """

    def __init__(self, base_extractor, config: CausalContextConfig) -> None:
        self.base = base_extractor
        self.config = config
        self.mode = "causal_context"

    def extract(self, episode):
        import torch

        if episode.ndim == 4:
            episode = episode.unsqueeze(0)
        B, T, C, H, W = episode.shape
        windows = (
            build_causal_windows(T, self.config.clip_len)
            if self.config.strategy == "per_timestep"
            else build_chunk_causal_windows(T, self.config.clip_len)
        )
        if self.config.verify_no_future:
            assert_no_future(windows)

        outs = []
        for w in windows:
            idx = torch.tensor(w, device=episode.device, dtype=torch.long)
            clip = episode.index_select(1, idx)   # (B, clip_len, C, H, W)
            feats = self.base.extract(clip)       # (B, clip_len, D)
            outs.append(feats[:, -1])              # take the last (t) step
        return torch.stack(outs, dim=1)            # (B, T, D)
