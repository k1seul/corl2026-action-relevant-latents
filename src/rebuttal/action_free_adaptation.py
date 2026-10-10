"""Optional action-free adaptation control (instruction §4.6).

Two augmented views of the same clip; a trainable "student" encoder is
matched to a frozen "teacher" copy of the initial encoder via cosine
similarity of a projector. Actions are never read.

This is provided as an OPTIONAL secondary control. It must never be used to
displace the primary forward-dynamics control, per the instruction, because
the fair cross-backbone control is a forward-dynamics head with the same
transitions and actions but no inverse-dynamics decoding.

The loss is deliberately written so that swapping in any random action tensor
leaves it unchanged exactly — see ``tests/test_action_free.py``.
"""
from __future__ import annotations

from typing import Callable


def action_free_cosine_loss(student_features, teacher_features, projector):
    """1 − cos(projector(student), stop_grad(teacher)), averaged over B, T.

    Both tensors must be shape ``(B, T, D)``. ``teacher_features`` is expected
    to already be detached; we detach again for safety. ``projector`` is a
    trainable ``nn.Module`` mapping D → D_proj.
    """
    import torch
    import torch.nn.functional as F  # noqa: N812

    if student_features.shape != teacher_features.shape:
        raise ValueError(
            f"shape mismatch: student={student_features.shape}, "
            f"teacher={teacher_features.shape}"
        )
    s = projector(student_features)
    t = teacher_features.detach()
    s = F.normalize(s, dim=-1)
    t = F.normalize(t, dim=-1)
    return (1.0 - (s * t).sum(dim=-1)).mean()


def make_two_view_batch(clip, augment_fn: Callable, *, seed_a: int, seed_b: int):
    """Build (view_a, view_b) with mild semantics-preserving augmentations.

    ``augment_fn(clip, seed)`` is provided by the caller and must be
    deterministic given its ``seed`` argument. Two different seeds should
    yield two independent augmentations of the same clip.
    """
    return augment_fn(clip, seed_a), augment_fn(clip, seed_b)
