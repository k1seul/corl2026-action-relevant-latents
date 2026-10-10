"""Deterministic action shuffling for the shuffled-ID matched control.

The purpose (see ``corl2026_coding_agent_instruction.md`` §4.4) is a control
that preserves the *action marginal* and the *entire optimization path* while
destroying the correct (z_t, z_{t+1}) → a_t correspondence.

We implement two modes:

  * ``circular_shift`` (primary, conservative):
        within each episode, roll the action sequence by a random *nonzero*
        offset that is bounded away from ±1-step alignment. The same offset
        applies to every clip drawn from that episode within a given run seed,
        so the action marginal is preserved exactly per episode.

  * ``within_task_cross_episode`` (secondary):
        deterministically permute episode IDs *within* the same task, then use
        the action sequence from the permuted episode. This preserves per-task
        action distribution while eliminating within-episode temporal
        correspondence. Cross-task permutation is intentionally *not* offered
        as the sole mode because it creates an easy mismatch.

Design principles:
  * pure-stdlib core (``shuffled_action_map``) so the shuffling logic can be
    verified without importing torch on the code-editing host;
  * a thin torch wrapper (``apply_shuffled_actions``) is applied *only* at
    dataloader-collate time, so upstream data pipelines are unchanged;
  * all randomness derives from ``(run_seed, episode_id, clip_start)`` via
    Python's ``hashlib.sha256`` — no reliance on ``random.Random`` state or
    torch RNG, so two processes with the same seed agree exactly.

This module does *not* touch actions used by the canonical probe or by the
downstream policy evaluation; those must always see the true actions.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable


# ── determinism ─────────────────────────────────────────────────────────────


def _deterministic_uint64(*parts: object) -> int:
    """Stable 64-bit int derived from arbitrary hashable parts.

    We use SHA-256 over ``|``-joined string representations so results are
    identical across Python versions, processes, and hardware.
    """
    key = "|".join(repr(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


# ── circular-shift core (stdlib only) ───────────────────────────────────────


@dataclass(frozen=True)
class ShiftPolicy:
    """Constraints on the random shift.

    ``min_abs_shift`` bounds the shift away from 1-step alignment on either
    side. Setting it to 2 means we never produce offsets in {-1, 0, +1}. The
    default of 2 matches the instruction ("bounded away from 1-step
    alignment").

    ``episode_length`` is the length of the action sequence to shift. Shifts
    are computed modulo this length so the marginal is preserved by
    construction (a circular rotation is a permutation of the index set).
    """
    episode_length: int
    min_abs_shift: int = 2

    def __post_init__(self) -> None:
        if self.episode_length <= 2 * self.min_abs_shift:
            raise ValueError(
                f"episode_length={self.episode_length} too small for "
                f"min_abs_shift={self.min_abs_shift}; increase episode length "
                "or lower the shift bound (but do not permit ±1 alignment)."
            )
        if self.min_abs_shift < 2:
            raise ValueError(
                "min_abs_shift must be ≥2 to keep the control away from the "
                "1-step-off degenerate case; see instruction §4.4."
            )


def sample_shift(run_seed: int, episode_id: object,
                 policy: ShiftPolicy) -> int:
    """Return a shift in ``[min_abs_shift, episode_length - min_abs_shift]``.

    The shift is a positive rotation; a negative rotation of the same
    magnitude would be equivalent under ``mod episode_length``. We restrict to
    strictly positive shifts for reporting simplicity.
    """
    lo = policy.min_abs_shift
    hi = policy.episode_length - policy.min_abs_shift
    span = hi - lo + 1
    if span <= 0:
        raise ValueError(
            f"no valid shift for episode_length={policy.episode_length}, "
            f"min_abs_shift={policy.min_abs_shift}"
        )
    r = _deterministic_uint64("shift", run_seed, episode_id) % span
    return lo + r


def circular_shifted_indices(run_seed: int, episode_id: object,
                             policy: ShiftPolicy) -> list[int]:
    """Return the shifted permutation of ``range(episode_length)``.

    Element ``i`` of the result is the index into the original action array
    that should be placed at position ``i`` in the shifted array. In other
    words, ``shifted[i] = original[result[i]]``.

    Rolling by ``k`` positions means position ``i`` reads from ``i + k``
    modulo the episode length, so this is exactly what
    ``numpy.roll(actions, -k)`` would produce.
    """
    k = sample_shift(run_seed, episode_id, policy)
    n = policy.episode_length
    return [(i + k) % n for i in range(n)]


# ── within-task cross-episode permutation ───────────────────────────────────


def within_task_permutation(run_seed: int, task_id: object,
                            episode_ids: Iterable[object]) -> dict[object, object]:
    """Return a derangement of ``episode_ids`` seeded by ``(run_seed, task_id)``.

    A derangement is a permutation with no fixed points, so no episode maps to
    itself and every episode receives someone else's action sequence. If the
    task has only one episode we return the identity (with a warning is left
    to the caller — this mode is inapplicable for singleton tasks).
    """
    eps = list(episode_ids)
    n = len(eps)
    if n <= 1:
        return {e: e for e in eps}

    # Fisher–Yates with a deterministic uint64 stream, then repair fixed
    # points by pairwise swap.
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        j = _deterministic_uint64("perm", run_seed, task_id, i) % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    return {eps[i]: eps[perm[i]] for i in range(n)}


# ── torch adaptor (imported lazily so this module is stdlib-testable) ──────


def apply_shuffled_actions(actions, *, run_seed: int, episode_id: object,
                           mode: str = "circular_shift",
                           min_abs_shift: int = 2):
    """Return a torch tensor of the same shape/dtype as ``actions``.

    Only the ordering of rows in the first (time) dimension is changed. Values
    are unchanged, so the per-dim mean/std of the returned tensor equals that
    of ``actions`` up to floating-point tolerance (they are literally the same
    numbers).

    Currently supports the ``circular_shift`` mode for a single episode-length
    clip. The cross-episode permutation mode is applied one level up, at the
    dataloader, where multiple episodes are visible simultaneously; that path
    lives in ``scripts/train_matched_control.py``.
    """
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover -- explicit failure
        raise RuntimeError(
            "apply_shuffled_actions requires torch; the pure-stdlib logic is "
            "in circular_shifted_indices."
        ) from exc
    import torch  # after guard above

    if not isinstance(actions, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(actions)}")
    if mode != "circular_shift":
        raise NotImplementedError(
            "cross-episode permutation is applied at the dataloader level; "
            "see scripts/train_matched_control.py"
        )

    T = int(actions.shape[0])
    policy = ShiftPolicy(episode_length=T, min_abs_shift=min_abs_shift)
    idx = circular_shifted_indices(run_seed, episode_id, policy)
    perm = torch.as_tensor(idx, dtype=torch.long, device=actions.device)
    return actions.index_select(0, perm)
