"""Within-task cross-episode derangement for the PRIMARY shuffled-ID control.

Per session-2 STEP 3 requirement, the primary shuffled-ID condition is:

    within_task_cross_episode_derangement

For every transition clip drawn during training we:

  * keep the original clip's frames (encoder input);
  * keep the original clip's task ID;
  * REPLACE the aligned action sequence with the action sequence taken from
    a DIFFERENT episode of the same task, chosen by a fixed derangement over
    that task's episode set;
  * derive the derangement deterministically from ``(run_seed, task_id)``.

This is scientifically stronger than a within-episode circular shift because
adjacent robot actions are highly correlated (~ smooth) — a small circular
shift barely changes the local action landscape. Sampling from a different
episode destroys the (z_t, z_{t+1}) → a_t correspondence at every offset.

Guarantees (all covered by unit tests in tests/test_derangement.py):

  D1  no source episode maps to itself (derangement, not permutation);
  D2  every episode is used as a target exactly once (bijection);
  D3  task ID is preserved for every clip;
  D4  the action multiset per task is exactly preserved (bijection over
      episodes ⇒ union of action sequences unchanged);
  D5  deterministic under the same ``(run_seed, task_id)``;
  D6  a different ``run_seed`` produces a materially different mapping;
  D7  transition-action alignment is destroyed even when the target
      episode has similar aggregate statistics (verified by an
      alignment-signal test).

The circular-shift permutation from ``action_shuffling.py`` remains
available as ``shuffled_id_circular_sensitivity`` (secondary). Do NOT use
it as the primary reported control.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence


# ── determinism helper ─────────────────────────────────────────────────────


def _deterministic_uint64(*parts: object) -> int:
    key = "|".join(repr(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big", signed=False)


# ── derangement ─────────────────────────────────────────────────────────────


def within_task_derangement(run_seed: int, task_id: object,
                            episode_ids: Sequence[object]) -> dict[object, object]:
    """Return a bijection ``episode_id → source_episode_id`` such that
    no episode maps to itself.

    Algorithm: seeded Fisher–Yates, followed by a repair pass that swaps
    any fixed point with its neighbour. For len==1 there is no valid
    derangement; we return an empty mapping so the caller knows to fall
    back to the untouched actions for that singleton task (or to skip the
    task entirely, depending on training policy).

    Determinism: identical output for identical ``(run_seed, task_id,
    tuple(episode_ids))``; NO dependence on Python's ``random`` state or
    torch RNG.
    """
    eps = list(episode_ids)
    n = len(eps)
    if n <= 1:
        return {}  # not derangeable — caller must handle

    perm = list(range(n))
    # Fisher–Yates
    for i in range(n - 1, 0, -1):
        j = _deterministic_uint64("dyk", run_seed, task_id, i) % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    # Repair fixed points by swapping with next index. For n>=2 this always
    # terminates because at most one fixed point can remain after a full pass.
    for _ in range(n):
        fixed = [i for i in range(n) if perm[i] == i]
        if not fixed:
            break
        for i in fixed:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]

    if any(perm[i] == i for i in range(n)):  # pragma: no cover
        raise RuntimeError(
            f"failed to derange episodes for task={task_id!r} "
            f"with {n} episodes; this is a bug in within_task_derangement.")

    return {eps[i]: eps[perm[i]] for i in range(n)}


def mapping_hash(mapping: dict) -> str:
    """Stable 12-char hash of a mapping — recorded in provenance.

    Two runs with the same seed on the same episode set produce identical
    hashes; different seeds produce different hashes with overwhelming
    probability.
    """
    items = sorted((repr(k), repr(v)) for k, v in mapping.items())
    h = hashlib.sha256("||".join(f"{k}=>{v}" for k, v in items).encode()).hexdigest()
    return h[:12]


# ── multi-task planner ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class DerangementPlan:
    """A frozen, complete mapping across every task in the training set.

    Built once per (run_seed, dataset_snapshot). Passed to the dataloader
    collate function; the collate replaces each clip's action array with
    the action array of ``plan.mapping[task][episode]``.
    """
    run_seed: int
    per_task: dict[object, dict[object, object]]
    per_task_hash: dict[object, str]
    skipped_singleton_tasks: list[object]

    def total_episodes(self) -> int:
        return sum(len(m) for m in self.per_task.values())

    def global_hash(self) -> str:
        combined = "||".join(
            f"{t}:{self.per_task_hash[t]}" for t in sorted(self.per_task_hash, key=repr)
        )
        return hashlib.sha256(combined.encode()).hexdigest()[:12]


def build_derangement_plan(run_seed: int,
                           episodes_per_task: dict[object, Sequence[object]]
                           ) -> DerangementPlan:
    per_task: dict[object, dict[object, object]] = {}
    per_task_hash: dict[object, str] = {}
    skipped: list[object] = []
    for task, eps in episodes_per_task.items():
        m = within_task_derangement(run_seed, task, eps)
        if not m:
            skipped.append(task)
            continue
        per_task[task] = m
        per_task_hash[task] = mapping_hash(m)
    return DerangementPlan(
        run_seed=run_seed,
        per_task=per_task,
        per_task_hash=per_task_hash,
        skipped_singleton_tasks=skipped,
    )


# ── torch adaptor (lazy import) ─────────────────────────────────────────────


def apply_derangement_actions(actions, *, task_id: object, episode_id: object,
                              plan: DerangementPlan,
                              action_bank: dict[object, dict[object, "torch.Tensor"]]):
    """Return the action tensor that should be used in the aux loss for this
    clip.

    ``action_bank[task_id][episode_id]`` must contain the full action
    sequence for the *source* episode. We slice it to the same length as
    ``actions``. If the incoming clip belongs to a singleton task that was
    skipped by the plan (or if the task/episode is not in the plan for any
    reason), we raise, forcing the training loop to skip the sample —
    silent fall-through to true-ID would corrupt the control.
    """
    import torch  # noqa: F401

    if task_id not in plan.per_task:
        raise KeyError(
            f"task {task_id!r} was not deranged (probably singleton). "
            "Skip the clip or exclude the task from the shuffled-ID split."
        )
    src_ep = plan.per_task[task_id].get(episode_id)
    if src_ep is None:
        raise KeyError(f"episode {episode_id!r} not in derangement plan for "
                       f"task {task_id!r}")
    src_actions = action_bank[task_id][src_ep]
    T = int(actions.shape[0])
    if src_actions.shape[0] < T:
        # Repeat the source's last action to match length — deterministic,
        # documented, and only ever hits when the source episode is shorter
        # than the requested clip. Never a rare silent fallback; caller
        # should log this event count in provenance.
        import torch
        pad = src_actions[-1:].expand(T - src_actions.shape[0], *src_actions.shape[1:])
        return torch.cat([src_actions, pad], dim=0)
    return src_actions[:T].to(actions.dtype).to(actions.device)
