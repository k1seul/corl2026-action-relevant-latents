"""Canonical eval-mode names and schema — STEP 5.

We MUST keep three evaluation modes strictly separate. Any result JSONL row
must carry a top-level ``eval_mode`` field taking one of:

  * ``canonical_offline_action_probe``  — original paper protocol, h_ψ(z_t)→a_t
                                          with the encoder's NATIVE offline
                                          context; must not force past-only.
  * ``causal_context_action_probe``     — h_ψ(z_t^causal)→a_t; past-only
                                          feature extraction via the wrapper
                                          in ``src/rebuttal/causal_context.py``.
  * ``closed_loop_policy_success``      — downstream policy under past-only
                                          observations at every timestep.

Downstream regeneration scripts refuse to include a row whose ``eval_mode``
is missing or unknown, and never merge cells from different modes into the
same table row.
"""
from __future__ import annotations

from enum import Enum
from typing import Iterable


class EvalMode(str, Enum):
    CANONICAL_OFFLINE = "canonical_offline_action_probe"
    CAUSAL_CONTEXT = "causal_context_action_probe"
    CLOSED_LOOP = "closed_loop_policy_success"

    @classmethod
    def values(cls) -> tuple[str, ...]:
        return tuple(m.value for m in cls)


def validate_eval_mode(mode: str) -> None:
    if mode not in EvalMode.values():
        raise ValueError(
            f"unknown eval_mode {mode!r}; expected one of {EvalMode.values()}"
        )


def separate_modes(rows: Iterable[dict]) -> dict[str, list[dict]]:
    """Split an iterable of result rows into a dict keyed by eval_mode.

    Rows lacking ``eval_mode`` go into the ``__unlabeled__`` bucket so
    regeneration can loudly refuse to consume them.
    """
    out: dict[str, list[dict]] = {m.value: [] for m in EvalMode}
    out["__unlabeled__"] = []
    for r in rows:
        m = r.get("eval_mode")
        if m in out:
            out[m].append(r)
        else:
            out["__unlabeled__"].append(r)
    return out
