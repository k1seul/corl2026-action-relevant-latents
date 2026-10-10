"""Per-dim action probe metrics for the Full Results table.

Action dim mapping (LIBERO 7-D end-effector control):
    0: dx       translation x (delta in world frame)
    1: dy       translation y
    2: dz       translation z
    3: drx      rotation x (axis-angle / euler depending on suite, but probe is
                blind to representation — same MSE regardless)
    4: dry      rotation y
    5: drz      rotation z
    6: gripper  binary-ish open/close command, continuous range typically [-1, 1]

For Robomimic the mapping is the same (7-D delta pose + gripper).
For CALVIN it is the same 7-D (xyz, rxryrz, gripper) per the CALVIN abs-action
spec.
For MetaWorld the native action is 4-D (xyz + gripper), zero-padded to 7-D
by the loader; the rotation dims (idx 3..5) are exactly zero and their MSE
will be ~0 by construction. Document accordingly when reporting.
"""
from __future__ import annotations

import torch


DIM_NAMES = ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")


def per_dim_mse(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Per-dim MSE between pred (N, A) and target (N, A).

    Returns dict keyed by DIM_NAMES (or "dim_{i}" if A != 7).
    """
    assert pred.shape == target.shape, f"shape mismatch: {pred.shape} vs {target.shape}"
    err = ((pred - target) ** 2).mean(dim=0)  # (A,)
    out: dict[str, float] = {}
    for i, v in enumerate(err.tolist()):
        name = DIM_NAMES[i] if (err.numel() == 7 and i < 7) else f"dim_{i}"
        out[name] = float(v)
    return out


def per_dim_r2(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Per-dim R² between pred (N, A) and target (N, A).

    R²_i = 1 - SSE_i / SS_total_i  where SS_total_i = Var(target_i) * N.
    Returns dict keyed by DIM_NAMES.
    """
    assert pred.shape == target.shape
    sse = ((pred - target) ** 2).sum(dim=0)  # (A,)
    var = target.var(dim=0, unbiased=False)  # (A,)
    n = target.shape[0]
    ss_total = var * n
    r2 = 1.0 - sse / ss_total.clamp_min(1e-12)
    out: dict[str, float] = {}
    for i, v in enumerate(r2.tolist()):
        name = DIM_NAMES[i] if (r2.numel() == 7 and i < 7) else f"dim_{i}"
        out[name] = float(v)
    return out


def aggregate_per_dim(per_dim: dict[str, float]) -> float:
    """Mean across dims (matches the existing aggregate MSE/R² convention)."""
    vals = list(per_dim.values())
    return float(sum(vals) / len(vals)) if vals else float("nan")
