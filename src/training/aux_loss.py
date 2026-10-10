"""Auxiliary representation losses for diffusion training.

All heads are training-time only — discarded at eval/probe time.
The aux head state is saved in ``ckpt_last_raw.pt`` under key ``"aux_head"``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Head modules ────────────────────────────────────────────────────────────


class InvDynAuxHead(nn.Module):
    """concat(f_t, f_{t+1}) → a_t  (inverse dynamics, L2 or L1)."""

    def __init__(self, in_dim: int, action_dim: int = 7,
                 hidden: tuple[int, ...] = (512, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim * 2
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """pooled: (B, T, D) → (B, T-1, A)"""
        pairs = torch.cat([pooled[:, :-1], pooled[:, 1:]], dim=-1)
        return self.net(pairs)


class ForwardDynAuxHead(nn.Module):
    """concat(f_t, action_embed(a_t)) → f_{t+1}  (forward dynamics)."""

    def __init__(self, in_dim: int, action_dim: int = 7,
                 action_embed_dim: int = 64,
                 hidden: tuple[int, ...] = (512, 256)) -> None:
        super().__init__()
        self.action_embed = nn.Linear(action_dim, action_embed_dim)
        layers: list[nn.Module] = []
        prev = in_dim + action_embed_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, in_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, pooled: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        """pooled: (B,T,D), actions: (B,T,A) → (B,T-1,D)"""
        a_emb = self.action_embed(actions[:, :-1])          # (B,T-1,E)
        inp = torch.cat([pooled[:, :-1], a_emb], dim=-1)    # (B,T-1,D+E)
        return self.net(inp)


class ContrastiveAuxHead(nn.Module):
    """Linear projection for temporal contrastive (InfoNCE) loss."""

    def __init__(self, in_dim: int, proj_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SingleFrameAuxHead(nn.Module):
    """f_t → a_t  (single-frame action regression, no concat)."""

    def __init__(self, in_dim: int, action_dim: int = 7,
                 hidden: tuple[int, ...] = (512, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """pooled: (B, T, D) → (B, T, A)"""
        return self.net(pooled)


class MultiStepInvDynHead(nn.Module):
    """concat(f_t, f_{t+k}) → (a_t, ..., a_{t+output_k-1}).

    k=2, output_k=1 → predict just a_t from a 2-frame gap.
    k=4, output_k=4 → predict the 4-action chunk from a 4-frame gap.
    """

    def __init__(self, in_dim: int, action_dim: int = 7,
                 k: int = 2, output_k: int = 1,
                 hidden: tuple[int, ...] = (512, 256)) -> None:
        super().__init__()
        self.k = k
        self.output_k = output_k
        self.action_dim = action_dim
        layers: list[nn.Module] = []
        prev = in_dim * 2
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, action_dim * output_k))
        self.net = nn.Sequential(*layers)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """pooled: (B, T, D) → (B, T-k, output_k * A)"""
        k = self.k
        pairs = torch.cat([pooled[:, :-k], pooled[:, k:]], dim=-1)
        return self.net(pairs)


# ─── Loss functions ───────────────────────────────────────────────────────────


def aux_invdyn_loss(features: torch.Tensor, aux_head: InvDynAuxHead,
                   actions: torch.Tensor,
                   loss_type: str = "l2") -> torch.Tensor:
    """features: (B,T,N,D), actions: (B,T,A) → scalar."""
    pooled = features.mean(dim=2)          # (B,T,D)
    pred = aux_head(pooled)                # (B,T-1,A)
    target = actions[:, :-1]              # (B,T-1,A)
    diff = pred - target
    return diff.abs().mean() if loss_type == "l1" else diff.pow(2).mean()


def forward_dyn_loss(features: torch.Tensor, aux_head: ForwardDynAuxHead,
                    actions: torch.Tensor) -> torch.Tensor:
    """Predict f_{t+1} from (f_t, a_t). Target is stop-gradient."""
    pooled = features.mean(dim=2)                  # (B,T,D)
    pred_next = aux_head(pooled, actions)           # (B,T-1,D)
    target_next = pooled[:, 1:].detach()            # (B,T-1,D) — no grad
    return (pred_next - target_next).pow(2).mean()


def contrastive_temporal_loss(features: torch.Tensor,
                               aux_head: ContrastiveAuxHead,
                               temperature: float = 0.1) -> torch.Tensor:
    """InfoNCE: anchor=f_t, positive=f_{t+1}, negatives=f_{t+1} from other clips."""
    pooled = features.mean(dim=2)               # (B,T,D)
    B, T, D = pooled.shape
    enc = F.normalize(aux_head(pooled.reshape(B * T, D)), dim=-1)
    enc = enc.reshape(B, T, -1)                 # (B,T,P)
    # per-timestep InfoNCE over batch dimension
    losses = []
    for t in range(T - 1):
        a = enc[:, t]        # (B,P) anchor
        p = enc[:, t + 1]    # (B,P) positive (same clip, next frame)
        logits = a @ p.T / temperature   # (B,B) — diagonal = positives
        tgt = torch.arange(B, device=logits.device)
        losses.append(F.cross_entropy(logits, tgt))
    return torch.stack(losses).mean()


def single_frame_action_loss(features: torch.Tensor,
                              aux_head: SingleFrameAuxHead,
                              actions: torch.Tensor) -> torch.Tensor:
    pooled = features.mean(dim=2)    # (B,T,D)
    pred = aux_head(pooled)          # (B,T,A)
    return (pred - actions).pow(2).mean()


def multistep_invdyn_loss(features: torch.Tensor,
                          aux_head: MultiStepInvDynHead,
                          actions: torch.Tensor) -> torch.Tensor:
    pooled = features.mean(dim=2)    # (B,T,D)
    k, output_k = aux_head.k, aux_head.output_k
    pred = aux_head(pooled)          # (B,T-k, output_k*A)
    valid_T = pooled.shape[1] - k
    if output_k == 1:
        target = actions[:, :valid_T]   # (B,T-k,A)
    else:
        # stack consecutive action chunks
        target = torch.stack(
            [actions[:, s:s + valid_T] for s in range(output_k)], dim=2
        ).reshape(pred.shape)           # (B,T-k, output_k*A)
    return (pred - target).pow(2).mean()


# ─── Unified dispatcher ───────────────────────────────────────────────────────


def compute_aux_loss(features: torch.Tensor,
                     aux_head: nn.Module,
                     actions: torch.Tensor,
                     head_type: str) -> torch.Tensor:
    if head_type in ("invdyn_l2", "invdyn"):
        return aux_invdyn_loss(features, aux_head, actions, loss_type="l2")
    if head_type == "invdyn_l1":
        return aux_invdyn_loss(features, aux_head, actions, loss_type="l1")
    if head_type == "forward_dyn":
        return forward_dyn_loss(features, aux_head, actions)
    if head_type == "contrastive":
        return contrastive_temporal_loss(features, aux_head)
    if head_type == "single_frame":
        return single_frame_action_loss(features, aux_head, actions)
    if head_type in ("multistep_invdyn", "multistep"):
        return multistep_invdyn_loss(features, aux_head, actions)
    raise ValueError(f"unknown aux_head_type: {head_type!r}")
