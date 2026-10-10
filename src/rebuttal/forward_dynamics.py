"""Forward-dynamics control with a non-drifting teacher target.

Session-2 STEP 4: the forward-dynamics head predicts ``z_{t+1}`` from
``(z_t, a_t)`` with the *target* branch driven by a frozen or EMA copy of
the initial encoder, so the target does not trivially drift with the
online encoder (which would let the loss collapse without learning any
action-conditional structure).

Loss:

    z_online     = encoder(clip)                               # (B,T,D)
    z_target     = target_encoder(clip).detach()               # (B,T,D)
    pred_next    = predictor(z_online[:, :-1], action_embed(a_t[:, :-1]))
    loss_fd      = mse( pred_next, z_target[:, 1:] )

We wrap `training.aux_loss.ForwardDynAuxHead` with a target-provider
callable so callers can pass either:

  * ``target_mode="frozen_initial"`` — a deep copy of the encoder at t=0
                                       held fixed for the whole run;
  * ``target_mode="ema"`` — an EMA (decay τ) of the online encoder,
                            updated after every optimizer step.

Collapse diagnostics (``diagnose_forward_dynamics``) computed each ``diag_every``
steps:

  D_fd_1  per-dim feature variance                       (fail if < min_var)
  D_fd_2  mean pairwise cosine over a batch              (fail if > max_cos)
  D_fd_3  predictor output variance                      (fail if < min_var)
  D_fd_4  encoder gradient norm                          (fail if == 0)
  D_fd_5  predictor gradient norm                        (fail if == 0)
  D_fd_6  loss vs constant-predictor baseline            (fail if not < baseline)

Every diagnostic is written to a JSONL alongside `provenance.json` so the
smoke-test gate (§4.6) can decide pass/fail from disk. Diagnostics never
raise; the caller checks and aborts.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


# ── configuration ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ForwardDynConfig:
    target_mode: str = "frozen_initial"   # or "ema"
    ema_decay: float = 0.999
    # collapse-detection thresholds
    min_feature_variance_per_dim: float = 1e-4
    max_mean_pairwise_cosine: float = 0.95
    min_predictor_output_variance: float = 1e-4
    diag_every_steps: int = 500

    def sanity_check(self) -> None:
        if self.target_mode not in ("frozen_initial", "ema"):
            raise ValueError(f"bad target_mode: {self.target_mode}")
        if not (0.0 < self.ema_decay < 1.0):
            raise ValueError(f"bad ema_decay: {self.ema_decay}")


# ── target-encoder wrapper ────────────────────────────────────────────────


def make_target_encoder(online_encoder, config: ForwardDynConfig):
    """Return an object with the same forward interface as ``online_encoder``
    whose parameters are frozen at construction time (``frozen_initial``) or
    updated by EMA (``ema``) via ``step_ema``.
    """
    import torch

    if config.target_mode not in ("frozen_initial", "ema"):
        raise ValueError(config.target_mode)

    target = copy.deepcopy(online_encoder).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    return target


@torch.no_grad() if False else (lambda f: f)  # sentinel so linters don't warn
def step_ema(online_encoder, target_encoder, decay: float) -> None:
    """In-place EMA update of the target encoder's parameters.

    Applied AFTER the optimizer step, so target does not lag by a fractional
    step. Callers under a DDP wrapper should ensure they pass the
    ``.module`` view of the online encoder.
    """
    import torch
    with torch.no_grad():
        for p_online, p_target in zip(online_encoder.parameters(),
                                       target_encoder.parameters()):
            p_target.mul_(decay).add_(p_online.data, alpha=1.0 - decay)


# ── loss ───────────────────────────────────────────────────────────────────


def forward_dynamics_loss(*,
                          online_features,
                          target_features,
                          predictor,
                          actions):
    """Return (loss, aux_stats).

    ``online_features``: (B, T, D)   with grad
    ``target_features``: (B, T, D)   detached
    ``predictor``:  callable (features_prev, actions_prev) -> features_next_pred
    ``actions``:  (B, T, A)

    Enforces ``target_features.requires_grad == False`` at the tensor level
    to make it structurally impossible for gradient to flow into the target
    encoder (defence-in-depth beyond the caller's .detach()).
    """
    import torch
    import torch.nn.functional as F  # noqa: N812

    if target_features.requires_grad:
        raise RuntimeError(
            "target_features must be detached before forward_dynamics_loss; "
            "call `.detach()` on the target-encoder output."
        )
    if online_features.shape != target_features.shape:
        raise ValueError(
            f"feature shape mismatch: online={online_features.shape}, "
            f"target={target_features.shape}")

    pred_next = predictor(online_features[:, :-1], actions[:, :-1])
    tgt_next = target_features[:, 1:]
    diff = pred_next - tgt_next
    loss = diff.pow(2).mean()

    # small aux stats (used by diagnose_forward_dynamics but cheap enough to
    # collect every step if desired)
    with torch.no_grad():
        aux = dict(
            loss=float(loss.detach().cpu()),
            pred_next_var=float(pred_next.detach().var().cpu()),
            tgt_next_var=float(tgt_next.detach().var().cpu()),
        )
    return loss, aux


# ── collapse diagnostics ──────────────────────────────────────────────────


@dataclass
class ForwardDynDiagnostics:
    step: int
    per_dim_variance_mean: float
    per_dim_variance_min: float
    mean_pairwise_cosine: float
    predictor_output_variance: float
    encoder_grad_norm: float
    predictor_grad_norm: float
    loss: float
    constant_baseline_loss: float
    passed: bool = field(init=False)
    failed_checks: list[str] = field(default_factory=list, init=False)

    def evaluate(self, config: ForwardDynConfig) -> None:
        checks: list[str] = []
        if self.per_dim_variance_min < config.min_feature_variance_per_dim:
            checks.append(
                f"per_dim_variance_min={self.per_dim_variance_min:.2e} "
                f"< min={config.min_feature_variance_per_dim:.2e}"
            )
        if self.mean_pairwise_cosine > config.max_mean_pairwise_cosine:
            checks.append(
                f"mean_pairwise_cosine={self.mean_pairwise_cosine:.4f} "
                f"> max={config.max_mean_pairwise_cosine}"
            )
        if self.predictor_output_variance < config.min_predictor_output_variance:
            checks.append(
                f"predictor_output_variance={self.predictor_output_variance:.2e} "
                f"< min={config.min_predictor_output_variance:.2e}"
            )
        if self.encoder_grad_norm == 0.0:
            checks.append("encoder_grad_norm==0 (encoder not learning)")
        if self.predictor_grad_norm == 0.0:
            checks.append("predictor_grad_norm==0 (predictor not learning)")
        if not (self.loss < self.constant_baseline_loss):
            checks.append(
                f"loss={self.loss:.4f} !< constant_baseline={self.constant_baseline_loss:.4f}"
            )
        self.failed_checks = checks
        self.passed = not checks


def _mean_pairwise_cosine(features) -> float:
    """features: (B, T, D) or (N, D). Returns the mean off-diagonal cosine
    similarity within a random subsample of at most 256 vectors."""
    import torch
    x = features.reshape(-1, features.shape[-1])
    n = x.shape[0]
    if n > 256:
        # deterministic subsample (first 256 rows) — this is a diagnostic
        # only, exact indices do not matter
        x = x[:256]
    x = torch.nn.functional.normalize(x.float(), dim=-1)
    sim = x @ x.T
    n = sim.shape[0]
    if n < 2:
        return 0.0
    off = sim - torch.diag(torch.diag(sim))
    return float(off.sum() / (n * (n - 1)))


def _param_grad_norm(module) -> float:
    import torch
    tot = 0.0
    for p in module.parameters():
        if p.grad is not None:
            tot += float(p.grad.detach().norm(2).cpu()) ** 2
    return math.sqrt(tot)


def diagnose_forward_dynamics(*,
                              step: int,
                              online_features,
                              predictor_output,
                              constant_baseline_loss: float,
                              loss: float,
                              encoder,
                              predictor,
                              config: ForwardDynConfig) -> ForwardDynDiagnostics:
    per_dim_var = online_features.detach().float().var(dim=(0, 1))
    diag = ForwardDynDiagnostics(
        step=step,
        per_dim_variance_mean=float(per_dim_var.mean().cpu()),
        per_dim_variance_min=float(per_dim_var.min().cpu()),
        mean_pairwise_cosine=_mean_pairwise_cosine(online_features),
        predictor_output_variance=float(predictor_output.detach().float().var().cpu()),
        encoder_grad_norm=_param_grad_norm(encoder),
        predictor_grad_norm=_param_grad_norm(predictor),
        loss=float(loss),
        constant_baseline_loss=float(constant_baseline_loss),
    )
    diag.evaluate(config)
    return diag


def append_diagnostics_jsonl(diag: ForwardDynDiagnostics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(diag)
    row["passed"] = diag.passed
    row["failed_checks"] = diag.failed_checks
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


# ── constant-predictor baseline (utility) ─────────────────────────────────


def constant_predictor_baseline(target_features) -> float:
    """MSE achieved by always predicting the batch-mean of the target
    features. Any competent predictor must beat this by a non-trivial
    margin; if not, the encoder/predictor is not learning action-conditional
    structure.

    ``target_features``: (B, T, D)  detached.
    """
    import torch
    tgt_next = target_features[:, 1:]
    mean = tgt_next.mean(dim=(0, 1), keepdim=True)
    return float(((tgt_next - mean) ** 2).mean().cpu())
