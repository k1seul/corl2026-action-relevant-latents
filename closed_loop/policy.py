"""pi0.5 (openpi PyTorch) policy construction and the token-coupling special token.

All closed-loop runs in the paper used openpi's fallback `Pi0Config` (gemma_2b VLM,
gemma_300m action expert, action horizon 16, 48 prompt tokens, action_dim 32), so the
config is built directly rather than through `openpi.training.config`.
"""
from __future__ import annotations

import logging
import os
import types
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

PI0_ACTION_DIM = 32          # pretrained pi0.5 action width; LIBERO's 7-D actions are zero-padded
LIBERO_ACTION_DIM = 7
MAX_TOKEN_LEN = 48


def make_pi0_config(action_horizon: int = 16, precision: str = "bfloat16"):
    import openpi.models.pi0_config as _pi0_cfg
    return _pi0_cfg.Pi0Config(
        dtype=precision,
        action_dim=PI0_ACTION_DIM,
        action_horizon=action_horizon,
        max_token_len=MAX_TOKEN_LEN,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=True,
    )


def build_pi0(pretrained: Optional[str], device, *, action_horizon: int = 16,
              precision: str = "bfloat16", gradient_checkpointing: bool = True):
    """Construct PI0Pytorch and load converted pi0.5 weights (`<dir>/model.safetensors`)."""
    os.environ.setdefault("PI0_VISION_ROPE", "0")
    import openpi.models_pytorch.pi0_pytorch as _pi0_torch

    cfg = make_pi0_config(action_horizon, precision)
    model = _pi0_torch.PI0Pytorch(cfg).to(device)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if pretrained:
        import safetensors.torch as _st
        path = Path(pretrained)
        weights = path / "model.safetensors" if path.is_dir() else path
        if not weights.exists():
            raise FileNotFoundError(f"pi0.5 weights not found: {weights} (see closed_loop/README.md)")
        missing, unexpected = _st.load_model(model, str(weights), strict=False)
        logging.info("pi0: loaded %s (missing=%d, unexpected=%d)", weights, len(missing or []), len(unexpected or []))
    return model, cfg


def get_tokenizer():
    try:
        from openpi.models_pytorch.preprocessing_pytorch import get_tokenizer as _get
        return _get()
    except Exception:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")


def _observation(base, wrist, state, language, tokenizer, device):
    from openpi.models.model import Observation
    B = base.shape[0]
    ones = torch.ones(B, dtype=torch.bool, device=device)
    zeros = torch.zeros(B, dtype=torch.bool, device=device)
    tok = tokenizer(language, padding="max_length", truncation=True, max_length=MAX_TOKEN_LEN,
                    return_tensors="pt")
    return Observation(
        # LIBERO has one wrist camera: the right-wrist slot is zeros with mask False
        images={"base_0_rgb": base, "left_wrist_0_rgb": wrist, "right_wrist_0_rgb": torch.zeros_like(base)},
        image_masks={"base_0_rgb": ones, "left_wrist_0_rgb": ones, "right_wrist_0_rgb": zeros},
        state=state,
        tokenized_prompt=tok["input_ids"].to(device),
        tokenized_prompt_mask=tok["attention_mask"].to(torch.bool).to(device),
    )


def make_train_observation(batch: Dict, *, image_size: int, tokenizer, device):
    """Batch from closed_loop.data -> (openpi Observation, actions padded to 32-D)."""
    def resize(x):
        if x.shape[-1] != image_size:
            x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
        return x.to(device, non_blocking=True)

    obs = _observation(resize(batch["image_main"]), resize(batch["image_wrist"]),
                       batch["state"].to(device, non_blocking=True).float(), batch["language"],
                       tokenizer, device)
    actions = batch["actions"].to(device).float()
    pad = torch.zeros(*actions.shape[:2], PI0_ACTION_DIM - actions.shape[-1], dtype=actions.dtype, device=device)
    return obs, torch.cat([actions, pad], dim=-1)


class PrefixCapture:
    """Keep the VLM prefix output of the last pi0 forward.

    PI0Pytorch calls `paligemma_with_expert.forward` directly (no module hooks fire),
    so the method is wrapped instead. During sampling the later calls carry no prefix
    and return None, which is ignored.
    """

    def __init__(self):
        self.prefix_out: Optional[torch.Tensor] = None

    def attach(self, pi0: nn.Module) -> None:
        target = pi0.paligemma_with_expert
        original = target.forward
        capture = self

        def _patched_forward(*args, **kwargs):
            output = original(*args, **kwargs)
            try:
                (prefix_out, _suffix_out), _kv = output
            except (TypeError, ValueError):
                prefix_out = output[0] if isinstance(output, (list, tuple)) else output
            if prefix_out is not None:
                capture.prefix_out = prefix_out
            return output

        target.forward = _patched_forward


# --------------------------------------------------------------------------------------
# Token coupling: one learned query token appended to the prefix
# --------------------------------------------------------------------------------------
def _embed_prefix_with_token(self, images, img_masks, lang_tokens, lang_masks):
    embs, pad_masks, att_masks = self._orig_embed_prefix(images, img_masks, lang_tokens, lang_masks)
    B = embs.shape[0]
    extra = self.future_query.expand(B, -1, -1).to(embs.dtype)                 # (B, 1, D)
    embs = torch.cat([embs, extra], dim=1)
    pad_masks = torch.cat([pad_masks, torch.ones(B, 1, dtype=pad_masks.dtype, device=pad_masks.device)], dim=1)
    # att_mask 0: the token is in the bidirectional prefix block (it attends to and is
    # attended by image/language tokens, and the action expert can attend to it)
    att_masks = torch.cat([att_masks, torch.zeros(B, 1, dtype=att_masks.dtype, device=att_masks.device)], dim=1)
    return embs, pad_masks, att_masks


def attach_special_token(pi0: nn.Module, hidden_dim: int, device) -> nn.Module:
    """Append one learned token to pi0's prefix, at training and at inference.

    Its output position (the last prefix position) is read by `SpecialTokenReadout`.
    Must be attached before loading a token-coupling checkpoint so that
    `future_query` exists as a parameter.
    """
    if hasattr(pi0, "_orig_embed_prefix"):
        raise RuntimeError("special token already attached")
    pi0._orig_embed_prefix = pi0.embed_prefix
    pi0.future_query = nn.Parameter(torch.zeros(1, 1, hidden_dim, device=device))
    nn.init.normal_(pi0.future_query, std=0.02)
    pi0.embed_prefix = types.MethodType(_embed_prefix_with_token, pi0)
    return pi0


class SpecialTokenReadout(nn.Module):
    """LayerNorm + linear map from the special token's output to the target space."""

    def __init__(self, in_dim: int, d_target: int, n_latents: int = 1, small_std: float = 0.01):
        super().__init__()
        self.n_latents, self.d_target = int(n_latents), int(d_target)
        self.norm = nn.LayerNorm(in_dim)
        self.out_proj = nn.Linear(in_dim, self.n_latents * self.d_target)
        # small random init keeps the cosine loss well defined from the first step
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=small_std)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, token_bd: torch.Tensor) -> torch.Tensor:
        x = self.out_proj(self.norm(token_bd.float()))
        return x.view(x.shape[0], self.n_latents, self.d_target)


def cosine_loss(pred_bld: torch.Tensor, target_bld: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of 1 - cos(pred, target), each sample flattened."""
    cos = F.cosine_similarity(pred_bld.float().flatten(1), target_bld.float().flatten(1), dim=-1)
    return (1.0 - cos).mean()
