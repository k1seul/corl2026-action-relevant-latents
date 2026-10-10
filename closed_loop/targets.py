"""Frozen target encoders and the clip-coupling auxiliary.

Clip coupling (`WorldModelClipAux`): the policy's main-camera prefix tokens are
mean-pooled and mapped by an MLP to the encoding of the two-camera clip o_t .. o_{t+16}
by the frozen DIFF + ID world model (last frame's tokens, mean-pooled).

Token coupling targets (`VJepa2TargetEncoder`, `WebDINOTargetEncoder`): the mean-pooled
feature of the main-camera frame o_{t+16}.

All encoders are frozen and used only to compute training targets.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.block_causal_transformer import BlockCausalVideoTransformer

DIFF_ID_REPO = "Dreamer-VLA/dreamer-vla-libero-aux-lam0p05"          # DIFF + ID (Table 1)
VJEPA2_ID_REPO = "Dreamer-VLA/dreamer-vla-best-libero-vjepa2-aux"    # V-JEPA 2 ViT-L + ID (Table 1)
VJEPA2_ARCH_REPO = "facebook/vjepa2-vitl-fpc64-256"
WEBDINO_REPO = "facebook/webssl-dino300m-full2b-224"                 # public frozen Web-DINO ViT-L

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _hf_ckpt(repo_or_path: str, filename: str = "ckpt_last.pt") -> Path:
    p = Path(repo_or_path)
    if p.is_file():
        return p
    if p.is_dir():
        return p / filename
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_or_path, filename))


# --------------------------------------------------------------------------------------
# DIFF + ID world model (block-causal video transformer, 2 cameras as 6 channels)
# --------------------------------------------------------------------------------------
class DreamerVLATargetEncoder(nn.Module):
    """Frozen DIFF + ID trunk. `random_init=True` keeps the architecture but skips the
    trained weights (the random-init control)."""

    def __init__(self, hf_repo: str = DIFF_ID_REPO, pool: str = "mean", diff_t: int = 0,
                 random_init: bool = False, device="cuda"):
        super().__init__()
        if pool not in ("mean", "spatial"):
            raise ValueError(f"pool must be 'mean'|'spatial', got {pool!r}")
        self.pool, self.diff_t = pool, int(diff_t)
        ckpt = torch.load(str(_hf_ckpt(hf_repo)), map_location="cpu", weights_only=False)
        m_cfg, d_cfg = ckpt["config"]["model"], ckpt["config"]["data"]
        is_diff = ckpt["config"].get("train", {}).get("objective", "diffusion") == "diffusion"
        in_channels = int(ckpt["model"]["patch_embed.proj.weight"].shape[1])     # 6 = 2 cameras
        backbone = BlockCausalVideoTransformer(
            in_channels=in_channels, patch_size=int(m_cfg["patch_size"]), num_frames=int(d_cfg["clip_len"]),
            image_size=int(d_cfg["resolution"]), dim=int(m_cfg["dim"]), depth=int(m_cfg["depth"]),
            num_heads=int(m_cfg["num_heads"]), mlp_ratio=float(m_cfg["mlp_ratio"]), dropout=0.0,
            enable_timestep=bool(is_diff), action_dim=0,
        )
        if random_init:
            logging.warning("DreamerVLATargetEncoder: random-init control, trained weights not loaded")
        else:
            missing, unexpected = backbone.load_state_dict(ckpt["model"], strict=False)
            if missing or unexpected:
                logging.warning("DreamerVLATargetEncoder: %d missing, %d unexpected keys", len(missing), len(unexpected))
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = backbone.to(device)
        self.image_size = int(d_cfg["resolution"])
        self.patch_size = int(m_cfg["patch_size"])
        self.n_patches = (self.image_size // self.patch_size) ** 2
        self.n_latents = 1 if pool == "mean" else self.n_patches
        self.d_bottleneck = int(m_cfg["dim"])

    @torch.no_grad()
    def encode_clip(self, main_clip, wrist_clip=None, reduce: str = "last") -> torch.Tensor:
        """(B, T, 3, H, W) clips in [0, 1] -> (B, L, D). `reduce='last'` takes the last
        frame's tokens, which attend to every earlier frame of the clip."""
        if reduce not in ("last", "mean"):
            raise ValueError(f"reduce must be 'last'|'mean', got {reduce!r}")
        B, T = main_clip.shape[:2]
        if wrist_clip is None:
            wrist_clip = torch.zeros_like(main_clip)
        clip = torch.cat([main_clip, wrist_clip], dim=2)                         # (B, T, 6, H, W)
        if T > self.backbone.num_frames:
            clip = clip[:, -self.backbone.num_frames:]
        t = (torch.full((B,), self.diff_t, dtype=torch.long, device=main_clip.device)
             if self.backbone.enable_timestep else None)
        feats = self.backbone.forward_features(clip, mask_indicator=None, t=t)    # (B, T, N, D)
        feats = feats[:, -1] if reduce == "last" else feats.mean(dim=1)
        if self.pool == "mean":
            feats = feats.mean(dim=1, keepdim=True)
        return feats.float()


def _l2norm_cost(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Per-token squared error divided by the batch's mean squared target."""
    p, t = pred.float(), tgt.float()
    return (p - t).pow(2).mean(-1) / t.pow(2).mean().clamp_min(1e-8)


class WorldModelClipAux(nn.Module):
    """Clip coupling: MLP(mean-pooled main-camera prefix tokens) -> DIFF + ID clip encoding.

    shuffle_target: batch-shuffled control (targets permuted across the mini-batch).
    random_init:    random-init control (frozen randomly initialized DIFF trunk).
    """

    def __init__(self, pi0_prefix_dim: int, hf_repo: str = DIFF_ID_REPO, reduce: str = "last",
                 head_hidden: int = 2048, shuffle_target: bool = False, random_init: bool = False,
                 diff_t: int = 0, device="cuda"):
        super().__init__()
        self.reduce = reduce
        self.shuffle_target = bool(shuffle_target)
        self.encoder = DreamerVLATargetEncoder(hf_repo=hf_repo, pool="mean", diff_t=diff_t,
                                               random_init=random_init, device=device)
        self.image_size = self.encoder.image_size                                   # 128
        self.head = nn.Sequential(
            nn.Linear(pi0_prefix_dim, head_hidden), nn.GELU(), nn.Linear(head_hidden, self.encoder.d_bottleneck),
        ).to(device)
        self.n_main_patches: Optional[int] = None    # set by the trainer: main-camera tokens lead the prefix

    def _resize(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.shape[-1] == self.image_size and clip.shape[-2] == self.image_size:
            return clip
        B, T = clip.shape[:2]
        x = F.interpolate(clip.reshape(B * T, *clip.shape[2:]), size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False)
        return x.reshape(B, T, *x.shape[1:])

    def loss(self, prefix_tokens: torch.Tensor, main_clip: torch.Tensor, wrist_clip: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            tgt = self.encoder.encode_clip(self._resize(main_clip), self._resize(wrist_clip), reduce=self.reduce)
        if self.shuffle_target:
            tgt = tgt[torch.randperm(tgt.shape[0], device=tgt.device)].detach()
        pred = self.head(prefix_tokens.mean(dim=1, keepdim=True))                  # (B, 1, D)
        return _l2norm_cost(pred, tgt).mean()


# --------------------------------------------------------------------------------------
# Token-coupling targets
# --------------------------------------------------------------------------------------
def _remap_vjepa2(enc_sd: dict) -> dict:
    """Packed-QKV checkpoint keys -> HF VJEPA2Model encoder keys."""
    out = {}
    for k, v in enc_sd.items():
        if k.startswith("patch_embed.proj."):
            out["encoder.embeddings.patch_embeddings.proj." + k[len("patch_embed.proj."):]] = v
        elif k in ("norm.weight", "norm.bias"):
            out["encoder.layernorm." + k.split(".")[1]] = v
        elif k.startswith("blocks."):
            _, i, rest = k.split(".", 2)
            if rest.startswith("attn.qkv."):
                suffix = rest[len("attn.qkv."):]
                q, kk, vv = torch.chunk(v, 3, dim=0)
                out[f"encoder.layer.{i}.attention.query.{suffix}"] = q
                out[f"encoder.layer.{i}.attention.key.{suffix}"] = kk
                out[f"encoder.layer.{i}.attention.value.{suffix}"] = vv
            elif rest.startswith("attn.proj."):
                out[f"encoder.layer.{i}.attention.proj.{rest[len('attn.proj.'):]}"] = v
            else:
                out[f"encoder.layer.{i}.{rest}"] = v
        else:
            out[k] = v
    return out


class VJepa2TargetEncoder(nn.Module):
    """Frozen V-JEPA 2 ViT-L + ID encoder (single frame repeated to one tubelet)."""

    def __init__(self, hf_repo: str = VJEPA2_ID_REPO, arch_repo: str = VJEPA2_ARCH_REPO,
                 pool: str = "mean", device="cuda"):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        self.pool = pool
        cfg = AutoConfig.from_pretrained(arch_repo)
        self.image_size = int(cfg.image_size)
        full = AutoModel.from_config(cfg)
        ckpt = torch.load(str(_hf_ckpt(hf_repo)), map_location="cpu", weights_only=False)
        missing, unexpected = full.load_state_dict(_remap_vjepa2(ckpt["encoder"]), strict=False)
        enc_missing = [k for k in missing if k.startswith("encoder.")]
        if enc_missing:
            raise RuntimeError(f"V-JEPA 2 encoder keys missing: {enc_missing[:5]}")
        enc = full.encoder.eval()
        for p in enc.parameters():
            p.requires_grad = False
        self.encoder_module = enc.to(device)
        del full
        self.register_buffer("img_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1).to(device))
        self.register_buffer("img_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1).to(device))
        self.tubelet_size = int(cfg.tubelet_size)
        self.n_latents, self.d_bottleneck = 1, int(cfg.hidden_size)

    @torch.no_grad()
    def encode(self, image_bchw: torch.Tensor) -> torch.Tensor:
        x = image_bchw
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = (x - self.img_mean) / self.img_std
        clip = x.unsqueeze(1).expand(x.shape[0], self.tubelet_size, -1, -1, -1).contiguous()
        feats = self.encoder_module(pixel_values_videos=clip).last_hidden_state
        return feats.mean(dim=1, keepdim=True).float()


class WebDINOTargetEncoder(nn.Module):
    """Frozen public Web-DINO ViT-L (no ID fine-tuning)."""

    def __init__(self, hf_repo: str = WEBDINO_REPO, image_size: int = 224, device="cuda"):
        super().__init__()
        from transformers import AutoModel
        self.image_size = int(image_size)
        try:
            m = AutoModel.from_pretrained(hf_repo, dtype=torch.float32)
        except TypeError:
            m = AutoModel.from_pretrained(hf_repo, torch_dtype=torch.float32)
        m = m.to(device).eval()
        for p in m.parameters():
            p.requires_grad = False
        self.model = m
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1))
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=device)
            hs = m(pixel_values=(dummy - self.mean) / self.std).last_hidden_state
        self.n_latents, self.d_bottleneck = 1, int(hs.shape[-1])

    @torch.no_grad()
    def encode(self, image_bchw: torch.Tensor) -> torch.Tensor:
        x = image_bchw
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, (self.image_size,) * 2, mode="bilinear", align_corners=False)
        x = ((x.to(self.mean.dtype) - self.mean) / self.std).to(next(self.model.parameters()).dtype)
        return self.model(pixel_values=x).last_hidden_state.mean(dim=1, keepdim=True).float()
