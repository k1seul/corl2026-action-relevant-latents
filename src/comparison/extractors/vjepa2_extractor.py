"""V-JEPA 2 extractor (frozen ViT-L).

Wraps the official Meta V-JEPA 2 ViT-L checkpoint. Handles:
  * resolution mismatch — our 128×128 → V-JEPA2 native 256×256 (bilinear)
  * tensor layout    — our (B,T,C,H,W) → V-JEPA2 (B,C,T,H,W)
  * temporal stride — V-JEPA2 tubelet_size=2 collapses 2 frames into 1
                      tube; we duplicate each tube feature back to both
                      frames so output stays (B, T, D)
  * normalization    — [0,1] → ImageNet mean/std

External setup required:
  1. Repo cloned at ``$EXTERNAL_MODELS_DIR/vjepa2`` (default
     ``external_models/vjepa2``)
  2. Checkpoint downloaded:
        wget https://dl.fbaipublicfiles.com/vjepa2/vitl.pt \\
             -O $EXTERNAL_MODELS_DIR/vjepa2_ckpts/vitl.pt
  3. Optional: ``pip install timm einops`` (V-JEPA2 deps)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from comparison.extractors.base import ExtractorMeta, RepresentationExtractor

EXTERNAL_DIR = Path(os.environ.get(
    "EXTERNAL_MODELS_DIR",
    "external_models",
))
VJEPA2_REPO = EXTERNAL_DIR / "vjepa2"
VJEPA2_CKPT_DIR = EXTERNAL_DIR / "vjepa2_ckpts"


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _ensure_vjepa2_on_path() -> None:
    if not VJEPA2_REPO.exists():
        raise FileNotFoundError(
            f"V-JEPA 2 repo not found at {VJEPA2_REPO}. "
            "Clone with: git clone https://github.com/facebookresearch/vjepa2 "
            f"{VJEPA2_REPO}"
        )
    p = str(VJEPA2_REPO)
    if p not in sys.path:
        sys.path.insert(0, p)


class VJEPA2Extractor(RepresentationExtractor):
    def __init__(self, device: torch.device,
                 ckpt_path: Optional[str] = None,
                 model_arch: str = "vit_large",
                 num_frames: int = 32,
                 img_size: int = 256) -> None:
        _ensure_vjepa2_on_path()
        from src.models import vision_transformer as vit_encoder  # noqa: E402

        # ViT-L kwargs that match the official ckpt's training config
        vit_kwargs = dict(
            patch_size=16,
            img_size=(img_size, img_size),
            num_frames=num_frames,
            tubelet_size=2,
            use_sdpa=True,
            use_silu=False,
            wide_silu=True,
            uniform_power=False,
            use_rope=True,
        )
        if model_arch == "vit_large":
            arch_fn = vit_encoder.vit_large
        elif model_arch == "vit_huge":
            arch_fn = vit_encoder.vit_huge
        else:
            raise ValueError(f"unsupported V-JEPA2 arch: {model_arch}")
        encoder = arch_fn(**vit_kwargs)

        ckpt = ckpt_path or str(VJEPA2_CKPT_DIR / "vitl.pt")
        if not Path(ckpt).exists():
            raise FileNotFoundError(
                f"V-JEPA 2 ckpt not found at {ckpt}. Download with:\n"
                "  mkdir -p $EXTERNAL_MODELS_DIR/vjepa2_ckpts\n"
                "  wget https://dl.fbaipublicfiles.com/vjepa2/vitl.pt "
                "-O $EXTERNAL_MODELS_DIR/vjepa2_ckpts/vitl.pt"
            )
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        # ckpt layout: {"target_encoder": ..., "context_encoder": ..., "predictor": ...}
        # we use target_encoder (the EMA encoder)
        sd = state.get("target_encoder", state.get("encoder"))
        sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
        msg = encoder.load_state_dict(sd, strict=False)
        # report missing keys briefly to stderr
        if msg.missing_keys or msg.unexpected_keys:
            sys.stderr.write(
                f"[vjepa2] load_state_dict missing={len(msg.missing_keys)} "
                f"unexpected={len(msg.unexpected_keys)}\n")

        encoder = encoder.to(device).eval()
        for p in encoder.parameters():
            p.requires_grad = False

        self._encoder = encoder
        self._device = device
        self._img_size = img_size
        self._num_frames = num_frames
        self._tubelet = 2
        n_params = sum(p.numel() for p in encoder.parameters())
        self._meta = ExtractorMeta(
            family="vjepa2",
            variant=f"{model_arch}_{img_size}",
            feat_dim=int(encoder.embed_dim),
            n_params=n_params,
            native_resolution=img_size,
            native_clip_len=num_frames,
            has_native_recon=False,  # JEPA-style: no pixel recon
        )

    @property
    def meta(self) -> ExtractorMeta:
        return self._meta

    def _preprocess(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: (B, T, C, H, W), float in [0, 1], C=6 (2-cam stacked) or 3
        B, T, C, H, W = clip.shape
        # if 2-cam stacked, drop eye_in_hand for V-JEPA2 fairness
        # (V-JEPA2 was pretrained on natural single-view video)
        if C == 6:
            clip = clip[:, :, :3]  # keep agentview only
            C = 3
        # resize spatial to img_size
        clip = clip.reshape(B * T, C, H, W)
        clip = F.interpolate(clip, size=(self._img_size, self._img_size),
                             mode="bilinear", align_corners=False)
        # imagenet normalize
        mean = torch.tensor(_IMAGENET_MEAN, device=clip.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=clip.device).view(1, 3, 1, 1)
        clip = (clip - mean) / std
        clip = clip.reshape(B, T, C, self._img_size, self._img_size)
        # transpose to (B, C, T, H, W)
        return clip.permute(0, 2, 1, 3, 4).contiguous()

    @torch.no_grad()
    def extract(self, clip: torch.Tensor,
                task_caption: Optional[str] = None) -> torch.Tensor:
        clip = clip.to(self._device)
        B, T = clip.shape[:2]
        x = self._preprocess(clip)            # (B, C, T, H_v, W_v)
        feats = self._encoder(x)              # (B, num_patches, D)
        # num_patches = (T/tubelet) * H_p * W_p
        D = feats.shape[-1]
        H_p = self._img_size // 16
        T_tubes = T // self._tubelet
        feats = feats.reshape(B, T_tubes, H_p * H_p, D)
        feats = feats.mean(dim=2)             # (B, T_tubes, D)
        # repeat each tube along T to expand back to original T
        feats = feats.repeat_interleave(self._tubelet, dim=1)  # (B, T, D)
        return feats[:, :T].float()  # safety crop in case T odd
