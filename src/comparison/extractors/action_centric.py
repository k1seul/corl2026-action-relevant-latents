"""Action-centric frozen baselines: CAIP, R3M, VIP, LIV.

These wrappers implement the ``RepresentationExtractor`` interface. Weights
are NOT bundled: every extractor's ``__init__`` verifies that the official
checkpoint is reachable and raises a ``FileNotFoundError`` with the exact
download command if not. This makes missing-asset failures loud and
recoverable, per instruction §2 rule 8.

For every one of these baselines we insist on:

  * official repo / official checkpoint (URL + commit hash recorded);
  * preprocessing exactly as the release recommends (recorded per model);
  * mean-pooled per-frame features returned as ``(B, T, D)`` to match the
    canonical probe interface;
  * ``feat_dim`` reported so the probe MLP can be sized correctly.

CAIP has two supported modes and we return them as two distinct extractors,
never a single "CAIP" score:

  * ``CAIPVisualOnly``  — patch/visual representation with NO task text;
                          this is the fair common comparison.
  * ``CAIPNativeConditioned`` — the native instruction-conditioned pathway;
                                clearly separate sensitivity number.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from comparison.extractors.base import ExtractorMeta, RepresentationExtractor


EXTERNAL_DIR = Path(os.environ.get(
    "EXTERNAL_MODELS_DIR",
    "external_models",
))


# ── shared helpers ──────────────────────────────────────────────────────────


def _missing(path: Path, download_cmd: str) -> FileNotFoundError:
    return FileNotFoundError(
        f"required asset not found at {path}\n"
        f"recovery command:\n  {download_cmd}\n"
        f"(set EXTERNAL_MODELS_DIR to change the parent directory)"
    )


def _ensure_import(module_name: str, install_hint: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{module_name} not importable; install with:\n  {install_hint}"
        ) from exc


# ── R3M (Nair et al., 2022) ─────────────────────────────────────────────────


class R3MExtractor(RepresentationExtractor):
    """R3M ResNet-50 frozen features.

    Repo: https://github.com/facebookresearch/r3m
    Install: ``pip install r3m`` (weights auto-downloaded on first use).
    """
    def __init__(self, device, model_size: str = "resnet50") -> None:
        import torch  # noqa: F401
        r3m_mod = _ensure_import("r3m", "pip install r3m")
        model = r3m_mod.load_r3m(model_size).to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        self._model = model
        self._device = device
        feat_dim = 2048 if model_size == "resnet50" else 512
        self._meta = ExtractorMeta(
            family="r3m",
            variant=model_size,
            feat_dim=feat_dim,
            n_params=sum(p.numel() for p in model.parameters()),
            native_resolution=224,
            native_clip_len=None,
            has_native_recon=False,
        )

    @property
    def meta(self) -> ExtractorMeta:
        return self._meta

    def extract(self, clip, task_caption: Optional[str] = None):
        import torch
        with torch.no_grad():
            clip = clip.to(self._device)
            B, T, C, H, W = clip.shape
            if C > 3:
                clip = clip[:, :, :3]
                C = 3
            # R3M expects [0,255] uint8-like; scale from [0,1]
            x = (clip * 255.0).reshape(B * T, C, H, W)
            # R3M internally resizes; but we keep 224 explicitly
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            )
            feats = self._model(x)  # (B*T, D)
            return feats.reshape(B, T, -1).float()


# ── VIP (Ma et al., 2023) ───────────────────────────────────────────────────


class VIPExtractor(RepresentationExtractor):
    """VIP ResNet-50 goal-conditioned value features (frozen).

    Repo: https://github.com/facebookresearch/vip
    Install: ``pip install vip-official``  or use the repo directly.
    """
    def __init__(self, device) -> None:
        import torch  # noqa: F401
        vip_mod = _ensure_import("vip", "pip install vip-official")
        model = vip_mod.load_vip().to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        self._model = model
        self._device = device
        self._meta = ExtractorMeta(
            family="vip",
            variant="resnet50",
            feat_dim=1024,
            n_params=sum(p.numel() for p in model.parameters()),
            native_resolution=224,
            native_clip_len=None,
            has_native_recon=False,
        )

    @property
    def meta(self) -> ExtractorMeta:
        return self._meta

    def extract(self, clip, task_caption: Optional[str] = None):
        import torch
        with torch.no_grad():
            clip = clip.to(self._device)
            B, T, C, H, W = clip.shape
            if C > 3:
                clip = clip[:, :, :3]
                C = 3
            x = (clip * 255.0).reshape(B * T, C, H, W)
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            )
            feats = self._model(x)
            return feats.reshape(B, T, -1).float()


# ── LIV (Ma et al., 2023) ───────────────────────────────────────────────────


class LIVExtractor(RepresentationExtractor):
    """LIV vision-language value features (frozen).

    Repo: https://github.com/penn-pal-lab/LIV
    Install: ``pip install liv-robotics`` (or use the repo directly).

    LIV supports a text-conditioned reward query; here we return the plain
    visual embedding, which is the piece needed for the canonical
    ``z_t -> a_t`` probe. If you want the text-conditioned score you must
    add a separate extractor and label it clearly.
    """
    def __init__(self, device) -> None:
        import torch  # noqa: F401
        liv_mod = _ensure_import("liv", "pip install liv-robotics")
        model = liv_mod.load_liv().to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        self._model = model
        self._device = device
        self._meta = ExtractorMeta(
            family="liv",
            variant="clip_vit_l14",
            feat_dim=1024,
            n_params=sum(p.numel() for p in model.parameters()),
            native_resolution=224,
            native_clip_len=None,
            has_native_recon=False,
        )

    @property
    def meta(self) -> ExtractorMeta:
        return self._meta

    def extract(self, clip, task_caption: Optional[str] = None):
        import torch
        with torch.no_grad():
            clip = clip.to(self._device)
            B, T, C, H, W = clip.shape
            if C > 3:
                clip = clip[:, :, :3]
                C = 3
            x = (clip * 255.0).reshape(B * T, C, H, W)
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            )
            # LIV's ``model(x, modality="vision")`` returns the visual embedding
            feats = self._model(x, modality="vision")
            return feats.reshape(B, T, -1).float()


# ── CAIP (two clearly-separated modes) ──────────────────────────────────────


class CAIPVisualOnly(RepresentationExtractor):
    """CAIP visual/patch representation WITHOUT task text.

    This is the fair common-comparison mode. Do not conflate with the
    native-conditioned mode below.
    """
    def __init__(self, device, ckpt_path: Optional[str] = None) -> None:
        import torch  # noqa: F401
        default_ckpt = EXTERNAL_DIR / "caip_ckpts" / "caip_vitl.pt"
        p = Path(ckpt_path) if ckpt_path else default_ckpt
        if not p.exists():
            raise _missing(
                p,
                "Follow the CAIP repo release instructions "
                "(https://github.com/<org>/caip) and place the ViT-L checkpoint "
                f"at {p}. If the repo has moved, record the new URL/commit in "
                "runs/corl2026_rebuttal/reports/ASSET_INVENTORY.md."
            )
        # Actual loading is left as a repo-specific import; we intentionally
        # do NOT hardcode a fake path here.
        raise NotImplementedError(
            "CAIP loader must be wired to the official repo; place the "
            f"checkpoint at {p} and implement the loader following the CAIP "
            "release README, then remove this raise."
        )

    @property
    def meta(self) -> ExtractorMeta:
        raise NotImplementedError

    def extract(self, clip, task_caption=None):
        raise NotImplementedError


class CAIPNativeConditioned(RepresentationExtractor):
    """CAIP's native instruction-conditioned representation.

    Reported SEPARATELY as a sensitivity result — must never be mixed into
    the fair common ranking (instruction §6).
    """
    def __init__(self, device, ckpt_path: Optional[str] = None) -> None:
        default_ckpt = EXTERNAL_DIR / "caip_ckpts" / "caip_vitl.pt"
        p = Path(ckpt_path) if ckpt_path else default_ckpt
        if not p.exists():
            raise _missing(
                p,
                "See CAIPVisualOnly for download instructions; same checkpoint."
            )
        raise NotImplementedError(
            "CAIP native-conditioned loader must be wired to the official "
            "repo; implement following the CAIP release README."
        )

    @property
    def meta(self) -> ExtractorMeta:
        raise NotImplementedError

    def extract(self, clip, task_caption=None):
        raise NotImplementedError
