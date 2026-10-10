"""Dreamer 4 (nicklashansen/dreamer4) tokenizer + dynamics extractor.

Two registerable flavours (per spec §4):
  - ``tokenizer``: feature = encoder output ``z`` (mean-pooled or flattened
                    over the L=16 latents). This is the "frozen V-JEPA"
                    analogue for Dreamer 4 — pure visual representation.
  - ``dynamics``:  feature = dynamics-model hidden state after a
                    zero-noise forward pass on the tokenizer latents.
                    Closer to the actual "world model representation"
                    that the agent would use.

We do NOT vendor nicklashansen's code into this repo; we expect the repo to
be cloned to ``external_models/dreamer4/`` and add it to sys.path here.

This file is intentionally a *stub* — the tokenizer/dynamics ckpts on
LIBERO have not yet been trained (Phase 2/3). The `.extract()` path will
fail loudly with FileNotFoundError until the user trains them via
``scripts/train_dreamer4_*_libero.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

from comparison.extractors.base import ExtractorMeta, RepresentationExtractor


# Path to the cloned dreamer4 repo. Adjust if you moved it.
DREAMER4_REPO = Path(
    "external_models/dreamer4"
)


def _add_dreamer4_to_path() -> None:
    pkg = DREAMER4_REPO / "dreamer4"
    if not pkg.exists():
        raise FileNotFoundError(
            f"dreamer4 repo not found at {DREAMER4_REPO}. "
            f"Clone via `git clone https://github.com/nicklashansen/dreamer4 {DREAMER4_REPO}`."
        )
    p = str(pkg)
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_tokenizer(ckpt_path: Path, device: torch.device,
                     img_size: int = 128, patch: int = 4, in_channels: int = 3):
    """Reconstruct nicklashansen's Tokenizer from its ckpt state dict.

    nicklashansen saves: {"args": <Namespace>, "model": state_dict, "step": int}
    We rebuild the Encoder + Decoder from `args` and load.
    """
    _add_dreamer4_to_path()
    from model import Encoder, Decoder, Tokenizer  # type: ignore

    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    args = state["args"] if "args" in state else None
    if args is None:
        # Fallback to default nicklashansen hyperparams (90k tokenizer config).
        # If the user trained with non-defaults, the ckpt will refuse to load
        # cleanly — that's the loud-failure signal we want.
        d_model = 256; n_heads = 4; depth = 8
        n_latents = 16; d_bottleneck = 32
        dropout = 0.05; mlp_ratio = 4.0; time_every = 1
        mae_p_min, mae_p_max = 0.0, 0.9
        scale_pos_embeds = True
    else:
        d_model = getattr(args, "d_model", 256)
        n_heads = getattr(args, "n_heads", 4)
        depth = getattr(args, "depth", 8)
        n_latents = getattr(args, "n_latents", 16)
        d_bottleneck = getattr(args, "d_bottleneck", 32)
        dropout = getattr(args, "dropout", 0.05)
        mlp_ratio = getattr(args, "mlp_ratio", 4.0)
        time_every = getattr(args, "time_every", 1)
        mae_p_min = getattr(args, "mae_p_min", 0.0)
        mae_p_max = getattr(args, "mae_p_max", 0.9)
        scale_pos_embeds = getattr(args, "scale_pos_embeds", True)

    n_patches = (img_size // patch) ** 2
    patch_dim = in_channels * patch * patch

    enc = Encoder(
        patch_dim=patch_dim, d_model=d_model, n_latents=n_latents,
        n_patches=n_patches, n_heads=n_heads, depth=depth,
        d_bottleneck=d_bottleneck, dropout=dropout, mlp_ratio=mlp_ratio,
        time_every=time_every, latents_only_time=True,
        mae_p_min=mae_p_min, mae_p_max=mae_p_max,
        scale_pos_embeds=scale_pos_embeds,
    )
    dec = Decoder(
        d_bottleneck=d_bottleneck, d_model=d_model, n_heads=n_heads,
        depth=depth, n_latents=n_latents, n_patches=n_patches,
        d_patch=patch_dim, dropout=dropout, mlp_ratio=mlp_ratio,
        time_every=time_every, latents_only_time=True,
        scale_pos_embeds=scale_pos_embeds,
    )
    tok = Tokenizer(enc, dec).to(device)
    # state_dict keys may be `model.encoder.*` (DDP) or `encoder.*`
    sd = state.get("model", state)
    cleaned = {k.removeprefix("module."): v for k, v in sd.items()}
    tok.load_state_dict(cleaned, strict=False)
    tok.eval()
    return tok, {
        "d_model": d_model, "n_latents": n_latents, "d_bottleneck": d_bottleneck,
        "n_patches": n_patches, "patch": patch, "img_size": img_size,
        "in_channels": in_channels,
    }


def _load_dynamics(ckpt_path, device: torch.device, *, tokenizer_meta: dict):
    """Reconstruct nicklashansen's Dynamics module from its train_dynamics
    ckpt. Saved format: {step, epoch, dynamics: <state_dict>, args, ...}.
    Falls back to defaults if `args` is missing."""
    _add_dreamer4_to_path()
    from model import Dynamics  # type: ignore

    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    args = state.get("args", {}) or {}
    if not isinstance(args, dict):
        args = vars(args) if hasattr(args, "__dict__") else {}

    d_model = int(args.get("d_model_dyn", 512))
    dyn_depth = int(args.get("dyn_depth", 8))
    n_heads = int(args.get("n_heads", 4))
    dropout = float(args.get("dropout", 0.0))
    mlp_ratio = float(args.get("mlp_ratio", 4.0))
    time_every = int(args.get("time_every", 1))
    k_max = int(args.get("k_max", 8))
    n_register = int(args.get("n_register", 4))
    n_agent = int(args.get("n_agent", 1))
    space_mode = args.get("space_mode", "wm_agent_isolated")
    scale_pos_embeds = bool(args.get("scale_pos_embeds", True))
    packing_factor = int(args.get("packing_factor", 2))

    d_bottleneck = int(tokenizer_meta["d_bottleneck"])
    n_latents = int(tokenizer_meta["n_latents"])
    n_spatial = n_latents // packing_factor
    d_spatial = d_bottleneck * packing_factor

    dyn = Dynamics(
        d_model=d_model, d_bottleneck=d_bottleneck, d_spatial=d_spatial,
        n_spatial=n_spatial, n_register=n_register, n_agent=n_agent,
        n_heads=n_heads, depth=dyn_depth, k_max=k_max, dropout=dropout,
        mlp_ratio=mlp_ratio, time_every=time_every, space_mode=space_mode,
        scale_pos_embeds=scale_pos_embeds,
    ).to(device)

    sd = state.get("dynamics", state.get("model", state))
    cleaned = {k.removeprefix("module."): v for k, v in sd.items()}
    dyn.load_state_dict(cleaned, strict=True)
    dyn.eval()
    return dyn, {
        "d_model": d_model, "n_spatial": n_spatial, "d_spatial": d_spatial,
        "packing_factor": packing_factor, "k_max": k_max,
        "n_register": n_register, "n_agent": n_agent,
    }


class Dreamer4Extractor(RepresentationExtractor):
    """Dreamer 4 representation extractor.

    Modes (``mode`` kw):
      - ``tokenizer``  : encoder output ``z`` → either flattened (L*D_b) or
                          mean-pooled across latents (D_b). Default: flatten.
      - ``dynamics``   : run tokenizer encoder, feed z through dynamics
                          model with shortcut_t=0 (clean / no denoising),
                          read the last hidden state. NOT YET IMPLEMENTED —
                          waiting on Phase 3 trained ckpt + dynamics module
                          probe interface; falls back to ``tokenizer``
                          when dynamics ckpt is absent.
    """
    def __init__(
        self,
        tokenizer_ckpt: str,
        device: torch.device,
        *,
        mode: str = "tokenizer",
        pool: str = "flatten",         # "flatten" or "mean"
        feature_source: str = "spatial",   # "spatial" (denoiser hidden) or
                                            # "agent" (Dreamer agent tokens h_t)
        dynamics_ckpt: Optional[str] = None,
        variant_name: Optional[str] = None,
        in_channels: int = 3,
        img_size: int = 128,
        patch: int = 4,
    ) -> None:
        if mode not in ("tokenizer", "dynamics"):
            raise ValueError(f"mode must be 'tokenizer' or 'dynamics', got {mode!r}")
        if pool not in ("flatten", "mean"):
            raise ValueError(f"pool must be 'flatten' or 'mean', got {pool!r}")
        if feature_source not in ("spatial", "agent"):
            raise ValueError(f"feature_source must be 'spatial' or 'agent', got {feature_source!r}")
        self._device = device
        self._mode = mode
        self._pool = pool
        self._feature_source = feature_source

        tok_path = Path(tokenizer_ckpt)
        if not tok_path.exists():
            raise FileNotFoundError(
                f"tokenizer ckpt missing at {tok_path} — train via "
                f"`scripts/train_dreamer4_tokenizer_libero.py` first."
            )
        self._tokenizer, self._tok_meta = _load_tokenizer(
            tok_path, device, img_size=img_size, patch=patch, in_channels=in_channels,
        )
        self._patch = patch
        self._in_channels = in_channels

        # Optional dynamics ckpt (Phase 3 output from nicklashansen/dreamer4)
        self._dynamics = None
        self._dyn_meta = None
        if mode == "dynamics":
            if dynamics_ckpt is None or not Path(dynamics_ckpt).exists():
                print(f"[Dreamer4Extractor] mode=dynamics requested but no "
                      f"dynamics_ckpt — falling back to mode=tokenizer.")
                self._mode = "tokenizer"
            else:
                self._dynamics, self._dyn_meta = _load_dynamics(
                    dynamics_ckpt, device,
                    tokenizer_meta=self._tok_meta,
                )

        if self._mode == "dynamics":
            d_model = self._dyn_meta["d_model"]
            n_spatial = self._dyn_meta["n_spatial"]
            n_agent = self._dyn_meta["n_agent"]
            if self._feature_source == "agent":
                # agent tokens: (B, T, n_agent, d_model)
                feat_dim = n_agent * d_model if self._pool == "flatten" else d_model
            else:
                # spatial hidden: (B, T, n_spatial, d_model)
                feat_dim = n_spatial * d_model if self._pool == "flatten" else d_model
        elif self._pool == "flatten":
            feat_dim = self._tok_meta["n_latents"] * self._tok_meta["d_bottleneck"]
        else:
            feat_dim = self._tok_meta["d_bottleneck"]
        n_params = sum(p.numel() for p in self._tokenizer.parameters())
        if self._dynamics is not None:
            n_params += sum(p.numel() for p in self._dynamics.parameters())
        self._meta = ExtractorMeta(
            family="dreamer4",
            variant=variant_name or f"dreamer4_{self._mode}_{self._pool}",
            feat_dim=int(feat_dim),
            n_params=int(n_params),
            native_resolution=int(img_size),
            native_clip_len=32,        # nicklashansen default seq_len=8 for tokenizer,
                                       # but we evaluate on our 32-frame clips.
            has_native_recon=True,
        )

    @property
    def meta(self) -> ExtractorMeta:
        return self._meta

    @torch.no_grad()
    def extract(self, clip: torch.Tensor,
                task_caption: Optional[str] = None) -> torch.Tensor:
        """clip: (B, T, C, H, W) float in [0, 1] → (B, T, D) feature."""
        from model import temporal_patchify, pack_bottleneck_to_spatial  # type: ignore
        clip = clip.to(self._device)
        patches = temporal_patchify(clip, self._patch)              # (B,T,Np,Dp)
        z, _ = self._tokenizer.encoder(patches)                    # (B,T,L,D_b)

        if self._mode == "tokenizer":
            if self._pool == "flatten":
                B, T, L, D = z.shape
                return z.reshape(B, T, L * D).float()              # (B, T, L*D_b)
            return z.mean(dim=2).float()                            # (B, T, D_b)

        # dynamics mode: run frozen dynamics on clean z, take spatial hidden
        B, T = z.shape[:2]
        z_packed = pack_bottleneck_to_spatial(
            z, n_spatial=self._dyn_meta["n_spatial"],
            k=self._dyn_meta["packing_factor"],
        )                                                            # (B,T,Sz,Dz)
        k_max = self._dyn_meta["k_max"]
        # clean input: sigma=0 → signal_idx=0; pick step_idx=log2(k_max)
        # ("end of denoising trajectory") to indicate clean-input forward.
        import math
        emax = int(round(math.log2(k_max)))
        step_idxs = torch.full((B, T), emax, device=self._device, dtype=torch.long)
        signal_idxs = torch.zeros((B, T), device=self._device, dtype=torch.long)

        # actions=None → ActionEncoder emits a learned base token (no info leak)
        _, h_t, spatial_hidden = self._dynamics(
            None, step_idxs, signal_idxs, z_packed,
            act_mask=None, agent_tokens=None,
            return_spatial_hidden=True,
        )
        if self._feature_source == "agent":
            # h_t: (B, T, n_agent, d_model) — Dreamer-style agent representation
            feats = h_t
        else:
            feats = spatial_hidden                                   # (B,T,n_spatial,d_model)

        if self._pool == "flatten":
            B, T, S, D = feats.shape
            return feats.reshape(B, T, S * D).float()
        return feats.mean(dim=2).float()                              # (B, T, d_model)

    @torch.no_grad()
    def native_recon_loss(self, clip: torch.Tensor) -> Optional[float]:
        """Tokenizer reconstruction MSE for sanity check."""
        from model import temporal_patchify, temporal_unpatchify  # type: ignore
        clip = clip.to(self._device)
        patches = temporal_patchify(clip, self._patch)
        z, _ = self._tokenizer.encoder(patches)
        pred = self._tokenizer.decoder(z)                          # (B,T,Np,Dp)
        recon = temporal_unpatchify(
            pred, H=self._tok_meta["img_size"], W=self._tok_meta["img_size"],
            C=self._in_channels, patch=self._patch,
        )
        return float((recon - clip).pow(2).mean().item())

    @torch.no_grad()
    def recon_rollout(self, prefix_clip: torch.Tensor, num_post: int,
                       actions: Optional[torch.Tensor] = None,
                       sched_schedule: str = "finest",
                       ) -> torch.Tensor:
        """Pixel rollout: prefix_clip → predicted clip for `num_post` frames.

        Mirrors nicklashansen `run_dynamics_eval`:
          1. encode prefix via frozen tokenizer
          2. pack to spatial tokens
          3. autoregressive shortcut-forcing rollout (sample one frame at a time)
          4. decode all latents back to pixels via frozen tokenizer decoder

        Returns (B, prefix_len + num_post, C, H, W) full clip in [0, 1].
        Requires dynamics_ckpt (raises if not loaded).
        """
        if self._dynamics is None:
            raise RuntimeError("recon_rollout requires dynamics_ckpt; pass it "
                                "into the Dreamer4Extractor constructor.")
        from model import (  # type: ignore
            temporal_patchify, pack_bottleneck_to_spatial,
        )
        import importlib.util
        # load nicklashansen's rollout helpers from their train_dynamics.py
        train_dyn = self._train_dynamics_module()

        prefix_clip = prefix_clip.to(self._device)
        B, L_pre, C, H, W = prefix_clip.shape
        T_full = L_pre + num_post

        # 1. encode prefix to latents
        patches = temporal_patchify(prefix_clip, self._patch)         # (B,L_pre,Np,Dp)
        z_pre, _ = self._tokenizer.encoder(patches)                  # (B,L_pre,L,D_b)
        z_pre_packed = pack_bottleneck_to_spatial(
            z_pre, n_spatial=self._dyn_meta["n_spatial"],
            k=self._dyn_meta["packing_factor"],
        )  # (B,L_pre,n_spatial,d_spatial)

        # 2. autoregressive rollout for `num_post` frames
        sched = train_dyn.make_tau_schedule(
            k_max=self._dyn_meta["k_max"], schedule=sched_schedule,
        )
        # nicklashansen's helper expects (B,T,n_spatial,d_spatial) ground truth
        # for ctx; we don't have post-frame GT here so we synthesize a tensor
        # of the right shape with prefix in [:L_pre] and zeros elsewhere
        # (only the prefix slice is used as conditioning).
        z_dummy = torch.cat([
            z_pre_packed,
            torch.zeros((B, num_post, *z_pre_packed.shape[2:]),
                         device=self._device, dtype=z_pre_packed.dtype),
        ], dim=1)
        # Actions argument: if model is action-cond and actions provided, use
        # them; otherwise pass None and ActionEncoder emits its base token.
        roll_actions = None
        if actions is not None:
            roll_actions = actions.to(self._device)
            if roll_actions.shape[-1] < 16:
                pad = torch.zeros(B, T_full, 16 - roll_actions.shape[-1],
                                    device=self._device, dtype=roll_actions.dtype)
                roll_actions = torch.cat([roll_actions, pad], dim=-1)

        z_rollout = train_dyn.sample_autoregressive_packed_sequence(
            self._dynamics,
            z_gt_packed=z_dummy,
            ctx_length=L_pre,
            horizon=num_post,
            k_max=self._dyn_meta["k_max"],
            sched=sched,
            actions=roll_actions,
            act_mask=None,
        )  # (B, L_pre+num_post, n_spatial, d_spatial)

        # 3. decode all latents back to pixels
        pred_clip = train_dyn.decode_packed_to_frames(
            self._tokenizer.decoder,
            z_packed=z_rollout,
            H=H, W=W, C=C, patch=self._patch,
            packing_factor=self._dyn_meta["packing_factor"],
        )  # (B, L_pre+num_post, C, H, W) in [0, 1]

        return pred_clip

    def _train_dynamics_module(self):
        """Lazy-import + cache nicklashansen's train_dynamics module so we can
        reuse its rollout / decode helpers without copy-pasting."""
        if hasattr(self, "_train_dyn_mod"):
            return self._train_dyn_mod
        _add_dreamer4_to_path()
        import importlib.util
        path = DREAMER4_REPO / "dreamer4" / "train_dynamics.py"
        spec = importlib.util.spec_from_file_location("_d4_train_dyn", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._train_dyn_mod = mod
        return mod
