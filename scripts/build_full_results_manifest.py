"""Build runs/full_results/manifest.yaml for the Full Results table.

Catalogs every Stage A sweep ckpt + Stage B LIBERO comparison ckpts +
frozen baselines (V-JEPA 2 / V-JEPA 2.1 / LAPA, no ckpt).

For each row we record:
  base_model: dreamer_v4 | vjepa2 | vjepa2_1 | lapa
  variant: short stable name (used as row label)
  family: diff | token | vjepa2 | vjepa2_1 | lapa
  seq_len: training clip length (int, from config)
  model_size: human label like "91M, large"
  n_params_million: float
  pretrained: "scratch" | "internet video (V-JEPA 2)" | "Open X (LAPA)" |
              "LIBERO" (continued SSL) | ...
  ckpt: relative path to ckpt_last.pt (or null for frozen baselines)
  config_path: relative path to source config.yaml (or null)
  notes: optional human note
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml
from omegaconf import OmegaConf


# Map config['model']['dim'] -> human size label
def _size_label(dim: int) -> str:
    return {
        256: "small",
        384: "medium-small",
        512: "large",
        768: "xl",
        1024: "xxl",
    }.get(int(dim), f"dim{dim}")


def _params_million_from_dim_depth(dim: int, depth: int, patch_size: int = 16,
                                     img_size: int = 128, num_frames: int = 32) -> float:
    # rough proxy: 12 * d^2 per block + attn/ffn coeffs. We compute exact
    # values by instantiating the model when possible; falls back to this.
    return 12.0 * (dim ** 2) * depth / 1e6


def _exact_param_count(ckpt_path: Path) -> float | None:
    """Try to load ckpt + count params via the relevant model constructor.
    Returns params in millions, or None if it can't be derived."""
    try:
        import torch  # local import to keep manifest gen lightweight
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if "model" not in state:
            return None
        msd = state["model"]
        n = sum(int(t.numel()) for t in msd.values() if hasattr(t, "numel"))
        return float(n / 1e6)
    except Exception:
        return None


def _stage_b_training_args(ckpt_path: Path) -> dict:
    """Extract batch_size / max_steps / lr / actual step from a Stage B ckpt.
    Returns empty dict if anything's missing."""
    try:
        import torch
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        a = state.get("args", {}) or {}
        if not isinstance(a, dict):
            a = vars(a) if hasattr(a, "__dict__") else {}
        return {
            "batch_size": a.get("batch_size"),
            "max_steps": a.get("max_steps"),
            "lr": a.get("lr"),
            "training_step_at_ckpt": int(state.get("step", -1)) if state.get("step") is not None else None,
        }
    except Exception:
        return {}


def _stage_a_entry(ckpt_path: Path) -> dict | None:
    """Read a Stage A config + ckpt → manifest entry."""
    cfg_path = ckpt_path.parent / "config.yaml"
    if not cfg_path.exists():
        return None
    cfg = OmegaConf.load(cfg_path)

    obj = cfg.train.get("objective", "mae")
    if obj == "token_ce":
        family = "token"
        pretrained = "scratch (LIBERO tokenizer reused)"
        base_model = "TOKEN"
    elif obj == "diffusion":
        family = "diff"
        pretrained = "scratch"
        base_model = "DIFF"
    elif obj == "flow_matching":
        family = "diff"
        pretrained = "scratch"
        base_model = "FM"
    elif obj == "mae":
        family = "diff"
        pretrained = "scratch"
        base_model = "MAE"
    else:
        return None

    dim = int(cfg.model.dim)
    depth = int(cfg.model.depth)
    seq_len = int(cfg.data.clip_len)
    aux_lambda = cfg.train.get("aux_invdyn_lambda", None)
    n_cams = len(cfg.data.cameras)
    action_dim = int(cfg.model.get("action_dim", 0))

    variant = ckpt_path.parent.name

    # human size label
    model_size = f"{_size_label(dim)}"
    pm = _exact_param_count(ckpt_path) or _params_million_from_dim_depth(dim, depth)
    notes_bits = []
    if obj == "flow_matching":
        notes_bits.append("FM")
    if obj == "mae":
        notes_bits.append("MAE")
    if aux_lambda is not None and float(aux_lambda) > 0:
        notes_bits.append(f"aux λ={aux_lambda}")
    if action_dim > 0:
        notes_bits.append("action-conditioned")
    if n_cams == 2:
        notes_bits.append("2cam")

    # training hyperparams from the source config
    batch_size = int(cfg.train.get("batch_size", -1))
    max_steps = int(cfg.train.get("max_steps", -1))
    lr = float(cfg.train.get("lr", -1.0))
    # actual step at ckpt save (matches max_steps for completed runs)
    actual_step = None
    try:
        import torch
        s = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        actual_step = int(s.get("step", -1)) if s.get("step") is not None else None
    except Exception:
        pass

    return {
        "base_model": base_model,
        "variant": variant,
        "family": family,
        "objective": obj,
        "seq_len": seq_len,
        "model_dim": dim,
        "model_depth": depth,
        "model_size": model_size,
        "n_params_million": round(pm, 2),
        "n_cams": n_cams,
        "action_dim": action_dim,
        "pretrained": pretrained,
        "ckpt": str(ckpt_path.relative_to(REPO_ROOT)),
        "config_path": str(cfg_path.relative_to(REPO_ROOT)),
        "batch_size": batch_size,
        "max_steps": max_steps,
        "lr": lr,
        "training_step_at_ckpt": actual_step,
        "notes": ", ".join(notes_bits) if notes_bits else "",
    }


def _stage_b_libero_entry(name: str, ckpt_dir: Path, family: str,
                            arch: str, num_frames: int, img_size: int,
                            pretrained: str, notes: str,
                            n_params_m: float, ckpt_key: str | None = None) -> dict:
    ckpt = ckpt_dir / "ckpt_last.pt"
    train_args = _stage_b_training_args(ckpt) if ckpt.exists() else {}
    return {
        "base_model": family,
        "variant": name,
        "family": family,
        "objective": "ssl/aux",
        "seq_len": num_frames,
        "model_dim": None,
        "model_depth": None,
        "model_size": "vit_large" if arch == "vit_large" else "vit_base",
        "n_params_million": n_params_m,
        "n_cams": 1,
        "action_dim": 0,
        "pretrained": pretrained,
        "ckpt": str(ckpt.relative_to(REPO_ROOT)) if ckpt.exists() else None,
        "arch": arch,
        "num_frames": num_frames,
        "img_size": img_size,
        "ckpt_key": ckpt_key,
        "batch_size": train_args.get("batch_size"),
        "max_steps": train_args.get("max_steps"),
        "lr": train_args.get("lr"),
        "training_step_at_ckpt": train_args.get("training_step_at_ckpt"),
        "notes": notes,
    }


def _frozen_baseline(name: str, family: str, arch: str,
                      num_frames: int, img_size: int,
                      pretrained: str, n_params_m: float, notes: str) -> dict:
    return {
        "base_model": family,
        "variant": name,
        "family": family,
        "objective": "frozen",
        "seq_len": num_frames,
        "model_size": "vit_large" if arch == "vit_large" else "vit_base",
        "n_params_million": n_params_m,
        "n_cams": 1,
        "action_dim": 0,
        "pretrained": pretrained,
        "ckpt": None,
        "arch": arch,
        "num_frames": num_frames,
        "img_size": img_size,
        "batch_size": None,         # never trained — frozen baseline
        "max_steps": None,
        "lr": None,
        "training_step_at_ckpt": None,
        "notes": notes,
    }


def main():
    entries: list[dict] = []

    # ---- Stage A sweep: walk every runs/sweep/*/*/ckpt_last.pt ----
    sweep_root = REPO_ROOT / "runs" / "sweep"
    for ckpt in sorted(sweep_root.glob("*/*/ckpt_last.pt")):
        e = _stage_a_entry(ckpt)
        if e is not None:
            entries.append(e)

    # ---- Stage B LIBERO comparison ckpts ----
    comp_dir = REPO_ROOT / "runs" / "comparison"
    entries.append(_stage_b_libero_entry(
        name="vjepa2_aux_finetune", ckpt_dir=comp_dir / "vjepa2_aux_finetune",
        family="vjepa2", arch="vit_large", num_frames=32, img_size=256,
        pretrained="internet video (V-JEPA 2) + LIBERO aux fine-tune",
        notes="best LIBERO Stage B (304M)", n_params_m=304.0,
    ))
    entries.append(_stage_b_libero_entry(
        name="vjepa2_1_aux_finetune", ckpt_dir=comp_dir / "vjepa2_1_aux_finetune",
        family="vjepa2_1", arch="vit_base", num_frames=32, img_size=384,
        pretrained="V-JEPA 2.1 distilled-from-G + LIBERO aux fine-tune",
        notes="V-JEPA 2.1 ViT-B (87M)", n_params_m=87.0,
        ckpt_key="encoder",
    ))
    entries.append(_stage_b_libero_entry(
        name="vjepa2_ssl_pretrain", ckpt_dir=comp_dir / "vjepa2_ssl_pretrain",
        family="vjepa2", arch="vit_large", num_frames=32, img_size=256,
        pretrained="V-JEPA 2 → continued SSL on LIBERO",
        notes="SSL on target", n_params_m=304.0,
    ))
    entries.append(_stage_b_libero_entry(
        name="vitl_scratch_aux_libero", ckpt_dir=comp_dir / "vitl_scratch_aux",
        family="vjepa2", arch="vit_large", num_frames=32, img_size=256,
        pretrained="scratch (ViT-L random init)",
        notes="LIBERO init-effect lower bound", n_params_m=304.0,
    ))

    # ---- Frozen baselines (no ckpt; built from registry) ----
    entries.append(_frozen_baseline(
        name="vjepa2_vit_l_frozen", family="vjepa2", arch="vit_large",
        num_frames=32, img_size=256,
        pretrained="internet video (V-JEPA 2)", n_params_m=304.0,
        notes="frozen reference",
    ))
    entries.append(_frozen_baseline(
        name="vjepa2_1_vit_b_frozen", family="vjepa2_1", arch="vit_base",
        num_frames=32, img_size=384,
        pretrained="V-JEPA 2.1 distilled-from-G", n_params_m=87.0,
        notes="frozen reference",
    ))
    # ---- Dreamer 4 dynamics (Phase 3 output, probe spatial hidden) ----
    d4_tokenizer_ckpt = REPO_ROOT / "runs" / "dreamer4_libero" / "tokenizer_v2_seq16" / "latest.pt"
    d4_dynamics_ckpt = REPO_ROOT / "runs" / "dreamer4_libero" / "dynamics" / "latest.pt"
    if d4_tokenizer_ckpt.exists() and d4_dynamics_ckpt.exists():
        d4_args = _stage_b_training_args(d4_dynamics_ckpt)
        import torch
        s = torch.load(str(d4_dynamics_ckpt), map_location="cpu", weights_only=False)
        dyn_sd = s.get("dynamics", {})
        # combined params: tokenizer (21.4M) + dynamics (42.6M)
        n_params_m = 21.4 + sum(int(t.numel()) for t in dyn_sd.values() if hasattr(t, "numel")) / 1e6
        entries.append({
            "base_model": "dreamer_v4",
            "variant": "dreamer4_dynamics_v1",
            "family": "dreamer4",
            "objective": "shortcut forcing + actions (Phase 3 reproduction)",
            "seq_len": 32,
            "model_size": "dyn-512d-8L + tok-256d-8L",
            "n_params_million": round(n_params_m, 2),
            "n_cams": 1,
            "action_dim": 0,    # probe doesn't use actions (action info leakage)
            "pretrained": "scratch (nicklashansen/dreamer4 reproduction)",
            "ckpt": str(d4_tokenizer_ckpt.relative_to(REPO_ROOT)),
            "dynamics_ckpt": str(d4_dynamics_ckpt.relative_to(REPO_ROOT)),
            "mode": "dynamics",
            "pool": "flatten",
            "arch": "vit_base",
            "num_frames": 32,
            "img_size": 128,
            "batch_size": d4_args.get("batch_size"),
            "max_steps": d4_args.get("max_steps"),
            "lr": d4_args.get("lr"),
            "training_step_at_ckpt": d4_args.get("training_step_at_ckpt"),
            "notes": "Phase 3 dynamics over Phase 2 tokenizer; 5 GPUs batch 4; "
                     "step 30000 ckpt (hung at end); probe = dynamics transformer "
                     "spatial hidden on clean z, actions=None.",
        })

    # ---- Dreamer 4 dynamics + aux (Phase 6 output, probe spatial hidden) ----
    d4_dynamics_aux_ckpt = REPO_ROOT / "runs" / "dreamer4_libero" / "dynamics_aux_lambda0p05" / "latest.pt"
    if d4_tokenizer_ckpt.exists() and d4_dynamics_aux_ckpt.exists():
        d4_aux_args = _stage_b_training_args(d4_dynamics_aux_ckpt)
        import torch
        s = torch.load(str(d4_dynamics_aux_ckpt), map_location="cpu", weights_only=False)
        dyn_sd = s.get("dynamics", {})
        n_params_m = 21.4 + sum(int(t.numel()) for t in dyn_sd.values() if hasattr(t, "numel")) / 1e6
        entries.append({
            "base_model": "dreamer_v4",
            "variant": "dreamer4_dynamics_aux_lambda0p05",
            "family": "dreamer4",
            "objective": "shortcut forcing + actions + invdyn aux λ=0.05 (Phase 6)",
            "seq_len": 32,
            "model_size": "dyn-512d-8L + tok-256d-8L",
            "n_params_million": round(n_params_m, 2),
            "n_cams": 1,
            "action_dim": 0,
            "pretrained": "scratch (nicklashansen/dreamer4 + dreamer-vla aux patch)",
            "ckpt": str(d4_tokenizer_ckpt.relative_to(REPO_ROOT)),
            "dynamics_ckpt": str(d4_dynamics_aux_ckpt.relative_to(REPO_ROOT)),
            "mode": "dynamics",
            "pool": "flatten",
            "arch": "vit_base",
            "num_frames": 32,
            "img_size": 128,
            "batch_size": d4_aux_args.get("batch_size"),
            "max_steps": d4_aux_args.get("max_steps"),
            "lr": d4_aux_args.get("lr"),
            "training_step_at_ckpt": d4_aux_args.get("training_step_at_ckpt"),
            "notes": "Phase 6 dynamics + aux (λ=0.05); 4 GPUs batch 4; step 30000 ckpt; "
                     "aux_invdyn_mse: 0.217 → 0.00044 (500x ↓); aux_head saved separately, "
                     "evaluator uses latest.pt only (aux-free).",
        })

    # ---- Dreamer 4 param-matched (~275M, V-JEPA ViT-L scale) + aux ----
    d4_xl_ckpt = REPO_ROOT / "runs" / "dreamer4_libero" / "dynamics_aux_xl" / "latest.pt"
    if d4_tokenizer_ckpt.exists() and d4_xl_ckpt.exists():
        xl_args = _stage_b_training_args(d4_xl_ckpt)
        import torch
        s = torch.load(str(d4_xl_ckpt), map_location="cpu", weights_only=False)
        np_xl = 21.4 + sum(int(t.numel()) for t in s.get("dynamics", {}).values() if hasattr(t, "numel")) / 1e6
        for fs, label in (("spatial", "dreamer4_xl_aux_spatial"),
                            ("agent",   "dreamer4_xl_aux_agent")):
            entries.append({
                "base_model": "dreamer_v4",
                "variant": label,
                "family": "dreamer4",
                "objective": "param-matched (d_model=1024 dyn_depth=12) + aux λ=0.05",
                "seq_len": 32,
                "model_size": "dyn-1024d-12L + tok-256d-8L (param-matched to V-JEPA ViT-L)",
                "n_params_million": round(np_xl, 2),
                "n_cams": 1, "action_dim": 0,
                "pretrained": "scratch (param-matched D4 + dreamer-vla aux patch)",
                "ckpt": str(d4_tokenizer_ckpt.relative_to(REPO_ROOT)),
                "dynamics_ckpt": str(d4_xl_ckpt.relative_to(REPO_ROOT)),
                "mode": "dynamics", "pool": "mean", "feature_source": fs,
                "arch": "vit_base", "num_frames": 32, "img_size": 128,
                "batch_size": xl_args.get("batch_size"),
                "max_steps": xl_args.get("max_steps"),
                "lr": xl_args.get("lr"),
                "training_step_at_ckpt": xl_args.get("training_step_at_ckpt"),
                "notes": "Param-matched ablation: D4 ~275M total (vs V-JEPA ViT-L 304M). "
                         "d_model_dyn=1024, dyn_depth=12. Tests whether D4 deficit is just scale.",
            })

    # ---- LAPA + aux finetune ----
    lapa_aux_ckpt = REPO_ROOT / "runs" / "comparison" / "lapa_aux_finetune" / "ckpt_last.pt"
    if lapa_aux_ckpt.exists():
        la_args = _stage_b_training_args(lapa_aux_ckpt)
        entries.append({
            "base_model": "lapa",
            "variant": "lapa_aux_finetune",
            "family": "lapa",
            "objective": "ssl/aux (LAPA LAQ + InvDynAuxHead λ=1.0, encoder unfrozen)",
            "seq_len": 32,
            "model_size": "laq-1024d",
            "n_params_million": 344.0,
            "n_cams": 1, "action_dim": 0,
            "pretrained": "Open X-Embodiment (LAPA) + LIBERO aux finetune",
            "ckpt": str(lapa_aux_ckpt.relative_to(REPO_ROOT)),
            "arch": "laq",
            "num_frames": 32, "img_size": 128,
            "batch_size": la_args.get("batch_size"),
            "max_steps": la_args.get("max_steps"),
            "lr": la_args.get("lr"),
            "training_step_at_ckpt": la_args.get("training_step_at_ckpt"),
            "notes": "LAPA LAQ encoder unfrozen + InvDynAuxHead trained jointly on LIBERO "
                     "(30k steps, batch 2). Parallels V-JEPA+aux pipeline for LAPA family.",
        })

    # ---- Dreamer 4 DMControl-pretrained + LIBERO aux finetune (V-JEPA+aux mirror) ----
    pre_dyn_aux_ckpt = REPO_ROOT / "runs" / "dreamer4_libero" / "dynamics_aux_pretrained" / "latest.pt"
    pre_tok_ckpt_ref = REPO_ROOT / "runs" / "dreamer4_libero" / "pretrained_dmcontrol" / "tokenizer.pt"
    if pre_dyn_aux_ckpt.exists() and pre_tok_ckpt_ref.exists():
        d4_pre_args = _stage_b_training_args(pre_dyn_aux_ckpt)
        for fs, label in (("spatial", "dreamer4_pretrained_aux_spatial"),
                            ("agent",   "dreamer4_pretrained_aux_agent")):
            entries.append({
                "base_model": "dreamer_v4",
                "variant": label,
                "family": "dreamer4",
                "objective": "DMControl pretrain → LIBERO aux λ=0.05 finetune",
                "seq_len": 32,
                "model_size": "dyn-512d-8L + tok-256d-8L",
                "n_params_million": 64.0,
                "n_cams": 1, "action_dim": 0,
                "pretrained": "nicklashansen pretrained (DMControl) + LIBERO aux finetune",
                "ckpt": str(pre_tok_ckpt_ref.relative_to(REPO_ROOT)),
                "dynamics_ckpt": str(pre_dyn_aux_ckpt.relative_to(REPO_ROOT)),
                "mode": "dynamics", "pool": "mean", "feature_source": fs,
                "arch": "vit_base", "num_frames": 32, "img_size": 128,
                "batch_size": d4_pre_args.get("batch_size"),
                "max_steps": d4_pre_args.get("max_steps"),
                "lr": d4_pre_args.get("lr"),
                "training_step_at_ckpt": d4_pre_args.get("training_step_at_ckpt"),
                "notes": "V-JEPA+aux mirror: DMControl pretrained dynamics → "
                         "LIBERO aux λ=0.05 finetune (20k steps); probe=" + fs,
            })

    # ---- Dreamer 4 DMControl-pretrained (frozen, like V-JEPA frozen baseline) ----
    pre_tok_ckpt = REPO_ROOT / "runs" / "dreamer4_libero" / "pretrained_dmcontrol" / "tokenizer.pt"
    pre_dyn_ckpt = REPO_ROOT / "runs" / "dreamer4_libero" / "pretrained_dmcontrol" / "dynamics.pt"
    if pre_tok_ckpt.exists() and pre_dyn_ckpt.exists():
        entries.append({
            "base_model": "dreamer_v4",
            "variant": "dreamer4_pretrained_dmcontrol_agent",
            "family": "dreamer4",
            "objective": "DMControl pretrained (frozen, no LIBERO finetune)",
            "seq_len": 32,
            "model_size": "dyn-512d-8L + tok-256d-8L",
            "n_params_million": 64.0,
            "n_cams": 1, "action_dim": 0,
            "pretrained": "nicklashansen pretrained on DMControl 30 tasks (frozen)",
            "ckpt": str(pre_tok_ckpt.relative_to(REPO_ROOT)),
            "dynamics_ckpt": str(pre_dyn_ckpt.relative_to(REPO_ROOT)),
            "mode": "dynamics", "pool": "mean", "feature_source": "agent",
            "arch": "vit_base", "num_frames": 32, "img_size": 128,
            "batch_size": None, "max_steps": None, "lr": None,
            "training_step_at_ckpt": None,
            "notes": "DMControl pretrained tokenizer+dynamics, used as frozen baseline. "
                     "Equivalent to 'V-JEPA frozen' but trained on robotics sim, not internet video.",
        })

    # ---- Dreamer 4 dynamics probed via AGENT tokens (Dreamer-style h_t) ----
    if d4_tokenizer_ckpt.exists() and d4_dynamics_ckpt.exists():
        d4_args2 = _stage_b_training_args(d4_dynamics_ckpt)
        import torch
        _s = torch.load(str(d4_dynamics_ckpt), map_location="cpu", weights_only=False)
        _np = 21.4 + sum(int(t.numel()) for t in _s.get("dynamics", {}).values() if hasattr(t, "numel")) / 1e6
        entries.append({
            "base_model": "dreamer_v4",
            "variant": "dreamer4_dynamics_agent",
            "family": "dreamer4",
            "objective": "shortcut forcing + actions (probe = agent tokens h_t)",
            "seq_len": 32,
            "model_size": "dyn-512d-8L + tok-256d-8L",
            "n_params_million": round(_np, 2),
            "n_cams": 1, "action_dim": 0,
            "pretrained": "scratch (nicklashansen/dreamer4 reproduction)",
            "ckpt": str(d4_tokenizer_ckpt.relative_to(REPO_ROOT)),
            "dynamics_ckpt": str(d4_dynamics_ckpt.relative_to(REPO_ROOT)),
            "mode": "dynamics", "pool": "mean", "feature_source": "agent",
            "arch": "vit_base", "num_frames": 32, "img_size": 128,
            "batch_size": d4_args2.get("batch_size"),
            "max_steps": d4_args2.get("max_steps"),
            "lr": d4_args2.get("lr"),
            "training_step_at_ckpt": d4_args2.get("training_step_at_ckpt"),
            "notes": "Same Phase 3 ckpt as dreamer4_dynamics_v1; probe = h_t "
                     "(agent tokens, Dreamer canonical representation) instead of spatial.",
        })
    if d4_tokenizer_ckpt.exists() and d4_dynamics_aux_ckpt.exists():
        d4_args3 = _stage_b_training_args(d4_dynamics_aux_ckpt)
        import torch
        _s = torch.load(str(d4_dynamics_aux_ckpt), map_location="cpu", weights_only=False)
        _np = 21.4 + sum(int(t.numel()) for t in _s.get("dynamics", {}).values() if hasattr(t, "numel")) / 1e6
        entries.append({
            "base_model": "dreamer_v4",
            "variant": "dreamer4_dynamics_aux_agent",
            "family": "dreamer4",
            "objective": "shortcut forcing + actions + invdyn aux λ=0.05 (probe = agent tokens)",
            "seq_len": 32,
            "model_size": "dyn-512d-8L + tok-256d-8L",
            "n_params_million": round(_np, 2),
            "n_cams": 1, "action_dim": 0,
            "pretrained": "scratch (nicklashansen/dreamer4 + dreamer-vla aux patch)",
            "ckpt": str(d4_tokenizer_ckpt.relative_to(REPO_ROOT)),
            "dynamics_ckpt": str(d4_dynamics_aux_ckpt.relative_to(REPO_ROOT)),
            "mode": "dynamics", "pool": "mean", "feature_source": "agent",
            "arch": "vit_base", "num_frames": 32, "img_size": 128,
            "batch_size": d4_args3.get("batch_size"),
            "max_steps": d4_args3.get("max_steps"),
            "lr": d4_args3.get("lr"),
            "training_step_at_ckpt": d4_args3.get("training_step_at_ckpt"),
            "notes": "Phase 6 ckpt; agent-token probe instead of spatial; "
                     "checks whether aux signal flows into Dreamer's h_t (canonical world-model state).",
        })

    # ---- Dreamer 4 tokenizer-only (Phase 2 output, frozen for probe) ----
    if d4_tokenizer_ckpt.exists():
        d4_args = _stage_b_training_args(d4_tokenizer_ckpt)
        # InvDynAuxHead-free; nicklashansen tokenizer ckpt = encoder + decoder
        import torch
        s = torch.load(str(d4_tokenizer_ckpt), map_location="cpu", weights_only=False)
        sd = s.get("model", {})
        n_params_m = sum(int(t.numel()) for t in sd.values() if hasattr(t, "numel")) / 1e6
        entries.append({
            "base_model": "dreamer_v4",
            "variant": "dreamer4_tokenizer_v2_seq16",
            "family": "dreamer4",
            "objective": "mae+lpips (Phase 2 reproduction)",
            "seq_len": int(d4_args.get("max_steps", 0)) and 16 or 16,  # always 16 for v2
            "model_size": "tokenizer-256d-8L-16lat-32dB",
            "n_params_million": round(n_params_m, 2),
            "n_cams": 1,
            "action_dim": 0,
            "pretrained": "scratch (nicklashansen/dreamer4 reproduction)",
            "ckpt": str(d4_tokenizer_ckpt.relative_to(REPO_ROOT)),
            "arch": "vit_base",            # not really; field is just a label
            "num_frames": 32,              # we probe on 32-frame clips
            "img_size": 128,
            "batch_size": d4_args.get("batch_size"),
            "max_steps": d4_args.get("max_steps"),
            "lr": d4_args.get("lr"),
            "training_step_at_ckpt": d4_args.get("training_step_at_ckpt"),
            "notes": "Phase 2 tokenizer only (no dynamics yet); seq_len=16 mae_p_max=0.75; "
                     "final PSNR≈25.2 dB on LIBERO 128×128; resumed once after OOM at 65k",
        })

    # ---- Cosmos-1 Tokenizer CI16x16 (NVIDIA, reconstruction-aligned) ----
    entries.append({
        "base_model": "cosmos_tokenizer",
        "variant": "cosmos_tokenizer_frozen",
        "family": "cosmos_tokenizer",
        "objective": "frozen (Cosmos-1 CI16x16, NVIDIA reconstruction-aligned tokenizer)",
        "seq_len": 32,
        "model_size": "conv-encoder",
        "n_params_million": 33.5,
        "n_cams": 1, "action_dim": 0,
        "pretrained": "Cosmos-1 (nvidia/Cosmos-Tokenizer-CI16x16, TorchScript JIT)",
        "ckpt": None,
        "arch": "conv", "num_frames": 32, "img_size": 128,
        "batch_size": None, "max_steps": None, "lr": None,
        "training_step_at_ckpt": None,
        "notes": "Same family as SDXL VAE (reconstruction-aligned). Matches the exact "
                 "Cosmos tokenizer used by Nilaksh et al. (2026) for direct comparison.",
    })

    cosmos_aux_ckpt = REPO_ROOT / "runs" / "comparison" / "cosmos_tokenizer_aux_finetune" / "ckpt_last.pt"
    if cosmos_aux_ckpt.exists():
        ct_args = _stage_b_training_args(cosmos_aux_ckpt)
        entries.append({
            "base_model": "cosmos_tokenizer",
            "variant": "cosmos_tokenizer_aux_finetune",
            "family": "cosmos_tokenizer",
            "objective": "Cosmos-1 CI16x16 + InvDynAuxHead λ=1.0",
            "seq_len": 32,
            "model_size": "conv-encoder",
            "n_params_million": 33.5,
            "n_cams": 1, "action_dim": 0,
            "pretrained": "Cosmos-1 + LIBERO aux finetune",
            "ckpt": str(cosmos_aux_ckpt.relative_to(REPO_ROOT)),
            "arch": "conv", "num_frames": 32, "img_size": 128,
            "batch_size": ct_args.get("batch_size"),
            "max_steps": ct_args.get("max_steps"),
            "lr": ct_args.get("lr"),
            "training_step_at_ckpt": ct_args.get("training_step_at_ckpt"),
            "notes": "DDP 2 GPUs, full encoder unfrozen + aux head. 20k steps lr=5e-5. "
                     "Parallels SDXL VAE protocol for reconstruction-family parity.",
        })

    # ---- SDXL VAE encoder (reconstruction-aligned, conv) frozen ----
    entries.append({
        "base_model": "sd3vae",
        "variant": "sdvae_frozen",
        "family": "sd3vae",
        "objective": "frozen (SDXL VAE encoder, latent-diffusion reconstruction objective)",
        "seq_len": 32,
        "model_size": "conv-encoder",
        "n_params_million": 34.2,
        "n_cams": 1, "action_dim": 0,
        "pretrained": "SDXL VAE (stabilityai/sdxl-vae). SD3 VAE gated; SDXL is faithful proxy.",
        "ckpt": None,
        "arch": "conv", "num_frames": 32, "img_size": 128,
        "batch_size": None, "max_steps": None, "lr": None,
        "training_step_at_ckpt": None,
        "notes": "Reconstruction-aligned encoder axis test. Conv-based, 34M params "
                 "(NOT scale-matched to ViT-L 304M). Output (4,16,16) flattened to "
                 "1024-d (matching ViT-L feat_dim).",
    })

    sdvae_aux_ckpt = REPO_ROOT / "runs" / "comparison" / "sd3vae_aux_finetune" / "ckpt_last.pt"
    if sdvae_aux_ckpt.exists():
        sv_args = _stage_b_training_args(sdvae_aux_ckpt)
        entries.append({
            "base_model": "sd3vae",
            "variant": "sdvae_aux_finetune",
            "family": "sd3vae",
            "objective": "SDXL VAE encoder + InvDynAuxHead λ=1.0",
            "seq_len": 32,
            "model_size": "conv-encoder",
            "n_params_million": 34.2,
            "n_cams": 1, "action_dim": 0,
            "pretrained": "SDXL VAE + LIBERO aux finetune",
            "ckpt": str(sdvae_aux_ckpt.relative_to(REPO_ROOT)),
            "arch": "conv", "num_frames": 32, "img_size": 128,
            "batch_size": sv_args.get("batch_size"),
            "max_steps": sv_args.get("max_steps"),
            "lr": sv_args.get("lr"),
            "training_step_at_ckpt": sv_args.get("training_step_at_ckpt"),
            "notes": "DDP 2 GPUs, full encoder finetune (34M is cheap). 20k steps.",
        })

    # ---- VideoMAE V1 large (304M, video MAE pretrain on Kinetics-400) ----
    entries.append({
        "base_model": "videomae",
        "variant": "videomae_frozen",
        "family": "videomae",
        "objective": "frozen (VideoMAE V1 large, video masked autoencoding)",
        "seq_len": 32, "model_size": "vit_large",
        "n_params_million": 303.9,
        "n_cams": 1, "action_dim": 0,
        "pretrained": "VideoMAE V1 large (MCG-NJU/videomae-large) on Kinetics-400",
        "ckpt": None,
        "arch": "vit_large", "num_frames": 32, "img_size": 224,
        "batch_size": None, "max_steps": None, "lr": None,
        "training_step_at_ckpt": None,
        "notes": "Mechanism test: video pretrain WITHOUT JEPA-style predictor. "
                 "Tests whether video temporal context alone suffices vs V-JEPA's "
                 "predictor-based masked latent prediction.",
    })

    videomae_aux_ckpt = REPO_ROOT / "runs" / "comparison" / "videomae_aux_finetune" / "ckpt_last.pt"
    if videomae_aux_ckpt.exists():
        vm_args = _stage_b_training_args(videomae_aux_ckpt)
        entries.append({
            "base_model": "videomae",
            "variant": "videomae_aux_finetune",
            "family": "videomae",
            "objective": "VideoMAE V1 large unfrozen + InvDynAuxHead λ=1.0",
            "seq_len": 32, "model_size": "vit_large",
            "n_params_million": 303.9,
            "n_cams": 1, "action_dim": 0,
            "pretrained": "VideoMAE V1 large + LIBERO aux finetune",
            "ckpt": str(videomae_aux_ckpt.relative_to(REPO_ROOT)),
            "arch": "vit_large", "num_frames": 32, "img_size": 224,
            "batch_size": vm_args.get("batch_size"),
            "max_steps": vm_args.get("max_steps"),
            "lr": vm_args.get("lr"),
            "training_step_at_ckpt": vm_args.get("training_step_at_ckpt"),
            "notes": "Same recipe as V-JEPA aux finetune. Tests video-pretrain-alone vs JEPA mechanism.",
        })

    # ---- Web-DINO (DINOv2 ViT-L 304M, webssl-300m-full2b-224) ----
    entries.append({
        "base_model": "webdino",
        "variant": "webdino_frozen",
        "family": "webdino",
        "objective": "frozen (Web-DINO 300m, contrastive masked image modeling on 2B web images)",
        "seq_len": 32,
        "model_size": "vit_large",
        "n_params_million": 303.7,
        "n_cams": 1, "action_dim": 0,
        "pretrained": "Web-DINO (facebook/webssl-dino300m-full2b-224)",
        "ckpt": None,
        "arch": "vit_large", "num_frames": 32, "img_size": 224,
        "batch_size": None, "max_steps": None, "lr": None,
        "training_step_at_ckpt": None,
        "notes": "DINOv2 ViT-L architecture, 304M params, pretrained on 2B web images "
                 "(Fang et al. Web-DINO 2026). Resize LIBERO 128→224 for native input.",
    })

    # ---- SigLIP 2 ViT-L (vision tower, 316M) frozen ----
    entries.append({
        "base_model": "siglip2",
        "variant": "siglip2_frozen",
        "family": "siglip2",
        "objective": "frozen (SigLIP 2 vision tower, contrastive image-text alignment)",
        "seq_len": 32,
        "model_size": "vit_large",
        "n_params_million": 316.0,
        "n_cams": 1, "action_dim": 0,
        "pretrained": "SigLIP 2 (google/siglip2-large-patch16-256)",
        "ckpt": None,
        "arch": "vit_large", "num_frames": 32, "img_size": 256,
        "batch_size": None, "max_steps": None, "lr": None,
        "training_step_at_ckpt": None,
        "notes": "Vision-language pretraining axis test. Loads only vision tower (316M); "
                 "discards text tower (548M). Resize LIBERO 128→256 for native input.",
    })

    siglip2_aux_ckpt = REPO_ROOT / "runs" / "comparison" / "siglip2_aux_finetune" / "ckpt_last.pt"
    if siglip2_aux_ckpt.exists():
        sl_args = _stage_b_training_args(siglip2_aux_ckpt)
        entries.append({
            "base_model": "siglip2",
            "variant": "siglip2_aux_finetune",
            "family": "siglip2",
            "objective": "SigLIP 2 vision tower + InvDynAuxHead λ=1.0 (V-JEPA recipe parity)",
            "seq_len": 32,
            "model_size": "vit_large",
            "n_params_million": 316.0,
            "n_cams": 1, "action_dim": 0,
            "pretrained": "SigLIP 2 + LIBERO aux finetune",
            "ckpt": str(siglip2_aux_ckpt.relative_to(REPO_ROOT)),
            "arch": "vit_large", "num_frames": 32, "img_size": 256,
            "batch_size": sl_args.get("batch_size"),
            "max_steps": sl_args.get("max_steps"),
            "lr": sl_args.get("lr"),
            "training_step_at_ckpt": sl_args.get("training_step_at_ckpt"),
            "notes": "DDP 2 GPUs, effective batch 8, 20k steps. Same protocol as Web-DINO+aux.",
        })

    # ---- Web-DINO + aux finetune ----
    webdino_aux_ckpt = REPO_ROOT / "runs" / "comparison" / "webdino_aux_finetune" / "ckpt_last.pt"
    if webdino_aux_ckpt.exists():
        wd_args = _stage_b_training_args(webdino_aux_ckpt)
        entries.append({
            "base_model": "webdino",
            "variant": "webdino_aux_finetune",
            "family": "webdino",
            "objective": "Web-DINO ViT-L unfrozen + InvDynAuxHead λ=1.0 (V-JEPA recipe parity)",
            "seq_len": 32,
            "model_size": "vit_large",
            "n_params_million": 303.7,
            "n_cams": 1, "action_dim": 0,
            "pretrained": "Web-DINO 300m (2B web images) + LIBERO aux finetune",
            "ckpt": str(webdino_aux_ckpt.relative_to(REPO_ROOT)),
            "arch": "vit_large", "num_frames": 32, "img_size": 224,
            "batch_size": wd_args.get("batch_size"),
            "max_steps": wd_args.get("max_steps"),
            "lr": wd_args.get("lr"),
            "training_step_at_ckpt": wd_args.get("training_step_at_ckpt"),
            "notes": "DDP on 2 GPUs, effective batch 8, 20k steps. Parallel V-JEPA+aux protocol "
                     "for masked-prediction family generalization test.",
        })

    # ---- Exp B: λ sweep on Web-DINO + SigLIP 2 ----
    for lam, lam_tag in [(0.01, "0p01"), (0.025, "0p025"),
                          (0.1, "0p1"), (0.3, "0p3")]:
        for fam, model_size, n_p, img_sz, pretrained in [
            ("webdino", "vit_large", 303.7, 224, "Web-DINO 300m (2B web images) + LIBERO aux finetune"),
            ("siglip2", "vit_large", 303.5, 256, "SigLIP 2 ViT-L + LIBERO aux finetune"),
        ]:
            ck = REPO_ROOT / "runs" / "comparison" / f"{fam}_aux_lam{lam_tag}" / "ckpt_last.pt"
            if not ck.exists():
                continue
            args = _stage_b_training_args(ck)
            entries.append({
                "base_model": fam,
                "variant": f"{fam}_aux_lam{lam_tag}",
                "family": fam,
                "objective": f"{fam} ViT-L unfrozen + InvDynAuxHead λ={lam}",
                "seq_len": 32, "model_size": model_size,
                "n_params_million": n_p,
                "n_cams": 1, "action_dim": 0,
                "pretrained": pretrained,
                "ckpt": str(ck.relative_to(REPO_ROOT)),
                "arch": "vit_large", "num_frames": 32, "img_size": img_sz,
                "batch_size": args.get("batch_size"),
                "max_steps": args.get("max_steps"),
                "lr": args.get("lr"),
                "training_step_at_ckpt": args.get("training_step_at_ckpt"),
                "notes": f"Exp B: aux-lambda sweep (λ={lam}). 20k steps single GPU batch=4.",
            })

    entries.append({
        "base_model": "lapa",
        "variant": "lapa_frozen",
        "family": "lapa",
        "objective": "frozen",
        "seq_len": 32,
        "model_size": "344M",
        "n_params_million": 344.0,
        "n_cams": 1,
        "action_dim": 0,
        "pretrained": "Open X-Embodiment (LAPA)",
        "ckpt": None,
        "batch_size": None,
        "max_steps": None,
        "lr": None,
        "training_step_at_ckpt": None,
        "notes": "frozen reference",
    })

    out = REPO_ROOT / "runs" / "full_results"
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(entries, sort_keys=False, allow_unicode=True))
    print(f"wrote {manifest_path} ({len(entries)} rows)")

    # Quick summary by family
    by_family: dict[str, int] = {}
    for e in entries:
        by_family[e["family"]] = by_family.get(e["family"], 0) + 1
    print("\nrows by family:")
    for k, v in sorted(by_family.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
