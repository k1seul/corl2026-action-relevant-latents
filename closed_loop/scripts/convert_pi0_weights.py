#!/usr/bin/env python3
"""Convert a pi0/pi0.5 JAX checkpoint from openpi-assets into a PyTorch safetensors file.

This is a standalone wrapper around openpi's conversion logic that does NOT require
the full `openpi.training.config` system (which needs `openpi_client` installed).

Typical usage:

    # Convert the already-cached pi05_libero JAX checkpoint:
    python scripts/convert_pi0_weights.py \
        --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
        --output-dir ~/.cache/openpi/pi05_libero_pytorch \
        --pi05

    # Base model (no LIBERO fine-tuning):
    python scripts/convert_pi0_weights.py \
        --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_base \
        --output-dir ~/.cache/openpi/pi05_base_pytorch \
        --pi05

After conversion the output dir contains:
    model.safetensors   <- load with safetensors.torch.load_model(pi0, path)
    config.json         <- hyperparameters used during conversion
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch

# We import from openpi source directly — no openpi_client needed.
_OPENPI_SRC = os.environ.get("OPENPI_SRC", str(pathlib.Path.home() / "openpi-src" / "src"))
if _OPENPI_SRC not in sys.path:
    sys.path.insert(0, _OPENPI_SRC)

import safetensors.torch as _st

from flax.nnx import traversals
import openpi.models.model as _model
import openpi.models.gemma as _gemma
import openpi.models.pi0_config as _pi0_cfg
import openpi.models_pytorch.pi0_pytorch as _pi0_torch


# ---------------------------------------------------------------------------
# helpers (verbatim from openpi's examples/convert_jax_model_to_pytorch.py)
# ---------------------------------------------------------------------------

def _restore_orbax(checkpoint_dir: str) -> dict:
    """Load JAX params via orbax and return a flat dict keyed by '/'-joined path."""
    params = _model.restore_params(
        f"{checkpoint_dir}/params/", restore_type=np.ndarray, dtype="float32"
    )
    return {
        "paligemma_params": traversals.flatten_mapping(params["PaliGemma"], sep="/"),
        "projection_params": params,
    }


def _slice_paligemma(state_dict: dict, cfg) -> tuple[dict, dict]:
    """Convert PaliGemma JAX params → PyTorch format.  Returns (pg_dict, expert_dict)."""
    suffix = "/value" if "img/embedding/kernel/value" in state_dict else ""

    def pop(k):
        return state_dict.pop(k + suffix) if (k + suffix) in state_dict else state_dict.pop(k)

    # patch embedding
    state_dict["paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"] = (
        pop("img/embedding/kernel").transpose(3, 2, 0, 1)
    )
    state_dict["paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias"] = (
        pop("img/embedding/bias")
    )

    # positional embedding
    hs = cfg.vision_config.hidden_size
    state_dict["paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight"] = (
        pop("img/pos_embedding").reshape(-1, hs)
    )

    # vision encoder blocks
    ln0_w = pop("img/Transformer/encoderblock/LayerNorm_0/scale")
    ln0_b = pop("img/Transformer/encoderblock/LayerNorm_0/bias")
    ln1_w = pop("img/Transformer/encoderblock/LayerNorm_1/scale")
    ln1_b = pop("img/Transformer/encoderblock/LayerNorm_1/bias")
    mlp0_k = pop("img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel")
    mlp0_b = pop("img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias")
    mlp1_k = pop("img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel")
    mlp1_b = pop("img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias")
    qk = pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel")
    qb = pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias")
    kk = pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel")
    kb = pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias")
    vk = pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel")
    vb = pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias")
    ok = pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel")
    ob = pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias")

    n_vis = cfg.vision_config.num_hidden_layers
    for i in range(n_vis):
        base = f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}"
        state_dict[f"{base}.layer_norm1.weight"] = ln0_w[i].transpose()
        state_dict[f"{base}.layer_norm1.bias"] = ln0_b[i]
        state_dict[f"{base}.layer_norm2.weight"] = ln1_w[i].transpose()
        state_dict[f"{base}.layer_norm2.bias"] = ln1_b[i]
        state_dict[f"{base}.mlp.fc1.weight"] = mlp0_k[i].transpose()
        state_dict[f"{base}.mlp.fc1.bias"] = mlp0_b[i]
        state_dict[f"{base}.mlp.fc2.weight"] = mlp1_k[i].transpose()
        state_dict[f"{base}.mlp.fc2.bias"] = mlp1_b[i]
        state_dict[f"{base}.self_attn.q_proj.weight"] = qk[i].reshape(-1, hs).transpose()
        state_dict[f"{base}.self_attn.q_proj.bias"] = qb[i].reshape(-1, hs).reshape(-1)
        state_dict[f"{base}.self_attn.k_proj.weight"] = kk[i].reshape(-1, hs).transpose()
        state_dict[f"{base}.self_attn.k_proj.bias"] = kb[i].reshape(-1, hs).reshape(-1)
        state_dict[f"{base}.self_attn.v_proj.weight"] = vk[i].reshape(-1, hs).transpose()
        state_dict[f"{base}.self_attn.v_proj.bias"] = vb[i].reshape(-1, hs).reshape(-1)
        state_dict[f"{base}.self_attn.out_proj.weight"] = ok[i].reshape(-1, hs).transpose()
        state_dict[f"{base}.self_attn.out_proj.bias"] = ob[i].reshape(-1, hs).reshape(-1)

    state_dict["paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight"] = (
        pop("img/Transformer/encoder_norm/scale").transpose()
    )
    state_dict["paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias"] = (
        pop("img/Transformer/encoder_norm/bias")
    )

    # multimodal projector
    state_dict["paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"] = (
        pop("img/head/kernel").transpose()
    )
    state_dict["paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"] = (
        pop("img/head/bias")
    )

    # text decoder (gemma)
    state_dict["paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"] = (
        pop("llm/embedder/input_embedding")
    )

    attn_vec = pop("llm/layers/attn/attn_vec_einsum/w")
    kv_ein = pop("llm/layers/attn/kv_einsum/w")
    q_ein = pop("llm/layers/attn/q_einsum/w")
    mlp_gate = pop("llm/layers/mlp/gating_einsum")
    mlp_lin = pop("llm/layers/mlp/linear")
    ln_pre_attn = pop("llm/layers/pre_attention_norm/scale")
    ln_pre_ffn = pop("llm/layers/pre_ffw_norm/scale")

    tc = cfg.text_config
    for i in range(tc.num_hidden_layers):
        base = f"paligemma_with_expert.paligemma.model.language_model.layers.{i}"
        state_dict[f"{base}.self_attn.q_proj.weight"] = (
            q_ein[i].transpose(0, 2, 1).reshape(tc.num_attention_heads * tc.head_dim, tc.hidden_size)
        )
        state_dict[f"{base}.self_attn.k_proj.weight"] = kv_ein[i, 0, 0].transpose()
        state_dict[f"{base}.self_attn.v_proj.weight"] = kv_ein[i, 1, 0].transpose()
        state_dict[f"{base}.self_attn.o_proj.weight"] = (
            attn_vec[i].transpose(2, 0, 1).reshape(tc.num_attention_heads * tc.head_dim, tc.hidden_size)
        )
        state_dict[f"{base}.mlp.gate_proj.weight"] = mlp_gate[i, 0].transpose()
        state_dict[f"{base}.mlp.up_proj.weight"] = mlp_gate[i, 1].transpose()
        state_dict[f"{base}.mlp.down_proj.weight"] = mlp_lin[i].transpose()
        state_dict[f"{base}.input_layernorm.weight"] = ln_pre_attn[i]
        state_dict[f"{base}.post_attention_layernorm.weight"] = ln_pre_ffn[i]

    state_dict["paligemma_with_expert.paligemma.model.language_model.norm.weight"] = (
        pop("llm/final_norm/scale")
    )

    # split off expert keys
    expert_keys = [
        "llm/final_norm_1/scale", "llm/final_norm_1/Dense_0/bias", "llm/final_norm_1/Dense_0/kernel",
        "llm/layers/attn/attn_vec_einsum_1/w", "llm/layers/attn/kv_einsum_1/w", "llm/layers/attn/q_einsum_1/w",
        "llm/layers/mlp_1/gating_einsum", "llm/layers/mlp_1/linear",
        "llm/layers/pre_attention_norm_1/scale",
        "llm/layers/pre_attention_norm_1/Dense_0/bias", "llm/layers/pre_attention_norm_1/Dense_0/kernel",
        "llm/layers/pre_ffw_norm_1/scale",
        "llm/layers/pre_ffw_norm_1/Dense_0/bias", "llm/layers/pre_ffw_norm_1/Dense_0/kernel",
    ]
    # also handle /value suffix variants
    expert_keys_set = set(expert_keys) | {k + "/value" for k in expert_keys}
    expert_dict = {}
    final_dict = {}
    for k, v in state_dict.items():
        k_bare = k.rstrip("/value") if k.endswith("/value") else k
        if k in expert_keys_set or k_bare in expert_keys_set:
            expert_dict[k] = v
        else:
            final_dict[k] = torch.from_numpy(v) if not isinstance(v, torch.Tensor) else v

    return final_dict, expert_dict


def _slice_gemma_expert(state_dict: dict, cfg, *, is_pi05: bool) -> dict:
    """Convert action expert JAX params → PyTorch format."""
    suffix = "/value" if "llm/layers/attn/attn_vec_einsum_1/w/value" in state_dict else ""

    def pop(k):
        return state_dict.pop(k + suffix) if (k + suffix) in state_dict else state_dict.pop(k)

    attn_vec = pop("llm/layers/attn/attn_vec_einsum_1/w")
    kv_ein = pop("llm/layers/attn/kv_einsum_1/w")
    q_ein = pop("llm/layers/attn/q_einsum_1/w")
    mlp_gate = pop("llm/layers/mlp_1/gating_einsum")
    mlp_lin = pop("llm/layers/mlp_1/linear")

    if is_pi05:
        ln_attn_b = pop("llm/layers/pre_attention_norm_1/Dense_0/bias")
        ln_attn_k = pop("llm/layers/pre_attention_norm_1/Dense_0/kernel")
        ln_ffn_b = pop("llm/layers/pre_ffw_norm_1/Dense_0/bias")
        ln_ffn_k = pop("llm/layers/pre_ffw_norm_1/Dense_0/kernel")
    else:
        ln_attn = pop("llm/layers/pre_attention_norm_1/scale")
        ln_ffn = pop("llm/layers/pre_ffw_norm_1/scale")

    # gemma.Config uses depth/num_heads/width instead of num_hidden_layers/num_attention_heads/hidden_size
    n_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "depth", None))
    n_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "num_heads", None))
    hidden = getattr(cfg, "hidden_size", getattr(cfg, "width", None))

    for i in range(n_layers):
        base = f"paligemma_with_expert.gemma_expert.model.layers.{i}"
        state_dict[f"{base}.self_attn.q_proj.weight"] = (
            q_ein[i].transpose(0, 2, 1).reshape(n_heads * cfg.head_dim, hidden)
        )
        state_dict[f"{base}.self_attn.k_proj.weight"] = kv_ein[i, 0, 0].transpose()
        state_dict[f"{base}.self_attn.v_proj.weight"] = kv_ein[i, 1, 0].transpose()
        state_dict[f"{base}.self_attn.o_proj.weight"] = (
            attn_vec[i].reshape(n_heads * cfg.head_dim, hidden).transpose(1, 0)
        )
        state_dict[f"{base}.mlp.gate_proj.weight"] = mlp_gate[i, 0].transpose()
        state_dict[f"{base}.mlp.up_proj.weight"] = mlp_gate[i, 1].transpose()
        state_dict[f"{base}.mlp.down_proj.weight"] = mlp_lin[i].transpose()

        if is_pi05:
            state_dict[f"{base}.input_layernorm.dense.bias"] = ln_attn_b[i]
            state_dict[f"{base}.input_layernorm.dense.weight"] = ln_attn_k[i].transpose()
            state_dict[f"{base}.post_attention_layernorm.dense.bias"] = ln_ffn_b[i]
            state_dict[f"{base}.post_attention_layernorm.dense.weight"] = ln_ffn_k[i].transpose()
        else:
            state_dict[f"{base}.input_layernorm.weight"] = ln_attn[i]
            state_dict[f"{base}.post_attention_layernorm.weight"] = ln_ffn[i]

    if is_pi05:
        fn_b = pop("llm/final_norm_1/Dense_0/bias")
        fn_k = pop("llm/final_norm_1/Dense_0/kernel")
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.bias"] = fn_b
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.weight"] = fn_k.transpose()
    else:
        state_dict["paligemma_with_expert.gemma_expert.model.norm.weight"] = pop("llm/final_norm_1/scale")

    return {k: (torch.from_numpy(v) if not isinstance(v, torch.Tensor) else v)
            for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# main conversion
# ---------------------------------------------------------------------------

class _PaliGemmaConfig:
    """Minimal config object that mirrors what the full openpi config provides."""
    def __init__(self):
        self.vision_config = type("VC", (), {
            "hidden_size": 1152,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "intermediate_size": 4304,
            "patch_size": 14,
            "projection_dim": 2048,
        })()
        self.text_config = type("TC", (), {
            "hidden_size": 2048,
            "num_hidden_layers": 18,
            "num_attention_heads": 8,
            "head_dim": 256,
            "intermediate_size": 16384,
        })()


def convert(checkpoint_dir: str, output_dir: str, *, pi05: bool, precision: str,
            action_dim: int, action_horizon: int) -> None:
    print(f"Loading JAX checkpoint: {checkpoint_dir}")
    initial = _restore_orbax(checkpoint_dir)

    pg_params = initial["paligemma_params"]
    proj_params = initial["projection_params"]

    # projection layers (action_in_proj, action_out_proj, time_mlp_in/out)
    proj_keys = (
        ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]
        if pi05
        else ["state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"]
    )
    projection_state = {}
    for key in proj_keys:
        val = proj_params[key]
        w = val["kernel"]["value"] if isinstance(val["kernel"], dict) else val["kernel"]
        b = val["bias"]["value"] if isinstance(val["bias"], dict) else val["bias"]
        projection_state[f"{key}.weight"] = torch.from_numpy(np.array(w)).T
        projection_state[f"{key}.bias"] = torch.from_numpy(np.array(b))

    print("Converting PaliGemma weights...")
    pg_cfg = _PaliGemmaConfig()
    pg_dict, expert_dict = _slice_paligemma(pg_params, pg_cfg)

    print("Converting Gemma action expert weights...")
    exp_cfg = _gemma.get_config("gemma_300m")
    gem_dict = _slice_gemma_expert(expert_dict, exp_cfg, is_pi05=pi05)

    all_params = {**pg_dict, **gem_dict, **projection_state}

    print("Instantiating PI0Pytorch model and loading weights...")
    model_cfg = _pi0_cfg.Pi0Config(
        dtype=precision,
        action_dim=action_dim,
        action_horizon=action_horizon,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=pi05,
    )
    model = _pi0_torch.PI0Pytorch(model_cfg)
    missing, unexpected = model.load_state_dict(all_params, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")

    if precision == "bfloat16":
        model = model.to(torch.bfloat16)
    elif precision == "float32":
        model = model.to(torch.float32)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "model.safetensors")
    print(f"Saving safetensors → {out_path}")
    _st.save_model(model, out_path)

    cfg_json = {
        "pi05": pi05,
        "precision": precision,
        "action_dim": action_dim,
        "action_horizon": action_horizon,
        "source": checkpoint_dir,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg_json, f, indent=2)

    size_gb = os.path.getsize(out_path) / 1e9
    print(f"Done!  {out_path}  ({size_gb:.1f} GB)")


def parse_args():
    p = argparse.ArgumentParser(description="Convert pi0/pi0.5 JAX checkpoint to PyTorch safetensors")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Path to JAX checkpoint (directory containing params/)")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for model.safetensors and config.json")
    p.add_argument("--pi05", action="store_true", default=False,
                   help="Set for pi0.5 (adaptive norm) vs pi0 (standard RMSNorm)")
    p.add_argument("--precision", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument("--action-dim", type=int, default=32,
                   help="Action dimension matching the checkpoint (openpi pads LIBERO 7-DoF to 32)")
    p.add_argument("--action-horizon", type=int, default=10,
                   help="Action horizon matching the checkpoint (pi05_libero=10)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        pi05=args.pi05,
        precision=args.precision,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
    )
