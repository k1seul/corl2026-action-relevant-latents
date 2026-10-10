"""Experiment C2: evaluate the trained aux head directly (no probe training).

Loads ckpt_last_raw.pt (contains aux_head state_dict) + ckpt_last.pt
(EMA model), extracts frozen features from a test split, and reports
R² for the aux head's raw predictions — without any probe retraining.

Comparison:
  C1  frozen features → probe MLP trained from scratch  (eval_downstream.py)
  C2  frozen features → aux head applied directly        (this script)

Usage:
    uv run python scripts/eval_aux_direct.py \
        --ckpt-dir runs/sweep/aux_ablations/aux_abl_A0_invdyn_l2_repro \
        --test-split data/splits_taskheld/test_ood.json \
        --out-dir runs/downstream/C2_aux_direct
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.libero_video import LiberoVideoClips, worker_init_fn
from models.block_causal_transformer import BlockCausalVideoTransformer
from training.aux_loss import InvDynAuxHead


def build_model(cfg, in_channels: int) -> BlockCausalVideoTransformer:
    return BlockCausalVideoTransformer(
        in_channels=in_channels,
        patch_size=cfg.model.patch_size,
        num_frames=cfg.data.clip_len,
        image_size=cfg.data.resolution,
        dim=cfg.model.dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=0.0,
        enable_timestep=True,
        action_dim=int(cfg.model.get("action_dim", 0)),
    )


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    var = target.var(dim=0, unbiased=False).mean().item()
    return 1.0 - mse / max(var, 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True,
                    help="run dir with ckpt_last.pt and ckpt_last_raw.pt")
    ap.add_argument("--test-split", default="data/splits_taskheld/test_ood.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--test-clips", type=int, default=400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # load EMA model (ckpt_last.pt)
    state = torch.load(ckpt_dir / "ckpt_last.pt", map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["config"])
    cfg.data.test_split = args.test_split

    in_channels = 3 * len(cfg.data.cameras)
    model = build_model(cfg, in_channels).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"EMA model loaded from {ckpt_dir / 'ckpt_last.pt'}")

    # load aux_head from ckpt_last_raw.pt
    raw_state = torch.load(ckpt_dir / "ckpt_last_raw.pt", map_location=device, weights_only=False)
    if "aux_head" not in raw_state:
        print("ERROR: aux_head not found in ckpt_last_raw.pt")
        print("  Re-train with the updated train.py that saves aux_head state.")
        sys.exit(1)
    feat_dim = int(cfg.model.dim)
    aux_action_dim = int(cfg.train.get("aux_action_dim", 7))
    aux_head = InvDynAuxHead(in_dim=feat_dim, action_dim=aux_action_dim).to(device)
    aux_head.load_state_dict(raw_state["aux_head"])
    aux_head.eval()
    print("aux_head loaded from ckpt_last_raw.pt")

    # encode test split
    ds = LiberoVideoClips(
        split_path=cfg.data.test_split,
        datasets_dir=cfg.data.datasets_dir,
        clip_len=cfg.data.clip_len,
        cameras=cfg.data.cameras,
        resolution=cfg.data.resolution,
        seed=42,
        return_actions=True,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False,
                        num_workers=2, worker_init_fn=worker_init_fn)

    all_preds, all_targets = [], []
    seen = 0
    for clip, actions in loader:
        clip = clip.to(device)
        actions = actions.to(device)
        B, T = clip.shape[:2]
        with torch.no_grad(), torch.amp.autocast(device_type=device.type,
                                                  dtype=torch.bfloat16,
                                                  enabled=device.type == "cuda"):
            t = torch.zeros(B, T, device=device, dtype=torch.long)
            features = model.forward_features(clip, mask_indicator=None, t=t)  # (B,T,N,D)
            pooled = features.float().mean(dim=2)                              # (B,T,D)
            pred = aux_head(pooled)                                            # (B,T-1,A)
        all_preds.append(pred.reshape(-1, aux_action_dim).cpu())
        all_targets.append(actions[:, :-1].reshape(-1, aux_action_dim).cpu())
        seen += B
        if seen >= args.test_clips:
            break

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    action_r2 = r2_score(preds, targets)
    # inv-dyn R²: same thing — aux head IS an inverse dynamics predictor
    invdyn_r2 = action_r2

    print(f"\n=== C2: aux head direct eval ===")
    print(f"  clips:     {seen}")
    print(f"  action R²: {action_r2:.4f}")
    print(f"  invdyn R²: {invdyn_r2:.4f}")

    metrics = {
        "method": "aux_head_direct",
        "ckpt_dir": str(ckpt_dir),
        "test_split": cfg.data.test_split,
        "clips": seen,
        "action_r2": action_r2,
        "invdyn_r2": invdyn_r2,
    }
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    with (out_dir / "metrics.txt").open("w") as f:
        f.write(f"action_r2: {action_r2:.4f}\n")
        f.write(f"invdyn_r2: {invdyn_r2:.4f}\n")
    print(f"wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
