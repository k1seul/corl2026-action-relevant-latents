"""LAPA LAQ + inverse-dynamics aux finetune (parallels train_vjepa2_aux.py).

Initialises from the pretrained Open-X LAQ ckpt, unfreezes everything, and
trains the encoder + InvDynAuxHead jointly with the aux loss as the only
supervision signal. Output ckpt is the aux-finetuned encoder (with
aux_head stored separately for completeness, never used at probe time).

LAPA's LAQ was trained on frame pairs (T=2); we feed T=32 clips by
chunking into adjacent pairs and averaging per-frame features. See
``LapaExtractor.extract`` for the same pattern in inference.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_lapa_aux.py \\
        --out-dir runs/comparison/lapa_aux_finetune \\
        --max-steps 30000 --batch-size 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.libero_video import worker_init_fn
from data.build_train_dataset import build_train_dataset, DATASET_CHOICES
from training.aux_loss import InvDynAuxHead

EXTERNAL_DIR = Path(os.environ.get(
    "EXTERNAL_MODELS_DIR",
    "external_models",
))
LAPA_REPO = EXTERNAL_DIR / "LAPA"
LAPA_CKPT = EXTERNAL_DIR / "lapa_ckpts" / "laq_openx.pt"


def _ensure_lapa_on_path() -> None:
    if not LAPA_REPO.exists():
        raise FileNotFoundError(f"LAPA repo not at {LAPA_REPO}")
    p = str(LAPA_REPO / "laq")
    if p not in sys.path:
        sys.path.insert(0, p)


def encode_frame_features(model, clip: torch.Tensor, image_size: int) -> torch.Tensor:
    """clip: (B, T, C, H, W) → (B, T, D=1024) per-frame features.

    Mirrors LapaExtractor.extract but WITH gradients enabled.
    """
    B, T, C, H, W = clip.shape
    if C > 3:
        clip = clip[:, :, :3]
        C = 3
    x = clip.reshape(B * T, C, H, W)
    x = F.interpolate(x, size=(image_size, image_size), mode="bilinear",
                       align_corners=False)
    x = x.reshape(B, T, C, image_size, image_size)

    pair_first = x[:, :-1]                                          # (B, T-1, 3, H, W)
    pair_last = x[:, 1:]                                             # (B, T-1, 3, H, W)
    pairs = torch.stack([pair_first, pair_last], dim=3)             # (B, T-1, 3, 2, H, W)
    pairs = pairs.reshape(B * (T - 1), C, 2, image_size, image_size)

    first_frames = pairs[:, :, :1]
    last_frames = pairs[:, :, 1:]
    first_emb = model.to_patch_emb_first_frame(first_frames)        # (BN, 1, h, w, D)
    last_emb = model.to_patch_emb_first_frame(last_frames)
    tokens = torch.cat([first_emb, last_emb], dim=1)                # (BN, 2, h, w, D)

    first_tokens, last_tokens = model.encode(tokens)
    first_pooled = first_tokens.mean(dim=(2, 3))[:, 0]              # (BN, D)
    last_pooled = last_tokens.mean(dim=(2, 3))[:, 0]
    first_pooled = first_pooled.reshape(B, T - 1, -1)
    last_pooled = last_pooled.reshape(B, T - 1, -1)
    feats = torch.cat([first_pooled, last_pooled[:, -1:]], dim=1)   # (B, T, D)
    return feats


def lr_schedule(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    # cosine decay to 0.1× peak
    import math
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ckpt", default=str(LAPA_CKPT))
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--clip-len", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=256)   # LAPA native
    ap.add_argument("--aux-lambda", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-interval", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset", choices=DATASET_CHOICES, default="libero")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir / "tb"))

    # ── build LAPA LAQ model ──────────────────────────────────────────────────
    _ensure_lapa_on_path()
    from laq_model.latent_action_quantization import LatentActionQuantization

    model = LatentActionQuantization(
        dim=1024, quant_dim=32, codebook_size=8,
        image_size=args.image_size, patch_size=32,
        spatial_depth=8, temporal_depth=8,
        dim_head=64, heads=16, code_seq_len=4,
    )
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    msg = model.load_state_dict(state)
    print(f"loaded {args.ckpt}: missing={len(msg.missing_keys)} "
          f"unexpected={len(msg.unexpected_keys)}")
    model = model.to(device)
    # CRITICAL: unfreeze for finetuning (LapaExtractor sets requires_grad=False)
    for p in model.parameters():
        p.requires_grad = True
    model.train()

    train_ds, action_dim = build_train_dataset(
        args.dataset, clip_len=args.clip_len, seed=args.seed,
        resolution=128, return_actions=True)
    aux_head = InvDynAuxHead(in_dim=1024, action_dim=action_dim).to(device)

    n_enc = sum(p.numel() for p in model.parameters())
    n_aux = sum(p.numel() for p in aux_head.parameters())
    print(f"params: encoder={n_enc/1e6:.1f}M, aux={n_aux/1e6:.1f}M, dataset={args.dataset}")
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn,
        drop_last=True, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(aux_head.parameters()),
        lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )

    step = 0
    t0 = time.time()
    while step < args.max_steps:
        for clip, actions in loader:
            if step >= args.max_steps:
                break
            clip = clip.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)

            cur_lr = args.lr * lr_schedule(step, args.warmup_steps, args.max_steps)
            for g in opt.param_groups:
                g["lr"] = cur_lr

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                     enabled=device.type == "cuda"):
                feats = encode_frame_features(model, clip,
                                                image_size=args.image_size)   # (B, T, 1024)
                pred = aux_head(feats)                                         # (B, T-1, 7)
                target = actions[:, :-1, :action_dim]
                aux_loss = (pred - target).pow(2).mean()
                loss = args.aux_lambda * aux_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(aux_head.parameters()), 1.0)
            opt.step()

            if step % args.log_interval == 0:
                ips = (step + 1) * args.batch_size / max(1e-6, time.time() - t0)
                print(f"step {step:6d} | aux {aux_loss.item():.4f} | "
                      f"lr {cur_lr:.2e} | clips/s {ips:.1f}")
                writer.add_scalar("train/aux", aux_loss.item(), step)
                writer.add_scalar("train/lr", cur_lr, step)

            step += 1

    torch.save({
        "step": step,
        "encoder": model.state_dict(),
        "aux_head": aux_head.state_dict(),
        "args": vars(args),
    }, out_dir / "ckpt_last.pt")
    writer.close()
    print(f"done. saved to {out_dir / 'ckpt_last.pt'}")


if __name__ == "__main__":
    main()
