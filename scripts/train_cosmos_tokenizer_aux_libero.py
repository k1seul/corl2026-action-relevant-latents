"""Cosmos-1 Continuous Image Tokenizer (CI16x16) + inverse-dynamics aux finetune.

Parallels train_sd_vae_aux_libero.py for the reconstruction-aligned family.
Cosmos encoder is shipped as TorchScript JIT (~33.5M params); we unfreeze
all parameters and train end-to-end with aux loss as the only supervision.

Usage:
    CUDA_VISIBLE_DEVICES=2,3 uv run torchrun --nproc_per_node=2 \\
        scripts/train_cosmos_tokenizer_aux_libero.py \\
        --out-dir runs/comparison/cosmos_tokenizer_aux_finetune \\
        --batch-size 8
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from huggingface_hub import hf_hub_download

from data.libero_video import worker_init_fn
from data.build_train_dataset import build_train_dataset, DATASET_CHOICES
from training.aux_loss import InvDynAuxHead


HF_REPO = "nvidia/Cosmos-Tokenizer-CI16x16"
NATIVE_SIZE = 128


def is_torchrun(): return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def init_distributed():
    if not is_torchrun(): return False, 0, 1, 0
    rank = int(os.environ["RANK"]); world_size = int(os.environ["WORLD_SIZE"]); local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def is_rank0(r): return r == 0


def encode_frame_features(encoder, clip: torch.Tensor, native_size: int = NATIVE_SIZE) -> torch.Tensor:
    """clip (B,T,C,H,W) [0,1] → (B,T,D=1024)."""
    B, T, C, H, W = clip.shape
    if C > 3:
        clip = clip[:, :, :3]; C = 3
    x = clip.reshape(B * T, C, H, W)
    if (H, W) != (native_size, native_size):
        x = F.interpolate(x, size=(native_size, native_size), mode="bilinear", align_corners=False)
    x = (x * 2.0 - 1.0)                                    # [-1, 1]
    out = encoder(x)
    latent = out[0] if isinstance(out, (tuple, list)) else out
    latent = latent.flatten(1)                              # (BT, 1024)
    return latent.reshape(B, T, -1)


def lr_schedule(step, warmup, total):
    if step < warmup: return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hf-repo", default=HF_REPO)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="lower lr (matches SDXL VAE) to avoid disrupting conv features")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--clip-len", type=int, default=32)
    ap.add_argument("--native-image-size", type=int, default=NATIVE_SIZE)
    ap.add_argument("--aux-lambda", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-interval", type=int, default=200)
    ap.add_argument("--dataset", choices=DATASET_CHOICES, default="libero")
    args = ap.parse_args()

    ddp, rank, world_size, local_rank = init_distributed()
    torch.manual_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    if is_rank0(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(out_dir / "tb"))

    # ── load JIT encoder ──────────────────────────────────────────────────────
    enc_path = hf_hub_download(repo_id=args.hf_repo, filename="encoder.jit")
    encoder = torch.jit.load(enc_path, map_location=device)
    for p in encoder.parameters():
        p.requires_grad = True
    encoder.train()

    # latent dim: 16 * (native/16)^2 = 16 * 8^2 = 1024
    feat_dim = 16 * (args.native_image_size // 16) ** 2
    train_ds, action_dim = build_train_dataset(
        args.dataset, clip_len=args.clip_len, seed=args.seed,
        resolution=128, return_actions=True)
    aux_head = InvDynAuxHead(in_dim=feat_dim, action_dim=action_dim).to(device)

    if ddp:
        encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank,
                       broadcast_buffers=False, find_unused_parameters=True)
        aux_head = DDP(aux_head, device_ids=[local_rank], output_device=local_rank)

    if is_rank0(rank):
        n_enc = sum(p.numel() for p in (encoder.module if ddp else encoder).parameters())
        n_aux = sum(p.numel() for p in (aux_head.module if ddp else aux_head).parameters())
        print(f"params: cosmos_enc={n_enc/1e6:.1f}M  aux={n_aux/1e6:.1f}M "
              f"world={world_size}  batch={args.batch_size}*{world_size}={args.batch_size*world_size}")

    # train_ds built above
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
        num_workers=args.num_workers, worker_init_fn=worker_init_fn,
        drop_last=True, pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    train_params = [
        *((encoder.module if ddp else encoder).parameters()),
        *((aux_head.module if ddp else aux_head).parameters()),
    ]
    opt = torch.optim.AdamW(train_params, lr=args.lr, betas=(0.9, 0.95),
                              weight_decay=args.weight_decay)

    step = 0; t0 = time.time()
    while step < args.max_steps:
        for clip, actions in loader:
            if step >= args.max_steps: break
            clip = clip.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)

            cur_lr = args.lr * lr_schedule(step, args.warmup_steps, args.max_steps)
            for g in opt.param_groups: g["lr"] = cur_lr

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                     enabled=device.type == "cuda"):
                feats = encode_frame_features(encoder.module if ddp else encoder, clip,
                                                native_size=args.native_image_size)
                pred = (aux_head.module if ddp else aux_head)(feats)
                target = actions[:, :-1, :action_dim]
                aux_loss = (pred - target).pow(2).mean()
                loss = args.aux_lambda * aux_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()

            if is_rank0(rank) and step % args.log_interval == 0:
                ips = (step + 1) * args.batch_size * world_size / max(1e-6, time.time() - t0)
                print(f"step {step:6d} | aux {aux_loss.item():.4f} | lr {cur_lr:.2e} | clips/s {ips:.1f}")
                writer.add_scalar("train/aux", aux_loss.item(), step)
                writer.add_scalar("train/lr", cur_lr, step)
            step += 1
        if ddp and sampler is not None: sampler.set_epoch(step)

    if is_rank0(rank):
        # JIT modules support state_dict; reloader will apply it to a fresh JIT module.
        enc_sd = (encoder.module if ddp else encoder).state_dict()
        aux_sd = (aux_head.module if ddp else aux_head).state_dict()
        torch.save({"step": step, "encoder": enc_sd, "aux_head": aux_sd,
                     "args": vars(args)},
                    out_dir / "ckpt_last.pt")
        writer.close()
        print(f"done. saved to {out_dir / 'ckpt_last.pt'}")
    if ddp:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
