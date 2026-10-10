"""Web-DINO ViT-L + inverse-dynamics aux finetune on LIBERO (DDP-ready).

Parallels train_vjepa2_aux.py but for facebook/webssl-dino300m-full2b-224.
Single-process when run directly with python; multi-GPU when launched via
torchrun --nproc_per_node=N.

Usage:
    # 1 GPU
    CUDA_VISIBLE_DEVICES=3 uv run python scripts/train_webdino_aux_libero.py \\
        --out-dir runs/comparison/webdino_aux_finetune

    # 2 GPUs (DDP)
    CUDA_VISIBLE_DEVICES=2,3 uv run torchrun --nproc_per_node=2 \\
        scripts/train_webdino_aux_libero.py \\
        --out-dir runs/comparison/webdino_aux_finetune --batch-size 4
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

from data.libero_video import worker_init_fn
from data.build_train_dataset import build_train_dataset, DATASET_CHOICES
from training.aux_loss import InvDynAuxHead


HF_REPO = "facebook/webssl-dino300m-full2b-224"


def is_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def init_distributed():
    if not is_torchrun():
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def is_rank0(rank: int) -> bool:
    return rank == 0


def encode_frame_features(model, clip: torch.Tensor, native_size: int = 224) -> torch.Tensor:
    """clip: (B, T, C, H, W) in [0,1] → (B, T, D=1024) per-frame features.

    Mirrors WebDINOExtractor.extract but WITH gradients enabled so the
    encoder finetunes."""
    B, T, C, H, W = clip.shape
    if C > 3:
        clip = clip[:, :, :3]
        C = 3
    x = clip.reshape(B * T, C, H, W)
    if (H, W) != (native_size, native_size):
        x = F.interpolate(x, size=(native_size, native_size),
                            mode="bilinear", align_corners=False)
    # DDP wrapping unwraps automatically; call .module if needed
    out = model(x)
    tokens = out.last_hidden_state                  # (BT, 1+N, D)
    patch_tokens = tokens[:, 1:]                    # (BT, N, D)
    pooled = patch_tokens.mean(dim=1)               # (BT, D)
    return pooled.reshape(B, T, -1)                 # (B, T, D)


def lr_schedule(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hf-repo", default=HF_REPO)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="per-GPU batch size")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--clip-len", type=int, default=32)
    ap.add_argument("--native-image-size", type=int, default=224)
    ap.add_argument("--aux-lambda", type=float, default=1.0,
                    help="aux is the ONLY loss; default 1.0 (V-JEPA recipe parity)")
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

    # --- model ----------------------------------------------------------------
    from transformers import AutoModel
    try:
        model = AutoModel.from_pretrained(args.hf_repo, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModel.from_pretrained(args.hf_repo, torch_dtype=torch.bfloat16)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad = True
    model.train()

    feat_dim = int(model.config.hidden_size)
    train_ds, action_dim = build_train_dataset(
        args.dataset, clip_len=args.clip_len, seed=args.seed,
        resolution=128, return_actions=True)
    aux_head = InvDynAuxHead(in_dim=feat_dim, action_dim=action_dim).to(device)

    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    broadcast_buffers=False, find_unused_parameters=False)
        aux_head = DDP(aux_head, device_ids=[local_rank], output_device=local_rank)

    if is_rank0(rank):
        n_enc = sum(p.numel() for p in (model.module if ddp else model).parameters())
        n_aux = sum(p.numel() for p in (aux_head.module if ddp else aux_head).parameters())
        print(f"params: encoder={n_enc/1e6:.1f}M  aux={n_aux/1e6:.1f}M  "
              f"world_size={world_size}  per-GPU batch={args.batch_size}  "
              f"effective batch={args.batch_size * world_size}")

    # --- data -----------------------------------------------------------------
    # train_ds already built above for action_dim inference
    if ddp:
        sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=rank, shuffle=True)
    else:
        sampler = None
    loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=args.num_workers, worker_init_fn=worker_init_fn,
        drop_last=True, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    opt = torch.optim.AdamW(
        [*((model.module if ddp else model).parameters()),
         *((aux_head.module if ddp else aux_head).parameters())],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )

    # --- train ---------------------------------------------------------------
    step = 0
    t0 = time.time()
    while step < args.max_steps:
        for epoch_i, (clip, actions) in enumerate(loader):
            if step >= args.max_steps:
                break
            clip = clip.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)

            cur_lr = args.lr * lr_schedule(step, args.warmup_steps, args.max_steps)
            for g in opt.param_groups:
                g["lr"] = cur_lr

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                     enabled=device.type == "cuda"):
                feats = encode_frame_features(
                    model.module if ddp else model, clip,
                    native_size=args.native_image_size,
                )  # (B, T, D)
                pred = (aux_head.module if ddp else aux_head)(feats)   # (B, T-1, 7)
                target = actions[:, :-1, :action_dim]
                aux_loss = (pred - target).pow(2).mean()
                loss = args.aux_lambda * aux_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*((model.module if ddp else model).parameters()),
                 *((aux_head.module if ddp else aux_head).parameters())], 1.0,
            )
            opt.step()

            if is_rank0(rank) and step % args.log_interval == 0:
                ips = (step + 1) * args.batch_size * world_size / max(1e-6, time.time() - t0)
                print(f"step {step:6d} | aux {aux_loss.item():.4f} | "
                      f"lr {cur_lr:.2e} | clips/s {ips:.1f}")
                writer.add_scalar("train/aux", aux_loss.item(), step)
                writer.add_scalar("train/lr", cur_lr, step)

            step += 1
        if ddp and sampler is not None:
            sampler.set_epoch(step)   # reshuffle for next pass

    # --- save -----------------------------------------------------------------
    if is_rank0(rank):
        enc_sd = (model.module if ddp else model).state_dict()
        aux_sd = (aux_head.module if ddp else aux_head).state_dict()
        torch.save({
            "step": step,
            "encoder": enc_sd,
            "aux_head": aux_sd,
            "args": vars(args),
        }, out_dir / "ckpt_last.pt")
        writer.close()
        print(f"done. saved to {out_dir / 'ckpt_last.pt'}")

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
