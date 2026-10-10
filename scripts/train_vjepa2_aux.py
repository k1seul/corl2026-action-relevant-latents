"""V-JEPA2 ViT-L + inverse-dynamics aux loss fine-tune on LIBERO.

Tests hypothesis B from the V-JEPA2 probe-capacity discussion: does
V-JEPA2 lack action signal because of its self-supervised objective?
If end-to-end fine-tuning with our aux loss recovers ~0.85 OOD R²,
the features had the structure but needed to be re-shuffled.

Single-camera (agentview only) — V-JEPA2 was pretrained on natural
single-view video; 2-cam stack is unfair to it.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_vjepa2_aux.py \\
        --out-dir runs/comparison/vjepa2_aux_finetune \\
        --max-steps 30000 --batch-size 4 --lr 5e-5
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.libero_video import LiberoVideoClips, worker_init_fn
from training.aux_loss import InvDynAuxHead

import os

EXTERNAL_DIR = Path(os.environ.get(
    "EXTERNAL_MODELS_DIR",
    "external_models"))

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _ensure_vjepa2_on_path() -> None:
    p = str(EXTERNAL_DIR / "vjepa2")
    if p not in sys.path:
        sys.path.insert(0, p)


def preprocess(clip: torch.Tensor, img_size: int = 256) -> torch.Tensor:
    """Convert (B, T, C, H, W) [0,1] → V-JEPA2 (B, C, T, H_v, W_v)."""
    B, T, C, H, W = clip.shape
    if C > 3:
        clip = clip[:, :, :3]
        C = 3
    clip = clip.reshape(B * T, C, H, W)
    clip = F.interpolate(clip, size=(img_size, img_size),
                         mode="bilinear", align_corners=False)
    mean = torch.tensor(_IMAGENET_MEAN, device=clip.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=clip.device).view(1, 3, 1, 1)
    clip = (clip - mean) / std
    clip = clip.reshape(B, T, C, img_size, img_size)
    return clip.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)


def encode_frame_features(encoder, clip: torch.Tensor,
                           img_size: int = 256, tubelet: int = 2) -> torch.Tensor:
    """Run V-JEPA2 encoder, return per-frame features (B, T, D)."""
    x = preprocess(clip, img_size)
    feats = encoder(x)                                  # (B, num_patches, D)
    B = x.shape[0]
    T = clip.shape[1]
    H_p = img_size // 16
    T_tubes = T // tubelet
    feats = feats.reshape(B, T_tubes, H_p * H_p, -1).mean(dim=2)
    feats = feats.repeat_interleave(tubelet, dim=1)     # (B, T, D)
    return feats[:, :T]


def lr_schedule(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ckpt", default=str(EXTERNAL_DIR / "vjepa2_ckpts/vitl.pt"))
    ap.add_argument("--from-scratch", action="store_true",
                    help="skip ckpt loading; train ViT-L from random init")
    ap.add_argument("--dataset", choices=("libero", "robomimic", "metaworld", "calvin"), default="libero")
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--clip-len", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--aux-lambda", type=float, default=1.0)  # aux is the ONLY loss
    ap.add_argument("--id-mask-p", type=float, default=1.0,
                    help="fraction of samples per batch to apply ID loss to "
                         "(sparsity sweep). 1.0 = all samples (default), 0.10 = 10%%. "
                         "Per spec: loss = (mask * per_sample_loss).sum() / mask.sum().clamp(min=1).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-interval", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--libero-datasets-dir", default="datasets",
                    help="root that contains libero_<suite>/*.hdf5")
    ap.add_argument("--libero-split-path", default="data/splits_taskheld/train.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir / "tb"))

    # ── build V-JEPA2 ViT-L encoder, load weights ─────────────────────────────
    _ensure_vjepa2_on_path()
    from src.models import vision_transformer as vit_encoder
    encoder = vit_encoder.vit_large(
        patch_size=16, img_size=(args.img_size, args.img_size),
        num_frames=args.clip_len, tubelet_size=2,
        use_sdpa=True, use_silu=False, wide_silu=True,
        uniform_power=False, use_rope=True,
    )
    if args.from_scratch:
        print("RANDOM INIT (--from-scratch): skipping ckpt load")
    else:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = state.get("target_encoder", state.get("encoder", state))
        sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
        msg = encoder.load_state_dict(sd, strict=False)
        print(f"loaded {args.ckpt}: missing={len(msg.missing_keys)} "
              f"unexpected={len(msg.unexpected_keys)}")
    encoder = encoder.to(device)
    encoder.train()

    aux_head = InvDynAuxHead(in_dim=encoder.embed_dim, action_dim=7).to(device)

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_aux = sum(p.numel() for p in aux_head.parameters())
    print(f"params: encoder={n_enc/1e6:.1f}M, aux={n_aux/1e6:.1f}M")

    # ── data ──────────────────────────────────────────────────────────────────
    if args.dataset == "robomimic":
        from comparison.datasets.robomimic import RobomimicVideoClips
        train_ds = RobomimicVideoClips(
            split_path="data/splits_robomimic/train.json",
            datasets_dir="datasets/robomimic",
            clip_len=args.clip_len,
            cameras=["agentview_image"],   # 1cam for V-JEPA fairness
            resolution=128,
            seed=args.seed,
            return_actions=True,
        )
    elif args.dataset == "metaworld":
        from comparison.datasets.metaworld_lerobot import MetaWorldClips
        train_ds = MetaWorldClips(
            split_path="data/splits_metaworld/train.json",
            datasets_dir="external_models/metaworld_data/data",
            clip_len=args.clip_len,
            cameras=["observation.image"],  # MetaWorld is natively 1cam
            resolution=128,
            seed=args.seed,
            return_actions=True,
        )
    elif args.dataset == "calvin":
        from comparison.datasets.calvin import CalvinClips
        train_ds = CalvinClips(
            split_path="data/splits_calvin_taskheld/train.json",
            datasets_dir=None,  # adapter reads data_root from JSON
            clip_len=args.clip_len,
            cameras=["rgb_static"],  # 1cam for V-JEPA fairness
            resolution=128,
            seed=args.seed,
            return_actions=True,
        )
    else:
        # Session-2 STEP 3: honour REBUTTAL_CONDITION=shuffled_id by pulling
        # the within-task cross-episode derangement action source. Every
        # other condition (true_id, forward_dynamics, ...) leaves actions
        # untouched.
        action_source = None
        _cond = os.environ.get("REBUTTAL_CONDITION", "true_id")
        if _cond == "shuffled_id":
            from rebuttal.derangement_dataloader import build_plan_and_action_source
            _run_seed = int(os.environ.get("REBUTTAL_RUN_SEED", str(args.seed)))
            plan, action_source, _ = build_plan_and_action_source(
                run_seed=_run_seed,
                split_path=args.libero_split_path,
                datasets_dir=args.libero_datasets_dir,
                action_dim=7, strict=True,
            )
            print(f"[rebuttal] REBUTTAL_CONDITION=shuffled_id  "
                  f"derangement plan hash={plan.global_hash()}  "
                  f"tasks={len(plan.per_task)}  "
                  f"skipped_singletons={plan.skipped_singleton_tasks}",
                  flush=True)
        train_ds = LiberoVideoClips(
            split_path=args.libero_split_path,
            datasets_dir=args.libero_datasets_dir,
            clip_len=args.clip_len,
            cameras=["agentview_rgb"],
            resolution=128,
            seed=args.seed,
            return_actions=True,
            action_source_override=action_source,
        )
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn,
        drop_last=True, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # ── optimizer ────────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(aux_head.parameters()),
        lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )

    # ── train ────────────────────────────────────────────────────────────────
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
                feats = encode_frame_features(encoder, clip,
                                                img_size=args.img_size)  # (B, T, D)
                pred = aux_head(feats)                                    # (B, T-1, 7)
                target = actions[:, :-1, :7]
                # per-sample ID loss: mean over (T-1, action_dim) → (B,)
                per_sample = (pred - target).pow(2).mean(dim=tuple(range(1, pred.ndim)))
                if args.id_mask_p >= 1.0:
                    aux_loss = per_sample.mean()
                else:
                    B = per_sample.shape[0]
                    mask = (torch.rand(B, device=per_sample.device) < args.id_mask_p).float()
                    n_active = mask.sum().clamp(min=1.0)
                    aux_loss = (per_sample * mask).sum() / n_active
                loss = args.aux_lambda * aux_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(aux_head.parameters()), 1.0)
            opt.step()

            if step % args.log_interval == 0:
                ips = (step + 1) * args.batch_size / max(1e-6, time.time() - t0)
                print(f"step {step:6d} | aux {aux_loss.item():.4f} | "
                      f"lr {cur_lr:.2e} | clips/s {ips:.1f}")
                writer.add_scalar("train/aux", aux_loss.item(), step)
                writer.add_scalar("train/lr", cur_lr, step)

            step += 1

    # ── save ──────────────────────────────────────────────────────────────────
    torch.save({
        "step": step,
        "encoder": encoder.state_dict(),
        "aux_head": aux_head.state_dict(),
        "args": vars(args),
    }, out_dir / "ckpt_last.pt")
    writer.close()
    print(f"done. saved to {out_dir / 'ckpt_last.pt'}")


if __name__ == "__main__":
    main()
