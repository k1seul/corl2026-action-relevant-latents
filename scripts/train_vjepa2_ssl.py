"""V-JEPA2 SSL continued pretrain on LIBERO video.

Tests hypothesis A: does V-JEPA2's R²=0.58 ceiling come from domain
mismatch? Continue training V-JEPA2 (encoder + predictor + EMA target)
on LIBERO clips using V-JEPA2's own masked-prediction objective. No
action labels are used.

Simplifications vs official V-JEPA2 training:
  * Mask scheme: random 50/50 split (visible vs predict) over the
    flattened patch grid, instead of V-JEPA2's multi-block 3D mask.
    Simple but preserves the "predict masked features from visible
    features" SSL signal.
  * Single mask per batch (V-JEPA2 trains with 2 masks).
  * No multi-resolution, no curriculum.

Single-camera (agentview), 256×256 native, T=32 frames.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_vjepa2_ssl.py \\
        --out-dir runs/comparison/vjepa2_ssl_pretrain \\
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

EXTERNAL_DIR = Path(
    "external_models")

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _ensure_vjepa2_on_path() -> None:
    p = str(EXTERNAL_DIR / "vjepa2")
    if p not in sys.path:
        sys.path.insert(0, p)


def preprocess(clip: torch.Tensor, img_size: int = 256) -> torch.Tensor:
    B, T, C, H, W = clip.shape
    if C > 3:
        clip = clip[:, :, :3]; C = 3
    clip = clip.reshape(B * T, C, H, W)
    clip = F.interpolate(clip, size=(img_size, img_size),
                         mode="bilinear", align_corners=False)
    mean = torch.tensor(_IMAGENET_MEAN, device=clip.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=clip.device).view(1, 3, 1, 1)
    clip = (clip - mean) / std
    clip = clip.reshape(B, T, C, img_size, img_size)
    return clip.permute(0, 2, 1, 3, 4).contiguous()


def random_mask_split(B: int, N: int, mask_ratio: float,
                       device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Random per-batch split of N tokens into (context, predict).

    Returns:
      masks_x : (B, N_context) — token indices to KEEP as context
      masks_y : (B, N_pred)    — token indices to PREDICT
    """
    n_pred = int(N * mask_ratio)
    n_ctx = N - n_pred
    rand = torch.rand(B, N, device=device)
    perm = rand.argsort(dim=1)
    return perm[:, :n_ctx], perm[:, n_ctx:]


def lr_schedule(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def momentum_at(step: int, total: int,
                m_start: float = 0.996, m_end: float = 1.0) -> float:
    return m_start + (m_end - m_start) * min(step / total, 1.0)


@torch.no_grad()
def ema_update(target: torch.nn.Module, online: torch.nn.Module, m: float) -> None:
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.mul_(m).add_(p_o, alpha=1.0 - m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ckpt", default=str(EXTERNAL_DIR / "vjepa2_ckpts/vitl.pt"))
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--clip-len", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--ema-start", type=float, default=0.996)
    ap.add_argument("--dataset", choices=("libero", "robomimic", "metaworld", "calvin"), default="libero")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-interval", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir / "tb"))

    _ensure_vjepa2_on_path()
    from src.models import vision_transformer as vit_encoder
    from src.models import predictor as vit_predictor

    # ── instantiate encoder, predictor, target_encoder ────────────────────────
    enc_kwargs = dict(
        patch_size=16, img_size=(args.img_size, args.img_size),
        num_frames=args.clip_len, tubelet_size=2,
        use_sdpa=True, use_silu=False, wide_silu=True,
        uniform_power=False, use_rope=True,
    )
    encoder = vit_encoder.vit_large(**enc_kwargs)
    target_encoder = vit_encoder.vit_large(**enc_kwargs)

    # match official V-JEPA2 ViT-L predictor config
    # (see vjepa2/src/hub/backbones.py::_make_vjepa2_model)
    pred_kwargs = dict(
        img_size=(args.img_size, args.img_size),
        patch_size=16, num_frames=args.clip_len, tubelet_size=2,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=384,
        depth=12, num_heads=12,
        use_mask_tokens=True, num_mask_tokens=10,
        use_sdpa=True, use_silu=False, wide_silu=True,
        uniform_power=False, use_rope=True,
    )
    predictor = vit_predictor.vit_predictor(**pred_kwargs)

    # ── load ckpt ─────────────────────────────────────────────────────────────
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    def _clean(sd):
        return {k.replace("module.", "").replace("backbone.", ""): v
                for k, v in sd.items()}

    msg_e = encoder.load_state_dict(_clean(state["encoder"]), strict=False)
    msg_t = target_encoder.load_state_dict(_clean(state["target_encoder"]),
                                            strict=False)
    msg_p = predictor.load_state_dict(_clean(state["predictor"]), strict=False)
    print(f"loaded encoder: missing={len(msg_e.missing_keys)} "
          f"unexpected={len(msg_e.unexpected_keys)}")
    print(f"loaded target:  missing={len(msg_t.missing_keys)} "
          f"unexpected={len(msg_t.unexpected_keys)}")
    print(f"loaded pred:    missing={len(msg_p.missing_keys)} "
          f"unexpected={len(msg_p.unexpected_keys)}")

    encoder = encoder.to(device).train()
    target_encoder = target_encoder.to(device).eval()
    predictor = predictor.to(device).train()
    for p in target_encoder.parameters():
        p.requires_grad = False

    n_e = sum(p.numel() for p in encoder.parameters())
    n_p = sum(p.numel() for p in predictor.parameters())
    print(f"params: encoder={n_e/1e6:.1f}M, predictor={n_p/1e6:.1f}M (target shared shape)")

    # ── data ─────────────────────────────────────────────────────────────────
    if args.dataset == "robomimic":
        from comparison.datasets.robomimic import RobomimicVideoClips
        train_ds = RobomimicVideoClips(
            split_path="data/splits_robomimic/train.json",
            datasets_dir="datasets/robomimic",
            clip_len=args.clip_len,
            cameras=["agentview_image"],
            resolution=128,
            seed=args.seed,
            return_actions=False,
        )
    elif args.dataset == "metaworld":
        from comparison.datasets.metaworld_lerobot import MetaWorldClips
        train_ds = MetaWorldClips(
            split_path="data/splits_metaworld/train.json",
            datasets_dir="external_models/metaworld_data/data",
            clip_len=args.clip_len,
            cameras=["observation.image"],
            resolution=128,
            seed=args.seed,
            return_actions=False,
        )
    elif args.dataset == "calvin":
        from comparison.datasets.calvin import CalvinClips
        train_ds = CalvinClips(
            split_path="data/splits_calvin_taskheld/train.json",
            datasets_dir=None,
            clip_len=args.clip_len,
            cameras=["rgb_static"],
            resolution=128,
            seed=args.seed,
            return_actions=False,
        )
    else:
        train_ds = LiberoVideoClips(
            split_path="data/splits_taskheld/train.json",
            datasets_dir="datasets",
            clip_len=args.clip_len,
            cameras=["agentview_rgb"],
            resolution=128,
            seed=args.seed,
            return_actions=False,
        )
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn,
        drop_last=True, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # ── optimizer (encoder + predictor; target is EMA-updated only) ───────────
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )

    # token-grid sizes
    H_p = args.img_size // 16
    T_tubes = args.clip_len // 2
    N_total = T_tubes * H_p * H_p
    print(f"token grid: T_tubes={T_tubes}, H_p={H_p}, N_total={N_total}")

    step = 0
    t0 = time.time()
    while step < args.max_steps:
        for clip in loader:
            if step >= args.max_steps:
                break
            clip = clip.to(device, non_blocking=True)
            x = preprocess(clip, args.img_size)             # (B, C, T, H, W)
            B = x.shape[0]

            cur_lr = args.lr * lr_schedule(step, args.warmup_steps, args.max_steps)
            for g in opt.param_groups:
                g["lr"] = cur_lr

            masks_x, masks_y = random_mask_split(B, N_total,
                                                  args.mask_ratio, device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                     enabled=device.type == "cuda"):
                # context: encoder sees only visible tokens
                z = encoder(x, masks=[masks_x])             # (B, N_ctx, D)
                # predictor: predict at masks_y given z + masks_x
                z_pred = predictor(z, [masks_x], [masks_y])  # (B, N_pred, D)
                # target: full clean encoder, gather at masks_y
                with torch.no_grad():
                    h = target_encoder(x)                   # (B, N_total, D)
                    # apply layer-norm per V-JEPA2 protocol
                    h = F.layer_norm(h, (h.size(-1),))
                    h_at_y = torch.gather(
                        h, 1,
                        masks_y.unsqueeze(-1).expand(-1, -1, h.shape[-1]),
                    )                                        # (B, N_pred, D)
                loss = F.smooth_l1_loss(z_pred, h_at_y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(predictor.parameters()), 1.0)
            opt.step()

            # EMA update target_encoder ← encoder
            m = momentum_at(step, args.max_steps, args.ema_start, 1.0)
            ema_update(target_encoder, encoder, m)

            if step % args.log_interval == 0:
                ips = (step + 1) * args.batch_size / max(1e-6, time.time() - t0)
                print(f"step {step:6d} | loss {loss.item():.4f} | "
                      f"lr {cur_lr:.2e} | m {m:.4f} | clips/s {ips:.1f}")
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr", cur_lr, step)
                writer.add_scalar("train/m", m, step)
            step += 1

    torch.save({
        "step": step,
        "encoder": encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "args": vars(args),
    }, out_dir / "ckpt_last.pt")
    writer.close()
    print(f"done. saved to {out_dir / 'ckpt_last.pt'}")


if __name__ == "__main__":
    main()
