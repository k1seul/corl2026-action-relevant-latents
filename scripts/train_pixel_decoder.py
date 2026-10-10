"""Generic pixel-decoder trainer for any encoder-only backbone.

Uses the existing extractor abstraction (so backbone-specific preprocess +
loading is unchanged). Encoder is frozen — only the PixelDecoder learns.

Option A (single-frame recon): map per-frame pooled feature -> 128x128 RGB.
Loss: MSE + lpips_weight * LPIPS(AlexNet).

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_pixel_decoder.py \\
        --family vjepa2 --variant vjepa2_aux \\
        --encoder-ckpt runs/comparison/vjepa2_aux_finetune/ckpt_last.pt \\
        --out-dir runs/comparison/decoder_vjepa2_aux \\
        --batch-size 16 --max-steps 10000

Supported families: vjepa2, vjepa2_1, webdino, siglip2, lapa, sd3vae, cosmos_tokenizer, videomae.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "rebuttal"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os as _os
from data.libero_video import LiberoVideoClips, worker_init_fn
from comparison.decoders import PixelDecoder
# For P6 frozen mode use the light-weight extractor factory in
# scripts/rebuttal/eval_matched_control.py (only handles vjepa2 / videomae /
# webdino) — avoids the DIFF/TOKEN chain that pulls in
# `models.block_causal_transformer`.
from eval_matched_control import build_extractor as _build_wave1_extractor


def build_extractor(spec, device):
    family = spec.get("family")
    if family in ("vjepa2", "videomae", "webdino"):
        return _build_wave1_extractor(
            family, device, finetuned_encoder_state=None,
            external_models_dir=Path(_os.environ.get(
                "EXTERNAL_MODELS_DIR",
                str(REPO_ROOT / "datasets/external_models"))),
            clip_len=spec.get("num_frames", 32),
        )
    from comparison.eval_pipeline import build_extractor as _b
    return _b(spec, device)


def lr_schedule(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--family", required=True,
                    help="vjepa2|vjepa2_1|webdino|siglip2|lapa|sd3vae|cosmos_tokenizer|videomae")
    ap.add_argument("--variant", default=None, help="descriptive name for the run")
    ap.add_argument("--encoder-ckpt", default=None,
                    help="path to aux-finetune ckpt (with 'encoder' key); "
                         "omit for the frozen public encoder (P6 common decoder)")
    ap.add_argument("--frozen", action="store_true",
                    help="use the public frozen backbone with no fine-tuned "
                         "ckpt override (skips --encoder-ckpt)")
    ap.add_argument("--hf-repo", default=None, help="optional HF repo override for some families")
    ap.add_argument("--img-size", type=int, default=None,
                    help="optional override of the extractor's native image size")
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--clip-len", type=int, default=32)
    ap.add_argument("--lpips-weight", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-interval", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir / "tb"))

    if args.frozen and args.encoder_ckpt:
        raise SystemExit("--frozen and --encoder-ckpt are mutually exclusive")
    if not args.frozen and not args.encoder_ckpt:
        raise SystemExit("pass either --frozen (P6 common decoder over the "
                          "public frozen backbone) or --encoder-ckpt (paper's "
                          "aux-finetune)")

    spec = {
        "family": args.family,
        "name": args.variant or f"{args.family}_{'frozen' if args.frozen else 'aux'}",
    }
    if args.encoder_ckpt:
        spec["ckpt"] = args.encoder_ckpt
    if args.hf_repo:
        spec["hf_repo"] = args.hf_repo
    if args.img_size is not None:
        spec["img_size"] = args.img_size

    extractor = build_extractor(spec, device)
    feat_dim = int(extractor.meta.feat_dim)
    print(f"extractor: {extractor.meta.family}/{extractor.meta.variant} feat_dim={feat_dim}")

    decoder = PixelDecoder(latent_dim=feat_dim).to(device)
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"decoder: {n_dec/1e6:.2f}M params")

    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    import os
    train_ds = LiberoVideoClips(
        split_path=os.environ.get("LIBERO_SPLIT_PATH", "data/splits_taskheld/train.json"),
        datasets_dir=os.environ.get("LIBERO_DATASETS_DIR", "datasets"),
        clip_len=args.clip_len, cameras=["agentview_rgb"], resolution=128,
        seed=args.seed, return_actions=False,
    )
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn,
        drop_last=True, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    opt = torch.optim.AdamW(decoder.parameters(),
                            lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)

    step = 0
    t0 = time.time()
    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break
            clip = batch[0] if isinstance(batch, (list, tuple)) else batch
            clip = clip.to(device, non_blocking=True)
            if clip.shape[2] > 3:
                clip = clip[:, :, :3]
            B, T, C, H, W = clip.shape

            cur_lr = args.lr * lr_schedule(step, args.warmup_steps, args.max_steps)
            for g in opt.param_groups:
                g["lr"] = cur_lr

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                     enabled=device.type == "cuda"):
                feats = extractor.extract(clip)
            feats = feats.float()
            recon = decoder(feats)

            target = clip
            mse = F.mse_loss(recon, target)
            recon_flat = recon.reshape(B * T, 3, recon.shape[-2], recon.shape[-1])
            target_flat = target.reshape(B * T, 3, H, W)
            lp = lpips_fn(recon_flat * 2 - 1, target_flat * 2 - 1).mean()
            loss = mse + args.lpips_weight * lp

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            opt.step()

            if step % args.log_interval == 0:
                with torch.no_grad():
                    psnr = -10.0 * torch.log10(mse.clamp(min=1e-10))
                ips = (step + 1) * args.batch_size / max(1e-6, time.time() - t0)
                print(f"step {step:6d} | mse {mse.item():.4f} | lpips {lp.item():.4f} | "
                      f"psnr {psnr.item():.2f} | lr {cur_lr:.2e} | clips/s {ips:.1f}")
                writer.add_scalar("train/mse", mse.item(), step)
                writer.add_scalar("train/lpips", lp.item(), step)
                writer.add_scalar("train/psnr", psnr.item(), step)
                writer.add_scalar("train/lr", cur_lr, step)
            step += 1

    torch.save({"step": step, "decoder": decoder.state_dict(),
                 "args": vars(args)},
                out_dir / "ckpt_last.pt")
    writer.close()
    print(f"done. saved to {out_dir / 'ckpt_last.pt'}")


if __name__ == "__main__":
    main()
