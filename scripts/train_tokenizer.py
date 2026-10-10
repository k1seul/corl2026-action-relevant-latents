"""Train an FSQ image tokenizer on LIBERO frames.

Samples random frames from train clips, runs them through
encoder->FSQ->decoder, and minimizes pixel MSE. The trained tokenizer
is consumed by variant (B) — its discrete indices become the targets
for the transformer's token-CE objective.

Run:
    uv run python scripts/train_tokenizer.py --config configs/tokenizer.yaml
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
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.libero_video import LiberoVideoClips, worker_init_fn
from models.tokenizer import FSQTokenizer


def lr_schedule(step: int, warmup: int, max_steps: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def sample_frames(clip: torch.Tensor, frames_per_clip: int) -> torch.Tensor:
    """`(B, T, C, H, W)` -> `(B*frames_per_clip, C, H, W)` random frames."""
    B, T, C, H, W = clip.shape
    idx = torch.randint(0, T, (B, frames_per_clip), device=clip.device)
    arange = torch.arange(B, device=clip.device).unsqueeze(-1).expand_as(idx)
    flat = clip[arange, idx]  # (B, frames_per_clip, C, H, W)
    return flat.reshape(B * frames_per_clip, C, H, W)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/tokenizer.yaml")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args, overrides = ap.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    torch.manual_seed(cfg.train.seed)
    device = torch.device(args.device)

    out_dir = Path(cfg.train.ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    in_channels = 3 * len(cfg.data.cameras)
    train_ds = LiberoVideoClips(
        split_path=cfg.data.train_split,
        datasets_dir=cfg.data.datasets_dir,
        clip_len=cfg.data.clip_len,
        cameras=cfg.data.cameras,
        resolution=cfg.data.resolution,
        seed=cfg.train.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        worker_init_fn=worker_init_fn,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=cfg.train.num_workers > 0,
    )

    tok = FSQTokenizer(
        in_channels=in_channels,
        levels=list(cfg.model.levels),
        base_ch=cfg.model.base_ch,
    ).to(device)
    n_params = sum(p.numel() for p in tok.parameters() if p.requires_grad)
    print(f"params: {n_params/1e6:.2f}M | codebook: {tok.codebook_size}")

    opt = torch.optim.AdamW(
        tok.parameters(),
        lr=cfg.train.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.train.weight_decay,
    )

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.train.amp_dtype]
    use_amp = amp_dtype != torch.float32 and device.type == "cuda"

    step = 0
    t0 = time.time()
    tok.train()
    while step < cfg.train.max_steps:
        for clip in train_loader:
            if step >= cfg.train.max_steps:
                break
            clip = clip.to(device, non_blocking=True)
            frames = sample_frames(clip, cfg.train.frames_per_clip)

            for g in opt.param_groups:
                g["lr"] = cfg.train.lr * lr_schedule(step, cfg.train.warmup_steps, cfg.train.max_steps)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                recon, _idx = tok(frames)
                loss = (recon - frames).pow(2).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tok.parameters(), 1.0)
            opt.step()

            if step % cfg.train.log_interval == 0:
                cur_lr = opt.param_groups[0]["lr"]
                ips = (step + 1) * frames.shape[0] / max(1e-6, time.time() - t0)
                psnr = 10.0 * math.log10(1.0 / max(1e-12, loss.item()))
                print(f"step {step:6d} | mse {loss.item():.5f} | psnr {psnr:.2f} | lr {cur_lr:.2e} | imgs/s {ips:.0f}")
            step += 1

    torch.save(
        {"step": step, "model": tok.state_dict(), "config": OmegaConf.to_container(cfg)},
        out_dir / "tokenizer_last.pt",
    )
    print(f"done. saved to {out_dir/'tokenizer_last.pt'}")


if __name__ == "__main__":
    main()
