"""Evaluate pixel decoder reconstruction PSNR on held-out LIBERO clips.

Per-backbone score for the (PSNR, action-R^2) Pareto figure. Uses the
test split (data/splits_taskheld/test.json) — held-out tasks, fresh
episodes — and the same probe-eval clip count as the action probe so
the two axes are measured on comparable data.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_pixel_rollout.py \\
        --family vjepa2 --variant vjepa2_aux \\
        --encoder-ckpt runs/comparison/vjepa2_aux_finetune/ckpt_last.pt \\
        --decoder-ckpt runs/comparison/decoder_vjepa2_aux/ckpt_last.pt \\
        --out-jsonl runs/full_results/decoder_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.libero_video import LiberoVideoClips, worker_init_fn
from comparison.decoders import PixelDecoder
from comparison.eval_pipeline import build_extractor


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--encoder-ckpt", required=True)
    ap.add_argument("--decoder-ckpt", required=True)
    ap.add_argument("--split", default="data/splits_taskheld/test.json")
    ap.add_argument("--num-clips", type=int, default=200)
    ap.add_argument("--clip-len", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-jsonl", default="runs/full_results/decoder_results.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hf-repo", default=None)
    ap.add_argument("--img-size", type=int, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)

    spec = {"family": args.family, "name": args.variant, "ckpt": args.encoder_ckpt}
    if args.hf_repo:
        spec["hf_repo"] = args.hf_repo
    if args.img_size is not None:
        spec["img_size"] = args.img_size
    extractor = build_extractor(spec, device)
    feat_dim = int(extractor.meta.feat_dim)
    print(f"extractor: {extractor.meta.family}/{extractor.meta.variant} feat_dim={feat_dim}")

    decoder = PixelDecoder(latent_dim=feat_dim).to(device).eval()
    dec_ck = torch.load(args.decoder_ckpt, map_location="cpu", weights_only=False)
    decoder.load_state_dict(dec_ck["decoder"])
    print(f"decoder loaded from {args.decoder_ckpt}")

    ds = LiberoVideoClips(
        split_path=args.split, datasets_dir="datasets",
        clip_len=args.clip_len, cameras=["agentview_rgb"], resolution=128,
        seed=0, return_actions=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=2, worker_init_fn=worker_init_fn)

    mse_total = 0.0
    psnr_total = 0.0
    n_frames = 0
    n_clips = 0
    t0 = time.time()
    for batch in loader:
        clip = batch[0] if isinstance(batch, (list, tuple)) else batch
        clip = clip.to(device, non_blocking=True)
        if clip.shape[2] > 3:
            clip = clip[:, :, :3]
        B, T, C, H, W = clip.shape

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                 enabled=device.type == "cuda"):
            feats = extractor.extract(clip)
        feats = feats.float()
        recon = decoder(feats)
        diff = (recon - clip).reshape(B * T, -1)
        per_frame_mse = diff.pow(2).mean(dim=-1)
        per_frame_psnr = -10.0 * torch.log10(per_frame_mse.clamp(min=1e-10))

        mse_total += per_frame_mse.sum().item()
        psnr_total += per_frame_psnr.sum().item()
        n_frames += per_frame_mse.numel()
        n_clips += B
        if n_clips >= args.num_clips:
            break

    avg_mse = mse_total / max(1, n_frames)
    avg_psnr = psnr_total / max(1, n_frames)
    row = {
        "variant": args.variant,
        "family": args.family,
        "encoder_ckpt": args.encoder_ckpt,
        "decoder_ckpt": args.decoder_ckpt,
        "split": args.split,
        "n_clips": n_clips,
        "n_frames": n_frames,
        "psnr_test_mean": avg_psnr,
        "mse_test_mean": avg_mse,
        "seconds": time.time() - t0,
    }
    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"{args.variant}: PSNR={avg_psnr:.2f} dB  MSE={avg_mse:.4f}  "
          f"(n_clips={n_clips} n_frames={n_frames}, {time.time()-t0:.1f}s)")
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
