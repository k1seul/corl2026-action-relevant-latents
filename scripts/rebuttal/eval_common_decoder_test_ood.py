"""Deterministic test-OOD evaluation of the three common decoders (§6).

Loads each `decoder_{backbone}_frozen/ckpt_last.pt`, freezes it, runs it on
a fixed slice of the test-OOD split, and reports MSE / LPIPS / PSNR. The
frame set is recorded with an exact SHA-256 of the ordered
(suite, file, demo, frame_within_clip) tuples.

Deterministic: same encoder → same clip iteration order → same result on
any GPU/machine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "rebuttal"))

from comparison.decoders import PixelDecoder
from eval_matched_control import build_extractor


BACKBONES = ("vjepa2", "videomae", "webdino")


def frame_list_sha(ep_ids, clip_len):
    """SHA-256 over the ordered (suite, file, demo, frame_idx) list.

    ep_ids: list of (suite, file, demo). We record every clip_len frame per
    episode so future re-runs on the same encoder / clip_len are bit-identical.
    """
    h = hashlib.sha256()
    h.update(f"clip_len={clip_len}\n".encode())
    for suite, fname, demo in ep_ids:
        for f_i in range(clip_len):
            h.update(f"{suite}|{fname}|{demo}|frame={f_i}\n".encode())
    return h.hexdigest()


@torch.no_grad()
def run_one(backbone, ckpt_dir, args, device, lpips_fn):
    from data.libero_video import LiberoVideoClips, worker_init_fn

    print(f"\n=== {backbone} ===", flush=True)
    t0 = time.time()
    extractor = build_extractor(
        backbone, device, None,
        external_models_dir=REPO_ROOT / "datasets/external_models",
        clip_len=32,
    )
    feat_dim = int(extractor.meta.feat_dim)
    print(f"[{backbone}] extractor built {time.time()-t0:.1f}s feat_dim={feat_dim}", flush=True)

    # Load decoder
    ckpt_path = ckpt_dir / f"decoder_{backbone}_frozen" / "ckpt_last.pt"
    if not ckpt_path.exists():
        print(f"[{backbone}] SKIP — no ckpt at {ckpt_path}")
        return None
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    decoder = PixelDecoder(latent_dim=feat_dim).to(device).eval()
    decoder.load_state_dict(state["decoder"])
    print(f"[{backbone}] decoder loaded from {ckpt_path} (step={state.get('step','?')})")

    # Test-OOD dataset (deterministic — no shuffle, no random)
    ds = LiberoVideoClips(
        split_path=str(REPO_ROOT / "runs/corl2026_rebuttal/splits/test_ood.json"),
        datasets_dir=str(REPO_ROOT / "datasets/libero_downloaded"),
        clip_len=32, cameras=["agentview_rgb"], resolution=128,
        seed=0, return_actions=False,
    )
    n = min(args.n_episodes, len(ds))
    subset = Subset(ds, list(range(n)))
    ep_ids = [(ds.episodes[i]["suite"], ds.episodes[i]["file"], ds.episodes[i]["demo"])
              for i in range(n)]
    frame_sha = frame_list_sha(ep_ids, clip_len=32)
    print(f"[{backbone}] n_episodes={n} frame_list_sha256={frame_sha}", flush=True)

    loader = DataLoader(
        subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn,
        drop_last=False, pin_memory=True,
    )

    mse_sum = 0.0; lpips_sum = 0.0; n_frames_total = 0
    t0 = time.time()
    for batch in loader:
        clip = batch[0] if isinstance(batch, (list, tuple)) else batch
        clip = clip.to(device, non_blocking=True)
        if clip.shape[2] > 3:
            clip = clip[:, :, :3]
        B, T, C, H, W = clip.shape
        # bfloat16 encoder for parity with training
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                 enabled=device.type == "cuda"):
            feats = extractor.extract(clip)
        feats = feats.float()
        recon = decoder(feats)                                 # (B, T, 3, H, W)
        target = clip                                          # (B, T, 3, H, W)
        # MSE — pixel-mean per-batch; accumulate over frames
        mse_per_frame = ((recon - target) ** 2).mean(dim=(2, 3, 4))    # (B, T)
        mse_sum += mse_per_frame.sum().item()
        # LPIPS — per-frame, [-1, 1]
        recon_flat = recon.reshape(B * T, 3, recon.shape[-2], recon.shape[-1])
        target_flat = target.reshape(B * T, 3, H, W)
        lp = lpips_fn(recon_flat * 2 - 1, target_flat * 2 - 1).squeeze()  # (B*T,)
        lpips_sum += lp.sum().item()
        n_frames_total += B * T

    dur = time.time() - t0
    mse_mean   = mse_sum   / n_frames_total
    lpips_mean = lpips_sum / n_frames_total
    psnr_mean  = -10.0 * (mse_mean if mse_mean > 1e-10 else 1e-10).__abs__()
    import math
    psnr_mean = -10.0 * math.log10(max(mse_mean, 1e-10))
    print(f"[{backbone}] n_frames={n_frames_total} MSE={mse_mean:.4f} "
          f"LPIPS={lpips_mean:.4f} PSNR={psnr_mean:.2f}dB ({dur:.1f}s)")

    return {
        "backbone": backbone,
        "ckpt": str(ckpt_path),
        "ckpt_step": int(state.get("step", -1)),
        "n_episodes": n,
        "n_frames": n_frames_total,
        "clip_len": 32,
        "resolution": 128,
        "frame_list_sha256": frame_sha,
        "test_split": "runs/corl2026_rebuttal/splits/test_ood.json",
        "test_split_sha256_head_note": "42f26dac66acbaf0c5a153f15605b09f9f9ea772c5647b7d20518874f2c769ca",
        "test_mse":   float(mse_mean),
        "test_lpips": float(lpips_mean),
        "test_psnr_db": float(psnr_mean),
    }


def build_report(records, out_md):
    lines = ["# COMMON_DECODER_TEST_OOD.md — deterministic test-OOD decoder eval (§6)\n\n"]
    lines.append("> **PACKAGE**: `ORIGINAL_CLUSTER_SECONDARY_EVIDENCE`. "
                 "MSE / LPIPS / PSNR here are conditional on the "
                 "original-cluster decoder training + this deterministic "
                 "test-OOD frame set (SHA-256 below); **do not merge** with "
                 "the submitted paper's cross-family Figure 1 or with GCP "
                 "decoder metrics.\n\n")
    lines.append("Deterministic evaluation of each frozen-backbone common "
                 "decoder on a fixed slice of the test-OOD split. Reported "
                 "values are one-pass MSE / LPIPS / PSNR — **not** train-log "
                 "tail means (that earlier train-set summary in "
                 "`COMMON_DECODER.md` is superseded for reporting purposes).\n\n")

    if records:
        r0 = records[0]
        lines.append(f"* Test split: `{r0['test_split']}` (sha256 head "
                     f"`{r0['test_split_sha256_head_note'][:16]}…`)\n")
        lines.append(f"* n_episodes evaluated: {r0['n_episodes']} × 32 frames "
                     f"= {r0['n_frames']} frames per backbone\n")
        lines.append(f"* Frame-list SHA-256 (ordered (suite,file,demo,frame_idx) "
                     f"tuples, clip_len=32): `{r0['frame_list_sha256']}`\n")
        lines.append(f"* Decoder: `src/comparison/decoders/pixel_decoder.py` "
                     f"(17M params, trained 10 000 steps in the common-decoder "
                     f"training run — see `COMMON_DECODER.md` for training curves)\n")
        lines.append(f"* Encoder: frozen public backbone, bfloat16 forward\n\n")

    lines.append("## Test-OOD decoder metrics (one-shot, deterministic)\n\n")
    lines.append("| Backbone | Test-OOD MSE | Test-OOD LPIPS | Test-OOD PSNR (dB) | n_frames | ckpt_step |\n")
    lines.append("|----------|--------------|-----------------|---------------------|----------|-----------|\n")
    for bb in BACKBONES:
        for r in records:
            if r["backbone"] == bb:
                lines.append(f"| {bb} | {r['test_mse']:.4f} | {r['test_lpips']:.4f} | "
                             f"{r['test_psnr_db']:.2f} | {r['n_frames']} | {r['ckpt_step']} |\n")

    lines.append("\n## Label\n\n")
    lines.append("**Three-backbone common-decoder sensitivity.** The table above "
                 "is a sensitivity check across three vision backbones under an "
                 "identical decoder protocol. It is NOT a repair of the submitted "
                 "cross-family Figure 1 (SigLIP2, LAPA, SDXL VAE, Cosmos, and "
                 "other backbones are not included here). Absolute values must "
                 "not be merged with the submitted matrix.\n")

    Path(out_md).write_text("".join(lines))
    print(f"wrote {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root",
                    default=str(REPO_ROOT / "runs/corl2026_rebuttal/checkpoints"))
    ap.add_argument("--out-jsonl",
                    default=str(REPO_ROOT / "runs/corl2026_rebuttal/results/common_decoder_test_ood.jsonl"))
    ap.add_argument("--out-md",
                    default=str(REPO_ROOT / "runs/corl2026_rebuttal/reports/COMMON_DECODER_TEST_OOD.md"))
    ap.add_argument("--n-episodes", type=int, default=1300, help="deterministic first-N of test_ood.json (all 1300 by default)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    args = ap.parse_args()

    device = torch.device(args.device)
    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    for p in lpips_fn.parameters(): p.requires_grad = False

    records = []
    for bb in args.backbones:
        r = run_one(bb, Path(args.ckpt_root), args, device, lpips_fn)
        if r is not None:
            records.append(r)
            with open(args.out_jsonl, "w") as f:
                for x in records: f.write(json.dumps(x) + "\n")

    build_report(records, args.out_md)


if __name__ == "__main__":
    main()
