"""Temporal cosine curve sweep over DIFF/aux/TOKEN variants.

For each variant, loads the trunk, encodes ``--n-clips`` clips from
``--split``, mean-pools spatial tokens, and computes
``cos(f_t, f_{t+k})`` averaged over t and batch for k ∈ {1, 2, 4, 8, 16}.

Outputs runs/sweep/temporal_cosine/summary.md + raw.jsonl.
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
from comparison.extractors.diff_extractor import DiffExtractor
from comparison.extractors.token_extractor import TokenExtractor


VARIANTS = [
    # (name, family, ckpt_path)
    ("DIFF-2cam-large-noact (baseline, no aux)",
     "diff", "runs/sweep/variants/v6_large_noact_taskheld/ckpt_last.pt"),
    ("DIFF + aux λ=0.01",
     "diff", "runs/sweep/aux_loss_tiny/aux_v6_large_noact_taskheld_lam0p01/ckpt_last.pt"),
    ("DIFF + aux λ=0.025",
     "diff", "runs/sweep/aux_loss_tiny/aux_v6_large_noact_taskheld_lam0p025/ckpt_last.pt"),
    ("DIFF + aux λ=0.05 ★",
     "diff", "runs/sweep/aux_loss_extension/aux_v6_large_noact_taskheld_lam0p05/ckpt_last.pt"),
    ("DIFF + aux λ=0.1",
     "diff", "runs/sweep/aux_loss/aux_v6_large_noact_taskheld_lam0p1/ckpt_last.pt"),
    ("DIFF + aux λ=0.15",
     "diff", "runs/sweep/aux_loss_extension/aux_v6_large_noact_taskheld_lam0p15/ckpt_last.pt"),
    ("DIFF + aux λ=0.3",
     "diff", "runs/sweep/aux_loss_extension/aux_v6_large_noact_taskheld_lam0p3/ckpt_last.pt"),
    ("DIFF + aux λ=1.0",
     "diff", "runs/sweep/aux_loss/aux_v6_large_noact_taskheld_lam1/ckpt_last.pt"),
    ("DIFF + aux λ=10",
     "diff", "runs/sweep/aux_loss/aux_v6_large_noact_taskheld_lam10/ckpt_last.pt"),
    ("DIFF-FULL (1cam) baseline",
     "diff", "runs/sweep/variants/v_mix_v10_noact/ckpt_last.pt"),
    ("DIFF-FULL (1cam) + aux λ=0.1",
     "diff", "runs/sweep/aux_loss_extension/aux_v_mix_v10_noact_lam0p1/ckpt_last.pt"),
    ("TOKEN-200k",
     "token", "runs/sweep/variants/v_mix_token_noact_200k/ckpt_last.pt"),
]

# V-JEPA 2 reference rows (from existing libero_4model results)
VJEPA_REFERENCE = [
    ("V-JEPA 2 frozen (ref)",        "vjepa2", None,
     [0.9565, 0.9101, 0.8856, 0.8928, 0.8606]),
    ("V-JEPA 2 + SSL (ref)",         "vjepa2", None,
     [0.9477, 0.8920, 0.8549, 0.8574, 0.8187]),
    ("V-JEPA 2 + aux (ref)",         "vjepa2", None,
     [0.9708, 0.9396, 0.8647, 0.7154, 0.4745]),
]

KS = (1, 2, 4, 8, 16)


@torch.no_grad()
def temporal_cosine_curve(features_per_clip_t: torch.Tensor) -> dict:
    """features_per_clip_t: (B, T, D), already pooled over spatial tokens."""
    B, T, D = features_per_clip_t.shape
    f = F.normalize(features_per_clip_t.float(), dim=-1)
    out = {}
    out_std = {}
    for k in KS:
        if k >= T:
            continue
        cos_per = (f[:, :-k] * f[:, k:]).sum(dim=-1)   # (B, T-k)
        out[k] = cos_per.mean().item()
        out_std[k] = cos_per.std().item()
    return out, out_std


def encode_n_clips(extractor, libero_cfg, split: str, n_clips: int,
                    device: torch.device, seed: int) -> torch.Tensor:
    """Returns pooled per-frame features: (n_clips, T, D)."""
    ds = LiberoVideoClips(
        split_path=split,
        datasets_dir=libero_cfg["datasets_dir"],
        clip_len=libero_cfg["clip_len"],
        cameras=libero_cfg["cameras"],
        resolution=libero_cfg["resolution"],
        seed=seed,
        return_actions=False,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2,
                        worker_init_fn=worker_init_fn)
    feats_list = []
    seen = 0
    for clip in loader:
        clip = clip.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                 enabled=device.type == "cuda"):
            f = extractor.extract(clip)  # (B, T, D)
        feats_list.append(f.float().cpu())
        seen += f.shape[0]
        if seen >= n_clips:
            break
    return torch.cat(feats_list, dim=0)[:n_clips]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/sweep/temporal_cosine")
    ap.add_argument("--split", default="data/splits_taskheld/test_indist.json")
    ap.add_argument("--n-clips", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # libero config — match what variants were trained on
    # (2cam variants: agentview+eye_in_hand; 1cam: agentview only)
    # We'll auto-detect from the ckpt's saved cfg.

    rows = []
    for name, family, ckpt_rel in VARIANTS:
        ckpt = REPO_ROOT / ckpt_rel
        if not ckpt.exists():
            print(f"[skip] {name}: ckpt missing")
            continue
        print(f"\n=== {name} ===")
        t0 = time.time()
        try:
            if family == "diff":
                extractor = DiffExtractor(str(ckpt), device)
            elif family == "token":
                extractor = TokenExtractor(str(ckpt), device)
            else:
                print(f"[skip] {name}: unknown family")
                continue
            # use the ckpt's own libero cfg for cameras/clip_len/resolution
            cfg = extractor._cfg
            libero_cfg = {
                "datasets_dir": cfg.data.datasets_dir,
                "clip_len": int(cfg.data.clip_len),
                "cameras": list(cfg.data.cameras),
                "resolution": int(cfg.data.resolution),
            }
            print(f"  clip_len={libero_cfg['clip_len']} cameras={libero_cfg['cameras']}")
            feats = encode_n_clips(extractor, libero_cfg, args.split,
                                     args.n_clips, device, seed=42)
            print(f"  encoded {feats.shape[0]} clips in {time.time()-t0:.1f}s, "
                  f"feats shape={tuple(feats.shape)}")
            curve, curve_std = temporal_cosine_curve(feats)
            row = {"name": name, "family": family,
                   "ckpt": str(ckpt_rel),
                   "n_clips": int(feats.shape[0]),
                   "cos": {str(k): float(v) for k, v in curve.items()},
                   "cos_std": {str(k): float(v) for k, v in curve_std.items()}}
            rows.append(row)
            for k in KS:
                if k in curve:
                    print(f"  cos(k={k:2d}) = {curve[k]:.4f} ± {curve_std[k]:.4f}")
        finally:
            try:
                del extractor
            except Exception:
                pass
            torch.cuda.empty_cache()

        # write incrementally
        with (out_dir / "raw.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- summary.md ----
    md = ["# Temporal cosine curves\n",
          f"`{args.split}`, {args.n_clips} clips, mean ± std over batch×time.\n",
          "Higher = features more similar across time gap k. "
          "DIFF baseline tends to ~1.0 (collapse); aux loss should push curve down.\n",
          "| Variant | k=1 | k=2 | k=4 | k=8 | k=16 |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        c = r["cos"]
        cells = [f"{c.get(str(k), float('nan')):.4f}" for k in KS]
        md.append(f"| {r['name']} | {' | '.join(cells)} |")
    # reference rows
    for name, fam, _, vals in VJEPA_REFERENCE:
        md.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")

    md.append("")
    md.append("## Caveats\n")
    md.append("- Cosine on mean-pooled features only; spatial-token-level patterns may differ.")
    md.append("- Per-frame std hides per-trajectory variance.")
    md.append("- DIFF rollout PSNR is much lower than V-JEPA 2's, so direct cosine comparison is "
              "between *different generative regimes*.")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {out_dir/'summary.md'} and {out_dir/'raw.jsonl'}")


if __name__ == "__main__":
    main()
