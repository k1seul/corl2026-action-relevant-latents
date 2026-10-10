"""P6: probe-architecture sensitivity on cached wave1 features.

For a given wave1 checkpoint (or the frozen backbone), encode features
ONCE, then run multiple probe variants against the same features:

  * ridge         — closed-form ridge regression (sklearn or numpy)
  * canonical_mlp — [D -> 256 -> 128 -> A] (matches paper canonical)
  * wider_mlp     — [D -> 1024 -> 512 -> A]
  * (causal_temporal deferred — features are already per-frame single-clip;
     a proper causal-context extractor exists but requires re-encoding
     with the past-only wrapper)

Emits rows to matched_controls.jsonl with new field ``probe_variant``.
Skips writing per-episode sufficient stats (only the canonical variant
needs those for the primary bootstrap).

Usage:
    python scripts/rebuttal/probe_sensitivity.py \\
        --backbone vjepa2 --condition frozen --training-seed -1 \\
        --ckpt "" --probe-variants ridge canonical_mlp wider_mlp
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "rebuttal"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_matched_control import (
    build_extractor, encode_slice, CanonicalMLPProbe, _flatten_frames,
    _split_sha, _sha256_file, DIM_NAMES,
)


def l2norm(x, ref=None):
    """L2-normalize x. If ref provided (train_x), fits on ref and applies to x."""
    return F.normalize(x, dim=-1)


def ridge_probe(train_x, train_y, test_x, test_y, alpha=1.0):
    """Closed-form ridge: (X'X + αI)^-1 X'y."""
    # solve on cpu float32 for numerical stability
    X = train_x.cpu().numpy().astype(np.float64)
    Y = train_y.cpu().numpy().astype(np.float64)
    D = X.shape[1]
    XtX = X.T @ X + alpha * np.eye(D)
    XtY = X.T @ Y
    W = np.linalg.solve(XtX, XtY)
    Xt = test_x.cpu().numpy().astype(np.float64)
    Yt = test_y.cpu().numpy().astype(np.float64)
    pred = Xt @ W
    diff = pred - Yt
    per_dim_mse = (diff ** 2).mean(axis=0)
    var = Yt.var(axis=0).clip(min=1e-12)
    per_dim_r2 = 1.0 - per_dim_mse / var
    return {
        "aggregate_test_mse": float(per_dim_mse.mean()),
        "aggregate_test_r2": float(per_dim_r2.mean()),
        "per_dim_mse": per_dim_mse.tolist(),
        "per_dim_r2": per_dim_r2.tolist(),
    }


def mlp_probe(train_x, train_y, test_x, test_y, hidden, seed, steps, lr, wd, bs, device):
    torch.manual_seed(seed)
    probe = CanonicalMLPProbe(in_dim=train_x.shape[-1],
                              action_dim=train_y.shape[-1],
                              hidden=hidden).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    n = train_x.shape[0]
    train_x = train_x.to(device); train_y = train_y.to(device)
    test_x = test_x.to(device); test_y = test_y.to(device)
    probe.train()
    for step in range(steps):
        idx = torch.randint(0, n, (bs,), device=device)
        loss = F.mse_loss(probe(train_x[idx]), train_y[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        pred = probe(test_x)
    diff = pred - test_y
    per_dim_mse = (diff ** 2).mean(dim=0).cpu().tolist()
    var = test_y.var(dim=0, unbiased=False).cpu().clamp_min(1e-12).tolist()
    per_dim_r2 = [1.0 - m / v for m, v in zip(per_dim_mse, var)]
    return {
        "aggregate_test_mse": float(sum(per_dim_mse) / len(per_dim_mse)),
        "aggregate_test_r2": float(sum(per_dim_r2) / len(per_dim_r2)),
        "per_dim_mse": per_dim_mse,
        "per_dim_r2": per_dim_r2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=("vjepa2", "videomae", "webdino"))
    ap.add_argument("--condition", required=True)
    ap.add_argument("--training-seed", type=int, default=-1)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--probe-variants", nargs="+",
                    default=["ridge", "canonical_mlp", "wider_mlp"])
    ap.add_argument("--probe-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--probe-steps", type=int, default=3000)
    ap.add_argument("--probe-lr", type=float, default=1e-3)
    ap.add_argument("--probe-wd", type=float, default=1e-4)
    ap.add_argument("--probe-batch", type=int, default=256)
    ap.add_argument("--probe-train-clips", type=int, default=800)
    ap.add_argument("--probe-eval-clips-test", type=int, default=1300)
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--split-train", default=str(REPO_ROOT / "runs/corl2026_rebuttal/splits/train.json"))
    ap.add_argument("--split-test-ood", default=str(REPO_ROOT / "runs/corl2026_rebuttal/splits/test_ood.json"))
    ap.add_argument("--datasets-dir", default=str(REPO_ROOT / "datasets/libero_downloaded"))
    ap.add_argument("--external-models-dir",
                    default=os.environ.get("EXTERNAL_MODELS_DIR",
                                             str(REPO_ROOT / "datasets/external_models")))
    ap.add_argument("--out-jsonl",
                    default=str(REPO_ROOT / "runs/corl2026_rebuttal/results/probe_sensitivity.jsonl"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    ckpt_sha = None; ckpt_step = None; finetuned_state = None
    if args.ckpt:
        ckpt_path = Path(args.ckpt).resolve()
        sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        finetuned_state = sd.get("encoder")
        ckpt_step = int(sd.get("step", 0))
        ckpt_sha = _sha256_file(ckpt_path)

    t0 = time.time()
    extractor = build_extractor(args.backbone, device, finetuned_state,
                                 external_models_dir=Path(args.external_models_dir),
                                 clip_len=32)
    print(f"[sens] extractor built in {time.time()-t0:.1f}s", flush=True)

    split_train = Path(args.split_train); split_test = Path(args.split_test_ood)
    dsd = Path(args.datasets_dir)
    train_feats, train_acts, _ = encode_slice(extractor, split_train, dsd, (0, args.probe_train_clips),
                                               clip_len=32, seed=0, batch_size=8, num_workers=2,
                                               device=device)
    test_feats, test_acts, _ = encode_slice(extractor, split_test, dsd, (0, args.probe_eval_clips_test),
                                             clip_len=32, seed=0, batch_size=8, num_workers=2,
                                             device=device)
    print(f"[sens] encoded train {tuple(train_feats.shape)}, test {tuple(test_feats.shape)}", flush=True)

    tx, ty = _flatten_frames(train_feats, train_acts)
    txe, tye = _flatten_frames(test_feats, test_acts)
    # l2 normalize (fit on train)
    tx = F.normalize(tx, dim=-1)
    txe = F.normalize(txe, dim=-1)

    split_sha = _split_sha(split_test)
    out = []
    for variant in args.probe_variants:
        if variant == "ridge":
            t = time.time()
            res = ridge_probe(tx, ty, txe, tye, alpha=args.ridge_alpha)
            print(f"[sens] ridge alpha={args.ridge_alpha}: R²={res['aggregate_test_r2']:.4f} ({time.time()-t:.1f}s)", flush=True)
            out.append({"backbone": args.backbone, "condition": args.condition,
                         "training_seed": args.training_seed, "probe_seed": None,
                         "probe_variant": variant, "ridge_alpha": args.ridge_alpha,
                         "clean_task_ood_r2": res["aggregate_test_r2"],
                         "clean_task_ood_mse": res["aggregate_test_mse"],
                         "per_dim_r2": dict(zip(DIM_NAMES, res["per_dim_r2"])),
                         "probe_feature_normalize": "l2",
                         "n_probe_train_pairs": int(tx.shape[0]),
                         "n_probe_eval_test_pairs": int(txe.shape[0]),
                         "finetuned_ckpt_sha256": ckpt_sha,
                         "finetuned_ckpt_step": ckpt_step,
                         "split_sha256": split_sha,
                         "extractor_meta": {"family": extractor.meta.family,
                                             "feat_dim": extractor.meta.feat_dim}})
        else:
            hidden = (256, 128) if variant == "canonical_mlp" else (1024, 512)
            for ps in args.probe_seeds:
                t = time.time()
                res = mlp_probe(tx, ty, txe, tye, hidden=hidden,
                                seed=ps, steps=args.probe_steps,
                                lr=args.probe_lr, wd=args.probe_wd,
                                bs=args.probe_batch, device=device)
                print(f"[sens] {variant} pseed={ps}: R²={res['aggregate_test_r2']:.4f} ({time.time()-t:.1f}s)", flush=True)
                out.append({"backbone": args.backbone, "condition": args.condition,
                             "training_seed": args.training_seed, "probe_seed": ps,
                             "probe_variant": variant, "probe_hidden": list(hidden),
                             "clean_task_ood_r2": res["aggregate_test_r2"],
                             "clean_task_ood_mse": res["aggregate_test_mse"],
                             "per_dim_r2": dict(zip(DIM_NAMES, res["per_dim_r2"])),
                             "probe_feature_normalize": "l2",
                             "n_probe_train_pairs": int(tx.shape[0]),
                             "n_probe_eval_test_pairs": int(txe.shape[0]),
                             "finetuned_ckpt_sha256": ckpt_sha,
                             "finetuned_ckpt_step": ckpt_step,
                             "split_sha256": split_sha,
                             "extractor_meta": {"family": extractor.meta.family,
                                                 "feat_dim": extractor.meta.feat_dim}})

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "a") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"[sens] appended {len(out)} rows to {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
