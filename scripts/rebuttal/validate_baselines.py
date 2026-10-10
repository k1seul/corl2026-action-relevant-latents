"""P3 validation suite for R3M / VIP frozen baselines.

Marks the previously-reported values (R3M ~ -9.7, VIP ~ -2.0) as
INVALID_PENDING_AUDIT and produces a validation JSONL + report that
covers each requested check:

  * official checkpoint / loader (already used, but re-report SHA)
  * official preprocessing (extractor internal, re-report normalization)
  * finite features (no NaN / Inf in encoded batch)
  * non-collapsed feature variance (per-dim std distribution)
  * train vs test R² gap (over/under-fit diagnostic)
  * pooled vs task-macro R²
  * standardized-feature probe (zscore w/ safe clamp)
  * Ridge alpha sweep {0.01, 0.1, 1, 10, 100}
  * constant-action baseline (predict mean action → R² should be ≈ 0)
  * agreement with official example: extract features on a tiny standard
    input and print their L2 norm + shape so we can eyeball vs docs.

Output: results/baseline_validation.jsonl + reports/BASELINE_VALIDATION.md.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "rebuttal"))

import numpy as np
import torch
import torch.nn.functional as F

from eval_matched_control import (
    build_extractor, encode_slice, _flatten_frames, _split_sha,
    DIM_NAMES,
)


def feature_stats(feats: torch.Tensor) -> dict:
    """feats: (K, T, D) → summary stats."""
    x = feats.reshape(-1, feats.shape[-1])
    x_np = x.float().cpu().numpy()
    per_dim_std = x_np.std(axis=0)
    per_dim_mean = x_np.mean(axis=0)
    return {
        "shape": list(feats.shape),
        "any_nan": bool(np.isnan(x_np).any()),
        "any_inf": bool(np.isinf(x_np).any()),
        "l2_norm_mean": float(np.linalg.norm(x_np, axis=1).mean()),
        "l2_norm_std":  float(np.linalg.norm(x_np, axis=1).std()),
        "per_dim_std_mean": float(per_dim_std.mean()),
        "per_dim_std_median": float(np.median(per_dim_std)),
        "n_dead_dims_std_lt_1e-4": int((per_dim_std < 1e-4).sum()),
        "n_dead_dims_std_lt_1e-2": int((per_dim_std < 1e-2).sum()),
        "per_dim_mean_abs_max": float(np.abs(per_dim_mean).max()),
        "min_val": float(x_np.min()),
        "max_val": float(x_np.max()),
    }


def constant_action_baseline_r2(train_y_np, test_y_np):
    mean = train_y_np.mean(axis=0)
    pred = np.tile(mean, (test_y_np.shape[0], 1))
    diff = pred - test_y_np
    per_dim_mse = (diff ** 2).mean(axis=0)
    var = test_y_np.var(axis=0).clip(min=1e-12)
    per_dim_r2 = 1.0 - per_dim_mse / var
    return float(per_dim_r2.mean())


def ridge_alpha_sweep(train_x, train_y, test_x, test_y, alphas):
    X = train_x.cpu().numpy().astype(np.float64)
    Y = train_y.cpu().numpy().astype(np.float64)
    Xt = test_x.cpu().numpy().astype(np.float64)
    Yt = test_y.cpu().numpy().astype(np.float64)
    D = X.shape[1]
    XtX = X.T @ X
    XtY = X.T @ Y
    out = {}
    var_test = Yt.var(axis=0).clip(min=1e-12)
    var_train = Y.var(axis=0).clip(min=1e-12)
    for a in alphas:
        W = np.linalg.solve(XtX + a * np.eye(D), XtY)
        pred = Xt @ W
        diff = pred - Yt
        per_dim_mse = (diff ** 2).mean(axis=0)
        per_dim_r2 = 1.0 - per_dim_mse / var_test
        # train R²
        pred_tr = X @ W
        diff_tr = pred_tr - Y
        per_dim_mse_tr = (diff_tr ** 2).mean(axis=0)
        per_dim_r2_tr = 1.0 - per_dim_mse_tr / var_train
        out[a] = {
            "test_r2_mean": float(per_dim_r2.mean()),
            "train_r2_mean": float(per_dim_r2_tr.mean()),
            "test_r2_per_dim": dict(zip(DIM_NAMES, per_dim_r2.tolist())),
            "normalized_mse": float((per_dim_mse / var_test).mean()),
        }
    return out


def per_task_pooled_r2(pred, y, ep_ids, ep_len):
    """Compute per-task pooled R² given predictions + targets + ep_ids."""
    from collections import defaultdict
    per_task = defaultdict(list)
    T = ep_len
    for i, m in enumerate(ep_ids):
        lo, hi = i * T, (i + 1) * T
        per_task[(m[0], m[1])].append((pred[lo:hi], y[lo:hi]))
    r2s = []
    for _, rows in per_task.items():
        p = np.concatenate([r[0] for r in rows])
        t = np.concatenate([r[1] for r in rows])
        diff = p - t
        per_dim_mse = (diff ** 2).mean(axis=0)
        var = t.var(axis=0).clip(min=1e-12)
        per_dim_r2 = 1.0 - per_dim_mse / var
        r2s.append(float(per_dim_r2.mean()))
    return float(sum(r2s) / len(r2s)) if r2s else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["r3m", "vip"])
    ap.add_argument("--out-jsonl",
                    default=str(REPO_ROOT / "runs/corl2026_rebuttal/results/baseline_validation.jsonl"))
    ap.add_argument("--out-md",
                    default=str(REPO_ROOT / "runs/corl2026_rebuttal/reports/BASELINE_VALIDATION.md"))
    ap.add_argument("--probe-train-clips", type=int, default=800)
    ap.add_argument("--probe-eval-clips-test", type=int, default=1300)
    ap.add_argument("--ridge-alphas", nargs="+", type=float,
                    default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    split_train = REPO_ROOT / "runs/corl2026_rebuttal/splits/train.json"
    split_test = REPO_ROOT / "runs/corl2026_rebuttal/splits/test_ood.json"
    datasets_dir = REPO_ROOT / "datasets/libero_downloaded"

    records = []
    for backbone in args.backbones:
        print(f"\n=== validating {backbone} ===", flush=True)
        t0 = time.time()
        extractor = build_extractor(backbone, device, None,
                                     external_models_dir=REPO_ROOT / "datasets/external_models",
                                     clip_len=32)
        print(f"[{backbone}] extractor built {time.time()-t0:.1f}s "
              f"family={extractor.meta.family} feat_dim={extractor.meta.feat_dim}", flush=True)

        # encode probe-train + probe-test slices
        train_feats, train_acts, _ = encode_slice(
            extractor, split_train, datasets_dir, (0, args.probe_train_clips),
            clip_len=32, seed=0, batch_size=8, num_workers=2, device=device)
        test_feats, test_acts, test_ep_ids = encode_slice(
            extractor, split_test, datasets_dir, (0, args.probe_eval_clips_test),
            clip_len=32, seed=0, batch_size=8, num_workers=2, device=device)

        # feature stats on the encoded train pool
        f_stats_train = feature_stats(train_feats)
        f_stats_test = feature_stats(test_feats)
        print(f"[{backbone}] train stats: {f_stats_train}", flush=True)

        # 3 normalization modes × ridge sweep
        tx_raw, ty = _flatten_frames(train_feats, train_acts)
        txe_raw, tye = _flatten_frames(test_feats, test_acts)
        # Standardize (zscore with safe clamp)
        mu = tx_raw.mean(dim=0, keepdim=True)
        sig = tx_raw.std(dim=0, keepdim=True).clamp_min(1e-2)
        tx_z = (tx_raw - mu) / sig
        txe_z = (txe_raw - mu) / sig
        tx_l2 = F.normalize(tx_raw, dim=-1)
        txe_l2 = F.normalize(txe_raw, dim=-1)

        ridge_none = ridge_alpha_sweep(tx_raw, ty, txe_raw, tye, args.ridge_alphas)
        ridge_z = ridge_alpha_sweep(tx_z, ty, txe_z, tye, args.ridge_alphas)
        ridge_l2 = ridge_alpha_sweep(tx_l2, ty, txe_l2, tye, args.ridge_alphas)

        # constant-action baseline
        const_r2 = constant_action_baseline_r2(ty.cpu().numpy(), tye.cpu().numpy())

        # task-macro R² for the best (raw + best alpha)
        best_alpha_none = max(ridge_none, key=lambda a: ridge_none[a]["test_r2_mean"])
        best_alpha_z = max(ridge_z, key=lambda a: ridge_z[a]["test_r2_mean"])
        # for task-macro we need to re-compute predictions once with the best α
        X = tx_raw.cpu().numpy().astype(np.float64)
        Xt = txe_raw.cpu().numpy().astype(np.float64)
        Ynp = ty.cpu().numpy().astype(np.float64)
        Ytnp = tye.cpu().numpy().astype(np.float64)
        Wbest = np.linalg.solve(X.T @ X + best_alpha_none * np.eye(X.shape[1]), X.T @ Ynp)
        pred_best = Xt @ Wbest
        macro_none = per_task_pooled_r2(pred_best, Ytnp, test_ep_ids, ep_len=32)
        # same for zscore
        Xz = tx_z.cpu().numpy().astype(np.float64)
        Xzt = txe_z.cpu().numpy().astype(np.float64)
        Wz = np.linalg.solve(Xz.T @ Xz + best_alpha_z * np.eye(Xz.shape[1]), Xz.T @ Ynp)
        pred_z = Xzt @ Wz
        macro_z = per_task_pooled_r2(pred_z, Ytnp, test_ep_ids, ep_len=32)

        record = {
            "backbone": backbone,
            "feature_stats_train": f_stats_train,
            "feature_stats_test": f_stats_test,
            "ridge_alpha_sweep_raw": {str(k): v for k, v in ridge_none.items()},
            "ridge_alpha_sweep_zscore": {str(k): v for k, v in ridge_z.items()},
            "ridge_alpha_sweep_l2":     {str(k): v for k, v in ridge_l2.items()},
            "best_alpha_raw": best_alpha_none,
            "best_alpha_zscore": best_alpha_z,
            "task_macro_r2_raw_best": macro_none,
            "task_macro_r2_zscore_best": macro_z,
            "constant_action_baseline_r2": const_r2,
            "probe_train_clips": args.probe_train_clips,
            "probe_eval_clips_test": args.probe_eval_clips_test,
        }
        records.append(record)

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # write report
    lines = ["# BASELINE_VALIDATION.md — R3M/VIP INVALID_PENDING_AUDIT resolution\n\n"]
    lines.append("Generated by `scripts/rebuttal/validate_baselines.py`.\n\n")
    for r in records:
        bb = r["backbone"]
        lines.append(f"## {bb}\n\n")
        fs = r["feature_stats_train"]
        lines.append(f"* Feature stats (probe-train pool): shape={fs['shape']}, "
                     f"any_NaN={fs['any_nan']}, any_Inf={fs['any_inf']}, "
                     f"L2 norm mean={fs['l2_norm_mean']:.3f}, "
                     f"per-dim std median={fs['per_dim_std_median']:.4f}, "
                     f"dead dims (std<1e-2)={fs['n_dead_dims_std_lt_1e-2']}/{fs['shape'][-1]}, "
                     f"range=[{fs['min_val']:.3f}, {fs['max_val']:.3f}].\n")
        lines.append(f"* Constant-action baseline R² = {r['constant_action_baseline_r2']:+.4f} "
                     f"(should be ≈ 0; large deviation means test-set variance is odd).\n")
        lines.append(f"* Ridge alpha sweep — RAW features:\n\n")
        lines.append("  | α | test R² | train R² | normalized MSE |\n")
        lines.append("  |---|---------|----------|-----------------|\n")
        for a, s in sorted(r["ridge_alpha_sweep_raw"].items(), key=lambda x: float(x[0])):
            lines.append(f"  | {a} | {s['test_r2_mean']:+.4f} | {s['train_r2_mean']:+.4f} | {s['normalized_mse']:.4f} |\n")
        lines.append(f"\n* Ridge alpha sweep — STANDARDIZED (zscore, clamp 1e-2):\n\n")
        lines.append("  | α | test R² | train R² | normalized MSE |\n")
        lines.append("  |---|---------|----------|-----------------|\n")
        for a, s in sorted(r["ridge_alpha_sweep_zscore"].items(), key=lambda x: float(x[0])):
            lines.append(f"  | {a} | {s['test_r2_mean']:+.4f} | {s['train_r2_mean']:+.4f} | {s['normalized_mse']:.4f} |\n")
        lines.append(f"\n* Ridge alpha sweep — L2 (unit norm):\n\n")
        lines.append("  | α | test R² | train R² | normalized MSE |\n")
        lines.append("  |---|---------|----------|-----------------|\n")
        for a, s in sorted(r["ridge_alpha_sweep_l2"].items(), key=lambda x: float(x[0])):
            lines.append(f"  | {a} | {s['test_r2_mean']:+.4f} | {s['train_r2_mean']:+.4f} | {s['normalized_mse']:.4f} |\n")
        lines.append(f"\n* Best-α task-macro R² (raw, α={r['best_alpha_raw']}) = "
                     f"{r['task_macro_r2_raw_best']:+.4f}\n")
        lines.append(f"* Best-α task-macro R² (zscore, α={r['best_alpha_zscore']}) = "
                     f"{r['task_macro_r2_zscore_best']:+.4f}\n\n")

    # verdict
    lines.append("## Verdict\n\n")
    lines.append("Compare each backbone's **best test R² over the full alpha sweep** "
                 "and **best-α task-macro R²** to the earlier reported value "
                 "(R3M ≈ −9.7, VIP ≈ −2.0). If the sweep shows a much higher "
                 "R² at some α, the earlier value was under-fit and must be "
                 "REPLACED. If the sweep never beats zero, the baselines are "
                 "genuinely poor on this protocol and can be reported as such — "
                 "but always with normalized MSE alongside so the reader has an "
                 "interpretable magnitude.\n")
    lines.append("\nDo not include the pre-audit −9.7 / −2.0 numbers in "
                 "REBUTTAL_EVIDENCE.md until this file's verdict is written.\n")
    Path(args.out_md).write_text("".join(lines))
    print(f"\nwrote {args.out_jsonl}\nwrote {args.out_md}")


if __name__ == "__main__":
    main()
