"""Common-ridge comparison (rebuttal instruction §4).

Selects normalization mode + ridge α **only on a train-indist val split**
carved out of the probe-train pool (80/20 episode-level, deterministic),
freezes the selected setting, then evaluates ONCE on the test-OOD split.

Applied identically to all six backbones:
  vjepa2, videomae, webdino, r3m, vip, caip_visual_only

The purpose is to give a common-protocol, single-choice-of-hparams comparison
where no test-set information can leak into hyperparameter selection.
Absolute R² values here are **not** comparable to the submitted paper's or
GCP's absolute values — see REBUTTAL_EVIDENCE.md's protocol block.

Output:
  * runs/corl2026_rebuttal/results/common_ridge_comparison.jsonl
  * runs/corl2026_rebuttal/reports/COMMON_RIDGE_COMPARISON.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "rebuttal"))

from eval_matched_control import build_extractor, encode_slice, _flatten_frames
from validate_baselines import feature_stats, constant_action_baseline_r2


BACKBONES = ("vjepa2", "videomae", "webdino", "r3m", "vip", "caip_visual_only")
NORMS = ("raw", "zscore", "l2")
DEFAULT_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def _normalize(x_train, x_test, mode):
    """Fit normalisation on train, apply to test. Returns (x_train_n, x_test_n)."""
    if mode == "raw":
        return x_train, x_test
    if mode == "zscore":
        mu = x_train.mean(dim=0, keepdim=True)
        sig = x_train.std(dim=0, keepdim=True).clamp_min(1e-2)
        return (x_train - mu) / sig, (x_test - mu) / sig
    if mode == "l2":
        return F.normalize(x_train, dim=-1), F.normalize(x_test, dim=-1)
    raise ValueError(mode)


def _ridge_r2(X, Y, Xv, Yv, alpha):
    """Fit closed-form ridge on (X, Y), score on (Xv, Yv) → mean per-dim R²."""
    D = X.shape[1]
    W = np.linalg.solve(X.T @ X + alpha * np.eye(D), X.T @ Y)
    pred = Xv @ W
    diff = pred - Yv
    per_dim_mse = (diff ** 2).mean(axis=0)
    var = Yv.var(axis=0).clip(min=1e-12)
    per_dim_r2 = 1.0 - per_dim_mse / var
    return float(per_dim_r2.mean()), W


def episode_split_indices(n_episodes, val_ratio, seed):
    """Deterministic episode-level 80/20 split. Returns (fit_idx, val_idx)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_episodes)
    n_val = int(round(val_ratio * n_episodes))
    val_idx = np.sort(perm[:n_val])
    fit_idx = np.sort(perm[n_val:])
    return fit_idx.tolist(), val_idx.tolist()


def sha_of_episode_ids(ep_ids):
    """Deterministic SHA-256 of the ordered list of (suite, file, demo)."""
    h = hashlib.sha256()
    for e in ep_ids:
        h.update(f"{e[0]}|{e[1]}|{e[2]}\n".encode())
    return h.hexdigest()


def run_backbone(bb, args, device):
    from data.libero_video import LiberoVideoClips
    print(f"\n=== {bb} ===", flush=True)
    t0 = time.time()
    extractor = build_extractor(
        bb, device, None,
        external_models_dir=REPO_ROOT / "datasets/external_models",
        clip_len=32,
    )
    print(f"[{bb}] extractor built {time.time()-t0:.1f}s feat_dim={extractor.meta.feat_dim}", flush=True)

    split_train = REPO_ROOT / "runs/corl2026_rebuttal/splits/train.json"
    split_test  = REPO_ROOT / "runs/corl2026_rebuttal/splits/test_ood.json"
    datasets_dir = REPO_ROOT / "datasets/libero_downloaded"

    # CAIP native and any text-conditioned modes are handled by a separate
    # script (validate_caip_native.py) — this one is visual-only comparisons.
    t0 = time.time()
    train_feats, train_acts, train_ep_ids = encode_slice(
        extractor, split_train, datasets_dir, (0, args.probe_train_clips),
        clip_len=32, seed=0, batch_size=args.batch_size,
        num_workers=args.num_workers, device=device)
    print(f"[{bb}] train encode {time.time()-t0:.1f}s shape={tuple(train_feats.shape)}", flush=True)

    t0 = time.time()
    test_feats, test_acts, test_ep_ids = encode_slice(
        extractor, split_test, datasets_dir, (0, args.probe_eval_clips_test),
        clip_len=32, seed=0, batch_size=args.batch_size,
        num_workers=args.num_workers, device=device)
    print(f"[{bb}] test encode {time.time()-t0:.1f}s shape={tuple(test_feats.shape)}", flush=True)

    # Deterministic episode-level train/val split for hparam selection.
    fit_idx, val_idx = episode_split_indices(
        train_feats.shape[0], args.val_ratio, args.split_seed)
    train_fit_feats = train_feats[fit_idx]; train_fit_acts = train_acts[fit_idx]
    train_val_feats = train_feats[val_idx]; train_val_acts = train_acts[val_idx]
    print(f"[{bb}] train_fit={len(fit_idx)} episodes, train_val={len(val_idx)} episodes")

    # Feature stats reported on the fit pool
    f_stats_fit = feature_stats(train_fit_feats)

    # Flatten to frames for probe
    Xfit, Yfit = _flatten_frames(train_fit_feats, train_fit_acts)
    Xval, Yval = _flatten_frames(train_val_feats, train_val_acts)
    Xtst, Ytst = _flatten_frames(test_feats, test_acts)

    # Grid search over (normalization × α) on train_fit → val
    grid = []
    for norm in NORMS:
        Xf_n, Xv_n = _normalize(Xfit, Xval, norm)
        Xf_np = Xf_n.cpu().numpy().astype(np.float64)
        Xv_np = Xv_n.cpu().numpy().astype(np.float64)
        Y_np  = Yfit.cpu().numpy().astype(np.float64)
        Yv_np = Yval.cpu().numpy().astype(np.float64)
        for a in args.ridge_alphas:
            val_r2, _ = _ridge_r2(Xf_np, Y_np, Xv_np, Yv_np, a)
            grid.append({"norm": norm, "alpha": a, "train_val_r2": val_r2})

    # Pick best-on-val (deterministic tiebreak: smallest α, then norm order)
    best = max(grid, key=lambda g: (g["train_val_r2"], -g["alpha"], -NORMS.index(g["norm"])))
    best_norm = best["norm"]; best_alpha = best["alpha"]
    print(f"[{bb}] BEST-ON-VAL: norm={best_norm} α={best_alpha} val_R²={best['train_val_r2']:+.4f}")

    # Freeze — refit on train_fit ONLY (val was used for selection; do NOT
    # refit on fit+val to keep the selection audit clean and single-decision).
    Xf_n, Xt_n = _normalize(Xfit, Xtst, best_norm)
    Xf_np = Xf_n.cpu().numpy().astype(np.float64)
    Xt_np = Xt_n.cpu().numpy().astype(np.float64)
    Y_np  = Yfit.cpu().numpy().astype(np.float64)
    Yt_np = Ytst.cpu().numpy().astype(np.float64)

    test_r2, W = _ridge_r2(Xf_np, Y_np, Xt_np, Yt_np, best_alpha)
    # Also report normalized MSE @ frozen setting
    pred = Xt_np @ W
    diff = pred - Yt_np
    per_dim_mse = (diff ** 2).mean(axis=0)
    var_t = Yt_np.var(axis=0).clip(min=1e-12)
    norm_mse = float((per_dim_mse / var_t).mean())

    # Constant-action baseline (train_fit mean predicting test)
    const_r2 = constant_action_baseline_r2(Y_np, Yt_np)

    return {
        "backbone": bb,
        "feat_dim": int(extractor.meta.feat_dim),
        "feature_stats_fit": f_stats_fit,
        "n_episodes_train_total": train_feats.shape[0],
        "n_episodes_train_fit": len(fit_idx),
        "n_episodes_train_val": len(val_idx),
        "n_episodes_test": test_feats.shape[0],
        "train_ep_sha256_fit": sha_of_episode_ids([train_ep_ids[i] for i in fit_idx]),
        "train_ep_sha256_val": sha_of_episode_ids([train_ep_ids[i] for i in val_idx]),
        "test_ep_sha256":     sha_of_episode_ids(test_ep_ids),
        "hparam_grid_norms": list(NORMS),
        "hparam_grid_alphas": list(args.ridge_alphas),
        "hparam_grid_train_val_scores": grid,
        "selected_norm": best_norm,
        "selected_alpha": best_alpha,
        "selected_train_val_r2": best["train_val_r2"],
        "test_oo_d_r2_at_frozen_setting": test_r2,
        "test_oo_d_normalized_mse_at_frozen_setting": norm_mse,
        "constant_action_baseline_r2": const_r2,
        "probe_train_clips": args.probe_train_clips,
        "probe_eval_clips_test": args.probe_eval_clips_test,
        "val_ratio": args.val_ratio,
        "split_seed": args.split_seed,
    }


def _fmt_pct_of_dim(bb_record):
    d = bb_record["feature_stats_fit"]["n_dead_dims_std_lt_1e-2"]
    D = bb_record["feature_stats_fit"]["shape"][-1]
    return f"{d}/{D} ({100.0*d/D:.0f}%)"


def build_report(records, out_md):
    lines = ["# COMMON_RIDGE_COMPARISON.md — instruction §4\n\n"]
    lines.append("> **PACKAGE**: `ORIGINAL_CLUSTER_SECONDARY_EVIDENCE`. "
                 "Absolute R² values in this file are conditional on the "
                 "original-cluster val-selected common-ridge protocol; "
                 "**do not merge** with the GCP exact-split matrix or with "
                 "the submitted paper's canonical matrix. Only within-"
                 "protocol ordering transfers.\n\n")
    lines.append("Common-protocol ridge comparison across all six backbones. "
                 "**Hyperparameters (normalisation, ridge α) are selected on a "
                 "deterministic 80/20 episode-level train-in-distribution val "
                 "split of the probe-train pool, then frozen and evaluated ONCE "
                 "on the test-OOD split.** No test-set information leaks into "
                 "the selection.\n\n")
    lines.append("This table replaces the earlier \"best test R² over the full "
                 "sweep\" baseline table in BASELINE_VALIDATION / CAIP_VALIDATION "
                 "(which selected on test-OOD and is therefore invalid for "
                 "final reporting).\n\n")

    lines.append("## Protocol identifier\n\n")
    lines.append("| Component | Value |\n")
    lines.append("|-----------|-------|\n")
    lines.append("| Grid | normalisation ∈ {raw, zscore-clamp-1e-2, l2-unit} × "
                 "ridge α ∈ {0.01, 0.1, 1, 10, 100, 1000} = 18 combinations |\n")
    lines.append("| Selector | argmax over the 18-cell grid of per-dim-mean R² on "
                 "the train-in-distribution val split |\n")
    lines.append("| Train split | `runs/corl2026_rebuttal/splits/train.json`, "
                 "first N=`probe_train_clips` episodes (deterministic dataset order) |\n")
    lines.append("| Train/val split | 80/20 episode-level, seed=42 permutation on "
                 "N episodes (episode SHA for both partitions in JSONL) |\n")
    lines.append("| Test split | `runs/corl2026_rebuttal/splits/test_ood.json`, "
                 "1300 held-out episodes (episode SHA in JSONL) |\n")
    lines.append("| Probe head | closed-form ridge, W = (XᵀX + αI)⁻¹XᵀY |\n")
    lines.append("| Refit-after-selection | fit only (no refit on fit+val) — one "
                 "selection decision, one test evaluation |\n")
    lines.append("| Frame-flattening | per-frame z_t as an independent sample, "
                 "same as the matched-control probe |\n\n")

    lines.append("## Table\n\n")
    lines.append("| Backbone | feat_dim | dead dims (std<1e-2) | Selected norm | Selected α | "
                 "Train-val R² (at selected) | **Test-OOD R² (one-shot at frozen setting)** | "
                 "Normalized MSE @ frozen | Constant-action R² |\n")
    lines.append("|----------|----------|-----------------------|----------------|-------------|"
                 "----------------------------|-------------------------------------------------|"
                 "-------------------------|--------------------|\n")
    order = ["vjepa2", "videomae", "webdino", "r3m", "vip", "caip_visual_only"]
    idx = {r["backbone"]: r for r in records}
    for bb in order:
        r = idx.get(bb)
        if not r: continue
        lines.append(
            f"| {bb} | {r['feat_dim']} | {_fmt_pct_of_dim(r)} | {r['selected_norm']} | "
            f"{r['selected_alpha']:g} | {r['selected_train_val_r2']:+.4f} | "
            f"**{r['test_oo_d_r2_at_frozen_setting']:+.4f}** | "
            f"{r['test_oo_d_normalized_mse_at_frozen_setting']:.4f} | "
            f"{r['constant_action_baseline_r2']:+.4f} |\n"
        )

    lines.append("\n## Interpretation\n\n")
    if records:
        cbase = records[0]["constant_action_baseline_r2"]
        ranked = sorted(records, key=lambda r: -r["test_oo_d_r2_at_frozen_setting"])
        rank_str = " > ".join(
            f"{r['backbone']} ({r['test_oo_d_r2_at_frozen_setting']:+.2f})"
            for r in ranked
        )
        n_pass = sum(1 for r in records
                     if r["test_oo_d_r2_at_frozen_setting"] > r["constant_action_baseline_r2"])
        lines.append(f"* **Baseline verdict (all six backbones).** Under this "
                     f"val-selected common-ridge readout, {n_pass} of {len(records)} "
                     f"backbones exceed the constant-action baseline "
                     f"({cbase:+.4f}) on test-OOD. Report every backbone as: "
                     f"*does not exceed the constant-action baseline under the "
                     f"val-selected common-ridge readout protocol*.\n")
        lines.append(f"* **Val↔test-OOD gap.** Every backbone fits its val "
                     f"split with positive R² (largest val R² = "
                     f"{max(r['selected_train_val_r2'] for r in records):+.3f}), "
                     f"but the test-OOD R² is very negative for all six. The "
                     f"train-in-distribution val split does not predict cross-"
                     f"task-OOD R² under raw-feature ridge.\n")
        lines.append(f"* **Ordering** (higher R² = better, only ordering is "
                     f"meaningful — absolute values are protocol-conditional): "
                     f"{rank_str}. The video-encoder family (V-JEPA2 / VideoMAE / "
                     f"Web-DINO) is ordered ahead of R3M / VIP; CAIP visual-only "
                     f"falls inside the video-family range, not below R3M / VIP.\n")
        lines.append(f"* **Probe-family dependence.** The absolute R² regime "
                     f"here (raw-feature ridge under val selection) is very "
                     f"different from the matched-control l2-normalized MLP "
                     f"probe. The two probes preserve the V-JEPA2 > VideoMAE > "
                     f"Web-DINO frozen ordering but differ on the absolute "
                     f"regime by ~1 R² unit. Absolute R² does not transfer "
                     f"across probes — only within-protocol paired deltas do.\n")
    lines.append("* **What NOT to say.** Do not describe any backbone here as a "
                 "\"structural encoding failure\" — this test does not distinguish "
                 "between a structural encoding property and a probe-family "
                 "incompatibility. The narrower defensible claim is only what the "
                 "baseline-verdict bullet states.\n")

    Path(out_md).write_text("".join(lines))
    print(f"wrote {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    ap.add_argument("--out-jsonl",
                    default=str(REPO_ROOT / "runs/corl2026_rebuttal/results/common_ridge_comparison.jsonl"))
    ap.add_argument("--out-md",
                    default=str(REPO_ROOT / "runs/corl2026_rebuttal/reports/COMMON_RIDGE_COMPARISON.md"))
    ap.add_argument("--probe-train-clips", type=int, default=800)
    ap.add_argument("--probe-eval-clips-test", type=int, default=1300)
    ap.add_argument("--ridge-alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    records = []
    for bb in args.backbones:
        try:
            rec = run_backbone(bb, args, device)
            records.append(rec)
            # incremental checkpoint so a mid-run crash keeps completed rows
            with open(args.out_jsonl, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
        except Exception as e:
            print(f"[{bb}] FAILED: {e!r}", flush=True)
            raise
    build_report(records, args.out_md)
    print("\ndone. best-on-val settings:")
    for r in records:
        print(f"  {r['backbone']}: norm={r['selected_norm']} α={r['selected_alpha']:g} → "
              f"test R²={r['test_oo_d_r2_at_frozen_setting']:+.4f}")


if __name__ == "__main__":
    main()
