"""CAIP exact-protocol action-centric baseline (final one-shot).

Per user instruction — the only action-centric baseline experiment allowed;
NO new 20k-step fine-tuning. Only frozen-feature extraction + submitted-
compatible MLP action probe, applied identically to CAIP (two modes) and to
V-JEPA2 / VideoMAE / Web-DINO frozen (for direct within-protocol comparison).

Pinned artifacts (recorded in outputs):
  * CAIP: yuvansharma/caip-vitl256, HF commit
      614a7ea79c3c3ec9d992aa74824c036d1ce69a48
      model.safetensors SHA-256
      2be38f8568dc4dacb6595c8b4e0a67cf8dfac6052049f806cccc258a14980e4e
  * Splits: `data/splits_taskheld/{train,test_ood}.json`
      train SHA-256:    501ad00a68ee0d5d7cc50d2d7ca94160003ef6f9680d32f923dd3f7f858085f6
      test-OOD SHA-256: d27290c85c5ba211012543c2c915bd154689d6d42cceb2de34a58fd5470f576c

Two CAIP modes:
  * caip_visual_only        — no task text; patch_features mean-pooled over
                              256 patch tokens → 1024-D vector as z_t.
  * caip_native_conditioned — per-episode LIBERO task caption (verified
                              non-leakage: caption is task label only,
                              contains no per-frame action vector, no
                              episode-specific identifier, and test-OOD tasks
                              are disjoint from probe-train tasks);
                              image_pooled as z_t. Reported SEPARATELY;
                              never mixed into the vision-only ranking.

Submitted-compatible probe (applied identically to all 5 encoders):
  * MLP D → 256 → 128 → 7
  * AdamW, lr = 3e-4, weight decay = 1e-4, batch = 512, 3000 steps
  * NO feature normalization
  * NO action normalization
  * 3 probe seeds ∈ {0, 1, 2}
  * pooled R² = mean over the seven per-dimension R²

Eval scopes:
  * Submitted-compatible point estimate — first 200 episodes of the
    submitted test-OOD split (in submitted file order).
  * All-task 26 × 50 = 1300 episodes — for bootstrap CI.

Bootstrap: 10 000-replicate held-out-task cluster bootstrap over the 26 tasks
(with replacement); for each replicate we recompute pooled R² for the CAIP
readout AND for the constant-action baseline on the sampled tasks, then
report both R² CIs and the (CAIP − constant) difference CI.

Output:
  * results/caip_exact_protocol.jsonl — per-encoder × per-probe-seed rows
    with metadata + raw predictions + targets per episode.
  * tables/caip_exact_protocol.csv — flat summary.
  * reports/CAIP_EXACT_PROTOCOL.md — human-readable report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "rebuttal"))

from eval_matched_control import (
    build_extractor,
    CAIPVisualOnlyExtractor,
    CAIPNativeExtractor,
)

# Pinned split identifiers (submitted paper)
SUBMITTED_TRAIN = REPO_ROOT / "data/splits_taskheld/train.json"
SUBMITTED_TEST  = REPO_ROOT / "data/splits_taskheld/test_ood.json"

# Pinned CAIP model provenance
CAIP_HF_COMMIT = "614a7ea79c3c3ec9d992aa74824c036d1ce69a48"
CAIP_WEIGHTS_SHA256 = "2be38f8568dc4dacb6595c8b4e0a67cf8dfac6052049f806cccc258a14980e4e"
SUBMITTED_TRAIN_SHA = "501ad00a68ee0d5d7cc50d2d7ca94160003ef6f9680d32f923dd3f7f858085f6"
SUBMITTED_TEST_SHA  = "d27290c85c5ba211012543c2c915bd154689d6d42cceb2de34a58fd5470f576c"

DIM_NAMES = ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")

ENCODERS = (
    "caip_visual_only",
    "caip_native_conditioned",
    "vjepa2",
    "videomae",
    "webdino",
)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def caption_from_filename(fname: str) -> str:
    """LIBERO hdf5 filename → task-description caption (non-leakage-verified)."""
    stem = fname.replace("_demo.hdf5", "").replace(".hdf5", "")
    return stem.replace("_", " ")


class SubmittedCompatMLP(nn.Module):
    def __init__(self, in_dim: int, action_dim: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, action_dim),
        )
    def forward(self, x): return self.net(x)


@torch.no_grad()
def encode(extractor, split_path: Path, datasets_dir: Path,
            episode_slice, batch_size: int, num_workers: int,
            device: torch.device, needs_caption: bool):
    """Return (feats [K,T,D], acts [K,T,7], episode_ids list)."""
    from data.libero_video import LiberoVideoClips, worker_init_fn
    ds = LiberoVideoClips(
        split_path=str(split_path), datasets_dir=str(datasets_dir),
        clip_len=32, cameras=["agentview_rgb"], resolution=128,
        seed=0, return_actions=True,
    )
    lo, hi = episode_slice
    hi = min(hi, len(ds))
    idxs = list(range(lo, hi))
    episode_ids = [(ds.episodes[i]["suite"], ds.episodes[i]["file"],
                    ds.episodes[i]["demo"]) for i in idxs]
    captions_all = [caption_from_filename(ep[1]) for ep in episode_ids]
    subset = Subset(ds, idxs)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        worker_init_fn=worker_init_fn, drop_last=False, pin_memory=False,
    )
    feats_all, acts_all = [], []
    n_seen = 0
    for clip, actions in loader:
        B = clip.shape[0]
        clip = clip.to(device)
        if needs_caption:
            caps = captions_all[n_seen:n_seen + B]
            f = extractor.extract(clip, task_caption=caps)
        else:
            f = extractor.extract(clip)
        feats_all.append(f.detach().float().cpu())
        acts_all.append(actions[..., :7].float().cpu())
        n_seen += B
    return torch.cat(feats_all, 0), torch.cat(acts_all, 0), episode_ids


def train_probe_and_predict(train_feats, train_acts, test_feats, seed,
                             device, lr=3e-4, wd=1e-4, batch=512, steps=3000):
    """Return test predictions [K_test, T, 7]."""
    torch.manual_seed(seed); np.random.seed(seed)
    K_tr, T, D = train_feats.shape
    X = train_feats.reshape(-1, D).to(device)
    Y = train_acts.reshape(-1, 7).to(device)
    N = X.shape[0]
    model = SubmittedCompatMLP(D).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    for step in range(steps):
        idx = torch.randint(0, N, (batch,), device=device, generator=None)
        xb, yb = X[idx], Y[idx]
        pred = model(xb)
        loss = torch.nn.functional.mse_loss(pred, yb)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        K_te = test_feats.shape[0]
        Xt = test_feats.reshape(-1, D).to(device)
        pred_all = model(Xt).cpu().numpy().reshape(K_te, T, 7)
    return pred_all


def pooled_r2(pred_flat: np.ndarray, target_flat: np.ndarray) -> float:
    """Mean over 7 per-dim R² on flattened (N, 7) arrays."""
    var = target_flat.var(axis=0).clip(min=1e-12)
    per_dim_mse = ((pred_flat - target_flat) ** 2).mean(axis=0)
    per_dim_r2 = 1.0 - per_dim_mse / var
    finite = per_dim_r2[np.isfinite(per_dim_r2)]
    return float(finite.mean()) if finite.size else float("nan")


def per_episode_sufficient(pred_kt7: np.ndarray, target_kt7: np.ndarray):
    """For each episode, return dict(sse, y_sum, y_sqsum, n) — enables
    fast task-cluster bootstrap."""
    K, T, _ = pred_kt7.shape
    out = []
    for k in range(K):
        p, y = pred_kt7[k], target_kt7[k]
        d = p - y
        out.append({
            "sse":    (d ** 2).sum(axis=0),
            "y_sum":  y.sum(axis=0),
            "y_sqsum":(y ** 2).sum(axis=0),
            "n":      T,
        })
    return out


def pool_r2_from_stats(task_arrays, sampled_task_indices, tasks):
    """Given per-task list-of-episode-stats and a bootstrap task sample,
    return pooled R²."""
    sse = np.zeros(7); ys = np.zeros(7); ysq = np.zeros(7); n = 0
    for ti in sampled_task_indices:
        t = tasks[ti]
        for e in task_arrays[t]:
            sse += e["sse"]; ys += e["y_sum"]; ysq += e["y_sqsum"]; n += e["n"]
    sst = ysq - (ys ** 2) / max(n, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_dim = 1.0 - sse / np.where(sst > 1e-12, sst, np.nan)
    finite = per_dim[np.isfinite(per_dim)]
    return float(finite.mean()) if finite.size else float("nan")


def bootstrap_ci(per_task_stats_a, per_task_stats_b_or_none, tasks,
                  iters=10000, seed=0):
    """Return (pointA, ciA_lo, ciA_hi, [pointDiff, ci_diff_lo, ci_diff_hi]).

    stats_a: dict[task] -> [episode stats] for the CAIP readout.
    stats_b_or_none: same shape for constant-action baseline (for the
        (a - b) difference CI); None to skip the diff.
    """
    rng = np.random.default_rng(seed)
    K = len(tasks)
    idx = rng.integers(0, K, size=(iters, K))
    r2_a = np.empty(iters); r2_b = np.empty(iters) if per_task_stats_b_or_none else None
    for i in range(iters):
        r2_a[i] = pool_r2_from_stats(per_task_stats_a, idx[i], tasks)
        if per_task_stats_b_or_none is not None:
            r2_b[i] = pool_r2_from_stats(per_task_stats_b_or_none, idx[i], tasks)
    r2_a = r2_a[np.isfinite(r2_a)]; r2_a.sort()
    # Point estimate at the identity task-sample (each task once)
    point_a = pool_r2_from_stats(per_task_stats_a, np.arange(K), tasks)
    lo_a = float(r2_a[int(0.025 * len(r2_a))])
    hi_a = float(r2_a[int(0.975 * len(r2_a)) - 1])
    out = {"point": point_a, "lo95": lo_a, "hi95": hi_a}
    if per_task_stats_b_or_none is not None:
        r2_b = r2_b[np.isfinite(r2_b)]
        # For the paired difference, use the SAME bootstrap draws:
        diffs = r2_a - r2_b[:len(r2_a)]
        diffs.sort()
        point_diff = point_a - pool_r2_from_stats(per_task_stats_b_or_none,
                                                    np.arange(K), tasks)
        out["diff_vs_const_point"] = point_diff
        out["diff_vs_const_lo95"] = float(diffs[int(0.025 * len(diffs))])
        out["diff_vs_const_hi95"] = float(diffs[int(0.975 * len(diffs)) - 1])
    return out


def const_baseline_predictions(train_acts, K_test, T):
    """Constant-action baseline: predict the mean-of-training-frames action
    for every test frame."""
    mu = train_acts.reshape(-1, 7).mean(dim=0).numpy()  # (7,)
    return np.broadcast_to(mu, (K_test, T, 7)).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train-episodes", type=int, default=400)
    ap.add_argument("--n-test-compat", type=int, default=200,
                    help="first-N test episodes for the submitted-compatible point estimate")
    ap.add_argument("--n-test-full", type=int, default=1300)
    ap.add_argument("--probe-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--probe-lr", type=float, default=3e-4)
    ap.add_argument("--probe-wd", type=float, default=1e-4)
    ap.add_argument("--probe-batch", type=int, default=512)
    ap.add_argument("--probe-steps", type=int, default=3000)
    ap.add_argument("--encode-batch", type=int, default=4)
    ap.add_argument("--encode-workers", type=int, default=2)
    ap.add_argument("--boot-iters", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS))
    ap.add_argument("--out-jsonl",
                    default=str(REPO_ROOT / "runs/corl2026_rebuttal/results/caip_exact_protocol.jsonl"))
    ap.add_argument("--out-csv",
                    default=str(REPO_ROOT / "runs/corl2026_rebuttal/tables/caip_exact_protocol.csv"))
    ap.add_argument("--out-md",
                    default=str(REPO_ROOT / "runs/corl2026_rebuttal/reports/CAIP_EXACT_PROTOCOL.md"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)

    # Verify pinned split hashes at runtime
    train_sha = sha256_of(SUBMITTED_TRAIN)
    test_sha  = sha256_of(SUBMITTED_TEST)
    assert train_sha == SUBMITTED_TRAIN_SHA, (
        f"submitted train split SHA mismatch: {train_sha} vs {SUBMITTED_TRAIN_SHA}")
    assert test_sha  == SUBMITTED_TEST_SHA, (
        f"submitted test-OOD split SHA mismatch: {test_sha} vs {SUBMITTED_TEST_SHA}")
    print(f"[caip-exact] pinned split SHAs verified.")

    # Verify CAIP model weight SHA
    caip_weights = REPO_ROOT / "datasets/external_models/caip/model.safetensors"
    caip_w_sha = sha256_of(caip_weights)
    assert caip_w_sha == CAIP_WEIGHTS_SHA256, (
        f"CAIP weights SHA mismatch: {caip_w_sha} vs {CAIP_WEIGHTS_SHA256}")
    print(f"[caip-exact] CAIP weights SHA verified: {caip_w_sha[:16]}...")

    datasets_dir = REPO_ROOT / "datasets/libero_downloaded"

    per_encoder_records = []
    for enc_name in args.encoders:
        print(f"\n=== encoder: {enc_name} ===", flush=True)
        t0 = time.time()
        needs_caption = (enc_name == "caip_native_conditioned")
        if enc_name == "caip_visual_only":
            extractor = CAIPVisualOnlyExtractor(device, REPO_ROOT / "datasets/external_models/caip")
        elif enc_name == "caip_native_conditioned":
            extractor = CAIPNativeExtractor(device, REPO_ROOT / "datasets/external_models/caip")
        else:
            extractor = build_extractor(
                enc_name, device, None,
                external_models_dir=REPO_ROOT / "datasets/external_models",
                clip_len=32,
            )
        print(f"[{enc_name}] extractor built {time.time()-t0:.1f}s "
              f"feat_dim={extractor.meta.feat_dim}", flush=True)

        # Encode
        t0 = time.time()
        train_feats, train_acts, train_ep_ids = encode(
            extractor, SUBMITTED_TRAIN, datasets_dir, (0, args.n_train_episodes),
            args.encode_batch, args.encode_workers, device, needs_caption)
        print(f"[{enc_name}] train encode {time.time()-t0:.1f}s "
              f"shape={tuple(train_feats.shape)}", flush=True)

        t0 = time.time()
        test_feats, test_acts, test_ep_ids = encode(
            extractor, SUBMITTED_TEST, datasets_dir, (0, args.n_test_full),
            args.encode_batch, args.encode_workers, device, needs_caption)
        print(f"[{enc_name}] test encode {time.time()-t0:.1f}s "
              f"shape={tuple(test_feats.shape)}", flush=True)

        # Free extractor GPU memory
        del extractor
        torch.cuda.empty_cache()

        # Per-probe-seed: train probe → predict → save
        probe_seed_records = []
        for pseed in args.probe_seeds:
            t0 = time.time()
            pred_test = train_probe_and_predict(
                train_feats, train_acts, test_feats, pseed, device,
                lr=args.probe_lr, wd=args.probe_wd,
                batch=args.probe_batch, steps=args.probe_steps,
            )
            # pred_test: [K_test=1300, T=32, 7]
            target_test = test_acts.numpy()

            # Submitted-compat point estimate: first N_compat episodes
            pred_compat_flat = pred_test[:args.n_test_compat].reshape(-1, 7)
            targ_compat_flat = target_test[:args.n_test_compat].reshape(-1, 7)
            r2_compat = pooled_r2(pred_compat_flat, targ_compat_flat)

            # All-task per-episode sufficient stats
            per_ep_stats = per_episode_sufficient(pred_test, target_test)
            probe_seed_records.append({
                "probe_seed": pseed,
                "r2_submitted_compat_200ep": r2_compat,
                "per_episode_stats": per_ep_stats,
            })
            print(f"[{enc_name}] probe seed {pseed}: R²(compat-200ep)={r2_compat:+.4f} "
                  f"({time.time()-t0:.1f}s)", flush=True)

        # ── All-task 26 × 50 bootstrap CI ────────────────────────────
        # Group per-episode stats by task, MERGE across probe seeds (bootstrap
        # convention from finalize_wave1_stats.py)
        from collections import defaultdict
        tasks_set = sorted({(e[0], e[1]) for e in test_ep_ids})
        # merged pool: dict[task] -> list of per-episode-stats (probe seeds concatenated)
        merged_stats = defaultdict(list)
        for rec in probe_seed_records:
            for e_stats, e_id in zip(rec["per_episode_stats"], test_ep_ids):
                task = (e_id[0], e_id[1])
                merged_stats[task].append(e_stats)

        # Constant-action baseline (train mean predicts test)
        const_pred = const_baseline_predictions(train_acts, test_feats.shape[0], 32)
        const_r2_compat = pooled_r2(
            const_pred[:args.n_test_compat].reshape(-1, 7),
            test_acts.numpy()[:args.n_test_compat].reshape(-1, 7),
        )
        const_stats_by_task = defaultdict(list)
        # Constant is deterministic (no probe seed) — insert once per episode
        const_ep_stats = per_episode_sufficient(const_pred, test_acts.numpy())
        for e_stats, e_id in zip(const_ep_stats, test_ep_ids):
            const_stats_by_task[(e_id[0], e_id[1])].append(e_stats)

        # Bootstrap CI on CAIP R² and (CAIP - constant) difference
        boot = bootstrap_ci(merged_stats, const_stats_by_task, tasks_set,
                             iters=args.boot_iters, seed=args.boot_seed)

        # Also point estimate for compat scope (mean over probe seeds)
        compat_r2_per_seed = [rec["r2_submitted_compat_200ep"] for rec in probe_seed_records]
        compat_r2_mean = float(np.mean(compat_r2_per_seed))

        record = {
            "encoder": enc_name,
            "feat_dim": int(train_feats.shape[-1]),
            "n_train_episodes": args.n_train_episodes,
            "n_test_full": args.n_test_full,
            "n_test_compat": args.n_test_compat,
            "probe": {
                "arch": "MLP D->256->128->7 (GELU)",
                "optimizer": "AdamW",
                "lr": args.probe_lr, "weight_decay": args.probe_wd,
                "batch": args.probe_batch, "steps": args.probe_steps,
                "feature_normalize": "none", "action_normalize": "none",
                "seeds": args.probe_seeds,
            },
            "submitted_compat": {
                "n_episodes": args.n_test_compat,
                "r2_per_seed": compat_r2_per_seed,
                "r2_mean_over_seeds": compat_r2_mean,
                "constant_action_baseline_r2_at_compat": const_r2_compat,
            },
            "all_task_bootstrap": {
                "n_tasks": len(tasks_set),
                "n_test_episodes": args.n_test_full,
                "iters": args.boot_iters, "seed": args.boot_seed,
                "r2_point": boot["point"],
                "r2_ci95": [boot["lo95"], boot["hi95"]],
                "diff_vs_constant_baseline_point": boot["diff_vs_const_point"],
                "diff_vs_constant_baseline_ci95":
                    [boot["diff_vs_const_lo95"], boot["diff_vs_const_hi95"]],
            },
            "pinned_artifacts": {
                "submitted_train_split_sha256": SUBMITTED_TRAIN_SHA,
                "submitted_test_ood_split_sha256": SUBMITTED_TEST_SHA,
                "caip_hf_commit":       CAIP_HF_COMMIT if enc_name.startswith("caip_") else None,
                "caip_weights_sha256":  CAIP_WEIGHTS_SHA256 if enc_name.startswith("caip_") else None,
            },
            "captions_non_leakage_note": (
                "For caip_native_conditioned only: caption is the LIBERO "
                "task-description string (hdf5 filename with '_demo.hdf5' "
                "stripped and '_' → space). Verified to contain (a) no per-"
                "frame 7-D action vector, (b) no episode-specific identifier "
                "(same caption across all 50 demos of the same task), and "
                "(c) test-OOD tasks are disjoint from probe-train tasks by "
                "split construction."
            ) if enc_name == "caip_native_conditioned" else None,
        }
        per_encoder_records.append(record)

        # Incremental checkpoint so a mid-run crash keeps completed rows
        with open(args.out_jsonl, "w") as f:
            for r in per_encoder_records:
                f.write(json.dumps(r) + "\n")

    # ── CSV summary ──
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "encoder", "feat_dim",
            "r2_compat_200ep_mean", "r2_compat_per_seed_json",
            "r2_all26tasks_point", "r2_all26tasks_ci95_lo", "r2_all26tasks_ci95_hi",
            "diff_vs_const_point", "diff_vs_const_ci95_lo", "diff_vs_const_ci95_hi",
            "const_baseline_r2_at_compat", "reported_separately_from_vision_only",
        ])
        for r in per_encoder_records:
            w.writerow([
                r["encoder"], r["feat_dim"],
                f'{r["submitted_compat"]["r2_mean_over_seeds"]:+.6f}',
                json.dumps([round(x, 6) for x in r["submitted_compat"]["r2_per_seed"]]),
                f'{r["all_task_bootstrap"]["r2_point"]:+.6f}',
                f'{r["all_task_bootstrap"]["r2_ci95"][0]:+.6f}',
                f'{r["all_task_bootstrap"]["r2_ci95"][1]:+.6f}',
                f'{r["all_task_bootstrap"]["diff_vs_constant_baseline_point"]:+.6f}',
                f'{r["all_task_bootstrap"]["diff_vs_constant_baseline_ci95"][0]:+.6f}',
                f'{r["all_task_bootstrap"]["diff_vs_constant_baseline_ci95"][1]:+.6f}',
                f'{r["submitted_compat"]["constant_action_baseline_r2_at_compat"]:+.6f}',
                "YES" if r["encoder"] == "caip_native_conditioned" else "NO",
            ])

    # ── Markdown report ──
    lines = ["# CAIP_EXACT_PROTOCOL.md — final CAIP action-centric baseline\n\n"]
    lines.append("> **PACKAGE**: `ORIGINAL_CLUSTER_SECONDARY_EVIDENCE`. "
                 "Absolute R² values here are conditional on this exact "
                 "readout protocol. Do NOT merge with the GCP exact-split "
                 "matrix; only within-protocol ordering and paired deltas "
                 "transfer. `caip_native_conditioned` receives a distinct "
                 "information channel (task text) and is **reported "
                 "separately**; never mixed into the vision-only ranking.\n\n")

    lines.append("## Pinned artifacts\n\n")
    lines.append("| Component | Value |\n|-----------|-------|\n")
    lines.append(f"| CAIP HuggingFace commit | `{CAIP_HF_COMMIT}` |\n")
    lines.append(f"| CAIP model.safetensors SHA-256 | `{CAIP_WEIGHTS_SHA256}` |\n")
    lines.append(f"| Submitted train split (`data/splits_taskheld/train.json`) SHA-256 | `{SUBMITTED_TRAIN_SHA}` |\n")
    lines.append(f"| Submitted test-OOD split (`data/splits_taskheld/test_ood.json`) SHA-256 | `{SUBMITTED_TEST_SHA}` |\n")
    lines.append(f"| V-JEPA2 ViT-L ckpt | `datasets/external_models/vjepa2_ckpts/vitl.pt` (fbaipublicfiles) |\n")
    lines.append(f"| VideoMAE ViT-L | `MCG-NJU/videomae-large` (HF snapshot in `datasets/external_models/videomae`) |\n")
    lines.append(f"| Web-DINO 300M | `facebook/webssl-dino300m-full2b-224` (HF snapshot in `datasets/external_models/webdino`) |\n\n")

    lines.append("## Protocol\n\n")
    lines.append("Frozen features from each encoder → per-frame single-target "
                 "prediction with an identical MLP probe. **No 20k-step "
                 "fine-tuning was launched for this experiment** — encoders "
                 "are used as-is from their public checkpoints.\n\n")
    lines.append("| Component | Value |\n|-----------|-------|\n")
    lines.append("| Probe head | MLP D → 256 → 128 → 7 (GELU) |\n")
    lines.append("| Optimizer | AdamW |\n")
    lines.append(f"| Learning rate | {args.probe_lr:.0e} |\n")
    lines.append(f"| Weight decay | {args.probe_wd:.0e} |\n")
    lines.append(f"| Batch | {args.probe_batch} |\n")
    lines.append(f"| Steps | {args.probe_steps} |\n")
    lines.append("| Feature normalization | NONE |\n")
    lines.append("| Action normalization | NONE |\n")
    lines.append(f"| Probe seeds | {args.probe_seeds} |\n")
    lines.append(f"| Aggregate | mean over 7 per-dimension R² |\n")
    lines.append(f"| Train pool | first {args.n_train_episodes} episodes of submitted train split, one length-32 clip per episode |\n")
    lines.append(f"| Submitted-compat point-est test pool | first {args.n_test_compat} episodes of submitted test-OOD split |\n")
    lines.append(f"| All-task test pool | all {args.n_test_full} episodes (26 tasks × 50 demos) |\n")
    lines.append(f"| Bootstrap | {args.boot_iters}-replicate held-out-task cluster over 26 tasks with replacement, seed={args.boot_seed} |\n\n")

    lines.append("## CAIP two-mode extraction\n\n")
    lines.append("* `caip_visual_only` — **no task text input**; extract "
                 "`patch_features` from CAIP, mean-pool the 256 patch tokens "
                 "over the token dimension, resulting 1024-D vector is `z_t`.\n")
    lines.append("* `caip_native_conditioned` — per-episode LIBERO task "
                 "caption (from the hdf5 filename); extract `image_pooled` "
                 "(text-conditioned image pooling), resulting 1024-D vector "
                 "is `z_t`. Same caption tiled across all 32 frames.\n")
    lines.append("* **Non-leakage verification** for the native caption: "
                 "(a) captions contain no per-frame 7-D action vector; "
                 "(b) captions contain no episode-specific identifier "
                 "(demo index / trajectory id) — identical across all 50 "
                 "demos of the same task; (c) test-OOD task set (26 tasks) "
                 "is disjoint from the probe-train task set by construction "
                 "of the split. Therefore the caption channel cannot supply "
                 "held-out action information — it supplies the task label only.\n\n")

    lines.append("## Submitted-compatible point estimate (first "
                 f"{args.n_test_compat} test episodes)\n\n")
    lines.append("| Encoder | R² (probe seed 0) | R² (probe seed 1) | R² (probe seed 2) | R² mean over seeds |\n")
    lines.append("|---------|---------------------|---------------------|---------------------|--------------------|\n")
    # Order: vision-only first, CAIP-visual, then CAIP-native SEPARATELY
    order_vision = [e for e in per_encoder_records if e["encoder"] in ("vjepa2","videomae","webdino","caip_visual_only")]
    order_native = [e for e in per_encoder_records if e["encoder"] == "caip_native_conditioned"]
    for r in order_vision:
        s = r["submitted_compat"]["r2_per_seed"]
        m = r["submitted_compat"]["r2_mean_over_seeds"]
        lines.append(f"| {r['encoder']} | {s[0]:+.4f} | {s[1]:+.4f} | {s[2]:+.4f} | **{m:+.4f}** |\n")
    lines.append("\n### `caip_native_conditioned` — reported separately (uses task text)\n\n")
    lines.append("| Encoder | R² (probe seed 0) | R² (probe seed 1) | R² (probe seed 2) | R² mean over seeds |\n")
    lines.append("|---------|---------------------|---------------------|---------------------|--------------------|\n")
    for r in order_native:
        s = r["submitted_compat"]["r2_per_seed"]
        m = r["submitted_compat"]["r2_mean_over_seeds"]
        lines.append(f"| {r['encoder']} | {s[0]:+.4f} | {s[1]:+.4f} | {s[2]:+.4f} | **{m:+.4f}** |\n")
    lines.append(f"\n**Constant-action baseline** at this scope: "
                 f"R² = {per_encoder_records[0]['submitted_compat']['constant_action_baseline_r2_at_compat']:+.4f} "
                 f"(predict the training mean action on every test frame).\n\n")

    lines.append("## All-task 10 000-replicate held-out-task bootstrap "
                 "(all 26 OOD tasks × 50 episodes)\n\n")
    lines.append("Bootstrap = 10 000 resamples of the 26 tasks with "
                 "replacement, seed=0; for each replicate we recompute pooled "
                 "R² for the encoder readout AND for the constant-action "
                 "baseline on the sampled tasks; the paired difference CI "
                 "uses the same task draws for both quantities.\n\n")
    lines.append("| Encoder | R² point (26-task) | 95% CI | R² − constant point | 95% CI on difference |\n")
    lines.append("|---------|---------------------|--------|---------------------|-----------------------|\n")
    for r in order_vision:
        b = r["all_task_bootstrap"]
        lines.append(f"| {r['encoder']} | {b['r2_point']:+.4f} | "
                     f"[{b['r2_ci95'][0]:+.4f}, {b['r2_ci95'][1]:+.4f}] | "
                     f"{b['diff_vs_constant_baseline_point']:+.4f} | "
                     f"[{b['diff_vs_constant_baseline_ci95'][0]:+.4f}, "
                     f"{b['diff_vs_constant_baseline_ci95'][1]:+.4f}] |\n")
    lines.append("\n### `caip_native_conditioned` — reported separately\n\n")
    lines.append("| Encoder | R² point (26-task) | 95% CI | R² − constant point | 95% CI on difference |\n")
    lines.append("|---------|---------------------|--------|---------------------|-----------------------|\n")
    for r in order_native:
        b = r["all_task_bootstrap"]
        lines.append(f"| {r['encoder']} | {b['r2_point']:+.4f} | "
                     f"[{b['r2_ci95'][0]:+.4f}, {b['r2_ci95'][1]:+.4f}] | "
                     f"{b['diff_vs_constant_baseline_point']:+.4f} | "
                     f"[{b['diff_vs_constant_baseline_ci95'][0]:+.4f}, "
                     f"{b['diff_vs_constant_baseline_ci95'][1]:+.4f}] |\n")

    lines.append("\n## Wording rule\n\n")
    lines.append("Report only what the measured readout supports. Do NOT "
                 "describe a negative or below-baseline result as a "
                 "\"structural failure\" — this experiment measures the "
                 "R² of a specific frozen-feature MLP readout on a specific "
                 "split; a below-baseline result is a statement about this "
                 "readout, not about the encoder's internal geometry.\n")
    lines.append("\nCorrect phrasings:\n\n")
    lines.append("* *\"Encoder X's frozen features do not exceed the "
                 "constant-action baseline under the submitted-compatible "
                 "readout on this split (R² point [value], 95% CI …).\"*\n")
    lines.append("* *\"Encoder X's frozen features exceed the constant-"
                 "action baseline (paired difference [value], 95% CI …).\"*\n")
    lines.append("\nIncorrect phrasings to avoid: *\"structural encoding "
                 "failure\", \"cannot encode actions\", \"the model does not "
                 "work on LIBERO\"*.\n")

    Path(args.out_md).write_text("".join(lines))
    print(f"\nwrote {args.out_jsonl}\nwrote {args.out_csv}\nwrote {args.out_md}")


if __name__ == "__main__":
    main()
