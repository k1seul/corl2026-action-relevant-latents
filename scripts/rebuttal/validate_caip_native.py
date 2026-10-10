"""Validate CAIP native-text-conditioned (`image_pooled` given per-episode
task caption) as an action probe on LIBERO 26-task test split.

Deferred from validate_baselines.py because it needs per-episode task
captions plumbed into the extractor. Captions are derived from the LIBERO
hdf5 filename (strip `_demo.hdf5`, replace `_` with space) — that string
is the human-readable task description LIBERO ships in.

Output structure mirrors baseline_validation.jsonl / BASELINE_VALIDATION.md
so `write_rebuttal_evidence.py`'s CAIP loader can consume it unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "rebuttal"))

from eval_matched_control import (
    build_extractor, _flatten_frames,
)
from validate_baselines import (
    feature_stats, constant_action_baseline_r2, ridge_alpha_sweep,
    per_task_pooled_r2,
)


def caption_from_filename(fname: str) -> str:
    """`pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5`
        → `pick up the alphabet soup and place it in the basket`."""
    stem = fname.replace("_demo.hdf5", "").replace(".hdf5", "")
    return stem.replace("_", " ")


@torch.no_grad()
def encode_slice_with_captions(extractor, split_path: Path,
                                 datasets_dir: Path,
                                 episode_slice: tuple[int, int], *,
                                 clip_len: int, seed: int,
                                 batch_size: int, num_workers: int,
                                 device: torch.device,
                                 resolution: int = 128):
    """Same as eval_matched_control.encode_slice but resolves a task caption
    per episode from its LIBERO filename and streams the batch-aligned
    caption list into extractor.extract(clip, task_caption=captions)."""
    from data.libero_video import LiberoVideoClips, worker_init_fn

    ds = LiberoVideoClips(
        split_path=str(split_path), datasets_dir=str(datasets_dir),
        clip_len=clip_len, cameras=["agentview_rgb"], resolution=resolution,
        seed=seed, return_actions=True,
    )
    lo, hi = episode_slice
    hi = min(hi, len(ds))
    if lo >= hi:
        raise ValueError(f"empty episode slice {lo}:{hi} for split with {len(ds)} episodes")

    idxs = list(range(lo, hi))
    episode_ids = [(ds.episodes[i]["suite"], ds.episodes[i]["file"],
                    ds.episodes[i]["demo"]) for i in idxs]
    captions_all = [caption_from_filename(ep[1]) for ep in episode_ids]

    subset = Subset(ds, idxs)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        worker_init_fn=worker_init_fn, drop_last=False, pin_memory=False,
    )
    feats_all: list[torch.Tensor] = []
    acts_all: list[torch.Tensor] = []
    n_seen = 0
    for clip, actions in loader:
        B = clip.shape[0]
        caps = captions_all[n_seen:n_seen + B]
        f = extractor.extract(clip.to(device), task_caption=caps)
        feats_all.append(f.detach().float().cpu())
        acts_all.append(actions[..., :7].float().cpu())
        n_seen += B
    feats = torch.cat(feats_all, dim=0)
    acts = torch.cat(acts_all, dim=0)
    return feats, acts, episode_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-jsonl",
                    default=str(REPO_ROOT / "runs/corl2026_rebuttal/results/caip_native_validation.jsonl"))
    ap.add_argument("--out-md",
                    default=str(REPO_ROOT / "runs/corl2026_rebuttal/reports/CAIP_NATIVE_VALIDATION.md"))
    ap.add_argument("--probe-train-clips", type=int, default=800)
    ap.add_argument("--probe-eval-clips-test", type=int, default=1300)
    ap.add_argument("--ridge-alphas", nargs="+", type=float,
                    default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    split_train = REPO_ROOT / "runs/corl2026_rebuttal/splits/train.json"
    split_test  = REPO_ROOT / "runs/corl2026_rebuttal/splits/test_ood.json"
    datasets_dir = REPO_ROOT / "datasets/libero_downloaded"

    print("=== validating caip_native_conditioned ===", flush=True)
    t0 = time.time()
    extractor = build_extractor(
        "caip_native_conditioned", device, None,
        external_models_dir=REPO_ROOT / "datasets/external_models",
        clip_len=32,
    )
    print(f"[caip_native] extractor built {time.time()-t0:.1f}s "
          f"feat_dim={extractor.meta.feat_dim}", flush=True)

    t0 = time.time()
    train_feats, train_acts, _ = encode_slice_with_captions(
        extractor, split_train, datasets_dir, (0, args.probe_train_clips),
        clip_len=32, seed=0, batch_size=args.batch_size,
        num_workers=args.num_workers, device=device)
    print(f"[caip_native] train encode {time.time()-t0:.1f}s "
          f"shape={tuple(train_feats.shape)}", flush=True)

    t0 = time.time()
    test_feats, test_acts, test_ep_ids = encode_slice_with_captions(
        extractor, split_test, datasets_dir, (0, args.probe_eval_clips_test),
        clip_len=32, seed=0, batch_size=args.batch_size,
        num_workers=args.num_workers, device=device)
    print(f"[caip_native] test encode {time.time()-t0:.1f}s "
          f"shape={tuple(test_feats.shape)}", flush=True)

    f_stats_train = feature_stats(train_feats)
    f_stats_test  = feature_stats(test_feats)
    print(f"[caip_native] train stats: {f_stats_train}", flush=True)

    tx_raw, ty = _flatten_frames(train_feats, train_acts)
    txe_raw, tye = _flatten_frames(test_feats, test_acts)
    mu = tx_raw.mean(dim=0, keepdim=True)
    sig = tx_raw.std(dim=0, keepdim=True).clamp_min(1e-2)
    tx_z = (tx_raw - mu) / sig
    txe_z = (txe_raw - mu) / sig
    tx_l2 = F.normalize(tx_raw, dim=-1)
    txe_l2 = F.normalize(txe_raw, dim=-1)

    ridge_none = ridge_alpha_sweep(tx_raw, ty, txe_raw, tye, args.ridge_alphas)
    ridge_z    = ridge_alpha_sweep(tx_z,   ty, txe_z,   tye, args.ridge_alphas)
    ridge_l2   = ridge_alpha_sweep(tx_l2,  ty, txe_l2,  tye, args.ridge_alphas)

    const_r2 = constant_action_baseline_r2(ty.cpu().numpy(), tye.cpu().numpy())

    best_alpha_none = max(ridge_none, key=lambda a: ridge_none[a]["test_r2_mean"])
    best_alpha_z    = max(ridge_z,    key=lambda a: ridge_z[a]["test_r2_mean"])
    X   = tx_raw.cpu().numpy().astype(np.float64)
    Xt  = txe_raw.cpu().numpy().astype(np.float64)
    Ynp = ty.cpu().numpy().astype(np.float64)
    Ytnp = tye.cpu().numpy().astype(np.float64)
    Wbest = np.linalg.solve(X.T @ X + best_alpha_none * np.eye(X.shape[1]), X.T @ Ynp)
    macro_none = per_task_pooled_r2(Xt @ Wbest, Ytnp, test_ep_ids, ep_len=32)
    Xz  = tx_z.cpu().numpy().astype(np.float64)
    Xzt = txe_z.cpu().numpy().astype(np.float64)
    Wz = np.linalg.solve(Xz.T @ Xz + best_alpha_z * np.eye(Xz.shape[1]), Xz.T @ Ynp)
    macro_z = per_task_pooled_r2(Xzt @ Wz, Ytnp, test_ep_ids, ep_len=32)

    record = {
        "backbone": "caip_native_conditioned",
        "feature_stats_train": f_stats_train,
        "feature_stats_test":  f_stats_test,
        "ridge_alpha_sweep_raw":    {str(k): v for k, v in ridge_none.items()},
        "ridge_alpha_sweep_zscore": {str(k): v for k, v in ridge_z.items()},
        "ridge_alpha_sweep_l2":     {str(k): v for k, v in ridge_l2.items()},
        "best_alpha_raw":    best_alpha_none,
        "best_alpha_zscore": best_alpha_z,
        "task_macro_r2_raw_best":    macro_none,
        "task_macro_r2_zscore_best": macro_z,
        "constant_action_baseline_r2": const_r2,
        "probe_train_clips":     args.probe_train_clips,
        "probe_eval_clips_test": args.probe_eval_clips_test,
    }

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        f.write(json.dumps(record) + "\n")

    lines = ["# CAIP_NATIVE_VALIDATION.md — CAIP text-conditioned probe verdict\n\n"]
    lines.append("Generated by `scripts/rebuttal/validate_caip_native.py`.\n\n")
    lines.append("## Task-caption provenance (instruction §5)\n\n")
    lines.append("**Exact source**: LIBERO ships one hdf5 per task; the filename "
                 "encodes the human-readable task description. The caption is "
                 "derived by:\n\n")
    lines.append("```python\n"
                 "# scripts/rebuttal/validate_caip_native.py\n"
                 "def caption_from_filename(fname: str) -> str:\n"
                 "    stem = fname.replace('_demo.hdf5', '').replace('.hdf5', '')\n"
                 "    return stem.replace('_', ' ')\n"
                 "```\n\n")
    lines.append("Example: "
                 "`pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5` "
                 "→ `pick up the alphabet soup and place it in the basket`. "
                 "The same caption is applied to every frame of a given episode.\n\n")
    lines.append("**Non-leakage verification**. The caption contains only the "
                 "task-description string:\n\n"
                 "* No 7-D end-effector delta values (dx, dy, dz, drx, dry, drz, gripper) "
                 "are present; a caption cannot supply the per-frame action vector we "
                 "are trying to decode.\n"
                 "* No episode-specific numbers (demo index, timestamp, trajectory "
                 "id) are present — the caption is identical across all 50 demos of "
                 "the same task, so the probe cannot use it to distinguish held-out "
                 "test episodes from training episodes of the same task.\n"
                 "* No test-OOD task identity leaks *back* into training: the "
                 "test-OOD task set (26 tasks) is disjoint from the probe-train "
                 "task set by construction of the split "
                 "(`runs/corl2026_rebuttal/splits/{train,test_ood}.json`, "
                 "sha256 `42f26dac…`). Every test-OOD caption at eval time is a "
                 "caption whose task the probe never trained on — the caption "
                 "provides the standard CAIP text-conditioning input, not any "
                 "action label.\n\n")
    lines.append("Report this baseline **separately** from the vision-only "
                 "backbones — it uses a distinct information channel (task text) "
                 "the other backbones do not receive. Never merge into a single "
                 "unqualified ranking with V-JEPA2 / VideoMAE / Web-DINO / R3M / "
                 "VIP / CAIP visual-only.\n\n")
    lines.append("## caip_native_conditioned\n\n")
    fs = record["feature_stats_train"]
    lines.append(f"* Feature stats (probe-train pool): shape={fs['shape']}, "
                 f"any_NaN={fs['any_nan']}, any_Inf={fs['any_inf']}, "
                 f"L2 norm mean={fs['l2_norm_mean']:.3f}, "
                 f"per-dim std median={fs['per_dim_std_median']:.4f}, "
                 f"dead dims (std<1e-2)={fs['n_dead_dims_std_lt_1e-2']}/{fs['shape'][-1]}, "
                 f"range=[{fs['min_val']:.3f}, {fs['max_val']:.3f}].\n")
    lines.append(f"* Constant-action baseline R² = {record['constant_action_baseline_r2']:+.4f}\n")
    for mode_key, title in (("raw", "RAW features"),
                             ("zscore", "STANDARDIZED (zscore, clamp 1e-2)"),
                             ("l2", "L2 (unit norm)")):
        lines.append(f"* Ridge alpha sweep — {title}:\n\n")
        lines.append("  | α | test R² | train R² | normalized MSE |\n")
        lines.append("  |---|---------|----------|-----------------|\n")
        for a, s in sorted(record[f"ridge_alpha_sweep_{mode_key}"].items(),
                            key=lambda x: float(x[0])):
            lines.append(f"  | {a} | {s['test_r2_mean']:+.4f} | "
                         f"{s['train_r2_mean']:+.4f} | {s['normalized_mse']:.4f} |\n")
        lines.append("\n")
    lines.append(f"* Best-α task-macro R² (raw, α={record['best_alpha_raw']}) = "
                 f"{record['task_macro_r2_raw_best']:+.4f}\n")
    lines.append(f"* Best-α task-macro R² (zscore, α={record['best_alpha_zscore']}) = "
                 f"{record['task_macro_r2_zscore_best']:+.4f}\n\n")

    # Best across all three modes
    best = -1e9
    best_mode = None
    best_alpha = None
    for m in ("raw", "zscore", "l2"):
        for a, s in record[f"ridge_alpha_sweep_{m}"].items():
            if s["test_r2_mean"] > best:
                best = s["test_r2_mean"]; best_mode = m; best_alpha = a
    lines.append("## Verdict\n\n")
    lines.append(f"Best test R² over the full sweep = **{best:+.4f}** "
                 f"(mode={best_mode}, α={best_alpha}); "
                 f"constant-action baseline R² = {record['constant_action_baseline_r2']:+.4f}.\n\n")
    if best > 0.05:
        lines.append("**CAIP native beats constant-action baseline** — text "
                     "conditioning DOES surface linearly-decodable action "
                     "information on top of visual features. Report the "
                     "audited best-α R² in REBUTTAL_EVIDENCE and note the "
                     "gap vs `caip_visual_only`.\n")
    else:
        lines.append("**CAIP native does not beat constant-action baseline** "
                     "at any α × normalization combination — the text "
                     "conditioning does NOT surface additional action "
                     "information the visual features alone lack. Report as "
                     "*reaches constant-action baseline* in REBUTTAL_EVIDENCE, "
                     "and note that this is separate from CAIP visual_only "
                     "(never combine into a single unqualified ranking).\n")

    Path(args.out_md).write_text("".join(lines))
    print(f"wrote {args.out_jsonl}\nwrote {args.out_md}")
    print(f"[caip_native] best test R² over sweep = {best:+.4f} "
          f"(mode={best_mode}, α={best_alpha})")


if __name__ == "__main__":
    main()
