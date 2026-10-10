"""B0 smoke test: evaluate frozen extractors on Robomimic.

Reuses the cross-arch eval pipeline. Verifies:
  - Robomimic dataset loader works
  - Frozen V-JEPA 2 / V-JEPA 2.1 / LAPA / TOKEN encode Robomimic clips
    without crashing
  - Probe metrics (action R², inv-dyn R², temporal cos) compute
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from comparison.datasets.robomimic import RobomimicVideoClips, worker_init_fn
from comparison.eval_pipeline import build_extractor
from comparison.metrics import temporal_consistency
from eval_downstream import (
    MLPProbe, aggregate_seeds, make_horizon_dataset,
    make_inverse_dynamics_dataset, train_probe,
)


# task → int id, for task probe (multi-task R² baseline)
TASK_ID = {"lift": 0, "square": 1, "transport": 2}


@torch.no_grad()
def encode_robomimic(extractor, split_path: str, n_clips: int,
                     device: torch.device, seed: int) -> dict:
    ds = RobomimicVideoClips(
        split_path=split_path,
        datasets_dir=str(REPO_ROOT / "datasets" / "robomimic"),
        clip_len=32,
        cameras=("agentview_image", "robot0_eye_in_hand_image"),
        resolution=128,
        seed=seed,
        return_actions=True,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2,
                        worker_init_fn=worker_init_fn)
    feat_t, act_t, feat_clip, task_ids = [], [], [], []
    seen = 0
    for clip, actions in loader:
        clip = clip.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                 enabled=device.type == "cuda"):
            f = extractor.extract(clip)         # (B, T, D)
        f = f.float()
        feat_t.append(f.cpu())
        act_t.append(actions.cpu())
        feat_clip.append(f.mean(dim=1).cpu())
        # task ids per batch
        for j in range(clip.shape[0]):
            ep = ds.episodes[seen + j]
            task_ids.append(TASK_ID[ep["task"]])
        seen += clip.shape[0]
        if seen >= n_clips:
            break
    return {
        "feat_per_clip_t": torch.cat(feat_t, dim=0),
        "actions_per_clip": torch.cat(act_t, dim=0),
        "feat_per_clip": torch.cat(feat_clip, dim=0),
        "task_per_clip": torch.tensor(task_ids, dtype=torch.long),
        "n_clips": seen,
    }


def evaluate_extractor(name: str, extractor, train_split: str, test_split: str,
                       n_train: int, n_test: int, device: torch.device,
                       seeds: list[int], hidden_dims: tuple) -> dict:
    print(f"\n=== {name} ({extractor.meta.family}) ===")
    t0 = time.time()
    train = encode_robomimic(extractor, train_split, n_train, device, seed=53)
    test  = encode_robomimic(extractor, test_split, n_test, device, seed=54)
    print(f"  encoded train={train['n_clips']} test={test['n_clips']} "
          f"({time.time()-t0:.1f}s)")

    feat_dim = train["feat_per_clip"].shape[-1]
    action_dim = train["actions_per_clip"].shape[-1]

    def run_seeded(make_probe, tx, ty, vx, vy, *, obj, n_steps):
        runs = []
        for s in seeds:
            torch.manual_seed(s)
            probe = make_probe().to(device)
            m = train_probe(probe, tx, ty, vx, vy, objective=obj,
                             steps=n_steps, batch_size=512, lr=3e-4,
                             weight_decay=1e-4, device=device)
            runs.append(m)
        return aggregate_seeds(runs)

    a_tx, a_ty = make_horizon_dataset(train["feat_per_clip_t"],
                                       train["actions_per_clip"], 1)
    a_vx, a_vy = make_horizon_dataset(test["feat_per_clip_t"],
                                       test["actions_per_clip"], 1)
    action = run_seeded(
        lambda: MLPProbe(feat_dim, action_dim, hidden_dims=hidden_dims),
        a_tx, a_ty, a_vx, a_vy, obj="regression", n_steps=3000)

    i_tx, i_ty = make_inverse_dynamics_dataset(
        train["feat_per_clip_t"], train["actions_per_clip"])
    i_vx, i_vy = make_inverse_dynamics_dataset(
        test["feat_per_clip_t"], test["actions_per_clip"])
    invdyn = run_seeded(
        lambda: MLPProbe(2 * feat_dim, action_dim, hidden_dims=hidden_dims),
        i_tx, i_ty, i_vx, i_vy, obj="regression", n_steps=3000)

    # task probe: 3-class (lift/square/transport)
    task = run_seeded(
        lambda: MLPProbe(feat_dim, 3, hidden_dims=()),
        train["feat_per_clip"], train["task_per_clip"],
        test["feat_per_clip"], test["task_per_clip"],
        obj="classification", n_steps=1500)

    tc = temporal_consistency(test["feat_per_clip_t"])

    return {
        "name": name,
        "meta": asdict(extractor.meta),
        "feat_dim": int(feat_dim),
        "action_h1_r2_mean": float(action["test_r2_mean"]),
        "action_h1_r2_std":  float(action["test_r2_std"]),
        "invdyn_r2_mean":    float(invdyn["test_r2_mean"]),
        "invdyn_r2_std":     float(invdyn["test_r2_std"]),
        "task_top1_mean":    float(task["test_top1_mean"]),
        "task_top1_std":     float(task["test_top1_std"]),
        "temporal_cos_adjacent": float(tc.cos_adjacent),
        "temporal_cos_curve_ks": tc.cos_curve_ks,
        "temporal_cos_curve_vals": tc.cos_curve_vals,
    }


def main():
    # full Stage B output dir (preserves smoke results in older dir)
    out_dir = REPO_ROOT / "runs" / "comparison" / "robomimic_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    train_split = str(REPO_ROOT / "data" / "splits_robomimic" / "train.json")
    test_split  = str(REPO_ROOT / "data" / "splits_robomimic" / "test_indist.json")

    # Full set: frozen baselines (kept from B0) + Robomimic-fine-tuned variants
    extractor_specs = [
        # ── frozen / LIBERO-trained reference rows ───────────────────────────
        {"name": "DIFF-LIBERO-aux0p05 (zero-shot transfer)", "family": "diff",
         "ckpt": "runs/sweep/aux_loss_extension/aux_v6_large_noact_taskheld_lam0p05/ckpt_last.pt"},
        {"name": "TOKEN-200k (LIBERO-trained, zero-shot)", "family": "token",
         "ckpt": "runs/sweep/variants/v_mix_token_noact_200k/ckpt_last.pt"},
        {"name": "V-JEPA2-vit_large frozen", "family": "vjepa2", "arch": "vit_large",
         "num_frames": 32, "img_size": 256},
        {"name": "V-JEPA2.1-vit_base_384 frozen", "family": "vjepa2_1",
         "arch": "vit_base", "num_frames": 32, "img_size": 384},
        {"name": "LAPA-laq_openx", "family": "lapa"},
        # ── Robomimic-trained variants (Stage B) ─────────────────────────────
        {"name": "DIFF-Robomimic-aux0p05", "family": "diff",
         "ckpt": "runs/comparison_robomimic/diff_aux/ckpt_last.pt"},
        {"name": "TOKEN-Robomimic", "family": "token",
         "ckpt": "runs/comparison_robomimic/token/ckpt_last.pt"},
        {"name": "V-JEPA2-vit_large + aux (Robomimic)", "family": "vjepa2",
         "arch": "vit_large", "num_frames": 32, "img_size": 256,
         "ckpt": "runs/comparison_robomimic/vjepa2_aux/ckpt_last.pt"},
        {"name": "V-JEPA2.1-vit_base + aux (Robomimic)", "family": "vjepa2_1",
         "arch": "vit_base", "num_frames": 32, "img_size": 384,
         "ckpt": "runs/comparison_robomimic/vjepa2_1_aux/ckpt_last.pt",
         "ckpt_key": "encoder"},
        {"name": "ViT-L scratch + aux (Robomimic)", "family": "vjepa2",
         "arch": "vit_large", "num_frames": 32, "img_size": 256,
         "ckpt": "runs/comparison_robomimic/vitl_scratch_aux/ckpt_last.pt"},
        {"name": "V-JEPA2 + SSL (Robomimic)", "family": "vjepa2",
         "arch": "vit_large", "num_frames": 32, "img_size": 256,
         "ckpt": "runs/comparison_robomimic/vjepa2_ssl/ckpt_last.pt"},
    ]

    rows = []
    results_path = out_dir / "results.json"
    if results_path.exists():
        rows = json.loads(results_path.read_text())
        done_names = {r["name"] for r in rows}
        print(f"resuming, already done: {sorted(done_names)}")
    else:
        done_names = set()

    for spec in extractor_specs:
        if spec["name"] in done_names:
            print(f"[skip] {spec['name']}")
            continue
        try:
            ext = build_extractor(spec, device)
        except Exception as e:
            print(f"[skip] {spec['name']}: build failed — {e}")
            rows.append({"name": spec["name"], "error": f"build: {e}"})
            continue
        try:
            r = evaluate_extractor(
                spec["name"], ext, train_split, test_split,
                n_train=300, n_test=100,  # smoke test sample size
                device=device, seeds=[0, 1, 2],
                hidden_dims=(256, 128),
            )
            rows.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            rows.append({"name": spec["name"], "error": f"eval: {e}"})
        finally:
            del ext
            torch.cuda.empty_cache()
        results_path.write_text(json.dumps(rows, indent=2))

    # markdown summary
    md = ["# Robomimic B0 smoke test (frozen extractors, LIBERO-trained DIFF/TOKEN re-used)\n",
          "Demo-level held-out, 3-task multi-task probe.\n",
          "| Model | feat_dim | action R² (H=1) | inv-dyn R² | task top-1 (3-cls) | temp cos_adj |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            md.append(f"| {r['name']} | — | ERR ({r['error'][:40]}) | | | |")
            continue
        md.append(
            f"| {r['name']} | {r['feat_dim']} | "
            f"{r['action_h1_r2_mean']:.4f}±{r['action_h1_r2_std']:.4f} | "
            f"{r['invdyn_r2_mean']:.4f}±{r['invdyn_r2_std']:.4f} | "
            f"{r['task_top1_mean']:.4f} | "
            f"{r['temporal_cos_adjacent']:.4f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {out_dir/'summary.md'}")


if __name__ == "__main__":
    main()
