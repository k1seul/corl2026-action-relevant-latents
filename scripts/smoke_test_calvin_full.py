"""CALVIN-D task-OOD smoke test (frozen extractors).

7 of 34 tasks held out. Uses lang-annotation-defined 64-frame windows
as 'demos'. Train = training subset's 27-task annotations (4069 eps),
test_OOD = validation subset's 7 held-out task annotations (207 eps).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch
from torch.utils.data import DataLoader

from comparison.datasets.calvin import CalvinClips, worker_init_fn
from comparison.eval_pipeline import build_extractor
from comparison.metrics import temporal_consistency
from eval_downstream import (
    MLPProbe, aggregate_seeds, make_horizon_dataset,
    make_inverse_dynamics_dataset, train_probe,
)


@torch.no_grad()
def encode_calvin(extractor, split_path: str, n_clips: int,
                   device: torch.device, seed: int,
                   expect_2cam: bool) -> dict:
    cameras = ("rgb_static", "rgb_gripper") if expect_2cam else ("rgb_static",)
    ds = CalvinClips(
        split_path=split_path, datasets_dir=None,   # use data_root from JSON
        clip_len=32, cameras=cameras,
        resolution=128, seed=seed, return_actions=True,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2,
                         worker_init_fn=worker_init_fn)
    feat_t, act_t, feat_clip, task_ids = [], [], [], []
    seen = 0
    for clip, actions in loader:
        clip = clip.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                 enabled=device.type == "cuda"):
            f = extractor.extract(clip).float()
        feat_t.append(f.cpu())
        act_t.append(actions.cpu())
        feat_clip.append(f.mean(dim=1).cpu())
        for j in range(clip.shape[0]):
            ep = ds.episodes[seen + j]
            task_ids.append(ep.get("task_index", -1))
        seen += clip.shape[0]
        if seen >= n_clips: break
    return {"feat_per_clip_t": torch.cat(feat_t),
            "actions_per_clip": torch.cat(act_t),
            "feat_per_clip": torch.cat(feat_clip),
            "task_per_clip": torch.tensor(task_ids, dtype=torch.long),
            "n_clips": seen}


def main():
    out_dir = REPO_ROOT / "runs" / "comparison" / "calvin_smoke_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    train_split = str(REPO_ROOT / "data" / "splits_calvin_taskheld" / "train.json")
    test_split = str(REPO_ROOT / "data" / "splits_calvin_taskheld" / "test_ood.json")

    extractor_specs = [
        {"name": "DIFF-LIBERO-aux0p05", "family": "diff",
         "ckpt": "runs/sweep/aux_loss_extension/aux_v6_large_noact_taskheld_lam0p05/ckpt_last.pt",
         "_2cam": True},
        {"name": "TOKEN-200k LIBERO", "family": "token",
         "ckpt": "runs/sweep/variants/v_mix_token_noact_200k/ckpt_last.pt",
         "_2cam": False},
        {"name": "V-JEPA2-vit_large frozen", "family": "vjepa2",
         "arch": "vit_large", "num_frames": 32, "img_size": 256, "_2cam": False},
        {"name": "V-JEPA2.1-vit_base_384 frozen", "family": "vjepa2_1",
         "arch": "vit_base", "num_frames": 32, "img_size": 384, "_2cam": False},
        {"name": "LAPA-laq_openx", "family": "lapa", "_2cam": False},
        # Stage B CALVIN-trained
        {"name": "V-JEPA 2 ViT-L + aux (CALVIN)", "family": "vjepa2",
         "arch": "vit_large", "num_frames": 32, "img_size": 256,
         "ckpt": "runs/comparison_calvin/vjepa2_aux/ckpt_last.pt", "_2cam": False},
        {"name": "V-JEPA 2.1 ViT-B + aux (CALVIN)", "family": "vjepa2_1",
         "arch": "vit_base", "num_frames": 32, "img_size": 384,
         "ckpt": "runs/comparison_calvin/vjepa2_1_aux/ckpt_last.pt",
         "ckpt_key": "encoder", "_2cam": False},
        {"name": "ViT-L scratch + aux (CALVIN)", "family": "vjepa2",
         "arch": "vit_large", "num_frames": 32, "img_size": 256,
         "ckpt": "runs/comparison_calvin/vitl_scratch_aux/ckpt_last.pt", "_2cam": False},
        {"name": "V-JEPA 2 + SSL (CALVIN)", "family": "vjepa2",
         "arch": "vit_large", "num_frames": 32, "img_size": 256,
         "ckpt": "runs/comparison_calvin/vjepa2_ssl/ckpt_last.pt", "_2cam": False},
        {"name": "DIFF + aux (CALVIN)", "family": "diff",
         "ckpt": "runs/comparison_calvin/diff_aux/ckpt_last.pt", "_2cam": True},
        {"name": "TOKEN (CALVIN)", "family": "token",
         "ckpt": "runs/comparison_calvin/token/ckpt_last.pt", "_2cam": False},
    ]

    rows = []
    results_path = out_dir / "results.json"
    if results_path.exists():
        rows = json.loads(results_path.read_text())
        done_names = {r["name"] for r in rows if "error" not in r}
        print(f"resuming, done: {sorted(done_names)}")
    else:
        done_names = set()

    for spec in extractor_specs:
        if spec["name"] in done_names:
            print(f"[skip] {spec['name']}"); continue
        expect_2cam = spec.pop("_2cam")
        print(f"\n=== {spec['name']} (2cam={expect_2cam}) ===")
        try:
            ext = build_extractor(spec, device)
        except Exception as e:
            print(f"  build failed: {e}"); continue
        try:
            t0 = time.time()
            train = encode_calvin(ext, train_split, n_clips=400,
                                    device=device, seed=53, expect_2cam=expect_2cam)
            test = encode_calvin(ext, test_split, n_clips=200,
                                   device=device, seed=54, expect_2cam=expect_2cam)
            print(f"  encoded train={train['n_clips']} test_OOD={test['n_clips']} "
                  f"({time.time()-t0:.1f}s)")

            feat_dim = train["feat_per_clip"].shape[-1]
            ad = train["actions_per_clip"].shape[-1]

            # action probe (H=1)
            tx, ty = make_horizon_dataset(train["feat_per_clip_t"],
                                            train["actions_per_clip"], 1)
            vx, vy = make_horizon_dataset(test["feat_per_clip_t"],
                                            test["actions_per_clip"], 1)
            runs = []
            for s in (0, 1, 2):
                torch.manual_seed(s)
                p = MLPProbe(feat_dim, ad, hidden_dims=(256, 128)).to(device)
                m = train_probe(p, tx, ty, vx, vy, objective="regression",
                                  steps=3000, batch_size=512, lr=3e-4,
                                  weight_decay=1e-4, device=device)
                runs.append(m)
            ag = aggregate_seeds(runs)

            # inv-dyn probe
            itx, ity = make_inverse_dynamics_dataset(train["feat_per_clip_t"],
                                                       train["actions_per_clip"])
            ivx, ivy = make_inverse_dynamics_dataset(test["feat_per_clip_t"],
                                                       test["actions_per_clip"])
            iruns = []
            for s in (0, 1, 2):
                torch.manual_seed(s)
                p = MLPProbe(2 * feat_dim, ad, hidden_dims=(256, 128)).to(device)
                m = train_probe(p, itx, ity, ivx, ivy, objective="regression",
                                  steps=3000, batch_size=512, lr=3e-4,
                                  weight_decay=1e-4, device=device)
                iruns.append(m)
            iag = aggregate_seeds(iruns)

            tc = temporal_consistency(test["feat_per_clip_t"])
            rows.append({
                "name": spec["name"],
                "feat_dim": int(feat_dim),
                "action_h1_r2_mean": float(ag["test_r2_mean"]),
                "action_h1_r2_std": float(ag["test_r2_std"]),
                "invdyn_r2_mean": float(iag["test_r2_mean"]),
                "invdyn_r2_std": float(iag["test_r2_std"]),
                "temporal_cos_adjacent": float(tc.cos_adjacent),
            })
            print(f"  action R²={ag['test_r2_mean']:.4f}±{ag['test_r2_std']:.4f}")
            print(f"  inv-dyn R²={iag['test_r2_mean']:.4f}±{iag['test_r2_std']:.4f}")
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append({"name": spec["name"], "error": str(e)})
        finally:
            del ext; torch.cuda.empty_cache()
            results_path.write_text(json.dumps(rows, indent=2))

    md = ["# CALVIN-D task-OOD smoke test (frozen extractors)\n",
          "Task-level held-out: 7 of 34 tasks held out. 400 train / 200 OOD test clips "
          "(language-annotation 64-frame windows).\n",
          "| Model | feat_dim | OOD action R² | OOD inv-dyn R² | temp cos_adj |",
          "|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            md.append(f"| {r['name']} | — | ERR ({r['error'][:40]}) | | |")
            continue
        md.append(
            f"| {r['name']} | {r['feat_dim']} | "
            f"{r['action_h1_r2_mean']:.4f}±{r['action_h1_r2_std']:.4f} | "
            f"{r['invdyn_r2_mean']:.4f}±{r['invdyn_r2_std']:.4f} | "
            f"{r['temporal_cos_adjacent']:.4f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
