"""CALVIN debug smoke test — verify loader + frozen extractors work.

Uses calvin_debug_dataset (~1.3GB, 1 episode of 2770 frames split into
27 pseudo-episodes of 100 frames). Just validates the pipeline runs.
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
from torch.utils.data import DataLoader

from comparison.datasets.calvin import CalvinClips, worker_init_fn
from comparison.eval_pipeline import build_extractor
from comparison.metrics import temporal_consistency
from eval_downstream import (
    MLPProbe, aggregate_seeds, make_horizon_dataset,
    make_inverse_dynamics_dataset, train_probe,
)


CALVIN_TRAIN_DIR = "external_models/calvin_data/calvin_debug_dataset/training"
CALVIN_VAL_DIR = "external_models/calvin_data/calvin_debug_dataset/validation"


@torch.no_grad()
def encode_calvin(extractor, split_path: str, datasets_dir: str,
                   n_clips: int, device: torch.device, seed: int) -> dict:
    ds = CalvinClips(
        split_path=split_path, datasets_dir=datasets_dir,
        clip_len=32, cameras=("rgb_static", "rgb_gripper"),
        resolution=128, seed=seed, return_actions=True,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2,
                         worker_init_fn=worker_init_fn)
    feat_t, act_t, feat_clip = [], [], []
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
        seen += clip.shape[0]
        if seen >= n_clips: break
    return {"feat_per_clip_t": torch.cat(feat_t),
            "actions_per_clip": torch.cat(act_t),
            "feat_per_clip": torch.cat(feat_clip),
            "n_clips": seen}


def main():
    out_dir = REPO_ROOT / "runs" / "comparison" / "calvin_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    train_split = str(REPO_ROOT / "data" / "splits_calvin_smoke" / "train.json")
    test_split = str(REPO_ROOT / "data" / "splits_calvin_smoke" / "test_indist.json")

    extractor_specs = [
        {"name": "V-JEPA 2 ViT-L frozen", "family": "vjepa2",
         "arch": "vit_large", "num_frames": 32, "img_size": 256},
        {"name": "V-JEPA 2.1 ViT-base frozen", "family": "vjepa2_1",
         "arch": "vit_base", "num_frames": 32, "img_size": 384},
        {"name": "LAPA-laq_openx frozen", "family": "lapa"},
    ]

    rows = []
    for spec in extractor_specs:
        print(f"\n=== {spec['name']} ===")
        try:
            ext = build_extractor(spec, device)
        except Exception as e:
            print(f"  build failed: {e}"); continue
        try:
            t0 = time.time()
            train = encode_calvin(ext, train_split, CALVIN_TRAIN_DIR,
                                    n_clips=24, device=device, seed=53)
            test = encode_calvin(ext, test_split, CALVIN_VAL_DIR,
                                   n_clips=12, device=device, seed=54)
            print(f"  encoded train={train['n_clips']} test={test['n_clips']} "
                  f"({time.time()-t0:.1f}s)")

            feat_dim = train["feat_per_clip"].shape[-1]
            ad = train["actions_per_clip"].shape[-1]

            tx, ty = make_horizon_dataset(train["feat_per_clip_t"],
                                            train["actions_per_clip"], 1)
            vx, vy = make_horizon_dataset(test["feat_per_clip_t"],
                                            test["actions_per_clip"], 1)
            runs = []
            for s in (0, 1, 2):
                torch.manual_seed(s)
                p = MLPProbe(feat_dim, ad, hidden_dims=(256, 128)).to(device)
                m = train_probe(p, tx, ty, vx, vy, objective="regression",
                                  steps=2000, batch_size=256, lr=3e-4,
                                  weight_decay=1e-4, device=device)
                runs.append(m)
            ag = aggregate_seeds(runs)

            tc = temporal_consistency(test["feat_per_clip_t"])
            rows.append({
                "name": spec["name"],
                "feat_dim": int(feat_dim),
                "action_h1_r2_mean": float(ag["test_r2_mean"]),
                "action_h1_r2_std": float(ag["test_r2_std"]),
                "temporal_cos_adjacent": float(tc.cos_adjacent),
            })
            print(f"  R²={ag['test_r2_mean']:.4f}±{ag['test_r2_std']:.4f}, "
                  f"cos_adj={tc.cos_adjacent:.4f}")
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append({"name": spec["name"], "error": str(e)})
        finally:
            del ext; torch.cuda.empty_cache()

    md = ["# CALVIN debug smoke test (frozen extractors)\n",
          f"24 train + 12 test clips from `calvin_debug_dataset` "
          "(1 long episode per split, chunked into 100-frame sub-episodes).\n",
          "| Model | feat_dim | action R² (H=1) | temp cos_adj |",
          "|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            md.append(f"| {r['name']} | — | ERR ({r['error'][:40]}) | |")
            continue
        md.append(
            f"| {r['name']} | {r['feat_dim']} | "
            f"{r['action_h1_r2_mean']:.4f}±{r['action_h1_r2_std']:.4f} | "
            f"{r['temporal_cos_adjacent']:.4f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
