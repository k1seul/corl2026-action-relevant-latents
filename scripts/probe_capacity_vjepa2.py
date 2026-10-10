"""V-JEPA2 frozen probe capacity test.

Encodes V-JEPA2 features once on LIBERO test_ood, then trains probes
of varying capacity. Tests whether the 0.561 OOD R² we got with
[256,128] MLP was a probe-capacity ceiling or a feature-quality ceiling.

Variants tested (3 seeds each):
  linear         : hidden_dims = ()
  small_mlp      : (256, 128)              ← current Phase 1 setting
  deep_mlp       : (1024, 512, 256, 128)
  vdeep_mlp      : (2048, 1024, 512, 256)

Outputs runs/comparison/vjepa2_probe_capacity/summary.md.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch
from omegaconf import OmegaConf

from comparison.extractors.vjepa2_extractor import VJEPA2Extractor
from comparison.eval_pipeline import encode_split_with_extractor
from eval_downstream import (
    MLPProbe, aggregate_seeds, build_task_index,
    make_horizon_dataset, make_inverse_dynamics_dataset, train_probe,
)


def main() -> None:
    out_dir = REPO_ROOT / "runs" / "comparison" / "vjepa2_probe_capacity"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    libero_cfg = {
        "datasets_dir": "datasets",
        "clip_len": 32,
        "cameras": ["agentview_rgb", "eye_in_hand_rgb"],
        "resolution": 128,
        "batch_size": 8,
    }
    train_split = "data/splits_taskheld/train.json"
    test_split = "data/splits_taskheld/test_ood.json"
    task_index = build_task_index(train_split)

    # encode once
    print("loading V-JEPA2 ViT-L ...")
    t0 = time.time()
    extractor = VJEPA2Extractor(device=device, model_arch="vit_large",
                                 num_frames=32, img_size=256)
    print(f"  loaded in {time.time()-t0:.1f}s")

    print("encoding train (1000 clips) ...")
    t0 = time.time()
    train_data = encode_split_with_extractor(
        extractor, train_split, libero_cfg, 1000, device,
        task_index, seed=53,
    )
    print(f"  done in {time.time()-t0:.1f}s")

    print("encoding test_ood (400 clips) ...")
    t0 = time.time()
    test_data = encode_split_with_extractor(
        extractor, test_split, libero_cfg, 400, device,
        task_index, seed=54,
    )
    print(f"  done in {time.time()-t0:.1f}s")

    # free extractor memory before probes
    del extractor
    torch.cuda.empty_cache()

    feat_dim = train_data["feat_per_clip"].shape[-1]
    action_dim = train_data["actions_per_clip"].shape[-1]
    print(f"feat_dim={feat_dim}, action_dim={action_dim}")

    variants = [
        ("linear",     ()),
        ("small_mlp",  (256, 128)),
        ("deep_mlp",   (1024, 512, 256, 128)),
        ("vdeep_mlp",  (2048, 1024, 512, 256)),
    ]
    probe_steps = 5000  # bumped from 3000 for fairer comparison

    rows = []
    # action_h1 dataset
    a_tx, a_ty = make_horizon_dataset(
        train_data["feat_per_clip_t"], train_data["actions_per_clip"], 1)
    a_vx, a_vy = make_horizon_dataset(
        test_data["feat_per_clip_t"], test_data["actions_per_clip"], 1)

    # inv-dyn dataset
    i_tx, i_ty = make_inverse_dynamics_dataset(
        train_data["feat_per_clip_t"], train_data["actions_per_clip"])
    i_vx, i_vy = make_inverse_dynamics_dataset(
        test_data["feat_per_clip_t"], test_data["actions_per_clip"])

    for name, hidden in variants:
        print(f"\n--- {name} hidden={hidden} ---")
        action_runs = []
        invdyn_runs = []
        n_probe_params_action = None
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            probe = MLPProbe(feat_dim, action_dim, hidden_dims=hidden).to(device)
            if n_probe_params_action is None:
                n_probe_params_action = sum(p.numel() for p in probe.parameters())
            m = train_probe(probe, a_tx, a_ty, a_vx, a_vy,
                             objective="regression", steps=probe_steps,
                             batch_size=512, lr=3e-4, weight_decay=1e-4,
                             device=device)
            action_runs.append(m)

            torch.manual_seed(seed)
            probe2 = MLPProbe(2 * feat_dim, action_dim, hidden_dims=hidden).to(device)
            m2 = train_probe(probe2, i_tx, i_ty, i_vx, i_vy,
                              objective="regression", steps=probe_steps,
                              batch_size=512, lr=3e-4, weight_decay=1e-4,
                              device=device)
            invdyn_runs.append(m2)

        a_agg = aggregate_seeds(action_runs)
        i_agg = aggregate_seeds(invdyn_runs)
        print(f"  action_h1 R²={a_agg['test_r2_mean']:.4f}±{a_agg['test_r2_std']:.4f}")
        print(f"  inv-dyn   R²={i_agg['test_r2_mean']:.4f}±{i_agg['test_r2_std']:.4f}")
        rows.append({
            "variant": name,
            "hidden": str(hidden),
            "probe_params": n_probe_params_action,
            "action_r2_mean": a_agg["test_r2_mean"],
            "action_r2_std":  a_agg["test_r2_std"],
            "invdyn_r2_mean": i_agg["test_r2_mean"],
            "invdyn_r2_std":  i_agg["test_r2_std"],
        })

    # summary
    md = ["# V-JEPA2 frozen — probe capacity test (LIBERO test_ood)\n",
          f"V-JEPA2 ViT-L frozen, feat_dim={feat_dim}, probe steps={probe_steps}, 3 seeds.\n",
          "| Variant | hidden dims | probe params | action R² | inv-dyn R² |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['variant']} | {r['hidden']} | {r['probe_params']/1e6:.2f}M | "
            f"{r['action_r2_mean']:.4f}±{r['action_r2_std']:.4f} | "
            f"{r['invdyn_r2_mean']:.4f}±{r['invdyn_r2_std']:.4f} |"
        )
    md.append("")
    md.append("## Reference (Phase 1 baseline)\n")
    md.append("- DIFF (ours, 91M params, frozen): action R² = 0.878")
    md.append("- V-JEPA2 frozen + small_mlp probe (Phase 1): action R² = 0.561\n")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {out_dir / 'summary.md'}")

    import json
    with (out_dir / "results.json").open("w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
