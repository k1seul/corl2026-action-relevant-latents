"""Cross-architecture comparison eval pipeline.

For each (extractor × dataset) pair, runs the production probe protocol
(action regression, task classification, temporal consistency) and
emits per-cell metrics. Stage A scope: LIBERO only.

Usage:
    uv run python -m comparison.eval_pipeline --config configs/comparison_libero.yaml \\
        --out-dir runs/comparison/libero_4model
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.libero_video import LiberoVideoClips, worker_init_fn
from comparison.extractors.base import RepresentationExtractor
from comparison.metrics import temporal_consistency

# reuse probe utilities from eval_downstream
from eval_downstream import (
    MLPProbe,
    aggregate_seeds,
    build_task_index,
    feature_stats,
    make_horizon_dataset,
    make_inverse_dynamics_dataset,
    train_probe,
)


def _finetuned_ckpt(spec: dict) -> bool:
    """True if the spec names a fine-tuned checkpoint to load.

    A named checkpoint that does not exist is an error: silently falling back to
    the pretrained weights would report frozen numbers under a +ID row name.
    """
    ckpt = spec.get("ckpt")
    if not ckpt:
        return False
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"{spec.get('name', spec.get('family'))}: ckpt not found: {ckpt}")
    return True


def build_extractor(spec: dict, device: torch.device) -> RepresentationExtractor:
    family = spec["family"]
    if family == "diff":
        from comparison.extractors.diff_extractor import DiffExtractor
        return DiffExtractor(spec["ckpt"], device, variant_name=spec.get("name"))
    if family == "token":
        from comparison.extractors.token_extractor import TokenExtractor
        return TokenExtractor(spec["ckpt"], device, variant_name=spec.get("name"))
    if family == "vjepa2":
        from comparison.extractors.vjepa2_extractor import VJEPA2Extractor
        return VJEPA2Extractor(
            device=device,
            ckpt_path=spec.get("ckpt"),
            model_arch=spec.get("arch", "vit_large"),
            num_frames=spec.get("num_frames", 32),
            img_size=spec.get("img_size", 256),
        )
    if family == "vjepa2_1":
        from comparison.extractors.vjepa2_1_extractor import VJEPA21Extractor
        return VJEPA21Extractor(
            device=device,
            ckpt_path=spec.get("ckpt"),
            model_arch=spec.get("arch", "vit_base"),
            num_frames=spec.get("num_frames", 32),
            img_size=spec.get("img_size", 384),
            ckpt_key=spec.get("ckpt_key", "ema_encoder"),
        )
    if family == "lapa":
        from comparison.extractors.lapa_extractor import LapaExtractor
        return LapaExtractor(device=device, ckpt_path=spec.get("ckpt"))
    if family == "cosmos_tokenizer":
        from comparison.extractors.cosmos_tokenizer_extractor import CosmosTokenizerExtractor
        # finetuned JIT module saved whole; load with torch.load if ckpt exists
        finetuned = None
        if _finetuned_ckpt(spec):
            finetuned = torch.load(spec["ckpt"], map_location=device, weights_only=False)
            if isinstance(finetuned, dict) and "encoder" in finetuned:
                finetuned = finetuned["encoder"]
        return CosmosTokenizerExtractor(
            device=device,
            ckpt_path=spec.get("hf_repo"),
            native_image_size=spec.get("img_size", 128),
            finetuned_state_dict=finetuned,
            variant_name=spec.get("name"),
        )
    if family == "sd3vae":
        from comparison.extractors.sd_vae_extractor import SDVAEExtractor
        finetuned_sd = None
        if _finetuned_ckpt(spec):
            state = torch.load(spec["ckpt"], map_location=device, weights_only=False)
            finetuned_sd = state.get("encoder", state.get("model"))
        return SDVAEExtractor(
            device=device,
            ckpt_path=spec.get("hf_repo"),
            native_image_size=spec.get("img_size", 128),
            finetuned_state_dict=finetuned_sd,
            variant_name=spec.get("name"),
        )
    if family == "videomae":
        from comparison.extractors.videomae_extractor import VideoMAEExtractor
        finetuned_sd = None
        if _finetuned_ckpt(spec):
            state = torch.load(spec["ckpt"], map_location=device, weights_only=False)
            finetuned_sd = state.get("encoder", state.get("model"))
        return VideoMAEExtractor(
            device=device,
            ckpt_path=spec.get("hf_repo"),
            native_image_size=spec.get("img_size", 224),
            finetuned_state_dict=finetuned_sd,
            variant_name=spec.get("name"),
        )
    if family == "siglip2":
        from comparison.extractors.siglip2_extractor import SigLIP2Extractor
        finetuned_sd = None
        if _finetuned_ckpt(spec):
            state = torch.load(spec["ckpt"], map_location=device, weights_only=False)
            finetuned_sd = state.get("encoder", state.get("model"))
        return SigLIP2Extractor(
            device=device,
            ckpt_path=spec.get("hf_repo"),
            native_image_size=spec.get("img_size", 256),
            finetuned_state_dict=finetuned_sd,
            variant_name=spec.get("name"),
        )
    if family == "webdino":
        from comparison.extractors.webdino_extractor import WebDINOExtractor
        finetuned_sd = None
        if _finetuned_ckpt(spec):
            state = torch.load(spec["ckpt"], map_location=device, weights_only=False)
            # finetune ckpt saved by train_webdino_aux_libero.py uses "encoder" key
            finetuned_sd = state.get("encoder", state.get("model"))
        return WebDINOExtractor(
            device=device,
            ckpt_path=spec.get("hf_repo"),     # if None, falls back to DEFAULT_CKPT
            native_image_size=spec.get("img_size", 224),
            finetuned_state_dict=finetuned_sd,
            variant_name=spec.get("name"),
        )
    if family == "dreamer4":
        from comparison.extractors.dreamer4_extractor import Dreamer4Extractor
        return Dreamer4Extractor(
            tokenizer_ckpt=spec.get("tokenizer_ckpt") or spec["ckpt"],
            device=device,
            mode=spec.get("mode", "tokenizer"),
            pool=spec.get("pool", "flatten"),
            feature_source=spec.get("feature_source", "spatial"),
            dynamics_ckpt=spec.get("dynamics_ckpt"),
            variant_name=spec.get("name"),
            in_channels=spec.get("in_channels", 3),
            img_size=spec.get("img_size", 128),
            patch=spec.get("patch", 4),
        )
    raise ValueError(f"unknown extractor family: {family!r}")


@torch.no_grad()
def encode_split_with_extractor(
    extractor: RepresentationExtractor,
    split_path: str,
    libero_cfg,
    num_clips_cap: int,
    device: torch.device,
    task_index: dict,
    seed: int,
) -> dict:
    """Run an extractor over a split, return pooled features + actions + task ids."""
    ds = LiberoVideoClips(
        split_path=split_path,
        datasets_dir=libero_cfg["datasets_dir"],
        clip_len=libero_cfg["clip_len"],
        cameras=libero_cfg["cameras"],
        resolution=libero_cfg["resolution"],
        seed=seed,
        return_actions=True,
    )
    loader = DataLoader(ds, batch_size=libero_cfg["batch_size"],
                        shuffle=False, num_workers=2,
                        worker_init_fn=worker_init_fn)
    episode_iter = iter(ds.episodes)

    feat_per_frame, actions_per_frame, feat_per_clip, task_ids = [], [], [], []
    seen = 0
    for clip, actions in loader:
        clip = clip.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        B = clip.shape[0]

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                 enabled=device.type == "cuda"):
            feats = extractor.extract(clip)  # (B, T, D)
        feats = feats.float()

        feat_per_frame.append(feats.cpu())
        actions_per_frame.append(actions.cpu())
        feat_per_clip.append(feats.mean(dim=1).cpu())  # pool over T

        for _ in range(B):
            ep = next(episode_iter)
            task_ids.append(task_index.get((ep["suite"], ep["file"]), -1))

        seen += B
        if seen >= num_clips_cap:
            break

    return {
        "feat_per_clip_t": torch.cat(feat_per_frame, dim=0),
        "actions_per_clip": torch.cat(actions_per_frame, dim=0),
        "feat_per_clip": torch.cat(feat_per_clip, dim=0),
        "task_per_clip": torch.tensor(task_ids, dtype=torch.long),
        "n_clips": seen,
    }


def evaluate_extractor(
    name: str,
    extractor: RepresentationExtractor,
    libero_cfg: dict,
    train_split: str,
    test_split: str,
    n_train_clips: int,
    n_test_clips: int,
    probe_steps: int,
    probe_seeds: list[int],
    hidden_dims: tuple,
    device: torch.device,
    task_index: dict,
    base_seed: int = 42,
) -> dict:
    print(f"\n=== {name} ({extractor.meta.family}) ===")
    t0 = time.time()
    train_data = encode_split_with_extractor(
        extractor, train_split, libero_cfg, n_train_clips, device,
        task_index, seed=base_seed + 11,
    )
    test_data = encode_split_with_extractor(
        extractor, test_split, libero_cfg, n_test_clips, device,
        task_index, seed=base_seed + 12,
    )
    print(f"  encoded train={train_data['n_clips']} test={test_data['n_clips']} "
          f"({time.time()-t0:.1f}s)")

    feat_dim = train_data["feat_per_clip"].shape[-1]
    action_dim = train_data["actions_per_clip"].shape[-1]
    n_tasks = max(int(train_data["task_per_clip"].max().item()) + 1, 1)

    def run_seeded(make_probe, tx, ty, vx, vy, *, obj, n_steps):
        runs = []
        for seed in probe_seeds:
            torch.manual_seed(seed)
            probe = make_probe().to(device)
            m = train_probe(
                probe, tx, ty, vx, vy,
                objective=obj, steps=n_steps,
                batch_size=512, lr=3e-4, weight_decay=1e-4,
                device=device,
            )
            runs.append(m)
        return aggregate_seeds(runs)

    # Action probe H=1
    tx, ty = make_horizon_dataset(train_data["feat_per_clip_t"],
                                   train_data["actions_per_clip"], 1)
    vx, vy = make_horizon_dataset(test_data["feat_per_clip_t"],
                                   test_data["actions_per_clip"], 1)
    action_metrics = run_seeded(
        lambda: MLPProbe(feat_dim, action_dim, hidden_dims=hidden_dims),
        tx, ty, vx, vy, obj="regression", n_steps=probe_steps,
    )

    # Inverse dynamics
    idtx, idty = make_inverse_dynamics_dataset(
        train_data["feat_per_clip_t"], train_data["actions_per_clip"])
    idvx, idvy = make_inverse_dynamics_dataset(
        test_data["feat_per_clip_t"], test_data["actions_per_clip"])
    invdyn_metrics = run_seeded(
        lambda: MLPProbe(2 * feat_dim, action_dim, hidden_dims=hidden_dims),
        idtx, idty, idvx, idvy, obj="regression", n_steps=probe_steps,
    )

    # Task probe (skip if all test tasks are OOD)
    test_known = (test_data["task_per_clip"] >= 0).float().mean().item()
    if test_known < 1e-3:
        task_metrics = {"test_top1_mean": float("nan"), "test_top1_std": 0.0,
                        "skipped_reason": "all test tasks OOD"}
    else:
        task_metrics = run_seeded(
            lambda: MLPProbe(feat_dim, n_tasks, hidden_dims=()),
            train_data["feat_per_clip"], train_data["task_per_clip"],
            test_data["feat_per_clip"], test_data["task_per_clip"],
            obj="classification", n_steps=max(probe_steps // 2, 500),
        )

    # Temporal consistency on test split (per-frame features)
    tc = temporal_consistency(test_data["feat_per_clip_t"])

    fstats = feature_stats(test_data["feat_per_clip"], test_data["task_per_clip"])

    return {
        "name": name,
        "meta": asdict(extractor.meta),
        "feat_dim": int(feat_dim),
        "action_h1_r2_mean": float(action_metrics["test_r2_mean"]),
        "action_h1_r2_std": float(action_metrics["test_r2_std"]),
        "invdyn_r2_mean": float(invdyn_metrics["test_r2_mean"]),
        "invdyn_r2_std": float(invdyn_metrics["test_r2_std"]),
        "task_top1_mean": float(task_metrics.get("test_top1_mean", float("nan"))),
        "task_top1_std": float(task_metrics.get("test_top1_std", 0.0)),
        "temporal_cos_adjacent": float(tc.cos_adjacent),
        "temporal_cos_curve_ks": tc.cos_curve_ks,
        "temporal_cos_curve_vals": tc.cos_curve_vals,
        "feat_norm": float(fstats["norm_mean"]),
        "feat_task_separation": float(fstats["task_separation"])
                                  if fstats["task_separation"] == fstats["task_separation"]
                                  else float("nan"),
    }


def write_summary_md(results: list[dict], out_path: Path) -> None:
    lines = ["# Cross-architecture comparison — LIBERO\n"]
    lines.append("| Model | family | feat_dim | params | action R² (H=1) | inv-dyn R² | task top-1 | temporal cos_adj |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        m = r["meta"]
        lines.append(
            f"| {r['name']} | {m['family']} | {m['feat_dim']} | "
            f"{m['n_params']/1e6:.1f}M | "
            f"{r['action_h1_r2_mean']:.4f}±{r['action_h1_r2_std']:.4f} | "
            f"{r['invdyn_r2_mean']:.4f}±{r['invdyn_r2_std']:.4f} | "
            f"{r['task_top1_mean']:.4f} | "
            f"{r['temporal_cos_adjacent']:.4f} |"
        )
    lines.append("")
    lines.append("## Temporal cosine decay curve\n")
    lines.append("| Model | k=1 | k=2 | k=4 | k=8 | k=16 |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        ks, vs = r["temporal_cos_curve_ks"], r["temporal_cos_curve_vals"]
        row = {k: v for k, v in zip(ks, vs)}
        cells = [f"{row.get(k, float('nan')):.4f}" for k in (1, 2, 4, 8, 16)]
        lines.append(f"| {r['name']} | {' | '.join(cells)} |")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="YAML with extractor specs and libero settings")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    libero_cfg = OmegaConf.to_container(cfg.libero, resolve=True)
    task_index = build_task_index(cfg.train_split)
    print(f"task_index size: {len(task_index)}")

    results = []
    # resume from existing results.json if present
    existing_path = out_dir / "results.json"
    if existing_path.exists():
        try:
            with existing_path.open() as f:
                results = json.load(f)
            already = {r["name"] for r in results}
            print(f"resuming: {len(results)} prior results, "
                  f"already done: {sorted(already)}")
        except Exception:
            results = []
    done_names = {r["name"] for r in results}

    for spec in cfg.extractors:
        spec = OmegaConf.to_container(spec, resolve=True)
        if spec["name"] in done_names:
            print(f"[skip] {spec['name']} (resumed from results.json)")
            continue

        extractor = None
        try:
            extractor = build_extractor(spec, device)
        except (FileNotFoundError, NotImplementedError) as e:
            print(f"[skip] {spec['name']}: build failed — {e}")
            results.append({
                "name": spec["name"],
                "meta": {"family": spec.get("family", "?"),
                         "variant": spec.get("name", "?"),
                         "feat_dim": -1, "n_params": -1},
                "error": f"build_failed: {e}",
            })
        else:
            try:
                res = evaluate_extractor(
                    name=spec["name"],
                    extractor=extractor,
                    libero_cfg=libero_cfg,
                    train_split=cfg.train_split,
                    test_split=cfg.test_split,
                    n_train_clips=int(cfg.get("n_train_clips", 1000)),
                    n_test_clips=int(cfg.get("n_test_clips", 400)),
                    probe_steps=int(cfg.get("probe_steps", 3000)),
                    probe_seeds=list(cfg.get("probe_seeds", [0, 1, 2])),
                    hidden_dims=tuple(cfg.get("probe_hidden_dims", [256, 128])),
                    device=device,
                    task_index=task_index,
                )
                results.append(res)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"[FAIL] {spec['name']}: {e}\n{tb}", flush=True)
                results.append({
                    "name": spec["name"],
                    "meta": {"family": spec.get("family", "?"),
                             "variant": spec.get("name", "?"),
                             "feat_dim": -1, "n_params": -1},
                    "error": f"eval_failed: {e}",
                })
            finally:
                del extractor
                torch.cuda.empty_cache()

        # write incrementally so partial results are recoverable
        with (out_dir / "results.json").open("w") as f:
            json.dump(results, f, indent=2)
        write_summary_md([r for r in results if "error" not in r],
                         out_dir / "summary.md")

    print(f"\nwrote {out_dir/'summary.md'}")


if __name__ == "__main__":
    main()
