"""Full Results table pipeline (paper Appendix B).

For each row in ``runs/full_results/manifest.yaml`` compute:
  - per-dim action probe MSE / R² on (train, test_ood) splits
  - aggregate action probe MSE / R²
  - pixel rollout MSE / PSNR / LPIPS (DIFF / TOKEN only — V-JEPA / LAPA = NaN)

Appends results to ``runs/full_results/results.jsonl`` one line per
(manifest_entry × split). Resumes by skipping (variant, split) pairs
already present.

Probe split protocol (per spec § "Train vs test"):
  - probe-train: LIBERO taskheld train.json, episode indices [0 : probe_train_clips]
  - probe-eval-train (= "train" cell): train.json, indices [probe_train_clips :
                                       probe_train_clips + probe_eval_clips]
  - probe-eval-test  (= "test"  cell): test_ood.json, indices [0 : probe_eval_clips]

This guarantees no episode overlap between probe-train and the train-cell
evaluation, so the "train" cell measures fit, not memorization.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/full_results_pipeline.py \
        --manifest runs/full_results/manifest.yaml \
        --out-dir runs/full_results \
        --probe-train-clips 400 --probe-eval-clips 200 \
        --recon-clips 64
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from comparison.metrics.action_probe import per_dim_mse, per_dim_r2, aggregate_per_dim
from comparison.metrics.recon import recon_psnr_lpips, rollout_diff_or_fm
from data.libero_video import LiberoVideoClips, worker_init_fn
from data.build_eval_dataset import build_eval_dataset, benchmark_splits, BENCHMARK_CHOICES

# Stage B extractors
from comparison.eval_pipeline import build_extractor as build_stage_b_extractor
# Stage A model loaders + probe utilities
from eval_downstream import (
    MLPProbe, build_pixel_model, build_token_model,
    encode_clip_to_tokens, load_tokenizer, train_probe,
)

# Default LIBERO eval splits
TRAIN_SPLIT = REPO_ROOT / "data" / "splits_taskheld" / "train.json"
TEST_OOD_SPLIT = REPO_ROOT / "data" / "splits_taskheld" / "test_ood.json"


# -------------------------- feature encoding -----------------------------


@torch.no_grad()
def encode_subset(
    extract_fn,
    cameras: tuple[str, ...],
    resolution: int,
    clip_len: int,
    split_path: Path,
    episode_slice: tuple[int, int],
    seed: int,
    device: torch.device,
    batch_size: int = 8,
    perturb_kind: str = "none",
    perturb_strength: float = 0.0,
    benchmark: str = "libero",
) -> dict:
    """Encode a contiguous slice of episodes from a split.

    extract_fn(clip: (B,T,C,H,W)) -> (B, T, D) per-frame pooled features.

    Returns dict with:
      feat_per_clip_t : (K, T, D) per-frame features
      clips           : (K, T, C, H, W) raw clips (for recon eval; only
                         materialised for the first ``recon_subset`` clips
                         the caller may slice into)
      actions         : (K, T, A)
    """
    full_ds, _, _ = build_eval_dataset(
        benchmark=benchmark, split_path=str(split_path),
        clip_len=clip_len, resolution=resolution, seed=seed)
    lo, hi = episode_slice
    hi = min(hi, len(full_ds))
    if lo >= hi:
        raise ValueError(f"empty slice {lo}:{hi} for split with {len(full_ds)} episodes")
    ds = Subset(full_ds, list(range(lo, hi)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=2, worker_init_fn=worker_init_fn)
    feat_list, action_list, clip_list = [], [], []
    seen = 0
    if perturb_kind != "none" and perturb_strength > 0:
        from eval_downstream import perturb_clip as _perturb
    else:
        _perturb = None
    for clip, actions in loader:
        clip = clip.to(device, non_blocking=True)
        if _perturb is not None:
            clip = _perturb(clip, perturb_kind, perturb_strength)
        actions = actions.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
            feats = extract_fn(clip)
        feats = feats.float()
        feat_list.append(feats.cpu())
        action_list.append(actions.cpu())
        clip_list.append(clip.cpu())
        seen += clip.shape[0]
    return {
        "feat_per_clip_t": torch.cat(feat_list, dim=0),       # (K, T, D)
        "actions_per_clip": torch.cat(action_list, dim=0),     # (K, T, A)
        "clips": torch.cat(clip_list, dim=0),                  # (K, T, C, H, W)
        "n_clips": seen,
    }


# ---- extract_fn factories for Stage A / Stage B ------------------------


def _stage_a_extract_fn(model, objective: str, tokenizer):
    @torch.no_grad()
    def fn(clip):
        if objective == "mae":
            feats = model.forward_features(clip, mask_indicator=None)
        elif objective in ("diffusion", "flow_matching"):
            B, T = clip.shape[:2]
            t = torch.zeros(B, T, device=clip.device, dtype=torch.long)
            feats = model.forward_features(clip, mask_indicator=None, t=t)
        elif objective == "token_ce":
            token_ids = encode_clip_to_tokens(tokenizer, clip)
            feats = model.forward_features(token_ids)
        else:
            raise ValueError(objective)
        # pool over N spatial tokens -> (B, T, D)
        return feats.float().mean(dim=2)
    return fn


def _stage_b_extract_fn(extractor):
    @torch.no_grad()
    def fn(clip):
        # extractor.extract returns (B, T, D) directly
        return extractor.extract(clip).float()
    return fn


# -------------------------- probe driver --------------------------------


def run_per_dim_probe(
    probe_train: dict,
    probe_eval_train: dict,
    probe_eval_test: dict,
    *,
    horizon: int,
    hidden_dims: tuple[int, ...],
    probe_seeds: list[int],
    steps: int,
    batch_size: int,
    lr: float,
    wd: float,
    device: torch.device,
) -> tuple[dict, dict]:
    """Train action probe on probe_train features, evaluate on both eval
    splits. Returns (train_cell, test_cell) — each is the metric dict
    averaged over ``probe_seeds`` seeds.

    Each cell contains:
      action_mse, action_r2,
      per_dim_mse: {dx, dy, dz, drx, dry, drz, gripper},
      per_dim_r2:  {...},
    plus per-seed std on aggregate numbers.
    """
    feat_tr = probe_train["feat_per_clip_t"]
    act_tr = probe_train["actions_per_clip"]
    feat_et = probe_eval_train["feat_per_clip_t"]
    act_et = probe_eval_train["actions_per_clip"]
    feat_ev = probe_eval_test["feat_per_clip_t"]
    act_ev = probe_eval_test["actions_per_clip"]

    feat_dim = feat_tr.shape[-1]
    A = act_tr.shape[-1]

    def _h(feat, act):
        # H=1 horizon dataset: pair feat[t] with act[t]
        T = feat.shape[1]
        return feat.reshape(-1, feat_dim), act.reshape(-1, A)

    train_x, train_y = _h(feat_tr, act_tr)
    et_x, et_y = _h(feat_et, act_et)
    ev_x, ev_y = _h(feat_ev, act_ev)

    train_x = train_x.to(device); train_y = train_y.to(device)
    et_x = et_x.to(device); et_y = et_y.to(device)
    ev_x = ev_x.to(device); ev_y = ev_y.to(device)

    # accumulate per-seed metrics on each eval split
    et_per_dim_mse_runs: list[dict] = []
    et_per_dim_r2_runs: list[dict] = []
    ev_per_dim_mse_runs: list[dict] = []
    ev_per_dim_r2_runs: list[dict] = []
    for s in probe_seeds:
        torch.manual_seed(s)
        probe = MLPProbe(feat_dim, A * horizon, hidden_dims=hidden_dims).to(device)
        # use eval_downstream.train_probe to share optimization defaults
        _ = train_probe(probe, train_x, train_y, ev_x, ev_y,
                         objective="regression", steps=steps, batch_size=batch_size,
                         lr=lr, weight_decay=wd, device=device)
        probe.eval()
        with torch.no_grad():
            pred_et = probe(et_x)
            pred_ev = probe(ev_x)
        et_per_dim_mse_runs.append(per_dim_mse(pred_et, et_y))
        et_per_dim_r2_runs.append(per_dim_r2(pred_et, et_y))
        ev_per_dim_mse_runs.append(per_dim_mse(pred_ev, ev_y))
        ev_per_dim_r2_runs.append(per_dim_r2(pred_ev, ev_y))

    def _mean_dict(dicts: list[dict]) -> dict:
        keys = dicts[0].keys()
        return {k: float(sum(d[k] for d in dicts) / len(dicts)) for k in keys}

    def _std_dict(dicts: list[dict]) -> dict:
        keys = dicts[0].keys()
        out = {}
        for k in keys:
            vs = [d[k] for d in dicts]
            m = sum(vs) / len(vs)
            v = math.sqrt(sum((x - m) ** 2 for x in vs) / len(vs))
            out[k] = float(v)
        return out

    et_pdm = _mean_dict(et_per_dim_mse_runs)
    et_pdr = _mean_dict(et_per_dim_r2_runs)
    ev_pdm = _mean_dict(ev_per_dim_mse_runs)
    ev_pdr = _mean_dict(ev_per_dim_r2_runs)
    et_pdm_std = _std_dict(et_per_dim_mse_runs)
    ev_pdm_std = _std_dict(ev_per_dim_mse_runs)

    train_cell = {
        "action_probe_mse": aggregate_per_dim(et_pdm),
        "action_probe_r2": aggregate_per_dim(et_pdr),
        "per_dim_mse": et_pdm,
        "per_dim_r2": et_pdr,
        "per_dim_mse_std": et_pdm_std,
    }
    test_cell = {
        "action_probe_mse": aggregate_per_dim(ev_pdm),
        "action_probe_r2": aggregate_per_dim(ev_pdr),
        "per_dim_mse": ev_pdm,
        "per_dim_r2": ev_pdr,
        "per_dim_mse_std": ev_pdm_std,
    }
    return train_cell, test_cell


# -------------------------- recon driver ---------------------------------


@torch.no_grad()
def compute_recon_cell(
    family: str,
    raw_model,
    tokenizer,
    cfg,
    clips: torch.Tensor,                # (K, T, C, H, W) on CPU
    actions: torch.Tensor | None,        # (K, T, A) on CPU, or None
    device: torch.device,
    n_clips: int,
    batch_size: int = 4,
    rollout_prefix: int | None = None,
    extractor=None,                      # for family="dreamer4"
) -> dict:
    """Run rollout + compute recon_mse, psnr, lpips on up to ``n_clips``
    clips. ``family`` ∈ {diff, token, dreamer4}.

    Action-conditioned variants receive GT ``actions`` during the rollout
    (they assert otherwise). This is the protocol Dreamer/IRIS/etc. use
    when comparing pixel rollout fidelity for action-conditioned WMs.
    Non-action-conditioned variants ignore the ``actions`` arg entirely.

    For family="dreamer4", uses ``extractor.recon_rollout()`` which runs the
    nicklashansen shortcut-forcing autoregressive rollout + tokenizer decode.
    The dreamer4 dynamics is always action-conditioned (paper convention)
    so we always pass GT actions.
    """
    K = min(n_clips, clips.shape[0])
    if K == 0:
        return {"recon_mse": float("nan"), "psnr": float("nan"), "lpips": float("nan")}
    T = clips.shape[1]
    # Default rollout prefix per family
    if family == "dreamer4":
        rollout_prefix = rollout_prefix or 8
    else:
        rollout_prefix = rollout_prefix or int(cfg.eval.get("rollout_prefix", 8))
    rollout_prefix = min(rollout_prefix, T - 1)
    rollout_prefix = max(1, rollout_prefix)
    num_post = T - rollout_prefix
    is_action_cond = (family == "dreamer4") or (
        cfg is not None and int(cfg.model.get("action_dim", 0)) > 0
    )

    pred_list: list[torch.Tensor] = []
    gt_list: list[torch.Tensor] = []
    if family == "token":
        # token rollout is defined in scripts/eval.py
        import importlib.util
        eval_path = REPO_ROOT / "scripts" / "eval.py"
        spec = importlib.util.spec_from_file_location("_eval_mod", eval_path)
        eval_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eval_mod)
        _rollout_token = eval_mod.rollout_token
    for i in range(0, K, batch_size):
        batch = clips[i:i + batch_size].to(device)
        prefix = batch[:, :rollout_prefix]
        batch_actions = None
        if is_action_cond and actions is not None:
            batch_actions = actions[i:i + batch_size].to(device)
        if family == "diff":
            pred = rollout_diff_or_fm(raw_model, cfg, prefix, num_post, device,
                                        actions=batch_actions)
        elif family == "token":
            pred = _rollout_token(raw_model, tokenizer, prefix, num_post,
                                    sample_temperature=0.0, autoregressive=True,
                                    actions=batch_actions)
        elif family == "dreamer4":
            if extractor is None:
                raise RuntimeError("dreamer4 recon requires extractor arg")
            # full batch_actions covers prefix+post (B, T, A); rollout uses
            # them at every frame's position for action conditioning.
            pred = extractor.recon_rollout(prefix, num_post=num_post,
                                            actions=batch_actions)
        else:
            raise ValueError(family)
        pred_list.append(pred.detach().cpu())
        gt_list.append(batch.cpu())

    pred_all = torch.cat(pred_list, dim=0)
    gt_all = torch.cat(gt_list, dim=0)
    return recon_psnr_lpips(pred_all, gt_all, rollout_prefix, device, compute_lpips=True)


# -------------------------- per-row entry --------------------------------


def _camera_tuple(entry: dict) -> tuple[str, ...]:
    """Pick camera list. Stage A configs may use 2 cams; Stage B always 1."""
    if entry["family"] in ("vjepa2", "vjepa2_1", "lapa", "dreamer4", "webdino", "siglip2", "sd3vae", "cosmos_tokenizer", "videomae"):
        return ("agentview_rgb",)
    # Stage A: read from config
    cfg_path = REPO_ROOT / entry["config_path"]
    cfg = OmegaConf.load(cfg_path)
    return tuple(cfg.data.cameras)


def _resolution(entry: dict) -> int:
    if entry["family"] in ("vjepa2", "vjepa2_1", "lapa", "dreamer4", "webdino", "siglip2", "sd3vae", "cosmos_tokenizer", "videomae"):
        return 128
    cfg = OmegaConf.load(REPO_ROOT / entry["config_path"])
    return int(cfg.data.resolution)


def _clip_len(entry: dict) -> int:
    return int(entry["seq_len"])


def _build_stage_a(entry: dict, device: torch.device):
    """Load Stage A model + tokenizer for both extraction and recon."""
    cfg = OmegaConf.load(REPO_ROOT / entry["config_path"])
    state = torch.load(REPO_ROOT / entry["ckpt"], map_location=device, weights_only=False)
    objective = cfg.train.get("objective", "mae")
    in_channels = 3 * len(cfg.data.cameras)
    if objective == "token_ce":
        tokenizer = load_tokenizer(cfg.train.tokenizer_ckpt, in_channels, device)
        # peek at a tokenized frame to get codebook + N per frame
        peek_ds = LiberoVideoClips(
            split_path=str(TRAIN_SPLIT), datasets_dir=str(REPO_ROOT / "datasets"),
            clip_len=cfg.data.clip_len, cameras=cfg.data.cameras,
            resolution=cfg.data.resolution, seed=0,
        )
        peek = peek_ds[0][:1].unsqueeze(0).to(device)
        with torch.no_grad():
            sample_ids = encode_clip_to_tokens(tokenizer, peek)
        N_per_frame = sample_ids.shape[-1]
        model = build_token_model(cfg, tokenizer.codebook_size, N_per_frame).to(device)
        model.load_state_dict(state["model"])
    else:
        tokenizer = None
        model = build_pixel_model(cfg, in_channels).to(device)
        model.load_state_dict(state["model"])
    model.eval()
    return model, tokenizer, cfg, objective


def _build_stage_b(entry: dict, device: torch.device):
    spec = {
        "name": entry["variant"],
        "family": entry["family"],
        "arch": entry.get("arch"),
        "num_frames": entry.get("num_frames", 32),
        "img_size": entry.get("img_size", 256),
    }
    if entry.get("ckpt"):
        spec["ckpt"] = str(REPO_ROOT / entry["ckpt"])
    if entry.get("ckpt_key"):
        spec["ckpt_key"] = entry["ckpt_key"]
    # dreamer4-specific spec fields (mode/pool/feature_source/dynamics_ckpt)
    for k in ("mode", "pool", "feature_source"):
        if entry.get(k) is not None:
            spec[k] = entry[k]
    if entry.get("dynamics_ckpt"):
        spec["dynamics_ckpt"] = str(REPO_ROOT / entry["dynamics_ckpt"])
    if entry.get("patch") is not None:
        spec["patch"] = entry["patch"]
    if entry.get("in_channels") is not None:
        spec["in_channels"] = entry["in_channels"]
    return build_stage_b_extractor(spec, device)


# -------------------------- orchestrator --------------------------------


def process_entry(
    entry: dict,
    args,
    device: torch.device,
) -> dict:
    """Compute all metrics for one manifest row. Returns a dict with
    two sub-dicts ``train_cell`` and ``test_cell``.
    """
    name = entry["variant"]
    family = entry["family"]
    print(f"\n=== {name} (family={family}) ===")
    t0 = time.time()

    cameras = _camera_tuple(entry)
    resolution = _resolution(entry)
    clip_len = _clip_len(entry)

    raw_model = tokenizer = cfg = objective = stage_b_ext = None
    try:
        if family in ("diff", "token"):
            raw_model, tokenizer, cfg, objective = _build_stage_a(entry, device)
            extract_fn = _stage_a_extract_fn(raw_model, objective, tokenizer)
        else:
            stage_b_ext = _build_stage_b(entry, device)
            # for V-JEPA / LAPA we keep our LIBERO clip canonical at 128
            # (the extractor will upsample internally).
            extract_fn = _stage_b_extract_fn(stage_b_ext)
    except Exception as e:
        print(f"  BUILD FAILED: {e}")
        return {"error": f"build: {e}"}

    pt_lo, pt_hi = 0, args.probe_train_clips
    et_lo, et_hi = args.probe_train_clips, args.probe_train_clips + args.probe_eval_clips
    ev_lo, ev_hi = 0, args.probe_eval_clips
    bench = getattr(args, "benchmark", "libero")
    train_split_p, test_ood_split_p = benchmark_splits(bench)
    train_split_p = REPO_ROOT / train_split_p
    test_ood_split_p = REPO_ROOT / test_ood_split_p

    try:
        probe_train = encode_subset(
            extract_fn, cameras, resolution, clip_len,
            train_split_p, (pt_lo, pt_hi), seed=11, device=device,
            batch_size=args.encode_batch_size, benchmark=bench,
        )
        probe_eval_train = encode_subset(
            extract_fn, cameras, resolution, clip_len,
            train_split_p, (et_lo, et_hi), seed=12, device=device,
            batch_size=args.encode_batch_size, benchmark=bench,
        )
        probe_eval_test = encode_subset(
            extract_fn, cameras, resolution, clip_len,
            test_ood_split_p, (ev_lo, ev_hi), seed=13, device=device,
            batch_size=args.encode_batch_size,
            perturb_kind=args.perturb_test_kind,
            perturb_strength=args.perturb_test_strength,
            benchmark=bench,
        )
    except Exception as e:
        print(f"  ENCODE FAILED: {e}")
        if raw_model is not None: del raw_model
        if stage_b_ext is not None: del stage_b_ext
        torch.cuda.empty_cache()
        return {"error": f"encode: {e}"}
    print(f"  encoded probe_train={probe_train['n_clips']} "
          f"probe_eval_train={probe_eval_train['n_clips']} "
          f"probe_eval_test={probe_eval_test['n_clips']} "
          f"({time.time()-t0:.1f}s)")

    # --- per-dim action probe (H=1) ---
    train_cell, test_cell = run_per_dim_probe(
        probe_train, probe_eval_train, probe_eval_test,
        horizon=1,
        hidden_dims=(256, 128),
        probe_seeds=[0, 1, 2],
        steps=args.probe_steps, batch_size=512,
        lr=3e-4, wd=1e-4, device=device,
    )

    # --- recon / PSNR / LPIPS (DIFF / TOKEN / dreamer4 with dynamics ckpt) ---
    can_recon = family in ("diff", "token")
    if family == "dreamer4" and stage_b_ext is not None and getattr(stage_b_ext, "_dynamics", None) is not None:
        can_recon = True
    if can_recon and not args.skip_recon:
        try:
            t1 = time.time()
            recon_train = compute_recon_cell(
                family=family, raw_model=raw_model, tokenizer=tokenizer, cfg=cfg,
                clips=probe_eval_train["clips"],
                actions=probe_eval_train["actions_per_clip"],
                device=device,
                n_clips=args.recon_clips, batch_size=args.recon_batch_size,
                extractor=stage_b_ext,
            )
            recon_test = compute_recon_cell(
                family=family, raw_model=raw_model, tokenizer=tokenizer, cfg=cfg,
                clips=probe_eval_test["clips"],
                actions=probe_eval_test["actions_per_clip"],
                device=device,
                n_clips=args.recon_clips, batch_size=args.recon_batch_size,
                extractor=stage_b_ext,
            )
            train_cell.update(recon_train)
            test_cell.update(recon_test)
            print(f"  recon train PSNR={recon_train['psnr']:.2f} test PSNR={recon_test['psnr']:.2f} "
                  f"({time.time()-t1:.1f}s)")
        except Exception as e:
            print(f"  RECON FAILED: {e}")
            for cell in (train_cell, test_cell):
                cell["recon_mse"] = float("nan")
                cell["psnr"] = float("nan")
                cell["lpips"] = float("nan")
            cell["recon_error"] = str(e)
    else:
        for cell in (train_cell, test_cell):
            cell["recon_mse"] = float("nan")
            cell["psnr"] = float("nan")
            cell["lpips"] = float("nan")

    if raw_model is not None: del raw_model
    if stage_b_ext is not None: del stage_b_ext
    torch.cuda.empty_cache()
    print(f"  done in {time.time()-t0:.1f}s "
          f"(train R²={train_cell['action_probe_r2']:.3f} test R²={test_cell['action_probe_r2']:.3f})")
    return {"train_cell": train_cell, "test_cell": test_cell}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="runs/full_results/manifest.yaml")
    ap.add_argument("--out-dir", default="runs/full_results")
    ap.add_argument("--probe-train-clips", type=int, default=400)
    ap.add_argument("--probe-eval-clips", type=int, default=200)
    ap.add_argument("--recon-clips", type=int, default=64)
    ap.add_argument("--encode-batch-size", type=int, default=8)
    ap.add_argument("--recon-batch-size", type=int, default=4)
    ap.add_argument("--probe-steps", type=int, default=3000)
    ap.add_argument("--skip-recon", action="store_true")
    ap.add_argument("--perturb-test-kind", type=str, default="none",
                    choices=("none", "noise", "blur", "jitter", "mask_eye"),
                    help="LIBERO-PRO-style visual perturbation applied to "
                         "probe-eval-test clips only (probe-train stays clean)")
    ap.add_argument("--perturb-test-strength", type=float, default=0.0)
    ap.add_argument("--perturb-tag", type=str, default="",
                    help="suffix appended to variant name in results.jsonl "
                         "(e.g., '_perturb_noise05')")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated variant names to run; default = all rows")
    ap.add_argument("--benchmark", choices=BENCHMARK_CHOICES, default="libero",
                    help="which benchmark splits + dataset to use")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    manifest = yaml.safe_load((REPO_ROOT / args.manifest).read_text())
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    # Resume: skip variants already in results.jsonl with no error
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            try:
                row = json.loads(line)
                if "error" not in row:
                    done.add(row["variant"])
            except Exception:
                pass
        print(f"resuming, done: {len(done)} rows")

    only_set = set(s.strip() for s in args.only.split(",")) if args.only else None

    with results_path.open("a") as f_out:
        for entry in manifest:
            name = entry["variant"]
            if only_set is not None and name not in only_set:
                continue
            tagged_name_pre = name + (args.perturb_tag if args.perturb_tag else "")
            if tagged_name_pre in done:
                print(f"[skip] {tagged_name_pre} (done)")
                continue
            t0 = time.time()
            result = process_entry(entry, args, device)
            tagged_name = name + (args.perturb_tag if args.perturb_tag else "")
            row = {
                "variant": tagged_name,
                "base_model": entry["base_model"],
                "family": entry["family"],
                "seq_len": entry["seq_len"],
                "model_size": entry["model_size"],
                "n_params_million": entry.get("n_params_million"),
                "pretrained": entry["pretrained"],
                "ckpt": entry.get("ckpt"),
                "dynamics_ckpt": entry.get("dynamics_ckpt"),
                "mode": entry.get("mode"),
                "feature_source": entry.get("feature_source"),
                "pool": entry.get("pool"),
                "notes": entry.get("notes", ""),
                "perturb_test_kind": args.perturb_test_kind,
                "perturb_test_strength": args.perturb_test_strength,
                **result,
                "elapsed_s": round(time.time() - t0, 1),
            }
            f_out.write(json.dumps(row, default=str) + "\n")
            f_out.flush()
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
