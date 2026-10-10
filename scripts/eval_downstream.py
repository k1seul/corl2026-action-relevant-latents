"""Linear / MLP probe downstream evaluation for any trained variant.

For a given checkpoint we:

  1. Build the model + tokenizer (if needed) and load weights.
  2. Extract per-token features ``(B, T, N, D)`` for a sample of train
     and test clips using each variant's ``forward_features`` (no head,
     no masking). For action-conditioned models we deliberately pass
     ``actions=None`` so action information cannot leak into the probe.
  3. Train two small probes on the cached features:

       * **action_regression**: pooled per-frame features -> action
         vector (R^7). Linear MLP, MSE loss; reports test MSE / R^2.
       * **task_classification**: pooled per-clip features -> task_id
         (130 classes). Linear; reports top-1 / top-5 accuracy.

  4. Write per-task metrics and a small JSON to ``out_dir``.

Run:
    uv run python scripts/eval_downstream.py --ckpt <ckpt_path> \
        --out-dir runs/downstream/<run_name>
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.libero_video import LiberoVideoClips, worker_init_fn
from models.block_causal_transformer import BlockCausalVideoTransformer
from models.token_video_transformer import TokenVideoTransformer
from models.tokenizer import FSQTokenizer


# ---- model / tokenizer construction (mirrors eval.py) ----


def build_pixel_model(cfg, in_channels: int) -> BlockCausalVideoTransformer:
    objective = cfg.train.get("objective", "mae")
    return BlockCausalVideoTransformer(
        in_channels=in_channels,
        patch_size=cfg.model.patch_size,
        num_frames=cfg.data.clip_len,
        image_size=cfg.data.resolution,
        dim=cfg.model.dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=0.0,
        enable_timestep=(objective in ("diffusion", "flow_matching")),
        action_dim=int(cfg.model.get("action_dim", 0)),
    )


def build_token_model(cfg, codebook_size: int, num_tokens_per_frame: int) -> TokenVideoTransformer:
    return TokenVideoTransformer(
        codebook_size=codebook_size,
        num_tokens_per_frame=num_tokens_per_frame,
        num_frames=cfg.data.clip_len,
        dim=cfg.model.dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=0.0,
        action_dim=int(cfg.model.get("action_dim", 0)),
    )


def load_tokenizer(ckpt_path: str, in_channels: int, device: torch.device) -> FSQTokenizer:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    tcfg = state["config"]
    tok = FSQTokenizer(
        in_channels=in_channels,
        levels=list(tcfg["model"]["levels"]),
        base_ch=tcfg["model"]["base_ch"],
    ).to(device)
    tok.load_state_dict(state["model"])
    tok.eval()
    return tok


@torch.no_grad()
def encode_clip_to_tokens(tok: FSQTokenizer, clip: torch.Tensor) -> torch.Tensor:
    B, T, C, H, W = clip.shape
    flat = clip.reshape(B * T, C, H, W)
    _, idx_grid = tok.encode(flat)
    return idx_grid.reshape(B, T, -1)


# ---- feature extraction wrapper ----


class FeatureExtractor:
    """Variant-agnostic feature extractor.

    `extract(clip)` returns post-norm token features `(B, T, N, D)`.
    Action input is deliberately set to ``None`` so action-conditioned
    models do not leak action info into the downstream action probe.
    """

    def __init__(self, model, objective: str, tokenizer: FSQTokenizer | None) -> None:
        self.model = model
        self.objective = objective
        self.tokenizer = tokenizer

    @torch.no_grad()
    def extract(self, clip: torch.Tensor) -> torch.Tensor:
        device = clip.device
        if self.objective == "mae":
            return self.model.forward_features(clip, mask_indicator=None)
        if self.objective in ("diffusion", "flow_matching"):
            B, T = clip.shape[:2]
            t = torch.zeros(B, T, device=device, dtype=torch.long)
            return self.model.forward_features(clip, mask_indicator=None, t=t)
        if self.objective == "token_ce":
            assert self.tokenizer is not None
            token_ids = encode_clip_to_tokens(self.tokenizer, clip)
            return self.model.forward_features(token_ids)
        raise ValueError(f"unknown objective: {self.objective}")


# ---- probes ----


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: tuple[int, ...] = ()) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_horizon_dataset(
    feat_kt: torch.Tensor,
    actions_kt: torch.Tensor,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair per-frame features with the next ``horizon`` actions.

    feat_kt: ``(K, T, D)`` per-clip per-frame features
    actions_kt: ``(K, T, A)`` per-clip per-frame actions
    Returns ``(feat: (M, D), targets: (M, horizon * A))`` for valid
    frames ``t in [0, T - horizon]``.
    """
    K, T, D = feat_kt.shape
    A = actions_kt.shape[-1]
    valid_T = T - horizon + 1
    if valid_T <= 0:
        raise ValueError(f"horizon {horizon} > clip_len {T}")
    feat = feat_kt[:, :valid_T].reshape(-1, D)
    stacked = torch.stack(
        [actions_kt[:, h : h + valid_T] for h in range(horizon)], dim=-2
    )  # (K, valid_T, H, A)
    targets = stacked.reshape(-1, horizon * A)
    return feat, targets


def make_inverse_dynamics_dataset(
    feat_kt: torch.Tensor,
    actions_kt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse dynamics: ``(feat[t], feat[t+1])`` -> ``action[t]``.

    Tests whether the feature pair encodes the transition that *caused*
    the action. Concatenates the two features along the feature axis.
    """
    K, T, D = feat_kt.shape
    A = actions_kt.shape[-1]
    pairs = torch.cat([feat_kt[:, :-1], feat_kt[:, 1:]], dim=-1)  # (K, T-1, 2D)
    targets = actions_kt[:, :-1]  # (K, T-1, A)
    return pairs.reshape(-1, 2 * D), targets.reshape(-1, A)


# ---- visual / modality perturbations for robustness OOD ----


def _gaussian_kernel2d(k: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
    g = torch.exp(-coords.pow(2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    g2 = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    return g2.expand(channels, 1, k, k).contiguous()


def perturb_clip(clip: torch.Tensor, kind: str, strength: float) -> torch.Tensor:
    """Apply a test-time perturbation to ``(B, T, C, H, W)`` clip in ``[0, 1]``.

    Supported kinds:
      ``none`` (identity)
      ``noise``    -- additive Gaussian, std=strength (e.g. 0.05, 0.15)
      ``blur``     -- per-frame Gaussian blur with kernel size = int(strength)
      ``jitter``   -- additive brightness shift in [-strength, strength]
      ``mask_eye`` -- zero out eye_in_hand channels (channels 3..5) in 2cam inputs
    """
    if kind == "none" or strength == 0:
        return clip
    if kind == "noise":
        return (clip + torch.randn_like(clip) * strength).clamp(0.0, 1.0)
    if kind == "blur":
        k = int(round(strength))
        if k <= 1:
            return clip
        sigma = max(1.0, k / 6.0)
        B, T, C, H, W = clip.shape
        flat = clip.reshape(B * T, C, H, W)
        win = _gaussian_kernel2d(k, sigma, C, clip.device, clip.dtype)
        blurred = F.conv2d(flat, win, padding=k // 2, groups=C)
        return blurred.reshape(B, T, C, H, W).clamp(0.0, 1.0)
    if kind == "jitter":
        delta = (torch.rand(clip.shape[0], 1, 1, 1, 1, device=clip.device, dtype=clip.dtype) * 2.0 - 1.0) * strength
        return (clip + delta).clamp(0.0, 1.0)
    if kind == "mask_eye":
        if clip.shape[2] < 6:
            return clip
        out = clip.clone()
        out[:, :, 3:6] = 0.0
        return out
    raise ValueError(f"unknown perturbation: {kind!r}")


def feature_stats(
    feat_per_clip: torch.Tensor, task_per_clip: torch.Tensor, max_pairs_per_task: int = 200
) -> dict[str, float]:
    """Compute simple feature-distribution statistics.

    norm_mean       : average L2 norm of clip features
    spread_mean     : average per-dim std (after centering on the mean)
    within_task_cos : average cosine similarity between feature pairs
                      drawn from the SAME task
    across_task_cos : same, drawn from DIFFERENT tasks
    task_separation : within - across (higher = better task structure)
    """
    K, D = feat_per_clip.shape
    norm_mean = feat_per_clip.norm(dim=-1).mean().item()
    spread_mean = (feat_per_clip - feat_per_clip.mean(dim=0, keepdim=True)).std(dim=0).mean().item()
    feat_normed = feat_per_clip / feat_per_clip.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    within: list[float] = []
    across: list[float] = []
    unique_tasks = task_per_clip.unique().tolist()
    for tid in unique_tasks:
        same = (task_per_clip == tid)
        diff = ~same
        if same.sum() < 2:
            continue
        in_feats = feat_normed[same]
        out_feats = feat_normed[diff]
        sim_in = in_feats @ in_feats.T
        triu = torch.triu_indices(in_feats.shape[0], in_feats.shape[0], offset=1)
        within.extend(sim_in[triu[0], triu[1]].tolist()[:max_pairs_per_task])
        if out_feats.shape[0] > 0:
            sim_out = (in_feats @ out_feats.T).flatten().tolist()
            across.extend(sim_out[:max_pairs_per_task])
    if not within or not across:
        return {"norm_mean": norm_mean, "spread_mean": spread_mean,
                "within_task_cos": float("nan"), "across_task_cos": float("nan"),
                "task_separation": float("nan")}
    w = sum(within) / len(within)
    a = sum(across) / len(across)
    return {
        "norm_mean": norm_mean,
        "spread_mean": spread_mean,
        "within_task_cos": w,
        "across_task_cos": a,
        "task_separation": w - a,
    }


def aggregate_seeds(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    """Mean and population std for each numeric key."""
    out: dict[str, float] = {}
    keys = set().union(*[set(m.keys()) for m in metrics_list])
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m]
        if not vals:
            continue
        m = sum(vals) / len(vals)
        v = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5
        out[f"{k}_mean"] = m
        out[f"{k}_std"] = v
    return out


def train_probe(
    probe: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    objective: str,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> dict[str, float]:
    """Train probe with AdamW; return final test metrics. ``objective``
    in ``{"regression", "classification"}``."""
    probe.train()
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    n = train_x.shape[0]
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), device=device)
        xb, yb = train_x[idx], train_y[idx]
        logits = probe(xb)
        if objective == "regression":
            loss = F.mse_loss(logits, yb)
        else:
            loss = F.cross_entropy(logits, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        out = probe(test_x)
        train_out = probe(train_x)
    if objective == "regression":
        test_mse = F.mse_loss(out, test_y).item()
        train_mse = F.mse_loss(train_out, train_y).item()
        # R^2 = 1 - MSE / Var
        var = test_y.var(dim=0, unbiased=False).mean().item()
        r2 = 1.0 - test_mse / max(1e-12, var)
        return {
            "test_mse": test_mse,
            "train_mse": train_mse,
            "test_r2": r2,
            "test_var": var,
        }
    else:
        test_pred = out.argmax(dim=-1)
        train_pred = train_out.argmax(dim=-1)
        topk = out.topk(min(5, out.shape[-1]), dim=-1).indices
        top1 = (test_pred == test_y).float().mean().item()
        top5 = (topk == test_y[:, None]).any(dim=-1).float().mean().item()
        train_top1 = (train_pred == train_y).float().mean().item()
        return {
            "test_top1": top1,
            "test_top5": top5,
            "train_top1": train_top1,
        }


# ---- task id mapping ----


def build_task_index(split_path: str) -> dict[tuple[str, str], int]:
    with open(split_path) as f:
        blob = json.load(f)
    seen = []
    seen_set = set()
    for ep in blob["episodes"]:
        key = (ep["suite"], ep["file"])
        if key not in seen_set:
            seen.append(key)
            seen_set.add(key)
    seen.sort()
    return {k: i for i, k in enumerate(seen)}


# ---- main encoding pass ----


def encode_split(
    extractor: FeatureExtractor,
    split_path: str,
    cfg,
    in_channels: int,
    num_clips_cap: int,
    device: torch.device,
    task_index: dict[tuple[str, str], int],
    seed: int,
    perturb_kind: str = "none",
    perturb_strength: float = 0.0,
):
    """Returns:
      feat_per_frame: (M, D)        -- pooled-over-N per-frame features
      action_per_frame: (M, A)      -- corresponding actions
      feat_per_clip:  (K, D)        -- pooled-over-T-and-N per-clip features
      task_per_clip:  (K,) long     -- task ids
      n_clips: int
    """
    ds = LiberoVideoClips(
        split_path=split_path,
        datasets_dir=cfg.data.datasets_dir,
        clip_len=cfg.data.clip_len,
        cameras=cfg.data.cameras,
        resolution=cfg.data.resolution,
        seed=seed,
        return_actions=True,
    )
    loader = DataLoader(
        ds, batch_size=cfg.eval.batch_size, shuffle=False,
        num_workers=2, worker_init_fn=worker_init_fn,
    )

    # we also need to know each batch element's (suite, file) to look up
    # task_id. We rely on the dataset's `episodes` list and the loader's
    # sequential traversal (shuffle=False).
    episode_iter = iter(ds.episodes)

    feat_per_clip_t_list, action_per_clip_list = [], []
    feat_per_clip_list, task_id_list = [], []
    seen = 0
    for batch in loader:
        clip, actions = batch
        clip = clip.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        B = clip.shape[0]

        if perturb_kind != "none":
            clip = perturb_clip(clip, perturb_kind, perturb_strength)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
            feats = extractor.extract(clip)  # (B, T, N, D)
        feats = feats.float()
        # pool over N for per-frame features
        f_frame = feats.mean(dim=2)  # (B, T, D)
        # pool over T and N for per-clip features
        f_clip = feats.mean(dim=(1, 2))  # (B, D)

        feat_per_clip_t_list.append(f_frame.cpu())
        action_per_clip_list.append(actions.cpu())
        feat_per_clip_list.append(f_clip.cpu())

        for _ in range(B):
            ep = next(episode_iter)
            # Episodes from an OOD split may reference tasks not in the
            # train-derived task_index; mark them with -1 so the task
            # probe can be skipped (action / inv-dyn probes still work).
            task_id_list.append(task_index.get((ep["suite"], ep["file"]), -1))

        seen += B
        if seen >= num_clips_cap:
            break

    return {
        "feat_per_clip_t": torch.cat(feat_per_clip_t_list, dim=0),  # (K, T, D)
        "actions_per_clip": torch.cat(action_per_clip_list, dim=0),  # (K, T, A)
        "feat_per_clip": torch.cat(feat_per_clip_list, dim=0),  # (K, D)
        "task_per_clip": torch.tensor(task_id_list, dtype=torch.long),
        "n_clips": seen,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--train-clips", type=int, default=1000)
    ap.add_argument("--test-clips", type=int, default=400)
    ap.add_argument("--probe-steps", type=int, default=3000)
    ap.add_argument("--probe-batch-size", type=int, default=512)
    ap.add_argument("--probe-lr", type=float, default=3e-4)
    ap.add_argument("--probe-weight-decay", type=float, default=1e-4)
    ap.add_argument("--probe-hidden-dims", type=str, default="256,128",
                    help="comma-separated hidden dims for the action MLP probe; '' = linear")
    ap.add_argument("--action-horizons", type=str, default="1,5",
                    help="comma-separated future-action horizons to probe")
    ap.add_argument("--probe-seeds", type=str, default="0,1,2",
                    help="comma-separated seeds for probe retraining; mean+/-std reported")
    ap.add_argument("--test-split", type=str, default=None,
                    help="override cfg.data.test_split (e.g. for OOD probe eval)")
    ap.add_argument("--test-perturbation", type=str, default="none",
                    help="visual / modality perturbation applied to test clips: "
                         "'none', 'noise_<sigma>', 'blur_<k>', 'jitter_<delta>', 'mask_eye'")
    args = ap.parse_args()
    hidden_dims = tuple(int(x) for x in args.probe_hidden_dims.split(",") if x.strip())
    horizons = [int(x) for x in args.action_horizons.split(",") if x.strip()]
    probe_seeds = [int(x) for x in args.probe_seeds.split(",") if x.strip()]

    device = torch.device(args.device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["config"])
    if args.test_split is not None:
        cfg.data.test_split = args.test_split

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "src_config.yaml")

    in_channels = 3 * len(cfg.data.cameras)
    objective = cfg.train.get("objective", "mae")

    # build model
    if objective == "token_ce":
        tokenizer = load_tokenizer(cfg.train.tokenizer_ckpt, in_channels, device)
        peek_ds = LiberoVideoClips(
            split_path=cfg.data.test_split, datasets_dir=cfg.data.datasets_dir,
            clip_len=cfg.data.clip_len, cameras=cfg.data.cameras,
            resolution=cfg.data.resolution, seed=cfg.train.seed + 1,
        )
        peek = peek_ds[0][:1].to(device).unsqueeze(0)  # (1, 1, C, H, W)
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
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"objective={objective} params={n_params/1e6:.1f}M clip_len={cfg.data.clip_len}")

    extractor = FeatureExtractor(model, objective, tokenizer)

    task_index = build_task_index(cfg.data.train_split)
    n_tasks = len(task_index)
    print(f"task_index size: {n_tasks}")

    print("encoding train ...")
    t0 = time.time()
    train_data = encode_split(
        extractor, cfg.data.train_split, cfg, in_channels, args.train_clips, device,
        task_index, seed=cfg.train.seed + 11,
    )
    print(f"  train clips={train_data['n_clips']} elapsed {time.time()-t0:.1f}s")

    # parse perturbation spec like 'noise_0.05', 'blur_5', 'mask_eye'
    spec = args.test_perturbation
    if spec in ("none", "mask_eye"):  # kinds with no numeric strength
        perturb_kind = spec
        perturb_strength = 0.0 if spec == "none" else 1.0
    elif "_" in spec:
        parts = spec.split("_", 1)
        perturb_kind = parts[0]
        try:
            perturb_strength = float(parts[1])
        except ValueError:
            perturb_strength = 1.0
    else:
        perturb_kind = spec
        perturb_strength = 1.0
    if perturb_kind != "none":
        print(f"test perturbation: kind={perturb_kind} strength={perturb_strength}")

    print("encoding test ...")
    t0 = time.time()
    test_data = encode_split(
        extractor, cfg.data.test_split, cfg, in_channels, args.test_clips, device,
        task_index, seed=cfg.train.seed + 12,
        perturb_kind=perturb_kind, perturb_strength=perturb_strength,
    )
    print(f"  test clips={test_data['n_clips']} elapsed {time.time()-t0:.1f}s")

    feat_dim = train_data["feat_per_clip"].shape[-1]
    action_dim = train_data["actions_per_clip"].shape[-1]

    def run_seeded_probe(make_probe, train_x, train_y, test_x, test_y, *,
                         obj: str, n_steps: int) -> dict[str, float]:
        runs = []
        for seed in probe_seeds:
            torch.manual_seed(seed)
            probe = make_probe().to(device)
            m = train_probe(
                probe, train_x, train_y, test_x, test_y,
                objective=obj, steps=n_steps,
                batch_size=args.probe_batch_size,
                lr=args.probe_lr, weight_decay=args.probe_weight_decay,
                device=device,
            )
            runs.append(m)
        return aggregate_seeds(runs)

    # action regression probes at each horizon (multi-seed)
    action_metrics: dict[int, dict[str, float]] = {}
    for H in horizons:
        print(f"training action probe H={H} (in={feat_dim}, out={H * action_dim}, hidden={hidden_dims}, seeds={len(probe_seeds)}) ...")
        t0 = time.time()
        train_x, train_y = make_horizon_dataset(
            train_data["feat_per_clip_t"], train_data["actions_per_clip"], H
        )
        test_x, test_y = make_horizon_dataset(
            test_data["feat_per_clip_t"], test_data["actions_per_clip"], H
        )
        action_metrics[H] = run_seeded_probe(
            lambda: MLPProbe(feat_dim, H * action_dim, hidden_dims=hidden_dims),
            train_x, train_y, test_x, test_y,
            obj="regression", n_steps=args.probe_steps,
        )
        print(f"  H={H}: R^2={action_metrics[H].get('test_r2_mean'):.4f}+/-{action_metrics[H].get('test_r2_std'):.4f} "
              f"({time.time()-t0:.1f}s)")

    # inverse dynamics probe: (feat[t], feat[t+1]) -> action[t]
    print(f"training inverse-dynamics probe (in={2*feat_dim}, out={action_dim}, seeds={len(probe_seeds)}) ...")
    t0 = time.time()
    id_train_x, id_train_y = make_inverse_dynamics_dataset(
        train_data["feat_per_clip_t"], train_data["actions_per_clip"]
    )
    id_test_x, id_test_y = make_inverse_dynamics_dataset(
        test_data["feat_per_clip_t"], test_data["actions_per_clip"]
    )
    inverse_metrics = run_seeded_probe(
        lambda: MLPProbe(2 * feat_dim, action_dim, hidden_dims=hidden_dims),
        id_train_x, id_train_y, id_test_x, id_test_y,
        obj="regression", n_steps=args.probe_steps,
    )
    print(f"  inv-dyn: R^2={inverse_metrics.get('test_r2_mean'):.4f}+/-{inverse_metrics.get('test_r2_std'):.4f} "
          f"({time.time()-t0:.1f}s)")

    # task classification probe (linear, multi-seed). Skip on OOD test
    # because task labels are out-of-vocabulary by construction.
    test_task_known = (test_data["task_per_clip"] >= 0).float().mean().item()
    if test_task_known < 1e-3:
        task_metrics = {
            "test_top1_mean": float("nan"), "test_top1_std": 0.0,
            "test_top5_mean": float("nan"), "train_top1_mean": float("nan"),
            "skipped_reason": "all test tasks OOD (not in train task_index)",
        }
        print(f"  task: SKIPPED (test_task_known={test_task_known:.3f})")
    else:
        print(f"training task probe (in={feat_dim}, classes={n_tasks}, seeds={len(probe_seeds)}) ...")
        t0 = time.time()
        task_metrics = run_seeded_probe(
            lambda: MLPProbe(feat_dim, n_tasks, hidden_dims=()),
            train_data["feat_per_clip"], train_data["task_per_clip"],
            test_data["feat_per_clip"], test_data["task_per_clip"],
            obj="classification", n_steps=max(args.probe_steps // 2, 500),
        )
        print(f"  task: top1={task_metrics.get('test_top1_mean'):.4f}+/-{task_metrics.get('test_top1_std'):.4f} "
              f"({time.time()-t0:.1f}s)")

    # feature distribution stats (single pass, no probe)
    print("computing feature stats (test split clip features) ...")
    t0 = time.time()
    fstats = feature_stats(test_data["feat_per_clip"], test_data["task_per_clip"])
    print(f"  norm={fstats['norm_mean']:.3f} spread={fstats['spread_mean']:.4f} "
          f"sep={fstats['task_separation']:.4f} ({time.time()-t0:.1f}s)")

    # write metrics
    metrics = {
        "objective": objective,
        "feat_dim": int(feat_dim),
        "n_tasks": int(n_tasks),
        "train_clips": int(train_data["n_clips"]),
        "test_clips": int(test_data["n_clips"]),
        "action_horizons": horizons,
        "probe_hidden_dims": list(hidden_dims),
        "probe_seeds": probe_seeds,
        "test_perturbation": {"kind": perturb_kind, "strength": perturb_strength},
        "action": {str(H): m for H, m in action_metrics.items()},
        "inverse_dynamics": inverse_metrics,
        "task": task_metrics,
        "feature_stats": fstats,
    }
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    with (out_dir / "metrics.txt").open("w") as f:
        f.write(f"objective: {objective}\n")
        f.write(f"feat_dim: {feat_dim}\n")
        for H, m in action_metrics.items():
            f.write(f"action_h{H}_r2_mean: {m['test_r2_mean']:.4f}\n")
            f.write(f"action_h{H}_r2_std: {m['test_r2_std']:.4f}\n")
            f.write(f"action_h{H}_mse_mean: {m['test_mse_mean']:.6f}\n")
        f.write(f"invdyn_r2_mean: {inverse_metrics['test_r2_mean']:.4f}\n")
        f.write(f"invdyn_r2_std: {inverse_metrics['test_r2_std']:.4f}\n")
        f.write(f"task_top1_mean: {task_metrics['test_top1_mean']:.4f}\n")
        f.write(f"task_top1_std: {task_metrics['test_top1_std']:.4f}\n")
        f.write(f"task_top5_mean: {task_metrics['test_top5_mean']:.4f}\n")
        f.write(f"feat_norm: {fstats['norm_mean']:.4f}\n")
        f.write(f"feat_spread: {fstats['spread_mean']:.4f}\n")
        f.write(f"feat_within_task_cos: {fstats['within_task_cos']:.4f}\n")
        f.write(f"feat_across_task_cos: {fstats['across_task_cos']:.4f}\n")
        f.write(f"feat_task_separation: {fstats['task_separation']:.4f}\n")
    print(f"wrote {out_dir/'metrics.json'} and {out_dir/'metrics.txt'}")


if __name__ == "__main__":
    main()
