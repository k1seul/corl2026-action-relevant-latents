"""Comprehensive probe-only sweep for the Tier-1 OOD finding.

Pushes how far probe-design alone can take task-OOD action R²
(baseline v6_large_noact_taskheld @ 0.717) without any model retrain.

Pipeline
--------
1. Per (model, split), encode pooled features once and pickle to disk
   (`cache/probe_features/<model>_<split>.pt`).
2. Phase A — frame stacking  : stack ∈ {1,3,5,7,9} × 3 models × 3 seeds
                                MLP [512,256,128], 10k steps, cosine LR
                                with 5% warmup, wd=1e-4, action H=1 + inv-dyn.
3. Phase B — MLP architecture : at best-stack-per-model, sweep
                                {[256,128], [512,256,128], [512,512,256,128],
                                 [1024,512,256]}.
4. Phase C — training hparams : steps × wd × dropout
                                = {10k,30k} × {0, 1e-4} × {0, 0.1}.
5. Best-config extras (one cell per model):
       long inv-dyn   : (s_t, s_{t+5}) -> a_{t..t+4}
       multi-frame ID : (s_{t-2..t}, s_{t+1..t+3}) -> a_t

Sanity
------
* Baseline reproduction: stack=1, [256,128], 1k steps, no scheduler ->
  v6_large_noact_taskheld OOD R²(H=1) ≈ 0.717.
* Random-init feature baseline (untrained backbone) -> R² near 0.
* Train-test MSE gap reported per cell for overfitting diagnosis.

Caveat about v_mix_token_noact_200k
-----------------------------------
That model was trained on the original `data/splits/train.json` (all
130 tasks, 40 demos/task), so the splits_taskheld/test_ood.json clips
include ~80% demos that the model already saw during training. Its
"OOD" R² therefore double-counts in-distribution data; we still run it
for completeness but flag it in the output table.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

# reuse model builders + extractor + encode_split from eval_downstream
import importlib.util
_es_spec = importlib.util.spec_from_file_location("eval_downstream", str(REPO_ROOT / "scripts" / "eval_downstream.py"))
_es = importlib.util.module_from_spec(_es_spec)
_es_spec.loader.exec_module(_es)

build_pixel_model = _es.build_pixel_model
build_token_model = _es.build_token_model
load_tokenizer = _es.load_tokenizer
encode_clip_to_tokens = _es.encode_clip_to_tokens
FeatureExtractor = _es.FeatureExtractor
encode_split = _es.encode_split
build_task_index = _es.build_task_index


# ---- spec ----

MODELS = {
    "v6_large_noact_taskheld": {
        "ckpt": "runs/sweep/variants/v6_large_noact_taskheld/ckpt_last.pt",
        "train_split": "data/splits_taskheld/train.json",
        "ood_split": "data/splits_taskheld/test_ood.json",
        "true_ood": True,
    },
    "v10_taskheld": {
        "ckpt": "runs/sweep/variants/v10_taskheld/ckpt_last.pt",
        "train_split": "data/splits_taskheld/train.json",
        "ood_split": "data/splits_taskheld/test_ood.json",
        "true_ood": True,
    },
    "v_mix_token_noact_200k": {
        "ckpt": "runs/sweep/variants/v_mix_token_noact_200k/ckpt_last.pt",
        # token model was trained on the original 130-task split; its
        # "OOD" overlaps with its training data
        "train_split": "data/splits/train.json",
        "ood_split": "data/splits_taskheld/test_ood.json",
        "true_ood": False,  # see caveat in module docstring
    },
}

CACHE_DIR = REPO_ROOT / "runs" / "downstream" / "_probe_cache"
OUT_DIR = REPO_ROOT / "runs" / "downstream"

PHASE_A_STACKS = [1, 3, 5, 7, 9]
PHASE_B_ARCHS = [(256, 128), (512, 256, 128), (512, 512, 256, 128), (1024, 512, 256)]
PHASE_C_STEPS = [10_000, 30_000]
PHASE_C_WDS = [0.0, 1e-4]
PHASE_C_DROPOUTS = [0.0, 0.1]
SEEDS = [0, 1, 2]


# ---- feature encoding & cache ----


def _build_extractor(ckpt_path: str, device: torch.device, randomize: bool = False):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["config"])
    in_channels = 3 * len(cfg.data.cameras)
    objective = cfg.train.get("objective", "mae")
    if objective == "token_ce":
        tok = load_tokenizer(cfg.train.tokenizer_ckpt, in_channels, device)
        # peek a frame to derive token-grid size
        from data.libero_video import LiberoVideoClips
        peek_ds = LiberoVideoClips(
            split_path=cfg.data.test_split, datasets_dir=cfg.data.datasets_dir,
            clip_len=cfg.data.clip_len, cameras=cfg.data.cameras,
            resolution=cfg.data.resolution, seed=0,
        )
        peek = peek_ds[0][:1].to(device).unsqueeze(0)
        with torch.no_grad():
            sample_ids = encode_clip_to_tokens(tok, peek)
        N_per_frame = sample_ids.shape[-1]
        model = build_token_model(cfg, tok.codebook_size, N_per_frame).to(device)
    else:
        tok = None
        model = build_pixel_model(cfg, in_channels).to(device)
    if not randomize:
        model.load_state_dict(state["model"])
    model.eval()
    return FeatureExtractor(model, objective, tok), cfg, in_channels


def cache_features(model_name: str, train_clips_cap: int, test_clips_cap: int,
                   device: torch.device, force: bool = False) -> dict:
    """Encode and cache features for one model on its train + OOD splits."""
    info = MODELS[model_name]
    cache_path = CACHE_DIR / f"{model_name}.pt"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force:
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    extractor, cfg, in_channels = _build_extractor(info["ckpt"], device)
    print(f"[encode] {model_name}: train_split={info['train_split']} (cap {train_clips_cap})")
    task_index = build_task_index(info["train_split"])
    cfg = cfg.copy()
    cfg.data.train_split = info["train_split"]

    t0 = time.time()
    train_data = encode_split(
        extractor, info["train_split"], cfg, in_channels,
        train_clips_cap, device, task_index, seed=int(cfg.train.seed) + 11,
    )
    print(f"  train clips={train_data['n_clips']} feat dim={train_data['feat_per_clip'].shape[-1]} "
          f"({time.time()-t0:.1f}s)")

    t0 = time.time()
    cfg.data.test_split = info["ood_split"]
    ood_data = encode_split(
        extractor, info["ood_split"], cfg, in_channels,
        test_clips_cap, device, task_index, seed=int(cfg.train.seed) + 12,
    )
    print(f"  ood   clips={ood_data['n_clips']} ({time.time()-t0:.1f}s)")

    cache = {
        "model": model_name,
        "true_ood": info["true_ood"],
        "feat_dim": int(train_data["feat_per_clip"].shape[-1]),
        "clip_len": int(cfg.data.clip_len),
        "action_dim": int(train_data["actions_per_clip"].shape[-1]),
        "train": {
            "feat_per_clip_t": train_data["feat_per_clip_t"],
            "actions_per_clip": train_data["actions_per_clip"],
            "task_per_clip": train_data["task_per_clip"],
            "n_clips": train_data["n_clips"],
        },
        "ood": {
            "feat_per_clip_t": ood_data["feat_per_clip_t"],
            "actions_per_clip": ood_data["actions_per_clip"],
            "task_per_clip": ood_data["task_per_clip"],
            "n_clips": ood_data["n_clips"],
        },
    }
    torch.save(cache, cache_path)
    print(f"  cached -> {cache_path}")
    return cache


def cache_random_init_features(model_name: str, train_clips_cap: int, test_clips_cap: int,
                                device: torch.device) -> dict:
    """Encode features with a randomly-initialized backbone (sanity baseline)."""
    info = MODELS[model_name]
    extractor, cfg, in_channels = _build_extractor(info["ckpt"], device, randomize=True)
    task_index = build_task_index(info["train_split"])
    cfg = cfg.copy()
    cfg.data.train_split = info["train_split"]
    cfg.data.test_split = info["ood_split"]
    train_data = encode_split(
        extractor, info["train_split"], cfg, in_channels,
        train_clips_cap, device, task_index, seed=int(cfg.train.seed) + 11,
    )
    ood_data = encode_split(
        extractor, info["ood_split"], cfg, in_channels,
        test_clips_cap, device, task_index, seed=int(cfg.train.seed) + 12,
    )
    return {
        "feat_dim": int(train_data["feat_per_clip"].shape[-1]),
        "train": train_data,
        "ood": ood_data,
    }


# ---- dataset builders ----


def _stack_with_repeat_pad(feat_kt: torch.Tensor, stack: int) -> torch.Tensor:
    """`(K, T, D) -> (K, T, stack*D)` where output[k, t] = concat(feat[k, t-stack+1..t]).

    First `stack-1` frames are filled by repeating feat[k, 0].
    """
    K, T, D = feat_kt.shape
    if stack <= 1:
        return feat_kt
    pad = feat_kt[:, :1].expand(K, stack - 1, D)
    padded = torch.cat([pad, feat_kt], dim=1)  # (K, T+stack-1, D)
    out = torch.stack([padded[:, i:i + T] for i in range(stack)], dim=-2)  # (K, T, stack, D)
    return out.reshape(K, T, stack * D)


def build_action_dataset(feat_kt: torch.Tensor, actions_kt: torch.Tensor, stack: int):
    """`(K, T, D), (K, T, A) -> (M, stack*D), (M, A)` for action regression."""
    feat_s = _stack_with_repeat_pad(feat_kt, stack)
    K, T, _ = feat_s.shape
    A = actions_kt.shape[-1]
    return feat_s.reshape(K * T, -1), actions_kt.reshape(K * T, A)


def build_invdyn_dataset(feat_kt: torch.Tensor, actions_kt: torch.Tensor, stack: int):
    """`(K, T, D), (K, T, A) -> (M, 2*stack*D), (M, A)` for inverse dynamics
    using stacked s_t and stacked s_{t+1}.
    """
    feat_s = _stack_with_repeat_pad(feat_kt, stack)
    pairs = torch.cat([feat_s[:, :-1], feat_s[:, 1:]], dim=-1)  # (K, T-1, 2*stack*D)
    targets = actions_kt[:, :-1]
    return pairs.reshape(-1, pairs.shape[-1]), targets.reshape(-1, actions_kt.shape[-1])


def build_long_invdyn_dataset(feat_kt: torch.Tensor, actions_kt: torch.Tensor, horizon: int = 5):
    """`(K, T, D), (K, T, A) -> (M, 2*D), (M, horizon*A)`:
    `(s_t, s_{t+horizon}) -> a_{t..t+horizon-1}` flattened.
    """
    K, T, D = feat_kt.shape
    A = actions_kt.shape[-1]
    if T <= horizon:
        raise ValueError(f"horizon {horizon} >= T {T}")
    pairs = torch.cat([feat_kt[:, :T - horizon], feat_kt[:, horizon:]], dim=-1)  # (K, T-h, 2D)
    target_chunks = torch.stack(
        [actions_kt[:, h:T - horizon + h] for h in range(horizon)], dim=-2
    )  # (K, T-h, h, A)
    return pairs.reshape(-1, 2 * D), target_chunks.reshape(-1, horizon * A)


def build_multiframe_invdyn_dataset(
    feat_kt: torch.Tensor, actions_kt: torch.Tensor, past: int = 3, future: int = 3
):
    """`(K, T, D), (K, T, A) -> (M, (past+future)*D), (M, A)`:
    `(s_{t-past+1..t}, s_{t+1..t+future}) -> a_t`. The targeted action
    is `a_t` with `t` ranging in `[past-1, T-future)`.
    """
    K, T, D = feat_kt.shape
    A = actions_kt.shape[-1]
    valid = T - past + 1 - future
    if valid <= 0:
        raise ValueError(f"clip_len {T} too short for past={past} future={future}")
    chunks = []
    for i in range(past + future):
        # for output index t in [past-1, past-1+valid), we read frame
        # past-1 - (past-1-i) = i (when i < past) and t+1+(i-past) for i>=past
        if i < past:
            offset = i  # 0, 1, ..., past-1 -> frames [t-past+1, ..., t]
        else:
            offset = i  # past, ..., past+future-1 -> frames [t+1, ..., t+future]
        # the start frame for that window is offset
        chunks.append(feat_kt[:, offset:offset + valid])
    pairs = torch.cat(chunks, dim=-1)  # (K, valid, (past+future)*D)
    targets = actions_kt[:, past - 1:past - 1 + valid]
    return pairs.reshape(-1, pairs.shape[-1]), targets.reshape(-1, A)


# ---- probe trainer ----


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: tuple[int, ...] = (),
                 dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def cosine_lr(step: int, max_steps: int, warmup_frac: float) -> float:
    warmup = max(1, int(max_steps * warmup_frac))
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train_probe(
    train_x: torch.Tensor, train_y: torch.Tensor,
    test_x: torch.Tensor, test_y: torch.Tensor,
    *,
    arch: tuple[int, ...],
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    use_scheduler: bool,
    warmup_frac: float,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    torch.manual_seed(seed)
    in_dim = train_x.shape[-1]
    out_dim = train_y.shape[-1]
    probe = MLPProbe(in_dim, out_dim, hidden_dims=arch, dropout=dropout).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    train_x = train_x.to(device); train_y = train_y.to(device)
    test_x = test_x.to(device); test_y = test_y.to(device)
    n = train_x.shape[0]
    probe.train()
    for step in range(steps):
        if use_scheduler:
            for g in opt.param_groups:
                g["lr"] = lr * cosine_lr(step, steps, warmup_frac)
        idx = torch.randint(0, n, (batch_size,), device=device)
        loss = F.mse_loss(probe(train_x[idx]), train_y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        pred_test = probe(test_x)
        pred_train = probe(train_x[:min(n, 8192)])  # subsample for train mse
        train_y_sub = train_y[:pred_train.shape[0]]
    test_mse = F.mse_loss(pred_test, test_y).item()
    train_mse = F.mse_loss(pred_train, train_y_sub).item()
    var = test_y.var(dim=0, unbiased=False).mean().item()
    r2 = 1.0 - test_mse / max(1e-12, var)
    train_r2 = 1.0 - train_mse / max(1e-12, train_y_sub.var(dim=0, unbiased=False).mean().item())
    return {
        "test_r2": r2,
        "train_r2": train_r2,
        "test_mse": test_mse,
        "train_mse": train_mse,
        "test_var": var,
        "params": int(sum(p.numel() for p in probe.parameters())),
    }


# ---- aggregation helpers ----


def mean_std(values: list[float]) -> tuple[float, float]:
    m = sum(values) / len(values)
    s = (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5
    return m, s


def best_by_mean_r2(rows: list[dict], group_keys: list[str], metric: str = "test_r2") -> dict:
    """Pick the cell with the highest mean test_r2 across seeds."""
    grouped: dict[tuple, list[float]] = {}
    for r in rows:
        if metric not in r: continue
        key_parts = []
        for k in group_keys:
            v = r[k]
            # lists (e.g. arch) aren't hashable; convert to tuples for grouping
            if isinstance(v, list):
                v = tuple(v)
            key_parts.append(v)
        key = tuple(key_parts)
        grouped.setdefault(key, []).append(r[metric])
    best_key, best_mean = None, -float("inf")
    for k, vs in grouped.items():
        m = sum(vs) / len(vs)
        if m > best_mean:
            best_mean = m; best_key = k
    return {"key": best_key, "mean_r2": best_mean,
            "values": grouped[best_key] if best_key in grouped else []}


# ---- phases ----


def run_phase(
    cache: dict, *, phase_name: str, configs: list[dict],
    seeds: list[int], device: torch.device, batch_size: int = 1024,
) -> list[dict]:
    """Run a list of probe configs, each over the given seeds. Returns
    flat list of result dicts.
    """
    rows: list[dict] = []
    feat_train = cache["train"]["feat_per_clip_t"]
    actions_train = cache["train"]["actions_per_clip"]
    feat_ood = cache["ood"]["feat_per_clip_t"]
    actions_ood = cache["ood"]["actions_per_clip"]

    # cache datasets per (stack, probe_kind) so we don't re-stack across seeds
    ds_cache: dict[tuple, tuple] = {}

    def get_ds(stack: int, kind: str):
        key = (stack, kind)
        if key in ds_cache:
            return ds_cache[key]
        if kind == "action":
            tx, ty = build_action_dataset(feat_train, actions_train, stack)
            vx, vy = build_action_dataset(feat_ood, actions_ood, stack)
        elif kind == "invdyn":
            tx, ty = build_invdyn_dataset(feat_train, actions_train, stack)
            vx, vy = build_invdyn_dataset(feat_ood, actions_ood, stack)
        else:
            raise ValueError(kind)
        ds_cache[key] = (tx, ty, vx, vy)
        return ds_cache[key]

    for cfg in configs:
        for kind in ("action", "invdyn"):
            tx, ty, vx, vy = get_ds(cfg["stack"], kind)
            for seed in seeds:
                t0 = time.time()
                m = train_probe(
                    tx, ty, vx, vy,
                    arch=cfg["arch"],
                    steps=cfg["steps"],
                    batch_size=batch_size,
                    lr=cfg.get("lr", 3e-4),
                    weight_decay=cfg["wd"],
                    dropout=cfg["dropout"],
                    use_scheduler=cfg["use_scheduler"],
                    warmup_frac=cfg.get("warmup_frac", 0.05),
                    seed=seed,
                    device=device,
                )
                row = {
                    "phase": phase_name, "kind": kind,
                    "stack": cfg["stack"], "arch": list(cfg["arch"]),
                    "steps": cfg["steps"], "wd": cfg["wd"], "dropout": cfg["dropout"],
                    "scheduler": cfg["use_scheduler"], "seed": seed,
                    "elapsed": time.time() - t0,
                    **m,
                }
                rows.append(row)
    return rows


# ---- output / formatting ----


def format_phase_table(rows: list[dict], group_keys: list[str], kind: str,
                       value_keys: list[str] | None = None) -> str:
    if value_keys is None:
        value_keys = ["test_r2"]
    rows = [r for r in rows if r["kind"] == kind]
    grouped: dict[tuple, list[dict]] = {}
    for r in rows:
        key_parts = []
        for k in group_keys:
            v = r[k]
            if isinstance(v, list):
                v = tuple(v)
            key_parts.append(v)
        key = tuple(key_parts)
        grouped.setdefault(key, []).append(r)
    out_lines = []
    header = "| " + " | ".join(group_keys + [f"{vk} (mean±std)" for vk in value_keys]) + " |"
    sep = "|" + "|".join(["---"] * (len(group_keys) + len(value_keys))) + "|"
    out_lines += [header, sep]
    sorted_keys = sorted(grouped.keys())
    for key in sorted_keys:
        values_per_metric = []
        for vk in value_keys:
            vals = [r[vk] for r in grouped[key]]
            m, s = mean_std(vals)
            values_per_metric.append(f"{m:+.4f}±{s:.4f}")
        out_lines.append("| " + " | ".join(list(map(str, key)) + values_per_metric) + " |")
    return "\n".join(out_lines)


# ---- main ----


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--train-clips", type=int, default=2000,
                    help="how many train demos to encode for probe training")
    ap.add_argument("--test-clips", type=int, default=1300,
                    help="how many OOD test demos to encode")
    ap.add_argument("--phase", type=str, default="all",
                    choices=["sanity", "A", "B", "C", "extras", "all"])
    ap.add_argument("--probe-batch-size", type=int, default=1024)
    ap.add_argument("--encode-only", action="store_true",
                    help="just build the feature cache, no probe training")
    ap.add_argument("--force-encode", action="store_true",
                    help="re-encode even if cache exists")
    args = ap.parse_args()

    device = torch.device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Encode features for each model
    caches: dict[str, dict] = {}
    for name in MODELS:
        caches[name] = cache_features(
            name, args.train_clips, args.test_clips, device, force=args.force_encode,
        )

    if args.encode_only:
        print("encoding done; skipping probes per --encode-only")
        return

    all_rows: list[dict] = []
    summaries: dict[str, dict] = {m: {} for m in MODELS}

    # 2. Sanity checks
    if args.phase in ("sanity", "all"):
        print("\n[sanity] reproducing baseline & random-init feature ...")
        for name, cache in caches.items():
            # baseline reproduction: stack=1, [256,128], 1k step, no scheduler
            rows = run_phase(
                cache, phase_name="sanity_baseline",
                configs=[{"stack": 1, "arch": (256, 128), "steps": 1000,
                          "wd": 1e-4, "dropout": 0.0,
                          "use_scheduler": False}],
                seeds=SEEDS, device=device,
                batch_size=args.probe_batch_size,
            )
            for r in rows:
                r["model"] = name
            all_rows += rows
            action_rows = [r for r in rows if r["kind"] == "action"]
            r2s = [r["test_r2"] for r in action_rows]
            m, s = mean_std(r2s)
            summaries[name]["sanity_baseline_r2"] = (m, s)
            print(f"  {name} sanity (stack=1, [256,128], 1k step): R²={m:+.4f}±{s:.4f}")
        # random-init feature baseline (encode once with random model)
        print("[sanity] random-init feature baseline ...")
        random_caches = {}
        for name in MODELS:
            rc = cache_random_init_features(name, 500, 500, device)
            random_caches[name] = rc
            tx, ty = build_action_dataset(rc["train"]["feat_per_clip_t"],
                                          rc["train"]["actions_per_clip"], 1)
            vx, vy = build_action_dataset(rc["ood"]["feat_per_clip_t"],
                                          rc["ood"]["actions_per_clip"], 1)
            m = train_probe(
                tx, ty, vx, vy, arch=(256, 128), steps=1000,
                batch_size=args.probe_batch_size, lr=3e-4, weight_decay=1e-4,
                dropout=0.0, use_scheduler=False, warmup_frac=0.05,
                seed=0, device=device,
            )
            summaries[name]["random_init_r2"] = (m["test_r2"], 0.0)
            print(f"  {name} random-init feature R² = {m['test_r2']:+.4f}")

    # 3. Phase A — frame stacking
    if args.phase in ("A", "all"):
        print("\n[Phase A] frame-stack sweep ...")
        for name, cache in caches.items():
            print(f"  -- {name} --")
            cfgs = [
                {"stack": s, "arch": (512, 256, 128), "steps": 10_000,
                 "wd": 1e-4, "dropout": 0.0,
                 "use_scheduler": True, "warmup_frac": 0.05}
                for s in PHASE_A_STACKS
            ]
            rows = run_phase(cache, phase_name="A", configs=cfgs,
                             seeds=SEEDS, device=device,
                             batch_size=args.probe_batch_size)
            for r in rows:
                r["model"] = name
            all_rows += rows
            best = best_by_mean_r2(
                [r for r in rows if r["kind"] == "action"], ["stack"],
            )
            summaries[name]["best_stack"] = best["key"][0]
            summaries[name]["A_best_r2"] = best["mean_r2"]
            print(f"    best stack = {best['key'][0]}, R²(action) = {best['mean_r2']:+.4f}")

    # 4. Phase B — MLP architecture
    if args.phase in ("B", "all"):
        print("\n[Phase B] MLP architecture sweep at best stack ...")
        for name, cache in caches.items():
            stack = summaries[name].get("best_stack", 1)
            print(f"  -- {name} (stack={stack}) --")
            cfgs = [
                {"stack": stack, "arch": arch, "steps": 10_000,
                 "wd": 1e-4, "dropout": 0.0,
                 "use_scheduler": True, "warmup_frac": 0.05}
                for arch in PHASE_B_ARCHS
            ]
            rows = run_phase(cache, phase_name="B", configs=cfgs,
                             seeds=SEEDS, device=device,
                             batch_size=args.probe_batch_size)
            for r in rows:
                r["model"] = name
            all_rows += rows
            best = best_by_mean_r2(
                [r for r in rows if r["kind"] == "action"], ["arch"],
            )
            summaries[name]["best_arch"] = best["key"][0]
            summaries[name]["B_best_r2"] = best["mean_r2"]
            print(f"    best arch = {best['key'][0]}, R²(action) = {best['mean_r2']:+.4f}")

    # 5. Phase C — training hparams
    if args.phase in ("C", "all"):
        print("\n[Phase C] training-hparam sweep at (best stack, best arch) ...")
        for name, cache in caches.items():
            stack = summaries[name].get("best_stack", 1)
            arch = tuple(summaries[name].get("best_arch", (512, 256, 128)))
            print(f"  -- {name} (stack={stack}, arch={arch}) --")
            cfgs = []
            for steps in PHASE_C_STEPS:
                for wd in PHASE_C_WDS:
                    for dr in PHASE_C_DROPOUTS:
                        cfgs.append({"stack": stack, "arch": arch, "steps": steps,
                                     "wd": wd, "dropout": dr,
                                     "use_scheduler": True, "warmup_frac": 0.05})
            rows = run_phase(cache, phase_name="C", configs=cfgs,
                             seeds=SEEDS, device=device,
                             batch_size=args.probe_batch_size)
            for r in rows:
                r["model"] = name
            all_rows += rows
            best = best_by_mean_r2(
                [r for r in rows if r["kind"] == "action"], ["steps", "wd", "dropout"],
            )
            summaries[name]["best_C"] = best["key"]
            summaries[name]["C_best_r2"] = best["mean_r2"]
            print(f"    best (steps,wd,dropout) = {best['key']}, R²(action) = {best['mean_r2']:+.4f}")

    # 6. Best-config extras: long inv-dyn, multi-frame inv-dyn
    extras: list[dict] = []
    if args.phase in ("extras", "all"):
        print("\n[extras] long + multi-frame inverse dynamics at best Phase-C config ...")
        for name, cache in caches.items():
            stack = summaries[name].get("best_stack", 1)
            arch = tuple(summaries[name].get("best_arch", (512, 256, 128)))
            steps, wd, dropout = summaries[name].get("best_C", (10_000, 1e-4, 0.0))
            feat_train = cache["train"]["feat_per_clip_t"]
            actions_train = cache["train"]["actions_per_clip"]
            feat_ood = cache["ood"]["feat_per_clip_t"]
            actions_ood = cache["ood"]["actions_per_clip"]
            for kind, builder in [
                ("long_invdyn_h5", lambda f, a: build_long_invdyn_dataset(f, a, horizon=5)),
                ("multiframe_invdyn", lambda f, a: build_multiframe_invdyn_dataset(f, a, past=3, future=3)),
            ]:
                tx, ty = builder(feat_train, actions_train)
                vx, vy = builder(feat_ood, actions_ood)
                for seed in SEEDS:
                    t0 = time.time()
                    m = train_probe(
                        tx, ty, vx, vy, arch=arch, steps=steps,
                        batch_size=args.probe_batch_size, lr=3e-4,
                        weight_decay=wd, dropout=dropout,
                        use_scheduler=True, warmup_frac=0.05,
                        seed=seed, device=device,
                    )
                    extras.append({
                        "phase": "extras", "kind": kind, "model": name,
                        "stack": stack, "arch": list(arch),
                        "steps": steps, "wd": wd, "dropout": dropout,
                        "seed": seed, "elapsed": time.time() - t0, **m,
                    })
            for kind in ("long_invdyn_h5", "multiframe_invdyn"):
                vals = [r["test_r2"] for r in extras if r["model"] == name and r["kind"] == kind]
                if vals:
                    mm, ss = mean_std(vals)
                    summaries[name][f"{kind}_r2"] = (mm, ss)
                    print(f"    {name} {kind}: R²={mm:+.4f}±{ss:.4f}")
    all_rows += extras

    # 7. write outputs
    rows_path = OUT_DIR / "probe_tier1_sweep.jsonl"
    with rows_path.open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {rows_path}")

    # markdown summary
    md = []
    md.append("# Probe Tier-1 sweep — comprehensive probe-only OOD R² exploration\n")
    md.append("Goal: push the v6_large_noact_taskheld task-OOD R² (baseline 0.717)")
    md.append("via probe-side improvements only — no model retraining.\n")
    md.append("**Models**: " + ", ".join(MODELS.keys()) + ".\n")
    md.append("> Caveat: `v_mix_token_noact_200k` was trained on the original 130-task split,")
    md.append("> so the splits_taskheld/test_ood demos overlap with its training data.")
    md.append("> Its 'OOD' R² is included for completeness but is *not* a true held-out generalization.\n")

    # headline
    md.append("## Headline — best probe per model (action H=1)\n")
    md.append("| model | sanity baseline R² | best-stack R² (A) | best-arch R² (B) | best-hparam R² (C) |")
    md.append("|---|---|---|---|---|")
    for name in MODELS:
        s = summaries[name]
        sb = s.get("sanity_baseline_r2", (None, 0))
        a = s.get("A_best_r2"); b = s.get("B_best_r2"); c = s.get("C_best_r2")
        def f(x): return f"{x:+.4f}" if x is not None else "—"
        sb_str = f"{sb[0]:+.4f}±{sb[1]:.4f}" if sb[0] is not None else "—"
        md.append(f"| {name} | {sb_str} | {f(a)} | {f(b)} | {f(c)} |")
    md.append("")

    # random-init baseline
    md.append("## Sanity — random-init feature baseline (action R²)\n")
    md.append("| model | R² |")
    md.append("|---|---|")
    for name in MODELS:
        ri = summaries[name].get("random_init_r2", (None, 0))
        md.append(f"| {name} | {ri[0]:+.4f} |" if ri[0] is not None else f"| {name} | — |")
    md.append("")

    # phase tables
    if args.phase in ("A", "all"):
        md.append("## Phase A — frame stacking\n")
        for name in MODELS:
            md.append(f"### {name}\n")
            md.append("**Action R² (H=1)**\n")
            md.append(format_phase_table(
                [r for r in all_rows if r["phase"] == "A" and r["model"] == name],
                ["stack"], "action",
            ))
            md.append("\n**Inverse dynamics R²**\n")
            md.append(format_phase_table(
                [r for r in all_rows if r["phase"] == "A" and r["model"] == name],
                ["stack"], "invdyn",
            ))
            md.append("\n")
    if args.phase in ("B", "all"):
        md.append("## Phase B — MLP architecture (at best stack)\n")
        for name in MODELS:
            md.append(f"### {name} (stack={summaries[name].get('best_stack', '?')})\n")
            md.append(format_phase_table(
                [r for r in all_rows if r["phase"] == "B" and r["model"] == name],
                ["arch"], "action",
            ))
            md.append("\n")
    if args.phase in ("C", "all"):
        md.append("## Phase C — training hparams (at best stack & arch)\n")
        for name in MODELS:
            md.append(f"### {name} (stack={summaries[name].get('best_stack', '?')}, "
                      f"arch={summaries[name].get('best_arch', '?')})\n")
            md.append(format_phase_table(
                [r for r in all_rows if r["phase"] == "C" and r["model"] == name],
                ["steps", "wd", "dropout"], "action",
            ))
            md.append("\n")
    if args.phase in ("extras", "all"):
        md.append("## Extras — long & multi-frame inverse dynamics (at best Phase-C cell)\n")
        md.append("| model | long inv-dyn R² (H=5) | multi-frame inv-dyn R² (3 past + 3 future) |")
        md.append("|---|---|---|")
        for name in MODELS:
            li = summaries[name].get("long_invdyn_h5_r2", (None, 0))
            mf = summaries[name].get("multiframe_invdyn_r2", (None, 0))
            li_s = f"{li[0]:+.4f}±{li[1]:.4f}" if li[0] is not None else "—"
            mf_s = f"{mf[0]:+.4f}±{mf[1]:.4f}" if mf[0] is not None else "—"
            md.append(f"| {name} | {li_s} | {mf_s} |")
        md.append("")

    # production recommendation
    md.append("## Production probe config recommendation\n")
    for name in MODELS:
        s = summaries[name]
        if "best_stack" not in s:
            continue
        md.append(f"**{name}**: stack={s['best_stack']}, arch={s.get('best_arch', '?')}, "
                  f"hparams={s.get('best_C', '?')} → R²={s.get('C_best_r2', '?')}\n")

    md_path = OUT_DIR / "probe_tier1_sweep.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
