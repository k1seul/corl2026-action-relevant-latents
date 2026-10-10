"""Surgical probe: per-layer R^2 of V-JEPA 2 ViT-L aux checkpoint.

Tests whether the JEPA pretraining objective shapes the trunk's later
layers away from action-relevance (peak-then-drop) or toward it
(monotonic increase).  Implementation uses forward hooks to record
each block's output during a single encoder forward pass.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/probe_vjepa_per_layer.py \\
        --ckpt runs/comparison/vjepa2_aux_finetune/ckpt_last.pt \\
        --out-jsonl runs/full_results/vjepa_per_layer.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data.libero_video import LiberoVideoClips, worker_init_fn
from comparison.metrics.action_probe import per_dim_mse, per_dim_r2, aggregate_per_dim

EXTERNAL_DIR = Path("external_models")
VJEPA2_REPO = EXTERNAL_DIR / "vjepa2"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _ensure_vjepa2_on_path():
    p = str(VJEPA2_REPO)
    if p not in sys.path:
        sys.path.insert(0, p)


def preprocess(clip: torch.Tensor, img_size: int = 256) -> torch.Tensor:
    B, T, C, H, W = clip.shape
    if C > 3:
        clip = clip[:, :, :3]; C = 3
    clip = clip.reshape(B * T, C, H, W)
    clip = F.interpolate(clip, size=(img_size, img_size), mode="bilinear", align_corners=False)
    mean = torch.tensor(_IMAGENET_MEAN, device=clip.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=clip.device).view(1, 3, 1, 1)
    clip = (clip - mean) / std
    clip = clip.reshape(B, T, C, img_size, img_size)
    return clip.permute(0, 2, 1, 3, 4).contiguous()


@torch.no_grad()
def encode_per_layer(encoder, clip, img_size=256, tubelet=2):
    """Returns dict {layer_index: (B, T, D)} per-frame features at every layer."""
    layer_outs = {}
    hooks = []

    def make_hook(i):
        def hook(mod, inp, out):
            layer_outs[i] = out
        return hook
    for i, blk in enumerate(encoder.blocks):
        hooks.append(blk.register_forward_hook(make_hook(i)))

    x = preprocess(clip, img_size)
    _ = encoder(x)
    for h in hooks:
        h.remove()

    B, T = clip.shape[:2]
    H_p = img_size // 16
    T_tubes = T // tubelet
    per_layer = {}
    for i, tok in layer_outs.items():
        # tok: (B, num_patches, D) where num_patches = T_tubes * H_p * H_p
        D = tok.shape[-1]
        tok = tok.reshape(B, T_tubes, H_p * H_p, D).mean(dim=2)         # (B, T_tubes, D)
        tok = tok.repeat_interleave(tubelet, dim=1)[:, :T]               # (B, T, D)
        per_layer[i] = tok.float().cpu()
    return per_layer


def encode_split(encoder, split_path, n_clips, clip_len, seed, device, img_size=256):
    ds = LiberoVideoClips(split_path=split_path, datasets_dir="datasets",
                          clip_len=clip_len, cameras=["agentview_rgb"],
                          resolution=128, seed=seed, return_actions=True)
    sub = Subset(ds, list(range(min(n_clips, len(ds)))))
    loader = DataLoader(sub, batch_size=4, shuffle=False, num_workers=2,
                          worker_init_fn=worker_init_fn)
    per_layer_chunks = {}
    actions_all = []
    for clip, actions in loader:
        clip = clip.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pl = encode_per_layer(encoder, clip, img_size=img_size)
        for i, t in pl.items():
            per_layer_chunks.setdefault(i, []).append(t)
        actions_all.append(actions)
    per_layer = {i: torch.cat(chunks, dim=0) for i, chunks in per_layer_chunks.items()}
    actions_all = torch.cat(actions_all, dim=0)
    return per_layer, actions_all


class MLPProbe(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 128)):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.GELU()]
            d = h
        layers.append(torch.nn.Linear(d, out_dim))
        self.net = torch.nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


def train_probe(in_dim, out_dim, train_x, train_y, ev_x, ev_y, steps=3000, lr=3e-4, device="cuda"):
    probe = MLPProbe(in_dim, out_dim).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    N = train_x.shape[0]
    for step in range(steps):
        idx = torch.randint(0, N, (512,))
        x = train_x[idx]; y = train_y[idx]
        pred = probe(x)
        loss = (pred - y).pow(2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        pred = probe(ev_x)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/comparison/vjepa2_aux_finetune/ckpt_last.pt")
    ap.add_argument("--out-jsonl", default="runs/full_results/vjepa_per_layer.jsonl")
    ap.add_argument("--ckpt-key", default="auto",
                    help="auto = prefer 'encoder' for aux ckpts, 'target_encoder' for original V-JEPA ckpts")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--clip-len", type=int, default=32)
    ap.add_argument("--probe-train-clips", type=int, default=400)
    ap.add_argument("--probe-eval-clips", type=int, default=200)
    ap.add_argument("--probe-steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    args = ap.parse_args()

    device = torch.device("cuda")
    _ensure_vjepa2_on_path()
    from src.models import vision_transformer as vit_encoder
    encoder = vit_encoder.vit_large(
        patch_size=16, img_size=(args.img_size, args.img_size),
        num_frames=args.clip_len, tubelet_size=2,
        use_sdpa=True, use_silu=False, wide_silu=True,
        uniform_power=False, use_rope=True,
    )
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if args.ckpt_key == "auto":
        # Aux ckpts have no 'target_encoder'; original V-JEPA ckpts have both
        # and we prefer the EMA target_encoder for inference (matches the
        # main extractor's behavior).
        key = "target_encoder" if "target_encoder" in ck else "encoder"
    else:
        key = args.ckpt_key
    print(f"loading state from ckpt key={key!r}")
    sd = ck.get(key, ck)
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
    msg = encoder.load_state_dict(sd, strict=False)
    print(f"loaded ckpt: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    encoder = encoder.to(device).eval()
    for p in encoder.parameters(): p.requires_grad = False
    n_layers = len(encoder.blocks)
    print(f"encoder has {n_layers} blocks, embed_dim={encoder.embed_dim}")

    print("encoding probe-train split...")
    t0 = time.time()
    pl_train, act_train = encode_split(encoder, "data/splits_taskheld/train.json",
                                         args.probe_train_clips, args.clip_len, seed=11, device=device)
    print(f"  train: {time.time()-t0:.1f}s, n={act_train.shape[0]}")
    t0 = time.time()
    pl_test, act_test = encode_split(encoder, "data/splits_taskheld/test_ood.json",
                                       args.probe_eval_clips, args.clip_len, seed=13, device=device)
    print(f"  test: {time.time()-t0:.1f}s, n={act_test.shape[0]}")

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i in range(n_layers):
            tr_x = pl_train[i].reshape(-1, pl_train[i].shape[-1]).to(device)
            tr_y = act_train[..., :7].reshape(-1, 7).to(device)
            te_x = pl_test[i].reshape(-1, pl_test[i].shape[-1]).to(device)
            te_y = act_test[..., :7].reshape(-1, 7).to(device)
            seed_r2 = []
            for s in args.seeds:
                torch.manual_seed(s)
                pred = train_probe(tr_x.shape[-1], 7, tr_x, tr_y, te_x, te_y,
                                     steps=args.probe_steps, device=device)
                pdr = per_dim_r2(pred, te_y)
                seed_r2.append(aggregate_per_dim(pdr))
            mean = sum(seed_r2) / len(seed_r2)
            print(f"  layer {i:2d}: R²={mean:.4f}  (seeds={seed_r2})")
            f.write(json.dumps({"layer": i, "r2_test_mean": mean, "r2_seeds": seed_r2}) + "\n")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
