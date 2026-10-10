"""LIBERO probe data-efficiency sweep on Web-DINO + SigLIP 2 (frozen + aux).

Encodes 3200 train + 200 test_ood clips once per (backbone, condition),
then trains a fresh MLP probe over probe-train sizes {400, 800, 1600, 3200},
3 seeds each. Outputs R² vs probe-train size curves.

Hypothesis under test: image-SSL backbones' low R² on LIBERO probe (~0.16
at probe-train=400) is partly a probe-budget artifact; CALVIN's larger
training pool let them reach R²~0.80.

Usage:
    CUDA_VISIBLE_DEVICES=2 uv run python scripts/probe_data_efficiency.py \\
        --out-jsonl runs/full_results/libero_probe_data_efficiency.jsonl
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
from torch.utils.data import DataLoader, Subset

from data.libero_video import LiberoVideoClips, worker_init_fn
from comparison.eval_pipeline import build_extractor
from comparison.metrics.action_probe import per_dim_r2, aggregate_per_dim


class MLPProbe(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 128)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.GELU()]
            d = h
        layers.append(torch.nn.Linear(d, out_dim))
        self.net = torch.nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


def train_probe(in_dim, out_dim, tr_x, tr_y, te_x, te_y,
                  steps=3000, lr=3e-4, batch=512, device="cuda"):
    probe = MLPProbe(in_dim, out_dim).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    N = tr_x.shape[0]
    for s in range(steps):
        idx = torch.randint(0, N, (min(batch, N),))
        x = tr_x[idx]; y = tr_y[idx]
        loss = (probe(x) - y).pow(2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        pred = probe(te_x)
    return pred


@torch.no_grad()
def encode_n_clips(extractor, n_clips, split_path, clip_len, seed, device,
                     batch_size=4):
    ds = LiberoVideoClips(split_path=split_path, datasets_dir="datasets",
                          clip_len=clip_len, cameras=["agentview_rgb"],
                          resolution=128, seed=seed, return_actions=True)
    n_clips = min(n_clips, len(ds))
    sub = Subset(ds, list(range(n_clips)))
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False,
                          num_workers=2, worker_init_fn=worker_init_fn)
    feats_chunks, act_chunks = [], []
    for clip, actions in loader:
        clip = clip.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            f = extractor.extract(clip)
        feats_chunks.append(f.float().cpu())
        act_chunks.append(actions)
    return torch.cat(feats_chunks, dim=0), torch.cat(act_chunks, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-jsonl", default="runs/full_results/libero_probe_data_efficiency.jsonl")
    ap.add_argument("--sizes", type=int, nargs="+", default=[400, 800, 1600, 3200])
    ap.add_argument("--max-train", type=int, default=3200)
    ap.add_argument("--probe-eval-clips", type=int, default=200)
    ap.add_argument("--clip-len", type=int, default=32)
    ap.add_argument("--probe-steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    args = ap.parse_args()

    device = torch.device("cuda")
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    runs = [
        ("webdino", "webdino_frozen", None, "facebook/webssl-dino300m-full2b-224"),
        ("webdino", "webdino_aux",    "runs/comparison/webdino_aux_finetune/ckpt_last.pt",
                                       "facebook/webssl-dino300m-full2b-224"),
        ("siglip2", "siglip2_frozen", None, "google/siglip2-large-patch16-256"),
        ("siglip2", "siglip2_aux",    "runs/comparison/siglip2_aux_finetune/ckpt_last.pt",
                                       "google/siglip2-large-patch16-256"),
    ]

    out_f = out_path.open("w")
    for family, variant, ckpt, hf_repo in runs:
        spec = {"family": family, "name": variant, "hf_repo": hf_repo}
        if ckpt: spec["ckpt"] = ckpt
        print(f"\n=== {variant} ===")
        t0 = time.time()
        extractor = build_extractor(spec, device)
        print(f"  build: {time.time()-t0:.1f}s")

        t0 = time.time()
        tr_f, tr_a = encode_n_clips(extractor, args.max_train,
                                      "data/splits_taskheld/train.json",
                                      args.clip_len, seed=11, device=device)
        print(f"  encode train ({tr_f.shape[0]} clips): {time.time()-t0:.1f}s")
        t0 = time.time()
        te_f, te_a = encode_n_clips(extractor, args.probe_eval_clips,
                                      "data/splits_taskheld/test_ood.json",
                                      args.clip_len, seed=13, device=device)
        print(f"  encode test  ({te_f.shape[0]} clips): {time.time()-t0:.1f}s")

        del extractor
        torch.cuda.empty_cache()

        D = tr_f.shape[-1]
        te_x = te_f.reshape(-1, D).to(device)
        te_y = te_a[..., :7].reshape(-1, 7).to(device)

        for size in args.sizes:
            n = min(size, tr_f.shape[0])
            tr_sub_x = tr_f[:n].reshape(-1, D).to(device)
            tr_sub_y = tr_a[:n, ..., :7].reshape(-1, 7).to(device)
            seed_r2 = []
            for s in args.seeds:
                torch.manual_seed(s)
                pred = train_probe(D, 7, tr_sub_x, tr_sub_y, te_x, te_y,
                                     steps=args.probe_steps, device=device)
                seed_r2.append(aggregate_per_dim(per_dim_r2(pred, te_y)))
            mean = sum(seed_r2)/len(seed_r2)
            print(f"  size={size:5d} -> R²={mean:.4f}  seeds={[f'{r:.4f}' for r in seed_r2]}")
            out_f.write(json.dumps({
                "variant": variant, "probe_train_size": size,
                "r2_test_mean": mean, "r2_test_seeds": seed_r2,
            }) + "\n")
            out_f.flush()
    out_f.close()
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
