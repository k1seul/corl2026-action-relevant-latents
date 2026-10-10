"""Quick single-ckpt probe for the ID sparsity sweep.

Encodes LIBERO task-OOD probe-train + test_ood with the requested backbone +
ckpt, trains a 3000-step MLP probe (3 seeds), prints aggregate + per-dim R^2,
and appends to summary.jsonl.
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


@torch.no_grad()
def encode(extractor, split_path, n_clips, clip_len, device, bs=4):
    ds = LiberoVideoClips(split_path=split_path, datasets_dir="datasets",
                          clip_len=clip_len, cameras=["agentview_rgb"],
                          resolution=128, seed=0, return_actions=True)
    sub = Subset(ds, list(range(min(n_clips, len(ds)))))
    loader = DataLoader(sub, batch_size=bs, shuffle=False, num_workers=2,
                          worker_init_fn=worker_init_fn)
    feats_chunks, acts_chunks = [], []
    for clip, acts in loader:
        clip = clip.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            f = extractor.extract(clip)
        feats_chunks.append(f.float().cpu()); acts_chunks.append(acts)
    return torch.cat(feats_chunks, 0), torch.cat(acts_chunks, 0)


def train_probe_3seed(D, tr_x, tr_y, te_x, te_y, device, steps=3000, seeds=(11,12,13)):
    r2s, per_dim_acc = [], []
    for s in seeds:
        torch.manual_seed(s)
        probe = MLPProbe(D, 7).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=3e-4, weight_decay=1e-4)
        for _ in range(steps):
            idx = torch.randint(0, tr_x.shape[0], (512,))
            loss = (probe(tr_x[idx]) - tr_y[idx]).pow(2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            pred = probe(te_x)
        pdr = per_dim_r2(pred, te_y)
        r2s.append(aggregate_per_dim(pdr))
        per_dim_acc.append(pdr)
    mean_r2 = sum(r2s) / len(r2s)
    # mean per-dim across seeds
    keys = per_dim_acc[0].keys()
    mean_per_dim = {k: float(sum(d[k] for d in per_dim_acc)/len(per_dim_acc)) for k in keys}
    return mean_r2, mean_per_dim, r2s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=("vjepa2", "videomae"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--p", type=float, required=True,
                    help="id-mask-p value (for summary.jsonl row)")
    ap.add_argument("--variant", default=None, help="variant name, e.g., vjepa2_aux_p0p10")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--clip-len", type=int, default=32)
    ap.add_argument("--summary-path", default="runs/id_sparsity_sweep/summary.jsonl")
    args = ap.parse_args()

    device = torch.device("cuda")
    if args.family == "vjepa2":
        spec = {"family": "vjepa2", "name": "vjepa2_sweep", "ckpt": args.ckpt}
    elif args.family == "videomae":
        spec = {"family": "videomae", "name": "videomae_sweep",
                 "hf_repo": "MCG-NJU/videomae-large", "ckpt": args.ckpt}
    extractor = build_extractor(spec, device)
    print(f"extractor: {extractor.meta.family} feat_dim={extractor.meta.feat_dim}")

    t0 = time.time()
    tr_f, tr_a = encode(extractor, "data/splits_taskheld/train.json",
                         args.n_train, args.clip_len, device)
    te_f, te_a = encode(extractor, "data/splits_taskheld/test_ood.json",
                         args.n_test, args.clip_len, device)
    print(f"encoded in {time.time()-t0:.1f}s; train={tr_f.shape}, test={te_f.shape}")

    del extractor
    torch.cuda.empty_cache()

    D = tr_f.shape[-1]
    tr_x = tr_f.reshape(-1, D).to(device)
    tr_y = tr_a[..., :7].reshape(-1, 7).to(device)
    te_x = te_f.reshape(-1, D).to(device)
    te_y = te_a[..., :7].reshape(-1, 7).to(device)

    mean_r2, mean_per_dim, seed_r2 = train_probe_3seed(D, tr_x, tr_y, te_x, te_y, device)
    print(f"\n{args.family} p={args.p}: test R²={mean_r2:.4f}  seeds={[f'{r:.3f}' for r in seed_r2]}")
    for k, v in mean_per_dim.items():
        print(f"  {k}: {v:.4f}")

    summary = Path(args.summary_path)
    summary.parent.mkdir(parents=True, exist_ok=True)
    variant = args.variant or f"{args.family}_aux_p{str(args.p).replace('.', 'p')}"
    row = {
        "variant": variant, "backbone": args.family, "p": args.p,
        "test_r2_aggregate": mean_r2,
        "test_r2_per_dim": mean_per_dim,
        "test_r2_seeds": seed_r2,
        "n_probe_seeds": 3, "fine_tune_steps": 20000,
        "ckpt": args.ckpt,
    }
    with summary.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"appended to {summary}")


if __name__ == "__main__":
    main()
