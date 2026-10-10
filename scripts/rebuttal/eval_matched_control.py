"""Canonical action-probe evaluation for a matched-control cell.

Given one wave1 checkpoint (``<backbone>/<condition>/<run_id>/ckpt_last.pt``)
this script:

  1. Loads the corresponding frozen backbone extractor, overriding its state
     dict with the fine-tuned encoder weights from the ckpt.
  2. Encodes the LIBERO train and test_ood splits (episode-slice contiguous)
     into per-frame ``z_t`` features.
  3. Trains a canonical MLP probe ``[D -> 256 -> 128 -> action_dim]`` for
     3000 steps against ``z_t -> a_t``, for each probe seed in {0,1,2}.
  4. Writes one JSONL row per (backbone, condition, training_seed,
     probe_seed) with ``eval_mode = canonical_offline_action_probe``.

Row schema conforms to ``src/rebuttal/provenance.py:REQUIRED_RESULT_ROW_FIELDS``
so ``regenerate_matched_controls_table.py`` picks it up automatically.

Usage:
    python scripts/rebuttal/eval_matched_control.py \\
        --backbone videomae --condition true_id --training-seed 0 \\
        --ckpt runs/corl2026_rebuttal/checkpoints/videomae/true_id/wave1_true_id_videomae_s0/ckpt_last.pt \\
        --out-jsonl runs/corl2026_rebuttal/results/matched_controls.jsonl

Pass ``--backbone <name> --condition frozen`` (with no --ckpt) to evaluate
the vanilla frozen backbone (no fine-tuning) as the frozen matched-control
cell.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


# ── extractor factory ─────────────────────────────────────────────────────


class _SimpleMeta:
    """Duck-typed ExtractorMeta for Phase-3 baselines that don't ship one."""
    def __init__(self, family, variant, feat_dim, n_params, native_resolution=224,
                 native_clip_len=None):
        self.family = family; self.variant = variant
        self.feat_dim = feat_dim; self.n_params = n_params
        self.native_resolution = native_resolution
        self.native_clip_len = native_clip_len
        self.has_native_recon = False


class R3MExtractor:
    """R3M ResNet-50 frozen features (Nair et al., 2022; Phase 3 baseline)."""
    def __init__(self, device, model_size="resnet50"):
        import r3m
        m = r3m.load_r3m(model_size)
        # r3m loads on cuda:0 by default; move to requested device
        m = m.to(device).eval()
        for p in m.parameters(): p.requires_grad = False
        self._model = m; self._device = device
        feat_dim = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}[model_size]
        self.meta = _SimpleMeta("r3m", model_size, feat_dim,
                                sum(p.numel() for p in m.parameters()),
                                native_resolution=224)

    @torch.no_grad()
    def extract(self, clip, task_caption=None):
        clip = clip.to(self._device)
        B, T, C, H, W = clip.shape
        if C > 3: clip = clip[:, :, :3]; C = 3
        # R3M expects [0, 255] float
        x = (clip * 255.0).reshape(B * T, C, H, W)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        feats = self._model(x)  # (B*T, D)
        return feats.reshape(B, T, -1).float()


class CAIPVisualOnlyExtractor:
    """CAIP visual-only: mean-pool over 256 patch tokens (no text conditioning).

    Uses `yuvansharma/caip-vitl256` (pinned by revision at load time).
    """
    def __init__(self, device, ckpt_dir):
        from transformers import AutoModel, AutoProcessor
        self._device = device
        self._model = AutoModel.from_pretrained(str(ckpt_dir), trust_remote_code=True).to(device).eval()
        for p in self._model.parameters(): p.requires_grad = False
        self._processor = AutoProcessor.from_pretrained(str(ckpt_dir), trust_remote_code=True)
        # native res per config = 256, mean/std imagenet-CLIP
        self._image_size = 256
        self._mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        self._std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
        self.meta = _SimpleMeta("caip_visual_only", "caip_vitl256_visual",
                                 feat_dim=1024,
                                 n_params=sum(p.numel() for p in self._model.parameters()),
                                 native_resolution=self._image_size)

    @torch.no_grad()
    def extract(self, clip, task_caption=None):
        # clip: (B, T, C, H, W) [0,1] agentview_rgb 128×128
        clip = clip.to(self._device)
        B, T, C, H, W = clip.shape
        if C > 3: clip = clip[:, :, :3]; C = 3
        x = clip.reshape(B * T, C, H, W)
        x = torch.nn.functional.interpolate(x, size=(self._image_size, self._image_size),
                                             mode="bicubic", align_corners=False)
        x = (x - self._mean) / self._std
        # CAIP forward without text: pass empty input_ids so the vision path runs
        # patch_features has shape (B, 256, 1024); mean-pool over patches
        try:
            out = self._model(pixel_values=x)
        except Exception:
            # Some forward paths require text; pass a 1-token PAD prompt
            pad_ids = torch.zeros((x.shape[0], 1), dtype=torch.long, device=self._device)
            out = self._model(pixel_values=x, input_ids=pad_ids)
        pf = out.patch_features                         # (B*T, 256, 1024)
        pooled = pf.mean(dim=1)                         # (B*T, 1024)
        return pooled.reshape(B, T, -1).float()


class CAIPNativeExtractor:
    """CAIP native task-conditioned: image_pooled given a task text prompt.

    The prompt is derived from the LIBERO task filename by the outer caller
    (task_caption). Reported SEPARATELY from CAIP visual-only — never in
    the same unqualified ranking.
    """
    def __init__(self, device, ckpt_dir):
        from transformers import AutoModel, AutoProcessor
        self._device = device
        self._model = AutoModel.from_pretrained(str(ckpt_dir), trust_remote_code=True).to(device).eval()
        for p in self._model.parameters(): p.requires_grad = False
        self._processor = AutoProcessor.from_pretrained(str(ckpt_dir), trust_remote_code=True)
        self._image_size = 256
        self._mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        self._std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
        self.meta = _SimpleMeta("caip_native_conditioned", "caip_vitl256_native",
                                 feat_dim=1024,
                                 n_params=sum(p.numel() for p in self._model.parameters()),
                                 native_resolution=self._image_size)

    @torch.no_grad()
    def extract(self, clip, task_caption=None):
        """task_caption: str (same caption for all B) OR List[str] length B
        (one caption per sample in the batch)."""
        assert task_caption, "CAIP native mode requires task_caption"
        clip = clip.to(self._device)
        B, T, C, H, W = clip.shape
        if C > 3: clip = clip[:, :, :3]; C = 3
        x = clip.reshape(B * T, C, H, W)
        x = torch.nn.functional.interpolate(x, size=(self._image_size, self._image_size),
                                             mode="bicubic", align_corners=False)
        x = (x - self._mean) / self._std
        if isinstance(task_caption, str):
            captions = [task_caption] * B
        else:
            captions = list(task_caption)
            assert len(captions) == B, f"expected {B} captions, got {len(captions)}"
        toks = self._processor.tokenizer(captions, return_tensors="pt",
                                          padding="max_length", truncation=True,
                                          max_length=self._model.config.context_length).to(self._device)
        # (B, L) → repeat_interleave to (B*T, L) so per-sample caption tiles across frames
        input_ids = toks["input_ids"].repeat_interleave(T, dim=0).contiguous()
        out = self._model(pixel_values=x, input_ids=input_ids)
        pooled = out.image_pooled                        # (B*T, 1024) — text-conditioned
        return pooled.reshape(B, T, -1).float()


class VIPExtractor:
    """VIP ResNet-50 goal-conditioned value features (Ma et al., 2023; Phase 3)."""
    def __init__(self, device):
        import vip
        m = vip.load_vip()
        m = m.to(device).eval()
        for p in m.parameters(): p.requires_grad = False
        self._model = m; self._device = device
        self.meta = _SimpleMeta("vip", "resnet50", 1024,
                                sum(p.numel() for p in m.parameters()),
                                native_resolution=224)

    @torch.no_grad()
    def extract(self, clip, task_caption=None):
        clip = clip.to(self._device)
        B, T, C, H, W = clip.shape
        if C > 3: clip = clip[:, :, :3]; C = 3
        x = (clip * 255.0).reshape(B * T, C, H, W)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        feats = self._model(x)  # (B*T, D)
        return feats.reshape(B, T, -1).float()


def build_extractor(backbone: str, device: torch.device,
                     finetuned_encoder_state: Optional[dict],
                     external_models_dir: Path,
                     clip_len: int = 32):
    """Build one of the matched-control (vjepa2/videomae/webdino) or
    Phase-3 (r3m/vip) backbone extractors, with an optional overridden
    encoder state dict for fine-tuned matched-control cells."""
    if backbone == "r3m":
        return R3MExtractor(device)
    if backbone == "vip":
        return VIPExtractor(device)
    if backbone == "caip_visual_only":
        return CAIPVisualOnlyExtractor(device, external_models_dir / "caip")
    if backbone == "caip_native_conditioned":
        return CAIPNativeExtractor(device, external_models_dir / "caip")
    if backbone == "vjepa2":
        # V-JEPA2 extractor loads its own ckpt from disk. Easiest is: write the
        # fine-tuned encoder state to a tiny temp file with the same key layout.
        from comparison.extractors.vjepa2_extractor import VJEPA2Extractor
        if finetuned_encoder_state is not None:
            import tempfile
            tmpdir = tempfile.mkdtemp(prefix="corl26_vjepa2_ft_")
            tmp_ckpt = Path(tmpdir) / "encoder_override.pt"
            torch.save({"encoder": finetuned_encoder_state,
                         "target_encoder": finetuned_encoder_state}, tmp_ckpt)
            return VJEPA2Extractor(device, ckpt_path=str(tmp_ckpt),
                                     num_frames=clip_len)
        return VJEPA2Extractor(device, num_frames=clip_len)

    if backbone == "videomae":
        from comparison.extractors.videomae_extractor import VideoMAEExtractor
        # HF snapshot is at $EXTERNAL_MODELS_DIR/videomae
        return VideoMAEExtractor(
            device,
            ckpt_path=str(external_models_dir / "videomae"),
            finetuned_state_dict=finetuned_encoder_state,
        )

    if backbone == "webdino":
        from comparison.extractors.webdino_extractor import WebDINOExtractor
        return WebDINOExtractor(
            device,
            ckpt_path=str(external_models_dir / "webdino"),
            finetuned_state_dict=finetuned_encoder_state,
        )

    raise ValueError(f"unknown backbone: {backbone}")


# ── feature encoding pass ─────────────────────────────────────────────────


@torch.no_grad()
def encode_slice(extractor, split_path: Path, datasets_dir: Path,
                 episode_slice: tuple[int, int], *, clip_len: int, seed: int,
                 batch_size: int, num_workers: int, device: torch.device,
                 resolution: int = 128):
    """Return (feats [K,T,D], acts [K,T,7], episode_ids [K] list of
    (suite, file, demo)). ``episode_ids`` is needed for the task-cluster
    bootstrap (P2)."""
    from data.libero_video import LiberoVideoClips, worker_init_fn

    ds = LiberoVideoClips(
        split_path=str(split_path), datasets_dir=str(datasets_dir),
        clip_len=clip_len, cameras=["agentview_rgb"], resolution=resolution,
        seed=seed, return_actions=True,
    )
    lo, hi = episode_slice
    hi = min(hi, len(ds))
    if lo >= hi:
        raise ValueError(f"empty episode slice {lo}:{hi} for split with {len(ds)} episodes")
    episode_ids = [(ds.episodes[i]["suite"], ds.episodes[i]["file"],
                    ds.episodes[i]["demo"]) for i in range(lo, hi)]
    subset = Subset(ds, list(range(lo, hi)))
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        worker_init_fn=worker_init_fn, drop_last=False, pin_memory=False,
    )
    feats_all: list[torch.Tensor] = []
    acts_all: list[torch.Tensor] = []
    for clip, actions in loader:
        # LiberoVideoClips returns clip (B,T,C,H,W) float32 in [0,1]
        f = extractor.extract(clip.to(device))     # (B, T, D)
        feats_all.append(f.detach().float().cpu())
        acts_all.append(actions[..., :7].float().cpu())
    feats = torch.cat(feats_all, dim=0)            # (K, T, D)
    acts = torch.cat(acts_all, dim=0)              # (K, T, 7)
    return feats, acts, episode_ids


# ── canonical MLP probe (single-frame z_t -> a_t) ─────────────────────────


class CanonicalMLPProbe(nn.Module):
    def __init__(self, in_dim: int, action_dim: int = 7,
                 hidden: tuple[int, ...] = (256, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _flatten_frames(feat_kt: torch.Tensor, act_kt: torch.Tensor):
    K, T, D = feat_kt.shape
    A = act_kt.shape[-1]
    return feat_kt.reshape(-1, D), act_kt.reshape(-1, A)


def train_and_score_probe(train_x: torch.Tensor, train_y: torch.Tensor,
                          test_x: torch.Tensor, test_y: torch.Tensor,
                          *, probe_seed: int, steps: int, batch_size: int,
                          lr: float, weight_decay: float, device: torch.device,
                          feature_normalize: str = "zscore",
                          test_ep_boundaries: list[tuple[int, int]] | None = None,
                          ) -> dict:
    """Train MLP probe with optional feature normalization.

    ``feature_normalize`` in {"none", "zscore", "l2"}. Statistics are computed
    on train_x and applied to both train and test — a standard linear-probe
    protocol that levels feature-scale differences across backbones without
    changing the encoder. Default ``l2`` because it's robust to dead ReLU
    dims (which explode zscore division) while still levelling feature
    magnitudes across backbones.
    """
    torch.manual_seed(probe_seed)
    # ---- feature normalization ------------------------------------------
    if feature_normalize == "zscore":
        mu = train_x.mean(dim=0, keepdim=True)
        # Clamp to 1e-3 (not 1e-6) so dead-ReLU dims don't explode.
        sig = train_x.std(dim=0, keepdim=True).clamp_min(1e-3)
        train_x = (train_x - mu) / sig
        test_x = (test_x - mu) / sig
    elif feature_normalize == "l2":
        train_x = torch.nn.functional.normalize(train_x, dim=-1)
        test_x = torch.nn.functional.normalize(test_x, dim=-1)
    elif feature_normalize == "none":
        pass
    else:
        raise ValueError(f"unknown feature_normalize: {feature_normalize}")

    probe = CanonicalMLPProbe(in_dim=train_x.shape[-1], action_dim=train_y.shape[-1]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    n = train_x.shape[0]
    train_x = train_x.to(device); train_y = train_y.to(device)
    test_x = test_x.to(device);   test_y = test_y.to(device)
    probe.train()
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), device=device)
        pred = probe(train_x[idx])
        loss = F.mse_loss(pred, train_y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        pred_test = probe(test_x)
    diff = (pred_test - test_y)
    per_dim_mse = (diff ** 2).mean(dim=0).cpu().tolist()
    var = test_y.var(dim=0, unbiased=False).cpu().clamp_min(1e-12).tolist()
    per_dim_r2 = [1.0 - m / v for m, v in zip(per_dim_mse, var)]

    out = {
        "aggregate_test_mse": float(sum(per_dim_mse) / len(per_dim_mse)),
        "aggregate_test_r2":  float(sum(per_dim_r2) / len(per_dim_r2)),
        "per_dim_mse": per_dim_mse,
        "per_dim_r2":  per_dim_r2,
    }
    # Per-episode sufficient statistics for task-cluster bootstrap (P2).
    # For episode k with frames [lo, hi): store per-dim SSE (sum of squared
    # residuals) and per-dim target sum + sum of squares + n_frames. These
    # let a bootstrap sampler compute correct pooled SSE / SS_total for any
    # subset of episodes without needing raw predictions.
    if test_ep_boundaries is not None:
        pred_np = pred_test.cpu().numpy()
        y_np = test_y.cpu().numpy()
        per_ep_stats = []
        for k, (lo, hi) in enumerate(test_ep_boundaries):
            n = hi - lo
            if n <= 0:
                continue
            resid = pred_np[lo:hi] - y_np[lo:hi]      # (n, A)
            per_ep_stats.append({
                "episode_index": k,
                "n_frames": int(n),
                "sse_per_dim": (resid ** 2).sum(axis=0).tolist(),
                "y_sum_per_dim": y_np[lo:hi].sum(axis=0).tolist(),
                "y_sqsum_per_dim": (y_np[lo:hi] ** 2).sum(axis=0).tolist(),
            })
        out["per_episode_sufficient_stats"] = per_ep_stats
    return out


DIM_NAMES = ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")


def _sha256_file(p: Path) -> Optional[str]:
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_sha(split_path: Path) -> Optional[str]:
    sha_file = split_path.with_suffix(split_path.suffix + ".sha256")
    if sha_file.exists():
        return sha_file.read_text().split()[0][:64]
    return _sha256_file(split_path)


# ── main ──────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True,
                    choices=("vjepa2", "videomae", "webdino", "r3m", "vip",
                             "caip_visual_only", "caip_native_conditioned"))
    ap.add_argument("--condition", required=True,
                    choices=("frozen", "true_id", "shuffled_id",
                             "shuffled_id_circular_sensitivity",
                             "forward_dynamics", "action_free_adaptation"))
    ap.add_argument("--training-seed", type=int, default=-1,
                    help="the training seed of the ckpt; -1 for the frozen condition")
    ap.add_argument("--ckpt", default=None,
                    help="path to wave1 ckpt_last.pt; omit for --condition frozen")
    ap.add_argument("--probe-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--probe-steps", type=int, default=3000)
    ap.add_argument("--probe-batch-size", type=int, default=256)
    ap.add_argument("--probe-lr", type=float, default=1e-3)
    ap.add_argument("--probe-weight-decay", type=float, default=1e-4)
    ap.add_argument("--probe-train-clips", type=int, default=400)
    ap.add_argument("--probe-eval-clips-train", type=int, default=200)
    ap.add_argument("--probe-eval-clips-test", type=int, default=500)
    ap.add_argument("--probe-feature-normalize", default="l2",
                    choices=("none", "zscore", "l2"),
                    help="feature-scale normalisation applied before probe "
                         "training. Statistics are computed on probe-train and "
                         "applied to both train and test. Default l2 (robust "
                         "to dead-ReLU dims that explode zscore).")
    ap.add_argument("--clip-len", type=int, default=32)
    ap.add_argument("--encode-batch-size", type=int, default=8)
    ap.add_argument("--encode-num-workers", type=int, default=2)
    ap.add_argument("--split-train", default=str(REPO_ROOT / "runs/corl2026_rebuttal/splits/train.json"))
    ap.add_argument("--split-test-ood", default=str(REPO_ROOT / "runs/corl2026_rebuttal/splits/test_ood.json"))
    ap.add_argument("--datasets-dir", default=str(REPO_ROOT / "datasets/libero_downloaded"))
    ap.add_argument("--external-models-dir",
                    default=os.environ.get("EXTERNAL_MODELS_DIR",
                                             str(REPO_ROOT / "datasets/external_models")))
    ap.add_argument("--out-jsonl", default=str(
        REPO_ROOT / "runs/corl2026_rebuttal/results/matched_controls.jsonl"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)

    # ── load fine-tuned encoder state if applicable ───────────────────
    finetuned_state = None
    ckpt_sha = None
    ckpt_step = None
    if args.ckpt:
        ckpt_path = Path(args.ckpt).resolve()
        if not ckpt_path.exists():
            sys.stderr.write(f"missing ckpt: {ckpt_path}\n"); return 2
        sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        finetuned_state = sd.get("encoder")
        if finetuned_state is None:
            sys.stderr.write(f"ckpt {ckpt_path} has no 'encoder' key; keys={list(sd)}\n"); return 3
        ckpt_step = int(sd.get("step", 0))
        ckpt_sha = _sha256_file(ckpt_path)
    else:
        if args.condition != "frozen":
            sys.stderr.write(f"non-frozen condition {args.condition!r} requires --ckpt\n"); return 4

    print(f"[eval_mc] backbone={args.backbone} condition={args.condition} "
          f"training_seed={args.training_seed} probe_seeds={args.probe_seeds}")
    if ckpt_sha:
        print(f"[eval_mc] ckpt={args.ckpt} sha256={ckpt_sha[:16]} step={ckpt_step}")

    # ── build extractor ───────────────────────────────────────────────
    t0 = time.time()
    extractor = build_extractor(
        args.backbone, device, finetuned_state,
        external_models_dir=Path(args.external_models_dir),
        clip_len=args.clip_len,
    )
    print(f"[eval_mc] extractor built in {time.time()-t0:.1f}s "
          f"family={extractor.meta.family} feat_dim={extractor.meta.feat_dim} "
          f"n_params={extractor.meta.n_params/1e6:.1f}M")

    # ── encode probe-train, probe-eval-train, probe-eval-test slices ──
    split_train = Path(args.split_train); split_test_ood = Path(args.split_test_ood)
    datasets_dir = Path(args.datasets_dir)

    t_enc = time.time()
    train_feats, train_acts, _ = encode_slice(
        extractor, split_train, datasets_dir,
        (0, args.probe_train_clips),
        clip_len=args.clip_len, seed=0,
        batch_size=args.encode_batch_size, num_workers=args.encode_num_workers,
        device=device,
    )
    print(f"[eval_mc] probe-train encoded: {tuple(train_feats.shape)} in {time.time()-t_enc:.1f}s")

    t_enc = time.time()
    train_eval_feats, train_eval_acts, train_eval_ep_ids = encode_slice(
        extractor, split_train, datasets_dir,
        (args.probe_train_clips, args.probe_train_clips + args.probe_eval_clips_train),
        clip_len=args.clip_len, seed=0,
        batch_size=args.encode_batch_size, num_workers=args.encode_num_workers,
        device=device,
    )
    print(f"[eval_mc] probe-eval-train encoded: {tuple(train_eval_feats.shape)} in {time.time()-t_enc:.1f}s")

    t_enc = time.time()
    test_feats, test_acts, test_ep_ids = encode_slice(
        extractor, split_test_ood, datasets_dir,
        (0, args.probe_eval_clips_test),
        clip_len=args.clip_len, seed=0,
        batch_size=args.encode_batch_size, num_workers=args.encode_num_workers,
        device=device,
    )
    print(f"[eval_mc] probe-eval-test  encoded: {tuple(test_feats.shape)} in {time.time()-t_enc:.1f}s "
          f"episodes={len(test_ep_ids)}")

    tx, ty = _flatten_frames(train_feats, train_acts)
    txe_tr, tye_tr = _flatten_frames(train_eval_feats, train_eval_acts)
    txe_te, tye_te = _flatten_frames(test_feats, test_acts)
    # Episode boundaries in the flattened test tensor: episode k spans
    # frames [k*T, (k+1)*T). Stored as (lo, hi) pairs.
    T = int(test_feats.shape[1])
    test_ep_boundaries = [(k * T, (k + 1) * T) for k in range(test_feats.shape[0])]

    print(f"[eval_mc] flat pairs: train={tx.shape[0]}, eval_train={txe_tr.shape[0]}, eval_test={txe_te.shape[0]}")

    # ── loop probe seeds ──────────────────────────────────────────────
    out_lines: list[dict] = []
    for probe_seed in args.probe_seeds:
        t_probe = time.time()
        clean_test = train_and_score_probe(
            tx, ty, txe_te, tye_te,
            probe_seed=probe_seed, steps=args.probe_steps,
            batch_size=args.probe_batch_size, lr=args.probe_lr,
            weight_decay=args.probe_weight_decay, device=device,
            feature_normalize=args.probe_feature_normalize,
            test_ep_boundaries=test_ep_boundaries,
        )
        clean_train = train_and_score_probe(
            tx, ty, txe_tr, tye_tr,
            probe_seed=probe_seed, steps=args.probe_steps,
            batch_size=args.probe_batch_size, lr=args.probe_lr,
            weight_decay=args.probe_weight_decay, device=device,
        )

        row = {
            "eval_mode": "canonical_offline_action_probe",
            "benchmark": "libero",
            "backbone": args.backbone,
            "condition": args.condition,
            "training_seed": args.training_seed,
            "probe_seed": probe_seed,
            "perturbation": "clean",
            "split_sha256": _split_sha(split_test_ood),
            # scientific values
            "clean_task_ood_r2": clean_test["aggregate_test_r2"],
            "clean_task_ood_mse": clean_test["aggregate_test_mse"],
            "per_dim_r2": dict(zip(DIM_NAMES, clean_test["per_dim_r2"])),
            "per_dim_mse": dict(zip(DIM_NAMES, clean_test["per_dim_mse"])),
            "clean_task_indist_r2": clean_train["aggregate_test_r2"],
            "clean_task_indist_mse": clean_train["aggregate_test_mse"],
            "per_dim_r2_train": dict(zip(DIM_NAMES, clean_train["per_dim_r2"])),
            # counts
            "n_probe_train_pairs": int(tx.shape[0]),
            "n_probe_eval_train_pairs": int(txe_tr.shape[0]),
            "n_probe_eval_test_pairs": int(txe_te.shape[0]),
            # provenance
            "extractor_meta": {
                "family": extractor.meta.family,
                "variant": extractor.meta.variant,
                "feat_dim": extractor.meta.feat_dim,
                "n_params": extractor.meta.n_params,
                "native_resolution": extractor.meta.native_resolution,
                "native_clip_len": extractor.meta.native_clip_len,
            },
            "finetuned_ckpt_path": args.ckpt,
            "finetuned_ckpt_sha256": ckpt_sha,
            "finetuned_ckpt_step": ckpt_step,
            "split_train_path": str(split_train),
            "split_test_ood_path": str(split_test_ood),
            "probe_steps": args.probe_steps,
            "probe_hidden": [256, 128],
            "probe_feature_normalize": args.probe_feature_normalize,
            # Per-episode sufficient statistics for cluster bootstrap
            "test_episode_ids": [
                {"suite": s, "file": f, "demo": d}
                for (s, f, d) in test_ep_ids
            ],
            "per_episode_sufficient_stats": clean_test.get("per_episode_sufficient_stats"),
            "wall_clock_probe_s": round(time.time() - t_probe, 2),
        }
        out_lines.append(row)
        print(f"[eval_mc] probe_seed={probe_seed}: "
              f"clean_task_ood_r2={row['clean_task_ood_r2']:.4f}  "
              f"per_dim R² gripper={row['per_dim_r2']['gripper']:.3f}  "
              f"dz={row['per_dim_r2']['dz']:.3f}  drx={row['per_dim_r2']['drx']:.3f}  "
              f"({row['wall_clock_probe_s']}s)")

    # ── append to JSONL ───────────────────────────────────────────────
    out_path = Path(args.out_jsonl); out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        for row in out_lines:
            f.write(json.dumps(row) + "\n")
    print(f"[eval_mc] appended {len(out_lines)} row(s) to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
