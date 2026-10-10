"""Train the block-causal video transformer with masked patch reconstruction.

Run:
    uv run python scripts/train.py --config configs/baseline.yaml
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.libero_video import LiberoVideoClips, worker_init_fn
from models.block_causal_transformer import BlockCausalVideoTransformer
from models.token_video_transformer import TokenVideoTransformer
from models.tokenizer import FSQTokenizer
from training.aux_loss import (
    InvDynAuxHead, ForwardDynAuxHead, ContrastiveAuxHead,
    SingleFrameAuxHead, MultiStepInvDynHead, compute_aux_loss,
)
from training.diffusion_objective import make_schedule, q_sample
from training.ema import EMA
from training.flow_matching import flow_matching_loss
from training.mask_objective import make_mask, masked_pixel_loss, unpatchify


def lr_schedule(step: int, warmup: int, max_steps: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_model(cfg, in_channels: int):
    objective = cfg.train.get("objective", "mae")
    if objective == "token_ce":
        return None  # token model needs codebook size from a loaded tokenizer
    return BlockCausalVideoTransformer(
        in_channels=in_channels,
        patch_size=cfg.model.patch_size,
        num_frames=cfg.data.clip_len,
        image_size=cfg.data.resolution,
        dim=cfg.model.dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=cfg.model.dropout,
        enable_timestep=(objective in ("diffusion", "flow_matching")),
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
    for p in tok.parameters():
        p.requires_grad = False
    return tok


def build_token_model(cfg, codebook_size: int, num_tokens_per_frame: int) -> TokenVideoTransformer:
    return TokenVideoTransformer(
        codebook_size=codebook_size,
        num_tokens_per_frame=num_tokens_per_frame,
        num_frames=cfg.data.clip_len,
        dim=cfg.model.dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=cfg.model.dropout,
        action_dim=int(cfg.model.get("action_dim", 0)),
    )


@torch.no_grad()
def encode_clip_to_tokens(tok: FSQTokenizer, clip: torch.Tensor) -> torch.Tensor:
    """`(B, T, C, H, W) -> (B, T, N=h*w)` token ids."""
    B, T, C, H, W = clip.shape
    flat = clip.reshape(B * T, C, H, W)
    _, idx_grid = tok.encode(flat)  # idx_grid: (B*T, h, w)
    return idx_grid.reshape(B, T, -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/baseline.yaml")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args, overrides = ap.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    torch.manual_seed(cfg.train.seed)
    device = torch.device(args.device)

    ckpt_dir = Path(cfg.train.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, ckpt_dir / "config.yaml")
    writer = SummaryWriter(log_dir=str(ckpt_dir / "tb"))

    in_channels = 3 * len(cfg.data.cameras)
    action_dim = int(cfg.model.get("action_dim", 0))
    aux_lambda = float(cfg.train.get("aux_invdyn_lambda", 0.0))
    aux_action_dim = int(cfg.train.get("aux_action_dim", 7))
    aux_head_type = str(cfg.train.get("aux_head_type", "invdyn_l2"))
    use_actions = action_dim > 0 or aux_lambda > 0
    loader_kind = str(cfg.data.get("loader", "libero"))
    if loader_kind == "robomimic":
        from comparison.datasets.robomimic import RobomimicVideoClips
        train_ds = RobomimicVideoClips(
            split_path=cfg.data.train_split,
            datasets_dir=cfg.data.datasets_dir,
            clip_len=cfg.data.clip_len,
            cameras=cfg.data.cameras,
            resolution=cfg.data.resolution,
            seed=cfg.train.seed,
            return_actions=use_actions,
        )
    elif loader_kind == "metaworld":
        from comparison.datasets.metaworld_lerobot import MetaWorldClips
        train_ds = MetaWorldClips(
            split_path=cfg.data.train_split,
            datasets_dir=cfg.data.datasets_dir,
            clip_len=cfg.data.clip_len,
            cameras=cfg.data.cameras,
            resolution=cfg.data.resolution,
            seed=cfg.train.seed,
            return_actions=use_actions,
        )
    elif loader_kind == "calvin":
        from comparison.datasets.calvin import CalvinClips
        train_ds = CalvinClips(
            split_path=cfg.data.train_split,
            datasets_dir=cfg.data.get("datasets_dir") or None,
            clip_len=cfg.data.clip_len,
            cameras=cfg.data.cameras,
            resolution=cfg.data.resolution,
            seed=cfg.train.seed,
            return_actions=use_actions,
        )
    else:
        train_ds = LiberoVideoClips(
            split_path=cfg.data.train_split,
            datasets_dir=cfg.data.datasets_dir,
            clip_len=cfg.data.clip_len,
            cameras=cfg.data.cameras,
            resolution=cfg.data.resolution,
            seed=cfg.train.seed,
            return_actions=use_actions,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        worker_init_fn=worker_init_fn,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=cfg.train.num_workers > 0,
    )

    objective = cfg.train.get("objective", "mae")
    tokenizer = None
    N_per_frame = (cfg.data.resolution // cfg.model.patch_size) ** 2
    if objective == "token_ce":
        tokenizer = load_tokenizer(cfg.train.tokenizer_ckpt, in_channels, device)
        # peek a sample to derive token grid size for the transformer
        peek_batch = next(iter(train_loader))
        peek_clip = peek_batch[0] if isinstance(peek_batch, (tuple, list)) else peek_batch
        peek = peek_clip[:1, :1].to(device)  # (1, 1, C, H, W)
        with torch.no_grad():
            sample_idx = encode_clip_to_tokens(tokenizer, peek)  # (1, 1, N)
        N_per_frame = sample_idx.shape[-1]
        model = build_token_model(cfg, tokenizer.codebook_size, N_per_frame).to(device)
    else:
        model = build_model(cfg, in_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n_params/1e6:.1f}M | tokens/clip: {cfg.data.clip_len * N_per_frame}")

    # auxiliary head (training-time only; see src/training/aux_loss.py)
    aux_head = None
    if aux_lambda > 0:
        feat_dim = int(cfg.model.dim)
        if aux_head_type in ("invdyn_l2", "invdyn", "invdyn_l1"):
            aux_head = InvDynAuxHead(in_dim=feat_dim, action_dim=aux_action_dim).to(device)
        elif aux_head_type == "forward_dyn":
            aux_head = ForwardDynAuxHead(in_dim=feat_dim, action_dim=aux_action_dim).to(device)
        elif aux_head_type == "contrastive":
            aux_head = ContrastiveAuxHead(in_dim=feat_dim).to(device)
        elif aux_head_type == "single_frame":
            aux_head = SingleFrameAuxHead(in_dim=feat_dim, action_dim=aux_action_dim).to(device)
        elif aux_head_type in ("multistep_invdyn", "multistep"):
            aux_k = int(cfg.train.get("aux_k", 2))
            aux_output_k = int(cfg.train.get("aux_output_k", aux_k))
            aux_head = MultiStepInvDynHead(
                in_dim=feat_dim, action_dim=aux_action_dim,
                k=aux_k, output_k=aux_output_k,
            ).to(device)
        else:
            raise ValueError(f"unknown aux_head_type: {aux_head_type!r}")
        n_aux = sum(p.numel() for p in aux_head.parameters())
        print(f"aux head: type={aux_head_type} lambda={aux_lambda} params={n_aux/1e6:.2f}M")

    opt_params = list(model.parameters())
    if aux_head is not None:
        opt_params += list(aux_head.parameters())
    opt = torch.optim.AdamW(
        opt_params,
        lr=cfg.train.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.train.weight_decay,
    )

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.train.amp_dtype]
    use_amp = amp_dtype != torch.float32 and device.type == "cuda"

    N = N_per_frame
    P = cfg.model.patch_size
    H = cfg.data.resolution
    diff_T = int(cfg.train.get("diff_num_steps", 1000))
    diff_schedule = cfg.train.get("diff_schedule", "linear")
    diff_prefix_cond = bool(cfg.train.get("diff_prefix_conditional", False))
    sched = (
        make_schedule(num_steps=diff_T, schedule=diff_schedule, device=device)
        if objective == "diffusion" else None
    )
    if objective == "diffusion":
        print(f"objective: diffusion (T_diff={diff_T}, schedule={diff_schedule}, prefix_conditional={diff_prefix_cond})")
    else:
        print(f"objective: {objective}")

    ema_decay = float(cfg.train.get("ema_decay", 0.0))
    ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None
    if ema is not None:
        print(f"ema decay: {ema_decay}")

    step = 0
    t0 = time.time()
    model.train()
    while step < cfg.train.max_steps:
        for batch in train_loader:
            if step >= cfg.train.max_steps:
                break
            if use_actions:
                clip, actions = batch
                clip = clip.to(device, non_blocking=True)
                actions = actions.to(device, non_blocking=True)
            else:
                clip = batch
                clip = clip.to(device, non_blocking=True)
                actions = None
            B, T = clip.shape[:2]

            for g in opt.param_groups:
                g["lr"] = cfg.train.lr * lr_schedule(step, cfg.train.warmup_steps, cfg.train.max_steps)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                if objective == "mae":
                    mask = make_mask(cfg.train.mask_type, B, T, N, cfg.train.mask_ratio, device=device)
                    pred = model(clip, mask, actions=actions)
                    loss = masked_pixel_loss(pred, clip, mask, P)
                elif objective == "diffusion":
                    if diff_prefix_cond:
                        L_pre = torch.randint(0, T, (B,), device=device)
                        is_noised = (
                            torch.arange(T, device=device)[None, :] >= L_pre[:, None]
                        )  # (B, T)
                        t_idx = torch.randint(0, diff_T, (B,), device=device)
                        noisy_full, _ = q_sample(clip, t_idx, sched)
                        noisy_clip = torch.where(
                            is_noised[:, :, None, None, None], noisy_full, clip
                        )
                        # per-frame t: 0 for clean prefix, t_idx for noised post
                        t_per_frame = torch.where(
                            is_noised,
                            t_idx[:, None].expand(B, T),
                            torch.zeros(B, T, dtype=torch.long, device=device),
                        )
                        # split forward into trunk + head so the post-norm
                        # features are available for the aux inv-dyn head
                        features = model._trunk(noisy_clip, mask_indicator=None, t=t_per_frame, actions=actions)
                        pred_patches = model.head(features)
                        pred_clip = unpatchify(pred_patches, P, H, in_channels)
                        sq = (pred_clip - clip).pow(2).mean(dim=(2, 3, 4))  # (B, T)
                        denom = is_noised.float().sum().clamp_min(1.0)
                        loss = (sq * is_noised.float()).sum() / denom
                    else:
                        t_idx = torch.randint(0, diff_T, (B,), device=device)
                        noisy, _ = q_sample(clip, t_idx, sched)
                        features = model._trunk(noisy, mask_indicator=None, t=t_idx, actions=actions)
                        pred_patches = model.head(features)
                        pred_clip = unpatchify(pred_patches, P, H, in_channels)
                        loss = (pred_clip - clip).pow(2).mean()
                    if aux_head is not None and aux_lambda > 0:
                        aux_loss_val = compute_aux_loss(
                            features, aux_head,
                            actions[..., :aux_action_dim], aux_head_type,
                        )
                        loss = loss + aux_lambda * aux_loss_val
                elif objective == "flow_matching":
                    fm_pc = bool(cfg.train.get("fm_prefix_conditional", True))
                    loss = flow_matching_loss(
                        model, clip, in_channels, P, H,
                        prefix_conditional=fm_pc,
                        t_distribution=cfg.train.get("fm_t_distribution", "uniform"),
                        t_logit_mean=float(cfg.train.get("fm_t_logit_mean", 0.0)),
                        t_logit_std=float(cfg.train.get("fm_t_logit_std", 1.0)),
                        actions=actions,
                    )
                elif objective == "token_ce":
                    with torch.no_grad():
                        token_ids = encode_clip_to_tokens(tokenizer, clip)  # (B, T, N)
                    mask = make_mask(cfg.train.mask_type, B, T, N, cfg.train.mask_ratio, device=device)
                    logits = model(token_ids, mask, actions=actions)  # (B, T, N, V)
                    loss = torch.nn.functional.cross_entropy(
                        logits[mask], token_ids[mask]
                    )
                else:
                    raise ValueError(f"unknown objective: {objective!r}")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema is not None:
                ema.update(model)

            if step % cfg.train.log_interval == 0:
                cur_lr = opt.param_groups[0]["lr"]
                ips = (step + 1) * B / max(1e-6, time.time() - t0)
                print(f"step {step:6d} | loss {loss.item():.4f} | lr {cur_lr:.2e} | clips/s {ips:.1f}")
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr", cur_lr, step)

            if step > 0 and step % cfg.train.ckpt_interval == 0:
                torch.save(
                    {"step": step, "model": model.state_dict(), "config": OmegaConf.to_container(cfg)},
                    ckpt_dir / f"ckpt_{step:06d}.pt",
                )
            step += 1

    if ema is not None:
        # save raw weights + aux_head as sidecar; EMA weights as final ckpt
        raw_state: dict = {"step": step, "model": model.state_dict(),
                           "config": OmegaConf.to_container(cfg)}
        if aux_head is not None:
            raw_state["aux_head"] = aux_head.state_dict()
        torch.save(raw_state, ckpt_dir / "ckpt_last_raw.pt")
        ema.copy_to(model)
    torch.save(
        {"step": step, "model": model.state_dict(), "config": OmegaConf.to_container(cfg)},
        ckpt_dir / "ckpt_last.pt",
    )
    writer.close()
    suffix = " (EMA)" if ema is not None else ""
    print(f"done. saved to {ckpt_dir/'ckpt_last.pt'}{suffix}")


if __name__ == "__main__":
    main()
