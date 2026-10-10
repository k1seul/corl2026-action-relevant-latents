"""Fine-tune pi0.5 on LIBERO with an auxiliary future-target loss (Section 4.6, Appendix N).

    loss = flow-matching action loss + weight * auxiliary target loss

--coupling none   pi0.5 baseline (no auxiliary loss)
--coupling clip   mean-pooled main-camera prefix tokens -> MLP -> frozen DIFF + ID encoding
                  of the two-camera clip o_t .. o_{t+16}; normalized L2, weight 0.1.
                  Controls: --shuffle-target (batch-shuffled), --random-init-target (random-init trunk).
--coupling token  one learned token appended to the prefix; its output -> LayerNorm + linear ->
                  frozen V-JEPA 2 + ID (--target vjepa2) or Web-DINO (--target webdino) mean feature
                  of o_{t+16}; cosine loss, weight 0.01 (0.05 also reported for V-JEPA 2).

The launchers in closed_loop/scripts/ give the exact settings of each Table 4 row.

Run from the repository root:
    PYTHONPATH=src:. python -m closed_loop.train --coupling clip --out-dir runs/closed_loop/clip_s0
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch

from closed_loop.data import DEFAULT_SUITES, build_loader
from closed_loop.policy import (PrefixCapture, SpecialTokenReadout, attach_special_token, build_pi0,
                                cosine_loss, get_tokenizer, make_train_observation)
from closed_loop.targets import (DIFF_ID_REPO, VJEPA2_ID_REPO, WEBDINO_REPO, DreamerVLATargetEncoder,
                                 VJepa2TargetEncoder, WebDINOTargetEncoder, WorldModelClipAux)

PALIGEMMA_WIDTH = 2048       # gemma_2b hidden size = width of the prefix tokens
PI0_IMAGE_SIZE = 224


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coupling", choices=["none", "clip", "token"], required=True)
    p.add_argument("--out-dir", required=True)
    # data
    p.add_argument("--data-root", default="datasets/libero", help="directory with libero_{10,90,goal,object,spatial}/*.hdf5")
    p.add_argument("--suites", default=",".join(DEFAULT_SUITES))
    p.add_argument("--split-ratio", type=float, default=0.9, help="fraction of demos used for training")
    p.add_argument("--num-workers", type=int, default=6)
    # policy and optimization (shared by every row of Table 4)
    p.add_argument("--pretrained-pi0", required=True, help="converted pi0.5 base weights (dir with model.safetensors)")
    p.add_argument("--action-horizon", type=int, default=16)
    p.add_argument("--num-train-steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--peak-lr", type=float, default=1.2e-4)
    p.add_argument("--end-lr", type=float, default=3e-6,
                   help="cosine schedule floor; the paper's baseline run used 1.2e-5 (see README)")
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    # auxiliary target
    p.add_argument("--aux-weight", type=float, default=None, help="default: 0.1 (clip), 0.01 (token)")
    p.add_argument("--target", choices=["vjepa2", "webdino"], default="vjepa2", help="[token] target encoder")
    p.add_argument("--target-repo", default=None,
                   help="HF repo or local checkpoint of the target encoder (default: the paper's)")
    p.add_argument("--clip-horizon", type=int, default=16, help="[clip] clip = o_t .. o_{t+H}")
    p.add_argument("--head-hidden", type=int, default=2048, help="[clip] MLP hidden width")
    p.add_argument("--shuffle-target", action="store_true", help="[clip] batch-shuffled control")
    p.add_argument("--random-init-target", action="store_true", help="[clip] random-init DIFF control")
    p.add_argument("--token-horizon", type=int, default=16, help="[token] target frame o_{t+H}")
    # bookkeeping
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=2500)
    p.add_argument("--wandb-project", default="", help="log to Weights & Biases when set")
    p.add_argument("--wandb-run", default="")
    a = p.parse_args()
    if a.aux_weight is None:
        a.aux_weight = {"none": 0.0, "clip": 0.1, "token": 0.01}[a.coupling]
    if a.coupling == "none" and a.aux_weight != 0.0:
        p.error("--coupling none trains without an auxiliary loss")
    return a


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def lr_schedule(step: int, *, peak: float, end: float, warmup: int, total: int) -> float:
    if warmup > 0 and step < warmup:
        init = peak / (warmup + 1)
        return init + (peak - init) * step / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return end + (peak - end) * 0.5 * (1.0 + np.cos(np.pi * progress))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    # ---- frozen target encoder --------------------------------------------------------
    # Built before the policy, in the same order as the code that produced the paper's
    # runs, so that a given --seed reproduces their initialization and data order.
    # The clip coupling builds its own encoder inside WorldModelClipAux; the original code
    # also constructed (and never used) a default DIFF + ID encoder here, which only
    # advanced the random state. It is rebuilt and discarded for the same reason.
    target_encoder = None
    if args.coupling == "token":
        if args.target == "vjepa2":
            target_encoder = VJepa2TargetEncoder(hf_repo=args.target_repo or VJEPA2_ID_REPO, device=device)
        else:
            target_encoder = WebDINOTargetEncoder(hf_repo=args.target_repo or WEBDINO_REPO, device=device)
    elif args.coupling == "clip":
        del_me = DreamerVLATargetEncoder(DIFF_ID_REPO, device=device)
        del del_me
        torch.cuda.empty_cache()

    # ---- data ---------------------------------------------------------------------------
    aux_horizons = (tuple(range(1, args.clip_horizon + 1)) if args.coupling == "clip"
                    else (args.token_horizon if args.coupling == "token" else args.action_horizon,))
    loader = build_loader(
        args.data_root, suites=[s for s in args.suites.split(",") if s], action_horizon=args.action_horizon,
        aux_horizons=aux_horizons, batch_size=args.batch_size, num_workers=args.num_workers,
        split="train", split_ratio=args.split_ratio, split_seed=args.seed,
        shuffle=True, drop_last=True, random_clip=True)
    logging.info("train demos=%d", len(loader.dataset))

    # ---- policy ---------------------------------------------------------------------------
    pi0, _ = build_pi0(args.pretrained_pi0, device, action_horizon=args.action_horizon)
    tokenizer = get_tokenizer()

    aux_head = None
    if args.coupling == "token":
        attach_special_token(pi0, PALIGEMMA_WIDTH, device)
        aux_head = SpecialTokenReadout(PALIGEMMA_WIDTH, target_encoder.d_bottleneck).to(device)
    elif args.coupling == "clip":
        aux_head = WorldModelClipAux(PALIGEMMA_WIDTH, hf_repo=args.target_repo or DIFF_ID_REPO,
                                     head_hidden=args.head_hidden, shuffle_target=args.shuffle_target,
                                     random_init=args.random_init_target, device=device)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, PI0_IMAGE_SIZE, PI0_IMAGE_SIZE, device=device, dtype=torch.bfloat16)
            aux_head.n_main_patches = int(pi0.paligemma_with_expert.embed_image(dummy).shape[1])
    capture = None
    if aux_head is not None:
        capture = PrefixCapture()
        capture.attach(pi0)

    params = list(pi0.parameters()) + (list(aux_head.parameters()) if aux_head is not None else [])
    optim = torch.optim.AdamW(params, lr=args.peak_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    amp = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    wandb = None
    if args.wandb_project:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run or None, config=vars(args), dir=str(out_dir))

    pi0.train()
    if aux_head is not None:
        aux_head.train()
    step, t0 = 0, time.time()
    while step < args.num_train_steps:
        for batch in loader:
            if step >= args.num_train_steps:
                break
            lr = lr_schedule(step, peak=args.peak_lr, end=args.end_lr, warmup=args.warmup_steps,
                             total=args.num_train_steps)
            for pg in optim.param_groups:
                pg["lr"] = lr
            obs, actions = make_train_observation(batch, image_size=PI0_IMAGE_SIZE, tokenizer=tokenizer, device=device)
            future_main = batch["image_main_aux"].to(device, non_blocking=True)      # (B, K, 3, H, W)

            with amp():
                pi0_loss = pi0(obs, actions)
                if isinstance(pi0_loss, (list, tuple)):
                    pi0_loss = torch.stack(list(pi0_loss))
                pi0_loss = pi0_loss.mean()

                aux_loss = torch.tensor(0.0, device=device)
                if args.coupling == "clip":
                    tokens = capture.prefix_out[:, :aux_head.n_main_patches, :].float()   # main camera leads the prefix
                    main_clip = torch.cat([batch["image_main"].to(device).unsqueeze(1), future_main], dim=1)
                    wrist_clip = torch.cat([batch["image_wrist"].to(device).unsqueeze(1),
                                            batch["image_wrist_aux"].to(device)], dim=1)
                    aux_loss = aux_head.loss(tokens, main_clip, wrist_clip)
                elif args.coupling == "token":
                    B, K = future_main.shape[:2]
                    z = target_encoder.encode(future_main.flatten(0, 1))
                    z = z.view(B, K, *z.shape[1:])[:, 0]                                  # (B, 1, D)
                    pred = aux_head(capture.prefix_out[:, -1, :])                          # special token = last
                    aux_loss = cosine_loss(pred, z)
                loss = pi0_loss + args.aux_weight * aux_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optim.step()

            if step % args.log_every == 0:
                ips = (step + 1) * args.batch_size / max(1e-6, time.time() - t0)
                logging.info("step %d/%d loss=%.4f pi0=%.4f aux=%.4f(w=%.4f) lr=%.2e gn=%.2f ips=%.1f",
                             step, args.num_train_steps, loss.item(), pi0_loss.item(), aux_loss.item(),
                             args.aux_weight, lr, float(gn), ips)
                if wandb is not None:
                    wandb.log({"train/loss": loss.item(), "train/pi0_flow_mse": pi0_loss.item(),
                               "train/aux": aux_loss.item(), "train/lr": lr, "train/grad_norm": float(gn)}, step=step)
            if step > 0 and args.save_every and step % args.save_every == 0:
                torch.save({"step": step, "pi0": pi0.state_dict(),
                            "aux_head": aux_head.state_dict() if aux_head is not None else None,
                            "optim": optim.state_dict(), "args": vars(args)},
                           out_dir / f"ckpt_step{step:08d}.pt")
            step += 1

    final = out_dir / "ckpt_final.pt"
    torch.save({"step": step, "pi0": pi0.state_dict(),
                "aux_head": aux_head.state_dict() if aux_head is not None else None, "args": vars(args)}, final)
    logging.info("[save] %s", final)


if __name__ == "__main__":
    main()
