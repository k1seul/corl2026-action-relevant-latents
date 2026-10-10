"""LIBERO and LIBERO-PRO rollouts for a closed_loop.train checkpoint (Table 4, Table 20).

Perturbations (LIBERO-PRO, https://github.com/Zxy-MLlab/LIBERO-PRO):
  none         standard LIBERO (original BDDL and benchmark init states)
  language     paraphrased instruction, same scene
  position     object start positions swapped
  object       objects replaced by visually different variants
  task         different goal (new instruction and goal predicate)
  environment  scene moved to other tables/floors; trials are split over all candidates

Settings used for the paper: 10 tasks x 10 trials per suite, replan every 5 steps,
10 denoising steps, 256 px renders resized to 224, --seed 7.

    PYTHONPATH=src:. python -m closed_loop.eval_libero_pro --checkpoint <ckpt_final.pt> \
        --task-suite libero_10 --perturbations none,language,position,object,task,environment \
        --out-dir runs/closed_loop_eval/<run>

Also accepts checkpoints written by the original training code (`args.aux_source`).
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import pathlib
import tempfile
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from closed_loop.perturbation import apply_environment_variants, apply_perturbation
from closed_loop.policy import (LIBERO_ACTION_DIM, MAX_TOKEN_LEN, _observation, attach_special_token,
                                build_pi0, get_tokenizer)

PRO_SUITES = ("libero_goal", "libero_spatial", "libero_10", "libero_object")
PERTURBATION_KINDS = ("none", "language", "position", "object", "task", "environment")
_BDDL_MODIFYING = {"position", "object", "task", "environment"}
# Fixed seed for choosing the perturbation instance and generating the perturbed scene's
# initial states, so that --seed only changes the policy's sampling noise.
_SCENE_FIXED_SEED = 0
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
SUITE_MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def _ensure_libero_config() -> None:
    """Write ~/.libero/config.yaml so `import libero.libero` does not prompt."""
    import importlib.util
    import yaml
    cfg_dir = pathlib.Path(os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")))
    cfg_file = cfg_dir / "config.yaml"
    if cfg_file.exists():
        return
    spec = importlib.util.find_spec("libero.libero")
    if spec is None or not spec.origin:
        return
    root = os.path.dirname(os.path.abspath(spec.origin))
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(yaml.safe_dump({
        "benchmark_root": root, "bddl_files": os.path.join(root, "./bddl_files"),
        "init_states": os.path.join(root, "./init_files"), "datasets": os.path.join(root, "../datasets"),
        "assets": os.path.join(root, "./assets"),
    }))


def _task_init_states(task_suite, task_id: int):
    """LIBERO stores init states as pickled numpy arrays and loads them with a bare
    torch.load, which torch >= 2.6 rejects (weights_only=True by default)."""
    orig = torch.load

    def _load(*a, **k):
        k.setdefault("weights_only", False)
        return orig(*a, **k)

    torch.load = _load
    try:
        return task_suite.get_task_init_states(task_id)
    finally:
        torch.load = orig


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).copy()
    q[3] = min(max(q[3], -1.0), 1.0)
    den = math.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((q[:3] * 2.0 * math.acos(q[3])) / den).astype(np.float32)


def state_from_env_obs(obs: Dict) -> np.ndarray:
    """15-D training state: ee_pos, ee_ori (axis-angle), gripper_states, joint_states."""
    return np.concatenate([
        np.asarray(obs["robot0_eef_pos"], dtype=np.float32), _quat2axisangle(obs["robot0_eef_quat"]),
        np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32), np.asarray(obs["robot0_joint_pos"], dtype=np.float32),
    ])


def _image(img_hwc_u8: np.ndarray, size: int, flip: str) -> torch.Tensor:
    img = {"vertical": img_hwc_u8[::-1, :, :], "180": img_hwc_u8[::-1, ::-1], "none": img_hwc_u8}[flip]
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).contiguous().float().div_(255.0)
    if t.shape[-1] != size:
        t = F.interpolate(t.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)
    return t


def load_policy(ckpt_path: str, device):
    """Rebuild pi0.5 (with the special token for token-coupling runs) and load weights."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    token = a.get("coupling") == "token" or a.get("aux_source") == "pi0_special_token"
    pi0, cfg = build_pi0(None, device, action_horizon=int(a.get("action_horizon", 16)), gradient_checkpointing=False)
    if token:
        attach_special_token(pi0, 2048, device)
    missing, unexpected = pi0.load_state_dict(ckpt["pi0"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: {len(missing)} missing, {len(unexpected)} unexpected keys")
    pi0.eval()
    return pi0


def _rollout_task(*, pi0, env, init_states, language, num_trials, max_steps, num_steps_wait, replan_steps,
                  inference_steps, image_size, flip, tokenizer, device, seed, task_id,
                  video_dir: Optional[pathlib.Path] = None):
    succ = 0
    for ep in range(num_trials):
        noise_seed = seed * 10000 + task_id * 100 + ep
        torch.manual_seed(noise_seed)
        np.random.seed(noise_seed)
        env.seed(noise_seed)
        env.reset()
        if init_states is not None:
            obs = env.set_init_state(init_states[ep % len(init_states)])
        else:
            obs, *_ = env.step(LIBERO_DUMMY_ACTION)
        plan: collections.deque = collections.deque()
        frames, done, t = [], False, 0
        while t < max_steps + num_steps_wait:
            if t < num_steps_wait:                                   # let objects settle
                obs, _r, _d, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue
            img_main, img_wrist = obs["agentview_image"], obs["robot0_eye_in_hand_image"]
            if video_dir is not None:
                frames.append(np.ascontiguousarray(img_main[::-1, :, :]))
            if not plan:
                observation = _observation(
                    _image(img_main, image_size, flip).unsqueeze(0).to(device),
                    _image(img_wrist, image_size, flip).unsqueeze(0).to(device),
                    torch.from_numpy(state_from_env_obs(obs)).float().unsqueeze(0).to(device),
                    [language], tokenizer, device)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    chunk = pi0.sample_actions(device, observation, num_steps=inference_steps)
                chunk = chunk.squeeze(0).float().cpu().numpy()[:, :LIBERO_ACTION_DIM]
                plan.extend(chunk[:replan_steps])
            obs, _r, done, _ = env.step(plan.popleft().tolist())
            if done:
                break
            t += 1
        succ += int(bool(done))
        if video_dir is not None and frames:
            import imageio
            imageio.mimwrite(str(video_dir / f"task{task_id:03d}_ep{ep:02d}_{'success' if done else 'failure'}.mp4"),
                             frames, fps=15)
    return succ, num_trials


def eval_suite(pi0, *, suite: str, perturbation: str, tokenizer, device, num_trials_per_task=10, num_tasks=-1,
               num_steps_wait=10, replan_steps=5, inference_steps=10, image_size=224, render_size=256,
               flip="vertical", video_base: Optional[pathlib.Path] = None) -> Dict:
    if perturbation != "none" and suite not in PRO_SUITES:
        raise ValueError(f"LIBERO-PRO perturbations only support {PRO_SUITES}")
    _ensure_libero_config()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_root = pathlib.Path(get_libero_path("bddl_files"))
    task_suite = benchmark.get_benchmark_dict()[suite]()
    n_tasks = task_suite.n_tasks
    max_steps = SUITE_MAX_STEPS[suite]
    common = dict(pi0=pi0, max_steps=max_steps, num_steps_wait=num_steps_wait, replan_steps=replan_steps,
                  inference_steps=inference_steps, image_size=image_size, flip=flip, tokenizer=tokenizer,
                  device=device, seed=_SCENE_FIXED_SEED)
    per_task, total_succ, total_eps, t_start = [], 0, 0, time.time()
    with tempfile.TemporaryDirectory(prefix="libero_pro_bddl_") as tmp:
        tmp = pathlib.Path(tmp)
        for task_id in range(n_tasks if num_tasks < 0 else min(num_tasks, n_tasks)):
            task = task_suite.get_task(task_id)
            orig_bddl = bddl_root / task.problem_folder / task.bddl_file
            content = orig_bddl.read_text(encoding="utf-8")
            if perturbation == "environment":
                variants = apply_environment_variants(content, suite, task.name)
                n_env = max(1, len(variants))
                split = [num_trials_per_task // n_env + (1 if i < num_trials_per_task % n_env else 0) for i in range(n_env)]
                s_task = n_task = 0
                for i_env, ((var_content, var_lang), n_tr) in enumerate(zip(variants, split)):
                    if n_tr <= 0:
                        continue
                    path = tmp / f"{task.bddl_file}.env{i_env}"
                    path.write_text(var_content, encoding="utf-8")
                    try:
                        env = OffScreenRenderEnv(bddl_file_name=str(path), camera_heights=render_size, camera_widths=render_size)
                    except Exception as e:                       # counted as failures
                        logging.warning("environment variant %d failed to load (%s); %d trials counted as failures", i_env, e, n_tr)
                        n_task += n_tr
                        continue
                    env.seed(_SCENE_FIXED_SEED)
                    s, n = _rollout_task(env=env, init_states=None, language=var_lang or str(task.language),
                                         num_trials=n_tr, task_id=task_id, **common)
                    env.close()
                    s_task += s; n_task += n
                per_task.append({"task_id": task_id, "task_original": str(task.language), "perturbation": perturbation,
                                 "episodes": n_task, "successes": s_task, "success_rate": s_task / max(1, n_task),
                                 "n_environments": n_env})
                total_succ += s_task; total_eps += n_task
                continue

            new_content, new_language = apply_perturbation(bddl_content=content, suite=suite, task=task.name,
                                                           kind=perturbation, seed=_SCENE_FIXED_SEED)
            language = new_language or str(task.language)
            if perturbation in _BDDL_MODIFYING:
                bddl = tmp / task.bddl_file
                bddl.write_text(new_content, encoding="utf-8")
                init_states = None                                     # env.reset() with a fixed seed
            else:
                bddl = orig_bddl
                init_states = _task_init_states(task_suite, task_id)
            try:
                env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=render_size, camera_widths=render_size)
            except Exception as e:
                logging.warning("task %d failed to load (%s); counted as %d failures", task_id, e, num_trials_per_task)
                per_task.append({"task_id": task_id, "task": language, "task_original": str(task.language),
                                 "perturbation": perturbation, "episodes": num_trials_per_task, "successes": 0,
                                 "success_rate": 0.0, "env_init_error": f"{type(e).__name__}: {e}"})
                total_eps += num_trials_per_task
                continue
            env.seed(_SCENE_FIXED_SEED)
            video_dir = None
            if video_base is not None:
                video_dir = video_base / f"task{task_id:03d}"
                video_dir.mkdir(parents=True, exist_ok=True)
            s, n = _rollout_task(env=env, init_states=init_states, language=language, num_trials=num_trials_per_task,
                                 task_id=task_id, video_dir=video_dir, **common)
            env.close()
            per_task.append({"task_id": task_id, "task": language, "task_original": str(task.language),
                             "perturbation": perturbation, "episodes": n, "successes": s, "success_rate": s / max(1, n)})
            total_succ += s; total_eps += n
            logging.info("[%s/%s] task %d/%d sr=%.2f cumulative %d/%d", suite, perturbation, task_id + 1, n_tasks,
                         s / max(1, n), total_succ, total_eps)
    return {"task_suite": suite, "perturbation": perturbation, "num_trials_per_task": num_trials_per_task,
            "total_episodes": total_eps, "total_successes": total_succ,
            "overall_success_rate": total_succ / max(1, total_eps), "per_task": per_task,
            "elapsed_sec": time.time() - t_start}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task-suite", default="libero_10", choices=list(PRO_SUITES) + ["libero_90"])
    p.add_argument("--perturbations", default="none", help="comma-separated subset of " + ",".join(PERTURBATION_KINDS))
    p.add_argument("--num-trials-per-task", type=int, default=10)
    p.add_argument("--num-tasks", type=int, default=-1)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--inference-steps", type=int, default=10)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--render-size", type=int, default=256)
    p.add_argument("--flip", choices=["vertical", "180", "none"], default="vertical", help="must match training")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--save-videos", action="store_true")
    a = p.parse_args()
    kinds = [k.strip() for k in a.perturbations.split(",") if k.strip()]
    for k in kinds:
        if k not in PERTURBATION_KINDS:
            raise SystemExit(f"unknown perturbation {k!r}")
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    _ensure_libero_config()
    pi0 = load_policy(a.checkpoint, device)
    tokenizer = get_tokenizer()
    run = pathlib.Path(a.checkpoint).parent.name
    for kind in kinds:
        out = pathlib.Path(a.out_dir) / f"{run}__{a.task_suite}__pro_{kind}"
        out.mkdir(parents=True, exist_ok=True)
        res = eval_suite(pi0, suite=a.task_suite, perturbation=kind, tokenizer=tokenizer, device=device,
                         num_trials_per_task=a.num_trials_per_task, num_tasks=a.num_tasks,
                         num_steps_wait=a.num_steps_wait, replan_steps=a.replan_steps,
                         inference_steps=a.inference_steps, image_size=a.image_size, render_size=a.render_size,
                         flip=a.flip, video_base=(out / "videos") if a.save_videos else None)
        logging.info("=== %s / %s: success %.3f (%d/%d) ===", a.task_suite, kind, res["overall_success_rate"],
                     res["total_successes"], res["total_episodes"])
        (out / "summary.json").write_text(json.dumps({**res, "checkpoint": a.checkpoint, "config": vars(a)}, indent=2))


if __name__ == "__main__":
    main()
