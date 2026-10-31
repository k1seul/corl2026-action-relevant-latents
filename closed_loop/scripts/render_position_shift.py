"""Render a LIBERO task as the policy first sees it: original layout vs. LIBERO-PRO position shift.

Uses the same perturbation and seeds as eval_libero_pro.py (scene seed 0, episode 0), lets
objects settle for 10 no-op steps, and saves the upright agent-view image.

    MUJOCO_GL=egl PYTHONPATH=src:. python closed_loop/scripts/render_position_shift.py \
        --suite libero_10 --task 2 --out closed_loop/assets --res 512
"""
import argparse
import pathlib
import tempfile

import numpy as np
import torch
from PIL import Image

from closed_loop.eval_libero_pro import (LIBERO_DUMMY_ACTION, _SCENE_FIXED_SEED, _ensure_libero_config,
                                         _task_init_states)
from closed_loop.perturbation import apply_perturbation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task", type=int, default=2)
    ap.add_argument("--out", default="closed_loop/assets")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--crop-top", type=float, default=150 / 512, help="fraction of rows dropped from the top")
    a = ap.parse_args()
    _ensure_libero_config()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[a.suite]()
    task = suite.get_task(a.task)
    orig = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    shifted, _ = apply_perturbation(bddl_content=orig.read_text(encoding="utf-8"), suite=a.suite,
                                    task=task.name, kind="position", seed=_SCENE_FIXED_SEED)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for cond in ("original", "position_shift"):
            bddl = orig
            if cond == "position_shift":
                bddl = pathlib.Path(tmp) / task.bddl_file
                bddl.write_text(shifted, encoding="utf-8")
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=a.res, camera_widths=a.res)
            env.seed(_SCENE_FIXED_SEED)
            ep_seed = _SCENE_FIXED_SEED * 10000 + a.task * 100              # episode 0
            np.random.seed(ep_seed); torch.manual_seed(ep_seed)
            env.seed(ep_seed); env.reset()
            if cond == "original":
                obs = env.set_init_state(_task_init_states(suite, a.task)[0])
            else:
                obs, *_ = env.step(LIBERO_DUMMY_ACTION)
            for _ in range(10):                                            # same settling steps as evaluation
                obs, *_ = env.step(LIBERO_DUMMY_ACTION)
            img = np.ascontiguousarray(obs["agentview_image"][::-1])        # upright, as the policy sees it
            img = img[int(a.crop_top * a.res):]
            name = f"{a.suite.replace('libero_', 'libero')}_task{a.task}_{cond}.png"
            Image.fromarray(img).save(out / name)
            env.close()
            print("saved", out / name)


if __name__ == "__main__":
    main()
