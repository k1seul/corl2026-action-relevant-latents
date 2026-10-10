"""Unified launcher for the Phase 1 matched-control study.

Dispatches to the appropriate backbone-specific ID fine-tuning script (which
lives under ``scripts/train_{vjepa2,videomae,webdino}_aux*.py``) after:

  * validating that (--backbone × --condition) is a legal combination;
  * enforcing step/batch matching across conditions within one backbone;
  * recording provenance BEFORE the training job starts;
  * injecting the shuffled-ID / forward-dynamics / frozen mode via env vars
    or CLI flags the wrapped script already understands.

Rationale for a separate wrapper (rather than editing each train script):
we want zero risk of breaking the existing 30k true-ID runs (which produced
the submitted numbers). Each wrapper call ends with an ``exec`` into the
original script so the training path is byte-identical to the historical
one, aside from the new controls.

Usage (single job, one A6000):

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_matched_control.py \\
        --backbone vjepa2 \\
        --condition true_id \\
        --run-id v0_vjepa2_true_id_s0 \\
        --training-seed 0 \\
        --probe-seed 0 \\
        --max-steps 20000

To generate the launch grid (9 jobs × 2 seeds = 18 primaries) see
``runs/corl2026_rebuttal/launchers/wave1_matched_controls.sh``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from rebuttal.provenance import record_provenance  # noqa: E402


BACKBONES = ("vjepa2", "videomae", "webdino")
# ``shuffled_id`` is aliased to the within-task cross-episode derangement.
# ``shuffled_id_circular_sensitivity`` remains available as a labeled
# secondary control (session-2 STEP 3).
CONDITIONS = ("frozen", "true_id", "shuffled_id",
              "shuffled_id_circular_sensitivity",
              "forward_dynamics", "action_free_adaptation")


# Mapping (backbone → underlying training script). "frozen" is not a
# training condition — it re-runs canonical feature extraction only.
TRAIN_SCRIPT = {
    "vjepa2":   "scripts/train_vjepa2_aux.py",
    "videomae": "scripts/train_videomae_aux_libero.py",
    "webdino":  "scripts/train_webdino_aux_libero.py",
}


def build_argv_for_child(backbone: str, condition: str, args: argparse.Namespace,
                        out_dir: Path) -> list[str]:
    """Convert wrapper args → argv for the backbone-specific script.

    The child scripts already accept ``--dataset libero --out-dir ... \\
    --max-steps N --batch-size B --lr LR``. Controls that don't yet exist
    at CLI level are passed through environment variables:

        REBUTTAL_CONDITION        one of CONDITIONS
        REBUTTAL_RUN_SEED         int
        REBUTTAL_SHUFFLE_MODE     circular_shift | within_task_cross_episode
        REBUTTAL_SHUFFLE_MIN_ABS  int (default 2)

    The child scripts must be patched to honour REBUTTAL_CONDITION before
    launch; see ``runs/corl2026_rebuttal/reports/TRAIN_SCRIPT_PATCH_NOTES.md``.
    """
    if backbone not in TRAIN_SCRIPT:
        raise ValueError(f"unknown backbone: {backbone}")
    child = TRAIN_SCRIPT[backbone]
    argv = [
        "uv", "run", "python", str(REPO_ROOT / child),
        "--dataset", "libero",
        "--out-dir", str(out_dir),
        "--max-steps", str(args.max_steps),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
    ]
    return argv


def main() -> int:
    p = argparse.ArgumentParser(description="Matched-control launcher wrapper.")
    p.add_argument("--backbone", required=True, choices=BACKBONES)
    p.add_argument("--condition", required=True, choices=CONDITIONS)
    p.add_argument("--run-id", required=True,
                   help="unique run identifier; used as sub-directory name")
    p.add_argument("--training-seed", type=int, required=True)
    p.add_argument("--probe-seed", type=int, required=True)
    p.add_argument("--max-steps", type=int, default=20000,
                   help="Must match across all matched-control conditions; "
                        "default 20k is the rebuttal-mandated common step count.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--effective-batch", type=int, default=None,
                   help="if using gradient accumulation, the effective batch")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr-schedule", default="cosine_warmup2000")
    p.add_argument("--out-root", default="runs/corl2026_rebuttal/checkpoints")
    p.add_argument(
        "--shuffle-mode",
        default="within_task_cross_episode_derangement",
        choices=("within_task_cross_episode_derangement", "circular_shift"),
        help=("primary is the within-task cross-episode derangement; "
              "circular_shift is retained only for the sensitivity condition."),
    )
    p.add_argument("--shuffle-min-abs", type=int, default=2)
    p.add_argument("--dry-run", action="store_true",
                   help="record provenance and print the child argv, don't exec")
    args = p.parse_args()

    if args.condition == "frozen":
        # frozen has no training; user should run the feature-extraction path
        # in scripts/full_results_pipeline.py directly.
        sys.stderr.write(
            "[matched_control] 'frozen' has no training step; run "
            "scripts/full_results_pipeline.py with backbone=" f"{args.backbone}\n"
        )
        return 0

    out_dir = REPO_ROOT / args.out_root / args.backbone / args.condition / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    effective_batch = args.effective_batch or args.batch_size

    record_provenance(
        out_dir,
        condition=args.condition,
        backbone=args.backbone,
        run_id=args.run_id,
        training_seed=args.training_seed,
        probe_seed=args.probe_seed,
        optimizer="adamw",
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        batch_size=args.batch_size,
        effective_batch=effective_batch,
        target_updates=args.max_steps,
        extra={"shuffle_mode": args.shuffle_mode,
               "shuffle_min_abs": args.shuffle_min_abs},
    )

    child_argv = build_argv_for_child(args.backbone, args.condition, args, out_dir)

    env = os.environ.copy()
    # Enforce shuffle_mode ↔ condition consistency at wrapper level so a
    # mis-typed CLI cannot silently produce a mislabeled result.
    if args.condition == "shuffled_id" and args.shuffle_mode != "within_task_cross_episode_derangement":
        raise SystemExit(
            "condition=shuffled_id requires --shuffle-mode="
            "within_task_cross_episode_derangement; use "
            "condition=shuffled_id_circular_sensitivity for the circular-shift "
            "sensitivity control."
        )
    if args.condition == "shuffled_id_circular_sensitivity" and args.shuffle_mode != "circular_shift":
        raise SystemExit(
            "condition=shuffled_id_circular_sensitivity requires "
            "--shuffle-mode=circular_shift.")

    env["REBUTTAL_CONDITION"] = args.condition
    env["REBUTTAL_RUN_SEED"] = str(args.training_seed)
    env["REBUTTAL_SHUFFLE_MODE"] = args.shuffle_mode
    env["REBUTTAL_SHUFFLE_MIN_ABS"] = str(args.shuffle_min_abs)

    print("[matched_control] child argv:")
    for a in child_argv:
        print(" ", a)
    print("[matched_control] env additions:")
    for k in ("REBUTTAL_CONDITION", "REBUTTAL_RUN_SEED",
              "REBUTTAL_SHUFFLE_MODE", "REBUTTAL_SHUFFLE_MIN_ABS"):
        print(f"  {k}={env[k]}")

    if args.dry_run:
        return 0

    os.execvpe(child_argv[0], child_argv, env)
    return 0  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
