#!/usr/bin/env bash
# Wave 1 — matched-control training jobs (first training seed).
#
# **PRIMARY LAUNCH PATH is Slurm/sbatch:**
#   REPO_ROOT=$(pwd) bash runs/corl2026_rebuttal/launchers/slurm/submit_wave1.sh
# This nohup script exists as a fallback for scheduler-less environments
# (interactive debugging on a single node); prefer the sbatch submitter.
#
# Layout: 9 primary jobs across 10 A6000 GPUs on ONE compute node.
#   * V-JEPA2 × {true_id, shuffled_id, forward_dynamics}   → GPU 0,1,2
#   * VideoMAE × {true_id, shuffled_id, forward_dynamics}  → GPU 3,4,5
#   * Web-DINO × {true_id, shuffled_id, forward_dynamics}  → GPU 6,7,8
#   * GPU 9 held as retry / OOM fallback.
#
# Prerequisites:
#   * REPO_ROOT is a clone of the migration bundle with rebuttal branch
#   * EXTERNAL_MODELS_DIR contains all frozen weights (see ASSET_INVENTORY.md)
#   * LIBERO videos + split JSON present under paths in src/data/build_train_dataset.py
#   * uv, torch, and backbone dependencies installed
#
# Usage:
#   bash runs/corl2026_rebuttal/launchers/wave1_matched_controls.sh
#
# Each job writes to
#   $LOGS/${BACKBONE}_${COND}_s${SEED}.log
# and produces its own provenance.json under
#   runs/corl2026_rebuttal/checkpoints/${BACKBONE}/${COND}/${RUN_ID}/

set -eu -o pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")"/../../.. && pwd)}"
cd "$REPO_ROOT"

LOGS="$REPO_ROOT/runs/corl2026_rebuttal/logs"
mkdir -p "$LOGS"

SEED=${SEED:-0}
MAX_STEPS=${MAX_STEPS:-20000}

launch() {
  local gpu="$1" backbone="$2" condition="$3"
  local run_id="wave1_${backbone}_${condition}_s${SEED}"
  local log="$LOGS/${backbone}_${condition}_s${SEED}.log"
  echo "[wave1] GPU $gpu :: $backbone $condition (seed=$SEED) -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup \
    uv run python "$REPO_ROOT/scripts/train_matched_control.py" \
      --backbone "$backbone" \
      --condition "$condition" \
      --run-id "$run_id" \
      --training-seed "$SEED" \
      --probe-seed 0 \
      --max-steps "$MAX_STEPS" \
    > "$log" 2>&1 &
  echo "  pid=$!"
}

# V-JEPA2 on GPUs 0,1,2
launch 0 vjepa2   true_id
launch 1 vjepa2   shuffled_id
launch 2 vjepa2   forward_dynamics

# VideoMAE on GPUs 3,4,5
launch 3 videomae true_id
launch 4 videomae shuffled_id
launch 5 videomae forward_dynamics

# Web-DINO on GPUs 6,7,8
launch 6 webdino  true_id
launch 7 webdino  shuffled_id
launch 8 webdino  forward_dynamics

echo "[wave1] all launched. tail $LOGS/*.log to follow."
echo "[wave1] use runs/corl2026_rebuttal/launchers/monitor.sh for status summary."
