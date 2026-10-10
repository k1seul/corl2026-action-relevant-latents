#!/bin/bash
#SBATCH --job-name=corl26-mc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --partition=<partition>
#SBATCH --time=48:00:00
#SBATCH -o runs/corl2026_rebuttal/logs/slurm-%x-%j.out
#SBATCH -e runs/corl2026_rebuttal/logs/slurm-%x-%j.err

# Matched-control training job (Phase 1 primary).
# One sbatch = one (backbone × condition × seed) cell.
#
# Env vars passed via sbatch --export=ALL,BACKBONE=...,CONDITION=...,...:
#   BACKBONE        vjepa2 | videomae | webdino
#   CONDITION       true_id | shuffled_id | forward_dynamics | action_free_adaptation
#   TRAIN_SEED      integer (predeclared in matched_controls.yaml)
#   PROBE_SEED      integer (default 0)
#   RUN_ID          unique identifier (predeclared)
#   MAX_STEPS       default 20000
#   BATCH_SIZE      per-backbone from matched_controls.yaml
#   LR              per-backbone from matched_controls.yaml
#   REPO_ROOT       absolute path to the cloned rebuttal repo

set -eu -o pipefail

: "${REPO_ROOT:?REPO_ROOT must be set (absolute path to the rebuttal repo)}"
: "${BACKBONE:?BACKBONE must be set}"
: "${CONDITION:?CONDITION must be set}"
: "${TRAIN_SEED:?TRAIN_SEED must be set}"
: "${RUN_ID:?RUN_ID must be set}"
PROBE_SEED="${PROBE_SEED:-0}"
MAX_STEPS="${MAX_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-5e-5}"

mkdir -p "$REPO_ROOT/runs/corl2026_rebuttal/logs"
cd "$REPO_ROOT"

# venv path is intentionally NOT hard-coded; use the venv the rest of the
# repo already uses, or set REBUTTAL_VENV before submission.
if [[ -n "${REBUTTAL_VENV:-}" ]]; then
  export PATH="$REBUTTAL_VENV/bin:$PATH"
fi
# LIBERO is installed as a nested-namespace package at ${LIBERO_SRC:-third_party/LIBERO}
# (nested libero/libero/ layout). Editable-install PYTHONPATH is required.
export PYTHONPATH="${LIBERO_SRC:-${LIBERO_SRC:-third_party/LIBERO}}:$REPO_ROOT/src:${PYTHONPATH:-}"
export EXTERNAL_MODELS_DIR="${EXTERNAL_MODELS_DIR:-$REPO_ROOT/datasets/external_models}"

echo "=========================================="
echo "CoRL 2026 matched-control :: $BACKBONE $CONDITION seed=$TRAIN_SEED"
echo "RUN_ID:      $RUN_ID"
echo "Node:        $(hostname)   GPU: $CUDA_VISIBLE_DEVICES"
echo "Job:         $SLURM_JOB_ID   Partition: $SLURM_JOB_PARTITION"
echo "Date start:  $(date -Iseconds)"
echo "Repo:        $REPO_ROOT"
echo "=========================================="

SHUFFLE_MODE="${SHUFFLE_MODE:-within_task_cross_episode_derangement}"

python "$REPO_ROOT/scripts/train_matched_control.py" \
    --backbone "$BACKBONE" \
    --condition "$CONDITION" \
    --run-id "$RUN_ID" \
    --training-seed "$TRAIN_SEED" \
    --probe-seed "$PROBE_SEED" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --shuffle-mode "$SHUFFLE_MODE"

echo "=========================================="
echo "Complete: $(date -Iseconds)"
echo "=========================================="
