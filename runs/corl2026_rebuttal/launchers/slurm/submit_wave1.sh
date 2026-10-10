#!/bin/bash
# Wave 1 submitter — enqueues the 9 primary matched-control jobs onto A6000
# partition, then optionally the second-seed batch.
#
# Env vars (override at call site):
#   REPO_ROOT           absolute path (required)
#   PART_A6000          Slurm partition name for A6000s (default: <partition>)
#   SEEDS               space-separated list of training seeds (default: "0 1")
#   MAX_STEPS           step count for every job (default: 20000)
#   REBUTTAL_VENV       path to venv to source (optional)
#
# Usage:
#   REPO_ROOT=$(pwd) bash runs/corl2026_rebuttal/launchers/slurm/submit_wave1.sh
#
# The script prints the sbatch submission command per job and captures job
# IDs into runs/corl2026_rebuttal/logs/wave1_job_ids.txt.

set -eu -o pipefail

: "${REPO_ROOT:?REPO_ROOT must be set}"
PART_A6000="${PART_A6000:-<partition>}"
SEEDS="${SEEDS:-0 1}"
MAX_STEPS="${MAX_STEPS:-20000}"

TEMPLATE="$REPO_ROOT/runs/corl2026_rebuttal/launchers/slurm/train_matched_control.sbatch"
JOBID_FILE="$REPO_ROOT/runs/corl2026_rebuttal/logs/wave1_job_ids.txt"
mkdir -p "$(dirname "$JOBID_FILE")"
: > "$JOBID_FILE"

declare -A BATCH_SIZE=([vjepa2]=4 [videomae]=2 [webdino]=4)
declare -A LR=([vjepa2]=5e-5 [videomae]=5e-5 [webdino]=5e-5)

# PRIMARY: shuffled_id is the within-task cross-episode derangement.
# We pass SHUFFLE_MODE explicitly so a hypothetical env inheritance from the
# caller cannot silently downgrade the control to circular_shift.
for SEED in $SEEDS; do
  for BB in vjepa2 videomae webdino; do
    for COND in true_id shuffled_id forward_dynamics; do
      RUN_ID="wave1_${BB}_${COND}_s${SEED}"
      BS="${BATCH_SIZE[$BB]}"
      LR_VAL="${LR[$BB]}"
      NAME="corl26-mc-${BB}-${COND}-s${SEED}"
      SHUFFLE_MODE=within_task_cross_episode_derangement
      echo "[submit_wave1] sbatch $NAME  (SHUFFLE_MODE=$SHUFFLE_MODE)"
      CMD=(sbatch --partition="$PART_A6000" --job-name="$NAME"
           --export=ALL,REPO_ROOT="$REPO_ROOT",BACKBONE="$BB",CONDITION="$COND",TRAIN_SEED="$SEED",PROBE_SEED=0,RUN_ID="$RUN_ID",MAX_STEPS="$MAX_STEPS",BATCH_SIZE="$BS",LR="$LR_VAL",SHUFFLE_MODE="$SHUFFLE_MODE",REBUTTAL_VENV="${REBUTTAL_VENV:-}"
           "$TEMPLATE")
      OUT=$("${CMD[@]}")
      echo "  $OUT"
      echo "$OUT | $NAME | $RUN_ID" >> "$JOBID_FILE"
    done
  done
done

echo "[submit_wave1] all jobs submitted. IDs → $JOBID_FILE"
echo "[submit_wave1] follow with: squeue -u \$USER   or   tail -f runs/corl2026_rebuttal/logs/slurm-*.out"
