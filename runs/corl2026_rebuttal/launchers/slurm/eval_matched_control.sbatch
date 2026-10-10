#!/bin/bash
#SBATCH --job-name=corl26-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --partition=<partition>
#SBATCH --time=04:00:00
#SBATCH -o runs/corl2026_rebuttal/logs/slurm-%x-%j.out
#SBATCH -e runs/corl2026_rebuttal/logs/slurm-%x-%j.err

# Canonical action-probe evaluation for one matched-control cell.
#
# Env vars (all mandatory except CKPT which is empty for frozen):
#   REPO_ROOT       absolute repo path
#   REBUTTAL_VENV   path to venv (default ~/anaconda3/envs/dreamer)
#   BACKBONE        vjepa2 | videomae | webdino
#   CONDITION       frozen | true_id | shuffled_id | ...
#   TRAIN_SEED      integer training seed (or -1 for frozen)
#   CKPT            absolute path to ckpt_last.pt, or empty for frozen

set -eu -o pipefail
: "${REPO_ROOT:?}"
: "${BACKBONE:?}"
: "${CONDITION:?}"
: "${TRAIN_SEED:?}"
CKPT="${CKPT:-}"

cd "$REPO_ROOT"
if [[ -n "${REBUTTAL_VENV:-}" ]]; then
  export PATH="$REBUTTAL_VENV/bin:$PATH"
fi
export PYTHONPATH="${LIBERO_SRC:-${LIBERO_SRC:-third_party/LIBERO}}:$REPO_ROOT/src:${PYTHONPATH:-}"
export EXTERNAL_MODELS_DIR="${EXTERNAL_MODELS_DIR:-$REPO_ROOT/datasets/external_models}"

echo "=========================================="
echo "eval matched-control :: $BACKBONE $CONDITION seed=$TRAIN_SEED"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date -Iseconds)"
[[ -n "$CKPT" ]] && echo "CKPT: $CKPT" || echo "CKPT: (frozen, no fine-tune)"
echo "=========================================="

CKPT_ARG=""
[[ -n "$CKPT" ]] && CKPT_ARG="--ckpt $CKPT"

python -u "$REPO_ROOT/scripts/rebuttal/eval_matched_control.py" \
    --backbone "$BACKBONE" \
    --condition "$CONDITION" \
    --training-seed "$TRAIN_SEED" \
    $CKPT_ARG \
    --datasets-dir "$REPO_ROOT/datasets/libero_downloaded" \
    --split-train "$REPO_ROOT/runs/corl2026_rebuttal/splits/train.json" \
    --split-test-ood "$REPO_ROOT/runs/corl2026_rebuttal/splits/test_ood.json" \
    --out-jsonl "$REPO_ROOT/runs/corl2026_rebuttal/results/matched_controls.jsonl"

echo "=========================================="
echo "DONE: $(date -Iseconds)"
echo "=========================================="
