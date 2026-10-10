#!/bin/bash
#SBATCH --job-name=corl26-cd
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --partition=<partition>
#SBATCH --time=24:00:00
#SBATCH -o runs/corl2026_rebuttal/logs/slurm-%x-%j.out
#SBATCH -e runs/corl2026_rebuttal/logs/slurm-%x-%j.err

# Common frozen pixel decoder (Phase 4). One sbatch = one extractor,
# identical decoder architecture / schedule / seed policy across extractors.
#
# Env vars:
#   EXTRACTOR   as in frozen_probe.sbatch
#   REPO_ROOT   absolute path

set -eu -o pipefail
: "${REPO_ROOT:?}"
: "${EXTRACTOR:?}"

mkdir -p "$REPO_ROOT/runs/corl2026_rebuttal/logs"
cd "$REPO_ROOT"
if [[ -n "${REBUTTAL_VENV:-}" ]]; then
  export PATH="$REBUTTAL_VENV/bin:$PATH"
fi
export PYTHONPATH="${LIBERO_SRC:-${LIBERO_SRC:-third_party/LIBERO}}:$REPO_ROOT/src:${PYTHONPATH:-}"
export EXTERNAL_MODELS_DIR="${EXTERNAL_MODELS_DIR:-$REPO_ROOT/datasets/external_models}"

echo "=========================================="
echo "Common decoder :: $EXTRACTOR"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date -Iseconds)"
echo "=========================================="

uv run python "$REPO_ROOT/scripts/train_pixel_decoder.py" \
    --extractor "$EXTRACTOR" \
    --frozen \
    --out-dir "$REPO_ROOT/runs/corl2026_rebuttal/results/common_decoder_${EXTRACTOR}"

echo "=========================================="
echo "Complete: $(date -Iseconds)"
echo "=========================================="
