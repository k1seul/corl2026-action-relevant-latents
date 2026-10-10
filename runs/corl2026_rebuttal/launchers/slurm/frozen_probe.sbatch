#!/bin/bash
#SBATCH --job-name=corl26-fp
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --partition=<partition>
#SBATCH --time=12:00:00
#SBATCH -o runs/corl2026_rebuttal/logs/slurm-%x-%j.out
#SBATCH -e runs/corl2026_rebuttal/logs/slurm-%x-%j.err

# Frozen-encoder canonical action probe (Phase 0 + Phase 3 baseline path).
# One sbatch = one extractor.
#
# Env vars:
#   EXTRACTOR   vjepa2 | videomae | webdino | siglip2 | lapa | sdxl_vae |
#               cosmos1 | r3m | vip | liv | caip_visual_only | caip_native
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
echo "Frozen probe :: $EXTRACTOR"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date -Iseconds)"
echo "=========================================="

uv run python "$REPO_ROOT/scripts/full_results_pipeline.py" \
    --backbone "$EXTRACTOR" \
    --frozen \
    --out-dir "$REPO_ROOT/runs/corl2026_rebuttal/results/frozen_probe_${EXTRACTOR}"

echo "=========================================="
echo "Complete: $(date -Iseconds)"
echo "=========================================="
