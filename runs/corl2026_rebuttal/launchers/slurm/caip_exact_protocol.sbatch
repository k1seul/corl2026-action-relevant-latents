#!/bin/bash
#SBATCH --job-name=corl26-caip-exact-protocol
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
#SBATCH --partition=<partition>
#SBATCH --time=06:00:00
#SBATCH -o runs/corl2026_rebuttal/logs/slurm-%x-%j.out
#SBATCH -e runs/corl2026_rebuttal/logs/slurm-%x-%j.err

set -eu -o pipefail
: "${REPO_ROOT:?}"

mkdir -p "$REPO_ROOT/runs/corl2026_rebuttal/logs"
cd "$REPO_ROOT"
export PATH="$HOME/anaconda3/envs/dreamer/bin:$PATH"
export PYTHONPATH="${LIBERO_SRC:-${LIBERO_SRC:-third_party/LIBERO}}:$REPO_ROOT/src:${PYTHONPATH:-}"
export EXTERNAL_MODELS_DIR="${EXTERNAL_MODELS_DIR:-$REPO_ROOT/datasets/external_models}"

echo "=========================================="
echo "CAIP exact-protocol final action-centric baseline"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date -Iseconds)"
echo "=========================================="

python "$REPO_ROOT/scripts/rebuttal/caip_exact_protocol.py"

echo "=========================================="
echo "Complete: $(date -Iseconds)"
echo "=========================================="
