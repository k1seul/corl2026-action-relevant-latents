#!/bin/bash
#SBATCH --job-name=corl26-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --partition=<partition>
#SBATCH --time=00:05:00
#SBATCH -o runs/corl2026_rebuttal/logs/slurm-%x-%j.out
#SBATCH -e runs/corl2026_rebuttal/logs/slurm-%x-%j.err

# Infra smoke — confirms the sbatch pipeline works end-to-end:
#   * A6000 partition is reachable
#   * dreamer conda env activates
#   * torch imports and sees CUDA
#   * repo is on the compute node's filesystem
# NO training, NO data access. Used only to validate the launch path.
#
# Submit with:
#   REPO_ROOT=$(pwd) sbatch --export=ALL,REPO_ROOT="$REPO_ROOT",REBUTTAL_VENV=$HOME/anaconda3/envs/dreamer \
#     runs/corl2026_rebuttal/launchers/slurm/sbatch_smoke.sbatch

set -eu -o pipefail
: "${REPO_ROOT:?REPO_ROOT must be set}"
cd "$REPO_ROOT"

if [[ -n "${REBUTTAL_VENV:-}" ]]; then
  export PATH="$REBUTTAL_VENV/bin:$PATH"
fi
export PYTHONPATH="${LIBERO_SRC:-${LIBERO_SRC:-third_party/LIBERO}}:$REPO_ROOT/src:${PYTHONPATH:-}"
export EXTERNAL_MODELS_DIR="${EXTERNAL_MODELS_DIR:-$REPO_ROOT/datasets/external_models}"

echo "=========================================="
echo "corl2026 rebuttal sbatch infra smoke"
echo "Node:        $(hostname)"
echo "Job:         $SLURM_JOB_ID   Partition: $SLURM_JOB_PARTITION"
echo "GPU env:     ${CUDA_VISIBLE_DEVICES:-}"
echo "Date:        $(date -Iseconds)"
echo "Python:      $(command -v python)  $(python --version 2>&1)"
echo "=========================================="
nvidia-smi -L
python - <<'PY'
import json, socket, torch, os
info = dict(
    hostname=socket.gethostname(),
    torch=torch.__version__,
    cuda_available=torch.cuda.is_available(),
    gpu_count=torch.cuda.device_count(),
    gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
    slurm_job=os.environ.get("SLURM_JOB_ID"),
    partition=os.environ.get("SLURM_JOB_PARTITION"),
)
print(json.dumps(info, indent=2))
PY
echo "=========================================="
echo "Smoke complete: $(date -Iseconds)"
echo "=========================================="
