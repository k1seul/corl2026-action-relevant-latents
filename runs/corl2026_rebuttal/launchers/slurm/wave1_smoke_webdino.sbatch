#!/bin/bash
#SBATCH --job-name=corl26-w1smoke-wdino
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --partition=<partition>
#SBATCH --time=00:30:00
#SBATCH -o runs/corl2026_rebuttal/logs/slurm-%x-%j.out
#SBATCH -e runs/corl2026_rebuttal/logs/slurm-%x-%j.err

# Wave-1 smoke for Web-DINO (50 steps).
set -eu -o pipefail
: "${REPO_ROOT:?}"
cd "$REPO_ROOT"

if [[ -n "${REBUTTAL_VENV:-}" ]]; then
  export PATH="$REBUTTAL_VENV/bin:$PATH"
fi
export PYTHONPATH="${LIBERO_SRC:-${LIBERO_SRC:-third_party/LIBERO}}:$REPO_ROOT/src:${PYTHONPATH:-}"
export EXTERNAL_MODELS_DIR="${EXTERNAL_MODELS_DIR:-$REPO_ROOT/datasets/external_models}"
export LIBERO_DATASETS_DIR="${LIBERO_DATASETS_DIR:-$REPO_ROOT/datasets/libero_downloaded}"
export LIBERO_SPLIT_PATH="${LIBERO_SPLIT_PATH:-$REPO_ROOT/data/splits_taskheld/train.json}"

OUT_DIR="$REPO_ROOT/runs/corl2026_rebuttal/checkpoints/webdino/smoke_true_id"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo "wave1 smoke :: Web-DINO true_id 50 steps"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date -Iseconds)"
echo "=========================================="

python "$REPO_ROOT/scripts/train_webdino_aux_libero.py" \
    --dataset libero \
    --out-dir "$OUT_DIR" \
    --max-steps 50 \
    --warmup-steps 20 \
    --batch-size 2 \
    --num-workers 2 \
    --lr 5e-5 \
    --log-interval 5

echo "=========================================="
echo "SMOKE COMPLETE: $(date -Iseconds)"
ls -la "$OUT_DIR"
python - <<'PY'
import torch
from pathlib import Path
ckpt = Path("runs/corl2026_rebuttal/checkpoints/webdino/smoke_true_id/ckpt_last.pt")
assert ckpt.exists(), f"ckpt not saved: {ckpt}"
sd = torch.load(ckpt, map_location="cpu", weights_only=False)
assert int(sd.get("step", 0)) > 0
for name, obj in sd.items():
    if isinstance(obj, dict):
        for k, v in obj.items():
            if torch.is_tensor(v):
                assert torch.isfinite(v).all(), f"NaN/Inf in {name}.{k}"
print(f"webdino smoke assertions PASSED. step={sd['step']}")
PY
