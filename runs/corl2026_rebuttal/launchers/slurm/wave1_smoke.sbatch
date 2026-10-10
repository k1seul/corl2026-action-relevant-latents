#!/bin/bash
#SBATCH --job-name=corl26-w1smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --partition=<partition>
#SBATCH --time=00:30:00
#SBATCH -o runs/corl2026_rebuttal/logs/slurm-%x-%j.out
#SBATCH -e runs/corl2026_rebuttal/logs/slurm-%x-%j.err

# Wave-1 pipeline smoke: 50-step V-JEPA2 true-ID fine-tune on LIBERO.
# Verifies end-to-end data → V-JEPA2 encoder → InvDynAuxHead → loss.backward → step.
#
# Env vars:
#   REPO_ROOT               absolute path
#   REBUTTAL_VENV           optional; path to venv to source
#   LIBERO_SRC              default ${LIBERO_SRC:-third_party/LIBERO}
#   EXTERNAL_MODELS_DIR     default $REPO_ROOT/datasets/external_models
#   LIBERO_DATASETS_DIR     default $REPO_ROOT/datasets/libero_downloaded
#   SPLIT_PATH              default $REPO_ROOT/data/splits_taskheld/train.json
#   MAX_STEPS               default 50

set -eu -o pipefail
: "${REPO_ROOT:?}"
cd "$REPO_ROOT"

if [[ -n "${REBUTTAL_VENV:-}" ]]; then
  export PATH="$REBUTTAL_VENV/bin:$PATH"
fi
export PYTHONPATH="${LIBERO_SRC:-${LIBERO_SRC:-third_party/LIBERO}}:$REPO_ROOT/src:${PYTHONPATH:-}"
export EXTERNAL_MODELS_DIR="${EXTERNAL_MODELS_DIR:-$REPO_ROOT/datasets/external_models}"

LIBERO_DATASETS_DIR="${LIBERO_DATASETS_DIR:-$REPO_ROOT/datasets/libero_downloaded}"
SPLIT_PATH="${SPLIT_PATH:-$REPO_ROOT/data/splits_taskheld/train.json}"
MAX_STEPS="${MAX_STEPS:-50}"

OUT_DIR="$REPO_ROOT/runs/corl2026_rebuttal/checkpoints/vjepa2/smoke_true_id"
mkdir -p "$OUT_DIR" "$REPO_ROOT/runs/corl2026_rebuttal/logs"

echo "=========================================="
echo "wave1 smoke :: V-JEPA2 true_id 50 steps"
echo "Job:         $SLURM_JOB_ID   Node: $(hostname)   Partition: $SLURM_JOB_PARTITION"
echo "GPU:         $CUDA_VISIBLE_DEVICES"
echo "REPO_ROOT:   $REPO_ROOT"
echo "LIBERO_DIR:  $LIBERO_DATASETS_DIR"
echo "SPLIT_PATH:  $SPLIT_PATH"
echo "CKPT:        $EXTERNAL_MODELS_DIR/vjepa2_ckpts/vitl.pt"
echo "MAX_STEPS:   $MAX_STEPS"
echo "Date:        $(date -Iseconds)"
echo "=========================================="

python "$REPO_ROOT/scripts/train_vjepa2_aux.py" \
    --dataset libero \
    --libero-datasets-dir "$LIBERO_DATASETS_DIR" \
    --libero-split-path "$SPLIT_PATH" \
    --ckpt "$EXTERNAL_MODELS_DIR/vjepa2_ckpts/vitl.pt" \
    --out-dir "$OUT_DIR" \
    --max-steps "$MAX_STEPS" \
    --warmup-steps 20 \
    --batch-size 2 \
    --num-workers 2 \
    --lr 5e-5 \
    --log-interval 5

echo "=========================================="
echo "SMOKE COMPLETE: $(date -Iseconds)"
echo "final ckpt: $OUT_DIR/ckpt_last.pt"
ls -la "$OUT_DIR"
echo "=========================================="

# Post-smoke assertions (soft; nonzero exit means the pipeline did NOT
# actually train, i.e. a REAL failure that squeue's COMPLETED would mask).
python - <<'PY'
import sys, torch
from pathlib import Path
ckpt = Path("runs/corl2026_rebuttal/checkpoints/vjepa2/smoke_true_id/ckpt_last.pt")
assert ckpt.exists(), f"ckpt not saved: {ckpt}"
sd = torch.load(ckpt, map_location="cpu", weights_only=False)
assert "encoder" in sd and "aux_head" in sd and "step" in sd
step = int(sd["step"])
assert step > 0, f"smoke exited before any optimizer step (step={step})"
# quick sanity: encoder & aux_head produced finite state_dicts
for name in ("encoder", "aux_head"):
    for k, v in sd[name].items():
        if torch.is_tensor(v):
            assert torch.isfinite(v).all(), f"NaN/Inf in {name}.{k}"
print(f"smoke assertions passed. step_reached={step}, encoder_tensors={len(sd['encoder'])}, aux_head_tensors={len(sd['aux_head'])}")
PY
