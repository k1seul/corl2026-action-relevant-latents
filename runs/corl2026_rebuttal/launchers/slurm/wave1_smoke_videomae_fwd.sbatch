#!/bin/bash
#SBATCH --job-name=corl26-w1smoke-vmae-fwd
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --partition=<partition>
#SBATCH --time=00:20:00
#SBATCH -o runs/corl2026_rebuttal/logs/slurm-%x-%j.out
#SBATCH -e runs/corl2026_rebuttal/logs/slurm-%x-%j.err

# 50-step VideoMAE forward_dynamics smoke. Verifies the STEP 4
# teacher-target formulation runs end-to-end.
set -eu -o pipefail
: "${REPO_ROOT:?}"
cd "$REPO_ROOT"
if [[ -n "${REBUTTAL_VENV:-}" ]]; then
  export PATH="$REBUTTAL_VENV/bin:$PATH"
fi
export PYTHONPATH="${LIBERO_SRC:-${LIBERO_SRC:-third_party/LIBERO}}:$REPO_ROOT/src:${PYTHONPATH:-}"
export EXTERNAL_MODELS_DIR="${EXTERNAL_MODELS_DIR:-$REPO_ROOT/datasets/external_models}"
export LIBERO_DATASETS_DIR="$REPO_ROOT/datasets/libero_downloaded"
export LIBERO_SPLIT_PATH="$REPO_ROOT/data/splits_taskheld/train.json"

export REBUTTAL_CONDITION=forward_dynamics
export REBUTTAL_RUN_SEED=0

OUT_DIR="$REPO_ROOT/runs/corl2026_rebuttal/checkpoints/videomae/smoke_forward_dynamics"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo "videomae forward_dynamics smoke (50 steps)"
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date -Iseconds)"
echo "=========================================="

python -u "$REPO_ROOT/scripts/train_videomae_aux_libero.py" \
    --dataset libero \
    --out-dir "$OUT_DIR" \
    --max-steps 50 --warmup-steps 20 \
    --batch-size 2 --num-workers 2 --lr 5e-5 \
    --log-interval 5

echo "=========================================="
echo "SMOKE COMPLETE: $(date -Iseconds)"
ls -la "$OUT_DIR"
python - <<'PY'
import torch, os
from pathlib import Path
p = Path("runs/corl2026_rebuttal/checkpoints/videomae/smoke_forward_dynamics/ckpt_last.pt")
sd = torch.load(str(p), map_location="cpu", weights_only=False)
step = int(sd.get("step", 0))
assert step > 0, f"fwd-dyn smoke exited before optimizer step: step={step}"
for name in ("encoder", "aux_head"):
    obj = sd.get(name)
    if isinstance(obj, dict):
        for k, v in obj.items():
            if torch.is_tensor(v):
                assert torch.isfinite(v).all(), f"NaN/Inf in {name}.{k}"
print(f"fwd-dyn smoke PASS step={step} encoder_tensors={len(sd['encoder'])} aux_head_tensors={len(sd['aux_head'])}")
PY
