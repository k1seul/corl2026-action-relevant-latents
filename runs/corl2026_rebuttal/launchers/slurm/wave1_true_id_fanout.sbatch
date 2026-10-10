#!/bin/bash
#SBATCH --job-name=corl26-w1trueid
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --partition=<partition>
#SBATCH --time=24:00:00
#SBATCH -o runs/corl2026_rebuttal/logs/slurm-%x-%j.out
#SBATCH -e runs/corl2026_rebuttal/logs/slurm-%x-%j.err

# Wave-1 primary true_id fine-tune. One sbatch = one (backbone, training_seed)
# cell. Only true_id is supported by the current train_*.py scripts;
# shuffled_id (derangement) and forward_dynamics require the dataloader
# extensions documented in TRAIN_SCRIPT_PATCH_NOTES.md and are held for
# session-3.
#
# Env vars:
#   BACKBONE            vjepa2 | videomae | webdino
#   TRAIN_SEED          integer
#   MAX_STEPS           default 20000
#   REPO_ROOT / REBUTTAL_VENV / LIBERO_SRC / EXTERNAL_MODELS_DIR
#   LIBERO_DATASETS_DIR / LIBERO_SPLIT_PATH   (default in-repo)

set -eu -o pipefail
: "${REPO_ROOT:?}"
: "${BACKBONE:?}"
: "${TRAIN_SEED:?}"
MAX_STEPS="${MAX_STEPS:-20000}"

cd "$REPO_ROOT"
if [[ -n "${REBUTTAL_VENV:-}" ]]; then
  export PATH="$REBUTTAL_VENV/bin:$PATH"
fi
export PYTHONPATH="${LIBERO_SRC:-${LIBERO_SRC:-third_party/LIBERO}}:$REPO_ROOT/src:${PYTHONPATH:-}"
export EXTERNAL_MODELS_DIR="${EXTERNAL_MODELS_DIR:-$REPO_ROOT/datasets/external_models}"
export LIBERO_DATASETS_DIR="${LIBERO_DATASETS_DIR:-$REPO_ROOT/datasets/libero_downloaded}"
export LIBERO_SPLIT_PATH="${LIBERO_SPLIT_PATH:-$REPO_ROOT/data/splits_taskheld/train.json}"

RUN_ID="wave1_true_id_${BACKBONE}_s${TRAIN_SEED}"
OUT_DIR="$REPO_ROOT/runs/corl2026_rebuttal/checkpoints/${BACKBONE}/true_id/${RUN_ID}"
mkdir -p "$OUT_DIR"

# Record provenance BEFORE training begins.
python - <<PY
import sys
sys.path.insert(0, "$REPO_ROOT/src")
from rebuttal.provenance import record_provenance
record_provenance(
    "$OUT_DIR",
    condition="true_id",
    backbone="$BACKBONE",
    run_id="$RUN_ID",
    training_seed=int("$TRAIN_SEED"),
    probe_seed=0,
    optimizer="adamw",
    lr=5e-5,
    lr_schedule="cosine_warmup2000",
    batch_size=2 if "$BACKBONE" in ("videomae", "vjepa2") else 4,
    effective_batch=2 if "$BACKBONE" in ("videomae", "vjepa2") else 4,
    target_updates=int("$MAX_STEPS"),
    split_paths={
        "train": "$REPO_ROOT/runs/corl2026_rebuttal/splits/train.json",
        "test_ood": "$REPO_ROOT/runs/corl2026_rebuttal/splits/test_ood.json",
        "test_indist": "$REPO_ROOT/runs/corl2026_rebuttal/splits/test_indist.json",
    },
    checkpoint_paths={
        "starting_frozen": (
            "$REPO_ROOT/datasets/external_models/vjepa2_ckpts/vitl.pt" if "$BACKBONE"=="vjepa2" else
            "$REPO_ROOT/datasets/external_models/videomae" if "$BACKBONE"=="videomae" else
            "$REPO_ROOT/datasets/external_models/webdino"
        )
    },
    extra={"note": "wave1 true_id fanout — reproduces submitted +ID matrix with re-generated splits"},
)
PY

echo "=========================================="
echo "wave1 true_id fanout :: $BACKBONE seed=$TRAIN_SEED"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "OUT_DIR: $OUT_DIR"
echo "Date: $(date -Iseconds)"
echo "=========================================="

case "$BACKBONE" in
  vjepa2)
    # bs=2 because bs=4 with 256px×32-frame clips exceeded 96GB cgroup;
    # smoke at bs=2 was proven fine at 96GB. mem bumped to 128GB for margin.
    python -u "$REPO_ROOT/scripts/train_vjepa2_aux.py" \
      --dataset libero \
      --libero-datasets-dir "$LIBERO_DATASETS_DIR" \
      --libero-split-path "$LIBERO_SPLIT_PATH" \
      --ckpt "$EXTERNAL_MODELS_DIR/vjepa2_ckpts/vitl.pt" \
      --out-dir "$OUT_DIR" \
      --max-steps "$MAX_STEPS" \
      --seed "$TRAIN_SEED" \
      --batch-size 2 --num-workers 2 --lr 5e-5 \
      --log-interval 200
    ;;
  videomae)
    python "$REPO_ROOT/scripts/train_videomae_aux_libero.py" \
      --dataset libero \
      --out-dir "$OUT_DIR" \
      --max-steps "$MAX_STEPS" \
      --seed "$TRAIN_SEED" \
      --batch-size 2 --num-workers 2 --lr 5e-5 \
      --log-interval 200
    ;;
  webdino)
    python "$REPO_ROOT/scripts/train_webdino_aux_libero.py" \
      --dataset libero \
      --out-dir "$OUT_DIR" \
      --max-steps "$MAX_STEPS" \
      --seed "$TRAIN_SEED" \
      --batch-size 4 --num-workers 2 --lr 5e-5 \
      --log-interval 200
    ;;
  *)
    echo "unknown backbone: $BACKBONE"
    exit 2
    ;;
esac

# Finalize provenance
python - <<PY
import sys, torch, os
sys.path.insert(0, "$REPO_ROOT/src")
from rebuttal.provenance import finalize_provenance
ckpt_path = os.path.join("$OUT_DIR", "ckpt_last.pt")
sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
finalize_provenance("$OUT_DIR", final_checkpoint_step=int(sd.get("step", 0)))
PY

echo "=========================================="
echo "DONE: $(date -Iseconds)"
echo "=========================================="
