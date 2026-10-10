#!/usr/bin/env bash
# Phase 2 launcher: train nicklashansen/dreamer4 tokenizer on LIBERO shards.
#
# Prerequisites:
#   1. external_models/dreamer4 cloned (see Phase 0).
#   2. scripts/preprocess_libero_for_dreamer4.py run on train.json — produces
#      data/libero_dreamer4/frames/<task>/*.pt and libero_tasks_list.json.
#
# Defaults (matching nicklashansen's published 90k tokenizer config):
#   d_model=256 n_heads=4 depth=8 n_latents=16 d_bottleneck=32
#   patch=4 H=W=128 C=3 mae_p_max=0.9
#   batch_size 8 per GPU, lr 1e-4, 90k steps
#
# Usage:
#   bash scripts/train_dreamer4_tokenizer_libero.sh
#   bash scripts/train_dreamer4_tokenizer_libero.sh 4              # 4 GPUs
#   NPROC=2 OUT=runs/dreamer4_libero/tokenizer_v2 bash <this>      # override
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT=$(pwd)

NPROC="${NPROC:-${1:-8}}"
D4_REPO="${D4_REPO:-external_models/dreamer4}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/libero_dreamer4/frames}"
TASKS_JSON="${TASKS_JSON:-$REPO_ROOT/data/libero_dreamer4/libero_tasks_list.json}"
OUT="${OUT:-$REPO_ROOT/runs/dreamer4_libero/tokenizer}"

# Build the flat task list JSON if it does not yet exist (consumes
# preprocess output's libero_tasks.json keys).
if [ ! -f "$TASKS_JSON" ]; then
  echo "building task list at $TASKS_JSON"
  uv run python - <<PY
import json, pathlib
src = pathlib.Path("$REPO_ROOT/data/libero_dreamer4/libero_tasks.json")
if not src.exists():
    raise SystemExit(f"missing {src}; run scripts/preprocess_libero_for_dreamer4.py first")
tasks = sorted(json.loads(src.read_text()).keys())
out = pathlib.Path("$TASKS_JSON")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(tasks))
print(f"wrote {len(tasks)} tasks -> {out}")
PY
fi

mkdir -p "$OUT"
export D4_TASK_SET="$TASKS_JSON"
export PYTHONUNBUFFERED=1
# Source .env for WANDB_API_KEY + WANDB_ENTITY. Override with
# WANDB_MODE=disabled to skip wandb syncing for a single run.
if [ -f "$REPO_ROOT/.env" ]; then
  set -a; source "$REPO_ROOT/.env"; set +a
fi
export WANDB_MODE="${WANDB_MODE:-online}"
# train_tokenizer.py imports `task_set`, `sharded_frame_dataset`, `model`
# from its own dir — add it to PYTHONPATH so we can launch from dreamer-vla.
export PYTHONPATH="$D4_REPO/dreamer4:${PYTHONPATH:-}"

cd "$REPO_ROOT"
exec uv run torchrun \
  --nproc_per_node="$NPROC" \
  "$D4_REPO/dreamer4/train_tokenizer.py" \
  --data_dirs "$DATA_DIR" \
  --batch_size 8 \
  --seq_len 8 \
  --max_steps 90000 \
  --ckpt_dir "$OUT" \
  --wandb_run_name libero_tokenizer \
  --wandb_project dreamer4-libero-tokenizer \
  "$@"
