#!/usr/bin/env bash
# Phase 3 launcher: train Dreamer 4 dynamics model on LIBERO with --use_actions.
#
# Prerequisites:
#   - Phase 2 complete: runs/dreamer4_libero/tokenizer/latest.pt exists
#   - Phase 1 complete: data/libero_dreamer4/frames/* and demos/*.pt exist
#
# Defaults match nicklashansen's published 40k dynamics config (DMControl).
# LIBERO is a different domain — flag any divergence from the published loss
# curves in runs/dreamer4_libero/dynamics/README.
#
# Usage:
#   bash scripts/train_dreamer4_dynamics_libero.sh
#   NPROC=4 bash scripts/train_dreamer4_dynamics_libero.sh
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT=$(pwd)

NPROC="${NPROC:-${1:-8}}"
D4_REPO="${D4_REPO:-external_models/dreamer4}"
DEMOS_DIR="${DEMOS_DIR:-$REPO_ROOT/data/libero_dreamer4/demos}"
FRAMES_DIR="${FRAMES_DIR:-$REPO_ROOT/data/libero_dreamer4/frames}"
TASKS_JSON="${TASKS_JSON:-$REPO_ROOT/data/libero_dreamer4/libero_tasks_list.json}"
TASKS_META="${TASKS_META:-$REPO_ROOT/data/libero_dreamer4/libero_tasks.json}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-$REPO_ROOT/runs/dreamer4_libero/tokenizer/latest.pt}"
OUT="${OUT:-$REPO_ROOT/runs/dreamer4_libero/dynamics}"

if [ ! -f "$TOKENIZER_CKPT" ]; then
  echo "ERROR: tokenizer ckpt missing at $TOKENIZER_CKPT" >&2
  echo "Run scripts/train_dreamer4_tokenizer_libero.sh first." >&2
  exit 1
fi

mkdir -p "$OUT"
export D4_TASK_SET="$TASKS_JSON"
export PYTHONUNBUFFERED=1
if [ -f "$REPO_ROOT/.env" ]; then
  set -a; source "$REPO_ROOT/.env"; set +a
fi
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONPATH="$D4_REPO/dreamer4:${PYTHONPATH:-}"

cd "$REPO_ROOT"
exec uv run torchrun \
  --nproc_per_node="$NPROC" \
  "$D4_REPO/dreamer4/train_dynamics.py" \
  --use_actions \
  --data_dirs "$DEMOS_DIR" \
  --frame_dirs "$FRAMES_DIR" \
  --tasks_json "$TASKS_META" \
  --tokenizer_ckpt "$TOKENIZER_CKPT" \
  --ckpt_dir "$OUT" \
  --max_steps 40000 \
  --wandb_run_name libero_dynamics \
  --wandb_project dreamer4-libero-dynamics \
  "$@"
