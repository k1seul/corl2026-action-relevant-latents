#!/usr/bin/env bash
# Phase 6 launcher: Dreamer-V4 + inverse-dynamics aux on LIBERO.
#
# Patched-in changes (dreamer-vla aux patch, see commit history of
# external_models/dreamer4):
#   1. Dynamics.forward gains `return_spatial_hidden=True` kwarg.
#   2. dynamics_pretrain_loss propagates spatial_hidden through aux dict.
#   3. train_dynamics.py builds InvDynAuxHead (from external_models/dreamer4/dreamer4/aux_loss.py)
#      when --aux_invdyn_lambda > 0; adds λ * MSE(pred_action, true_action[:, :7])
#      to the total loss.
#   4. aux_head saved to aux_head.pt (separate from latest.pt) — latest.pt is
#      used for evaluation and remains aux-head-free, parallel to our DIFF/V-JEPA
#      protocol.
#
# Defaults: λ=0.05 (LIBERO sweet spot from our DIFF sweep, paper §3.7).
#           Same training schedule as nicklashansen Phase 3 (40k steps).
#
# Prerequisites:
#   - Phase 2 complete: runs/dreamer4_libero/tokenizer/latest.pt
#   - Phase 1 complete: data/libero_dreamer4/{frames,demos}/
#   - external_models/dreamer4 patched (this script's effects depend on patches
#     applied at 2026-05-11; see runs/dreamer4_libero/README.md changelog).
#
# Usage:
#   bash scripts/train_dreamer4_aux_libero.sh
#   AUX_LAMBDA=0.025 OUT=runs/dreamer4_libero/dynamics_aux_lambda0p025 \
#       bash scripts/train_dreamer4_aux_libero.sh           # λ sweep
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT=$(pwd)

NPROC="${NPROC:-${1:-8}}"
AUX_LAMBDA="${AUX_LAMBDA:-0.05}"
D4_REPO="${D4_REPO:-external_models/dreamer4}"
DEMOS_DIR="${DEMOS_DIR:-$REPO_ROOT/data/libero_dreamer4/demos}"
FRAMES_DIR="${FRAMES_DIR:-$REPO_ROOT/data/libero_dreamer4/frames}"
TASKS_JSON="${TASKS_JSON:-$REPO_ROOT/data/libero_dreamer4/libero_tasks_list.json}"
TASKS_META="${TASKS_META:-$REPO_ROOT/data/libero_dreamer4/libero_tasks.json}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-$REPO_ROOT/runs/dreamer4_libero/tokenizer/latest.pt}"
OUT="${OUT:-$REPO_ROOT/runs/dreamer4_libero/dynamics_aux_lambda${AUX_LAMBDA/./p}}"

# Sanity: confirm the patch is applied. The marker is `--aux_invdyn_lambda`
# in train_dynamics.py argparse. If absent, this script's flags will error.
if ! grep -q "aux_invdyn_lambda" "$D4_REPO/dreamer4/train_dynamics.py"; then
  echo "ERROR: $D4_REPO/dreamer4/train_dynamics.py does not contain the aux" >&2
  echo "patch. Re-apply the dreamer-vla aux patch and try again." >&2
  exit 3
fi

if [ ! -f "$TOKENIZER_CKPT" ]; then
  echo "ERROR: tokenizer ckpt missing at $TOKENIZER_CKPT" >&2
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
  --aux_invdyn_lambda "$AUX_LAMBDA" \
  --aux_action_dim 7 \
  --wandb_run_name "libero_dynamics_aux_lambda${AUX_LAMBDA}" \
  --wandb_project dreamer4-libero-dynamics \
  "$@"
