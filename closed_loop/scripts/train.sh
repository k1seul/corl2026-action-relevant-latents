#!/usr/bin/env bash
# Train one row of Table 4.
#   bash closed_loop/scripts/train.sh <row> [seed]
# rows: baseline | clip | clip_shuffle | clip_randinit | vjepa2 | vjepa2_w0.05 | webdino
# Needs: PRETRAINED_PI0 (converted pi0.5 base, see setup_pretrained.sh), LIBERO HDF5s under DATA_ROOT.
# One 48 GB GPU (bf16, gradient checkpointing, batch 32) per run; about 12-24 h for 10k steps.
set -euo pipefail
ROW=${1:?row}; SEED=${2:-0}
cd "$(dirname "$0")/../.."
export PYTHONPATH="src:.:${OPENPI_SRC:-}${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
# NW = DataLoader workers of the original run. Clip start frames are drawn inside the
# workers, so the worker count is part of what a seed reproduces.
case "$ROW" in
  baseline)      NW=4; EXTRA=(--coupling none --end-lr 1.2e-5) ;;   # end LR of the paper's baseline run
  clip)          NW=3; EXTRA=(--coupling clip) ;;
  clip_shuffle)  NW=3; EXTRA=(--coupling clip --shuffle-target) ;;
  clip_randinit) NW=6; EXTRA=(--coupling clip --random-init-target) ;;
  vjepa2)        NW=6; EXTRA=(--coupling token --target vjepa2 --aux-weight 0.01) ;;
  vjepa2_w0.05)  NW=6; EXTRA=(--coupling token --target vjepa2 --aux-weight 0.05) ;;
  webdino)       NW=6; EXTRA=(--coupling token --target webdino --aux-weight 0.01) ;;
  *) echo "unknown row $ROW" >&2; exit 1 ;;
esac
python -m closed_loop.train "${EXTRA[@]}" --seed "$SEED" --num-workers "${NUM_WORKERS:-$NW}" \
  --pretrained-pi0 "${PRETRAINED_PI0:?set PRETRAINED_PI0}" --data-root "${DATA_ROOT:-datasets/libero}" \
  --out-dir "runs/closed_loop/${ROW}_s${SEED}"
