#!/usr/bin/env bash
# Evaluate a checkpoint on LIBERO (perturbation none) and LIBERO-PRO, all four suites.
#   bash closed_loop/scripts/eval.sh <ckpt_final.pt> <out_dir>
# Needs LIBERO installed and headless MuJoCo rendering (e.g. MUJOCO_GL=egl).
set -euo pipefail
CKPT=${1:?checkpoint}; OUT=${2:?out_dir}
cd "$(dirname "$0")/../.."
export PYTHONPATH="src:.:${OPENPI_SRC:-}${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
for SUITE in libero_spatial libero_object libero_goal libero_10; do
  python -m closed_loop.eval_libero_pro --checkpoint "$CKPT" --task-suite "$SUITE" \
    --perturbations none,language,position,object,task,environment \
    --num-trials-per-task 10 --replan-steps 5 --num-steps-wait 10 --inference-steps 10 \
    --image-size 224 --render-size 256 --flip vertical --seed 7 --out-dir "$OUT/$SUITE"
done
python closed_loop/summarize.py "$OUT"
