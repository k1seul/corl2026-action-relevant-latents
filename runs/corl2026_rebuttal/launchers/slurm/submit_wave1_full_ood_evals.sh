#!/bin/bash
# Re-run the 15 wave1 audited evals with probe_eval_clips_test=1300 so the
# entire 26-task held-out split contributes to the cluster bootstrap
# (default 500 clips gives only 10 of 26 tasks because the split JSON is
# ordered task-by-task with 50 demos each).
#
# Env: same as submit_wave1_audited_evals.sh
#
set -eu -o pipefail
: "${REPO_ROOT:?}"
: "${REBUTTAL_VENV:=$HOME/anaconda3/envs/dreamer}"
: "${PART:=<partition>}"

cd "$REPO_ROOT"
CKPT_ROOT="$REPO_ROOT/runs/corl2026_rebuttal/checkpoints"

# We tag with a JOB suffix and pass --probe-eval-clips-test 1300 via a
# small wrapper below. Rather than duplicating the sbatch template we
# invoke it with an extra env var EVAL_EXTRA that the template appends.
export EVAL_EXTRA="--probe-eval-clips-test 1300 --probe-train-clips 800"

# Ensure sbatch template forwards EVAL_EXTRA. We'll rely on the template
# passing all its EXPORT vars; the wrapper reads EVAL_EXTRA from env.
# Since eval_matched_control.sbatch doesn't currently splice extra args,
# we submit via --wrap so we can inject the flags directly and capture
# the exact same env exports.
submit() {
  local bb="$1" cond="$2" seed="$3" ckpt="$4"
  local name="corl26-eval-full-${bb}-${cond}-s${seed}"
  local cmd="cd $REPO_ROOT && \
    export PATH=$REBUTTAL_VENV/bin:\$PATH && \
    export PYTHONPATH=${LIBERO_SRC:-third_party/LIBERO}:$REPO_ROOT/src && \
    export EXTERNAL_MODELS_DIR=$REPO_ROOT/datasets/external_models && \
    python -u scripts/rebuttal/eval_matched_control.py \
      --backbone $bb --condition $cond --training-seed $seed \
      ${ckpt:+--ckpt $ckpt} \
      --datasets-dir $REPO_ROOT/datasets/libero_downloaded \
      --split-train $REPO_ROOT/runs/corl2026_rebuttal/splits/train.json \
      --split-test-ood $REPO_ROOT/runs/corl2026_rebuttal/splits/test_ood.json \
      --out-jsonl $REPO_ROOT/runs/corl2026_rebuttal/results/matched_controls.jsonl \
      $EVAL_EXTRA"
  sbatch --partition="$PART" --job-name="$name" \
    --nodes=1 --gres=gpu:1 --cpus-per-task=4 --mem=48GB --time=04:00:00 \
    -o "$REPO_ROOT/runs/corl2026_rebuttal/logs/slurm-%x-%j.out" \
    -e "$REPO_ROOT/runs/corl2026_rebuttal/logs/slurm-%x-%j.err" \
    --wrap "$cmd"
}

for bb in videomae vjepa2 webdino; do
  submit "$bb" frozen -1 ""
  for cond in true_id shuffled_id; do
    for seed in 0 1; do
      ck="$CKPT_ROOT/${bb}/${cond}/wave1_${cond}_${bb}_s${seed}/ckpt_last.pt"
      [[ ! -f "$ck" ]] && { echo "MISSING $ck"; continue; }
      submit "$bb" "$cond" "$seed" "$ck"
    done
  done
done

echo
squeue -u "$USER" -o "%.10i %.36j %.8T %.10M %.15R" | head -20
