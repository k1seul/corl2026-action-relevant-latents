#!/bin/bash
# Submit all 15 wave1 evals with the patched eval_matched_control.py so
# per_episode_sufficient_stats gets captured (required by the P2 task-cluster
# bootstrap). Appends to matched_controls.jsonl; the audit script filters
# on the presence of the new field.
#
# Env vars:
#   REPO_ROOT       required
#   REBUTTAL_VENV   default ~/anaconda3/envs/dreamer
#   PART            default <partition> (has spare GPU slots and works with
#                    all backbones including V-JEPA2 which failed on rtx6000)

set -eu -o pipefail
: "${REPO_ROOT:?}"
: "${REBUTTAL_VENV:=$HOME/anaconda3/envs/dreamer}"
: "${PART:=<partition>}"

cd "$REPO_ROOT"

CKPT_ROOT="$REPO_ROOT/runs/corl2026_rebuttal/checkpoints"

submit_frozen() {
  local bb="$1"
  local name="corl26-eval-audited-${bb}-frozen"
  sbatch --partition="$PART" --job-name="$name" \
    --export=ALL,REPO_ROOT="$REPO_ROOT",REBUTTAL_VENV="$REBUTTAL_VENV",BACKBONE="$bb",CONDITION=frozen,TRAIN_SEED=-1,CKPT="" \
    "$REPO_ROOT/runs/corl2026_rebuttal/launchers/slurm/eval_matched_control.sbatch"
}

submit_finetuned() {
  local bb="$1" cond="$2" seed="$3"
  local ck="$CKPT_ROOT/${bb}/${cond}/wave1_${cond}_${bb}_s${seed}/ckpt_last.pt"
  if [[ ! -f "$ck" ]]; then
    echo "MISSING ckpt: $ck ; skipping"; return
  fi
  local name="corl26-eval-audited-${bb}-${cond}-s${seed}"
  sbatch --partition="$PART" --job-name="$name" \
    --export=ALL,REPO_ROOT="$REPO_ROOT",REBUTTAL_VENV="$REBUTTAL_VENV",BACKBONE="$bb",CONDITION="$cond",TRAIN_SEED="$seed",CKPT="$ck" \
    "$REPO_ROOT/runs/corl2026_rebuttal/launchers/slurm/eval_matched_control.sbatch"
}

for bb in videomae vjepa2 webdino; do
  submit_frozen "$bb"
  for cond in true_id shuffled_id; do
    for seed in 0 1; do
      submit_finetuned "$bb" "$cond" "$seed"
    done
  done
done

echo
echo "== queued audited-eval submissions =="
squeue -u "$USER" -o "%.10i %.35j %.8T %.10M %.15R"
