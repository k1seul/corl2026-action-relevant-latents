#!/usr/bin/env bash
# Probe all aux ablation runs after training completes.
# Writes downstream results to runs/downstream/aux_ablations/.
#
# A runs: H=1 (standard single-step)
# B runs: H=1 (single-step) + H=k (chunk, where k is the run's k-value)
# C2:     eval_aux_direct.py using A0's saved aux_head
#
# Usage:
#   CUDA_VISIBLE_DEVICES=3 bash scripts/probe_aux_ablations.sh 2>&1 | tee /tmp/probe_aux_abl.log

set -uo pipefail
cd "$(dirname "$0")/.."

SWEEP_DIR="runs/sweep/aux_ablations"
DS_ROOT="runs/downstream/aux_ablations"
SPLIT_OOD="data/splits_taskheld/test_ood.json"
mkdir -p "$DS_ROOT"

PROBE_ARGS="--probe-steps 3000 --probe-hidden-dims 256,128 --probe-seeds 0,1,2 --train-clips 1000 --test-clips 400"

run_probe() {
  local name="$1"; local horizons="$2"
  local ckpt="$SWEEP_DIR/$name/ckpt_last.pt"
  local out="$DS_ROOT/${name}_ood"
  if [[ ! -f "$ckpt" ]]; then echo "[SKIP] $name — ckpt missing"; return; fi
  if [[ -f "$out/metrics.txt" ]]; then echo "[SKIP] $name/ood (done)"; return; fi
  echo "[RUN ] $name  horizons=$horizons"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    uv run python scripts/eval_downstream.py \
      --ckpt "$ckpt" --out-dir "$out" \
      --test-split "$SPLIT_OOD" \
      --action-horizons "$horizons" \
      $PROBE_ARGS >"$out.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] $name  rc=$rc  (see $out.log)"; return
  fi
  r2=$(grep "action_h1_r2_mean" "$out/metrics.txt" 2>/dev/null | awk '{print $2}')
  ivd=$(grep "invdyn_r2_mean"   "$out/metrics.txt" 2>/dev/null | awk '{print $2}')
  echo "[DONE] $name  action_h1_r2=$r2  invdyn_r2=$ivd"
}

echo "=== Experiment A probes ==="
for name in aux_abl_A0_invdyn_l2_repro aux_abl_A1_forward_dyn \
            aux_abl_A2_contrastive aux_abl_A3_single_frame aux_abl_A4_invdyn_l1; do
  run_probe "$name" "1"
done

echo ""
echo "=== Experiment B probes ==="
run_probe aux_abl_B1_multistep_k2 "1,2"
run_probe aux_abl_B2_multistep_k4 "1,4"
run_probe aux_abl_B3_multistep_k8 "1,8"

echo ""
echo "=== Experiment C2: aux head direct ==="
A0_DIR="$SWEEP_DIR/aux_abl_A0_invdyn_l2_repro"
C2_OUT="$DS_ROOT/C2_aux_direct"
if [[ -f "$A0_DIR/ckpt_last_raw.pt" ]]; then
  if [[ ! -f "$C2_OUT/metrics.txt" ]]; then
    echo "[RUN ] C2 aux head direct"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      uv run python scripts/eval_aux_direct.py \
        --ckpt-dir "$A0_DIR" \
        --test-split "$SPLIT_OOD" \
        --out-dir "$C2_OUT" \
        --test-clips 400 >"$C2_OUT.log" 2>&1
    if [[ $? -ne 0 ]]; then
      echo "[FAIL] C2 (see $C2_OUT.log)"
    else
      r2=$(grep "action_r2" "$C2_OUT/metrics.txt" 2>/dev/null | awk '{print $2}')
      echo "[DONE] C2 aux_head direct  action_r2=$r2"
    fi
  else
    echo "[SKIP] C2 (done)"
  fi
else
  echo "[SKIP] C2 — A0 not yet trained (run aux_ablations_a sweep first)"
fi

echo ""
echo "=== probe sweep complete. run make_aux_ablation_summary.py to build summary.md ==="
