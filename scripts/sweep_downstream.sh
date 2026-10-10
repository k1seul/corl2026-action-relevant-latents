#!/usr/bin/env bash
# Run scripts/eval_downstream.py on every variant ckpt and write
# results to runs/downstream/<run_name>/. Skips entries already done.
set -uo pipefail

cd "$(dirname "$0")/.."

OUT_ROOT="runs/downstream"
mkdir -p "$OUT_ROOT"

CKPTS=(
  "runs/sweep/strong_v1/strong_v1_clip32_medium_50k/ckpt_last.pt"
  "runs/sweep/variants/variant_A_diff/ckpt_last.pt"
  "runs/sweep/variants/variant_B_token/ckpt_last.pt"
  "runs/sweep/variants/A_diff_v2_pc_cos_ema/ckpt_last.pt"
  "runs/sweep/variants/A_diff_v3_pc_cos_ema_100k/ckpt_last.pt"
  "runs/sweep/variants/A_diff_v4_pc_cos_ema_large/ckpt_last.pt"
  "runs/sweep/variants/A_diff_v5_pc_cos_ema_action_100k/ckpt_last.pt"
  "runs/sweep/variants/A_diff_v6_action_2cam_100k/ckpt_last.pt"
  "runs/sweep/variants/A_diff_v7_action_200k/ckpt_last.pt"
  "runs/sweep/variants/A_diff_v8_action_clip64_50k/ckpt_last.pt"
  "runs/sweep/variants/A_diff_v9_action_clip64_200k/ckpt_last.pt"
  "runs/sweep/variants/A_diff_v10_action_large_200k/ckpt_last.pt"
  "runs/sweep/variants/A_fm_v2_lr5e-5_logitnormal_50k/ckpt_last.pt"
  "runs/sweep/variants/A_fm_v3_pc_ema_100k/ckpt_last.pt"
  "runs/sweep/variants/B_token_v2_action_100k/ckpt_last.pt"
  "runs/sweep/variants/A_mae_rollout50k/ckpt_last.pt"
  "runs/sweep/variants/A_diff_v6_large/ckpt_last.pt"
  "runs/sweep/variants/v_mix_v6_large_noact/ckpt_last.pt"
  "runs/sweep/variants/v_mix_v10_noact/ckpt_last.pt"
  "runs/sweep/variants/v_mix_token_noact_200k/ckpt_last.pt"
)

NAMES=(
  "strong_v1_A_mae"
  "variant_A_diff"
  "variant_B_token"
  "A_diff_v2_pc_cos_ema"
  "A_diff_v3_pc_cos_ema_100k"
  "A_diff_v4_pc_cos_ema_large"
  "A_diff_v5_pc_cos_ema_action_100k"
  "A_diff_v6_action_2cam_100k"
  "A_diff_v7_action_200k"
  "A_diff_v8_action_clip64_50k"
  "A_diff_v9_action_clip64_200k"
  "A_diff_v10_action_large_200k"
  "A_fm_v2_lr5e-5_logitnormal_50k"
  "A_fm_v3_pc_ema_100k"
  "B_token_v2_action_100k"
  "A_mae_rollout50k"
  "A_diff_v6_large"
  "v_mix_v6_large_noact"
  "v_mix_v10_noact"
  "v_mix_token_noact_200k"
)

for i in "${!CKPTS[@]}"; do
  ckpt="${CKPTS[$i]}"
  name="${NAMES[$i]}"
  out="$OUT_ROOT/$name"
  if [[ -f "$out/metrics.txt" ]]; then
    echo "[skip ] $name (already complete)"
    continue
  fi
  if [[ ! -f "$ckpt" ]]; then
    echo "[skip ] $name (ckpt missing: $ckpt)"
    continue
  fi
  echo "[start] $name"
  uv run python scripts/eval_downstream.py \
    --ckpt "$ckpt" --out-dir "$out" \
    --train-clips 1000 --test-clips 400 \
    --probe-steps 3000 \
    > "$out.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[FAIL ] $name (rc=$rc, see $out.log)"
  else
    metrics_line=$(grep -E "^action_h[15]_test_r2|^task_test_top1" "$out/metrics.txt" 2>/dev/null | tr '\n' ' ')
    echo "[done ] $name  $metrics_line"
  fi
done

echo ""
echo "all done"
