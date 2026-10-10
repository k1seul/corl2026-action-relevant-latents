#!/usr/bin/env bash
# Full perturbation sweep: ALL 84 backbones × 7 perturbations
# (probe-only; recon skipped to keep elapsed time bounded)
set -e
cd "$(dirname "$0")/.."

ALL_VARIANTS=$(cat /tmp/all_variants.txt 2>/dev/null || \
  uv run python -c "import yaml; m=yaml.safe_load(open('runs/full_results/manifest.yaml')); print(','.join(e['variant'] for e in m))")

GPU=${1:-1}
echo "using GPU $GPU"

for kind_strength in "noise:0.05" "noise:0.10" "noise:0.15" "jitter:0.10" "jitter:0.20" "blur:5" "blur:11"; do
  kind=${kind_strength%:*}; strength=${kind_strength#*:}
  tag="_pert_${kind}_${strength//./}"
  echo "===== $kind strength=$strength tag=$tag ====="
  CUDA_VISIBLE_DEVICES=$GPU uv run python scripts/full_results_pipeline.py \
    --only "$ALL_VARIANTS" \
    --probe-train-clips 400 --probe-eval-clips 200 \
    --skip-recon \
    --perturb-test-kind "$kind" \
    --perturb-test-strength "$strength" \
    --perturb-tag "$tag" 2>&1 | grep -E "===.*===|recon train|done in|RECON FAILED|ENCODE FAILED|BUILD FAILED|error|Traceback" | tail -50
done

echo "===final==="
wc -l runs/full_results/results.jsonl
