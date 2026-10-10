#!/usr/bin/env bash
# Run probe + recon under LIBERO-PRO-style visual perturbations
# (proxy for Object/Environment perturbation axes).
set -e
cd "$(dirname "$0")/.."

# Headline backbones (subset of 84) — one per family-tier
VARIANTS="vjepa2_aux_finetune,vjepa2_vit_l_frozen,vjepa2_1_aux_finetune,vjepa2_1_vit_b_frozen,lapa_aux_finetune,lapa_frozen,aux_v6_large_noact_taskheld_lam0p05,webdino_aux_finetune,webdino_frozen,siglip2_aux_finetune,siglip2_frozen,sdvae_aux_finetune,sdvae_frozen,cosmos_tokenizer_aux_finetune,cosmos_tokenizer_frozen,dreamer4_dynamics_aux_agent,dreamer4_pretrained_aux_agent,vitl_scratch_aux_libero"

GPU=${1:-1}

for kind_strength in "noise:0.05" "noise:0.10" "noise:0.15" "jitter:0.10" "jitter:0.20" "blur:5" "blur:11"; do
  kind=${kind_strength%:*}
  strength=${kind_strength#*:}
  tag="_pert_${kind}_${strength//./}"
  echo "===== perturbation $kind strength=$strength tag=$tag ====="
  CUDA_VISIBLE_DEVICES=$GPU uv run python scripts/full_results_pipeline.py \
    --only "$VARIANTS" \
    --probe-train-clips 400 --probe-eval-clips 200 \
    --skip-recon \
    --perturb-test-kind "$kind" \
    --perturb-test-strength "$strength" \
    --perturb-tag "$tag" 2>&1 | tail -3
done

echo "===final row count==="
wc -l runs/full_results/results.jsonl
