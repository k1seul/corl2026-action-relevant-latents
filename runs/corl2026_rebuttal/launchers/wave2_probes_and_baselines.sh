#!/usr/bin/env bash
# Wave 2 — RTX 3090 workload: feature extraction, probes, action-centric
# baselines, common frozen decoder, closed-loop rollouts.
#
# **PRIMARY LAUNCH PATH is Slurm/sbatch:**
#   REPO_ROOT=$(pwd) bash runs/corl2026_rebuttal/launchers/slurm/submit_wave2.sh
# This nohup script exists as a fallback for scheduler-less environments.
#
# All jobs are single-GPU on 3090s. Ordering matters ONLY in the sense that
# per-checkpoint feature extraction must precede the probe consuming it.
#
# NB: we do NOT dispatch all 16 GPUs from a single script; instead we group
# jobs by their compute profile and rely on the monitor to schedule the next
# job onto a free GPU. If your cluster has Slurm/Kubernetes, replace the
# CUDA_VISIBLE_DEVICES loop with the appropriate submission command.

set -eu -o pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")"/../../.. && pwd)}"
cd "$REPO_ROOT"

LOGS="$REPO_ROOT/runs/corl2026_rebuttal/logs"
mkdir -p "$LOGS"

# ── group A: canonical probe re-extraction on frozen public ckpts ──────────
#   V-JEPA2, VideoMAE, Web-DINO, SigLIP-2, LAPA, SDXL-VAE, Cosmos-1
#   → one 3090 each; produces frozen action-probe R² baselines.
BACKBONES_FROZEN=(vjepa2 videomae webdino siglip2 lapa sdxl_vae cosmos1)
GPU=0
for bb in "${BACKBONES_FROZEN[@]}"; do
  log="$LOGS/frozen_probe_${bb}.log"
  echo "[wave2A] GPU $GPU :: frozen probe $bb -> $log"
  CUDA_VISIBLE_DEVICES="$GPU" nohup \
    uv run python "$REPO_ROOT/scripts/full_results_pipeline.py" \
      --backbone "$bb" \
      --frozen \
      --out-dir "$REPO_ROOT/runs/corl2026_rebuttal/results/frozen_probe_$bb" \
    > "$log" 2>&1 &
  GPU=$((GPU + 1))
done

# ── group B: action-centric frozen baselines (§6) ──────────────────────────
BASELINES=(r3m vip liv caip_visual_only caip_native)
for bl in "${BASELINES[@]}"; do
  log="$LOGS/action_centric_${bl}.log"
  echo "[wave2B] GPU $GPU :: action-centric $bl -> $log"
  CUDA_VISIBLE_DEVICES="$GPU" nohup \
    uv run python "$REPO_ROOT/scripts/full_results_pipeline.py" \
      --extractor "$bl" \
      --frozen \
      --out-dir "$REPO_ROOT/runs/corl2026_rebuttal/results/action_centric_$bl" \
    > "$log" 2>&1 &
  GPU=$((GPU + 1))
done

# ── group C: common frozen pixel decoder (§7) ──────────────────────────────
#   Freeze encoder, train the SAME 17M decoder on identical splits/schedule.
COMMON_DECODER_BACKBONES=(vjepa2 videomae webdino siglip2 sdxl_vae cosmos1)
for bb in "${COMMON_DECODER_BACKBONES[@]}"; do
  log="$LOGS/common_decoder_${bb}.log"
  echo "[wave2C] GPU $GPU :: common decoder for $bb -> $log"
  CUDA_VISIBLE_DEVICES="$GPU" nohup \
    uv run python "$REPO_ROOT/scripts/train_pixel_decoder.py" \
      --extractor "$bb" \
      --frozen \
      --out-dir "$REPO_ROOT/runs/corl2026_rebuttal/results/common_decoder_$bb" \
    > "$log" 2>&1 &
  GPU=$((GPU + 1))
done

echo "[wave2] group A/B/C launched. total dispatched GPUs=$GPU"
echo "[wave2] the closed-loop policy study lives in closed_loop/ (see its README)"
