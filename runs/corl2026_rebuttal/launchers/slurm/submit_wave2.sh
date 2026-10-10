#!/bin/bash
# Wave 2 submitter — RTX 3090 workload: frozen probes, action-centric
# baselines, common frozen decoder.
#
# Env vars:
#   REPO_ROOT      absolute path (required)
#   PART_3090      Slurm partition name for 3090s (default: <partition>)
#   REBUTTAL_VENV  path to venv to source (optional)

set -eu -o pipefail
: "${REPO_ROOT:?}"
PART_3090="${PART_3090:-<partition>}"

PROBE_TEMPLATE="$REPO_ROOT/runs/corl2026_rebuttal/launchers/slurm/frozen_probe.sbatch"
DECODER_TEMPLATE="$REPO_ROOT/runs/corl2026_rebuttal/launchers/slurm/common_decoder.sbatch"
JOBID_FILE="$REPO_ROOT/runs/corl2026_rebuttal/logs/wave2_job_ids.txt"
mkdir -p "$(dirname "$JOBID_FILE")"
: > "$JOBID_FILE"

# 2A: frozen probes on public backbones
for EXTR in vjepa2 videomae webdino siglip2 lapa sdxl_vae cosmos1; do
  NAME="corl26-fp-${EXTR}"
  echo "[submit_wave2A] sbatch $NAME"
  OUT=$(sbatch --partition="$PART_3090" --job-name="$NAME" \
    --export=ALL,REPO_ROOT="$REPO_ROOT",EXTRACTOR="$EXTR",REBUTTAL_VENV="${REBUTTAL_VENV:-}" \
    "$PROBE_TEMPLATE")
  echo "  $OUT"
  echo "$OUT | $NAME" >> "$JOBID_FILE"
done

# 2B: action-centric baselines (§6). CAIP has two clearly-labeled modes.
for EXTR in r3m vip liv caip_visual_only caip_native; do
  NAME="corl26-ac-${EXTR}"
  echo "[submit_wave2B] sbatch $NAME"
  OUT=$(sbatch --partition="$PART_3090" --job-name="$NAME" \
    --export=ALL,REPO_ROOT="$REPO_ROOT",EXTRACTOR="$EXTR",REBUTTAL_VENV="${REBUTTAL_VENV:-}" \
    "$PROBE_TEMPLATE")
  echo "  $OUT"
  echo "$OUT | $NAME" >> "$JOBID_FILE"
done

# 2C: common frozen decoder (§7)
for EXTR in vjepa2 videomae webdino siglip2 sdxl_vae cosmos1; do
  NAME="corl26-cd-${EXTR}"
  echo "[submit_wave2C] sbatch $NAME"
  OUT=$(sbatch --partition="$PART_3090" --job-name="$NAME" \
    --export=ALL,REPO_ROOT="$REPO_ROOT",EXTRACTOR="$EXTR",REBUTTAL_VENV="${REBUTTAL_VENV:-}" \
    "$DECODER_TEMPLATE")
  echo "  $OUT"
  echo "$OUT | $NAME" >> "$JOBID_FILE"
done

echo "[submit_wave2] all jobs submitted. IDs → $JOBID_FILE"
