#!/usr/bin/env bash
# Download a pi0/pi0.5 base checkpoint from gs://openpi-assets and convert to
# PyTorch safetensors. Idempotent — skips steps whose output already exists.
#
# Supported models (see https://github.com/Physical-Intelligence/openpi):
#   pi0_base      — base π₀ (flow-matching, NOT pi05)
#   pi05_base     — base π₀.₅ (adaptive norm)
#   pi05_libero   — π₀.₅ already fine-tuned on LIBERO (our default)
#
# (pi0_fast_base is an autoregressive model — different architecture, not supported here.)
#
# Usage:
#   bash closed_loop/scripts/setup_pretrained.sh pi05_base
#   bash closed_loop/scripts/setup_pretrained.sh pi05_libero
#   bash closed_loop/scripts/setup_pretrained.sh pi0_base
#
# Output:
#   ~/.cache/openpi/openpi-assets/checkpoints/<MODEL>/         (JAX, ~12 GB)
#   ~/.cache/openpi/<MODEL>_pytorch/model.safetensors          (PyTorch, ~7 GB)

set -euo pipefail

MODEL="${1:-}"
if [[ -z "${MODEL}" ]]; then
    echo "usage: $0 <model>   (pi0_base | pi05_base | pi05_libero)" >&2
    exit 1
fi

case "${MODEL}" in
    pi0_base)     IS_PI05=0 ;;
    pi05_base)    IS_PI05=1 ;;
    pi05_libero)  IS_PI05=1 ;;
    pi0_fast_base)
        echo "ERROR: pi0_fast_base is an autoregressive model and is not supported" >&2
        echo "       by this PyTorch wrapper. Use pi0_base or pi05_base instead." >&2
        exit 1
        ;;
    *)
        echo "ERROR: unknown model '${MODEL}'. Choices: pi0_base, pi05_base, pi05_libero" >&2
        exit 1
        ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
OPENPI_SRC="${OPENPI_SRC:-${HOME}/openpi-src/src}"
export PYTHONPATH="${ROOT}:${OPENPI_SRC}:${PYTHONPATH:-}"

JAX_DIR="${HOME}/.cache/openpi/openpi-assets/checkpoints/${MODEL}"
PT_DIR="${HOME}/.cache/openpi/${MODEL}_pytorch"
PT_FILE="${PT_DIR}/model.safetensors"

echo "============================================================"
echo " setup_pretrained.sh  model=${MODEL}  pi05=${IS_PI05}"
echo "------------------------------------------------------------"
echo " JAX dir  : ${JAX_DIR}"
echo " PyT dir  : ${PT_DIR}"
echo "============================================================"

# --- step 1: download JAX checkpoint
# openpi-assets is a *public* GCS bucket. We prefer gsutil if present; otherwise
# fall back to gcsfs with anonymous credentials (no GCP login required).
if [[ -d "${JAX_DIR}/params" ]]; then
    echo "[1/2] JAX checkpoint already present, skipping download."
elif command -v gsutil >/dev/null 2>&1; then
    echo "[1/2] Downloading via gsutil ..."
    mkdir -p "${JAX_DIR}"
    gsutil -m cp -r "gs://openpi-assets/checkpoints/${MODEL}/*" "${JAX_DIR}/"
else
    echo "[1/2] gsutil not found; using anonymous gcsfs (public bucket, no login needed) ..."
    python - <<PY
from openpi.shared.download import maybe_download
# gs={"token": "anon"} → gcsfs uses anonymous credentials; works for public buckets
local = maybe_download("gs://openpi-assets/checkpoints/${MODEL}", gs={"token": "anon"})
print(f"downloaded -> {local}")
PY
fi

# --- step 2: convert JAX → PyTorch safetensors
if [[ -f "${PT_FILE}" ]]; then
    echo "[2/2] PyTorch safetensors already present, skipping conversion."
    echo "      (delete ${PT_DIR} to force re-conversion)"
else
    echo "[2/2] Converting JAX → PyTorch safetensors ..."
    PI05_FLAG=()
    [[ ${IS_PI05} -eq 1 ]] && PI05_FLAG=(--pi05)
    python closed_loop/scripts/convert_pi0_weights.py \
        --checkpoint-dir "${JAX_DIR}" \
        --output-dir "${PT_DIR}" \
        "${PI05_FLAG[@]}" \
        --precision bfloat16 \
        --action-dim 32 \
        --action-horizon 10
fi

echo ""
echo "Done. Use this checkpoint with:"
echo "  PRETRAINED_PI0=${PT_DIR} bash closed_loop/scripts/train.sh clip 0"
