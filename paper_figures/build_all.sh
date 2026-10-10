#!/usr/bin/env bash
# Rebuild the camera-ready figures and tables in paper_figures/out/ from paper_figures/data/.
# Usage: bash paper_figures/build_all.sh
# Needs matplotlib + numpy. Set PY=/path/to/python to use another interpreter.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"

echo "== input checksums"
(cd data && sha256sum -c SHA256SUMS.txt)

echo "== Fig. 1 (option A: +ID R^2 vs record PSNR) + comparison variants"
"$PY" src/build_fig1_pixel_vs_action.py
"$PY" src/build_fig1_pixel_vs_action.py --with-diff-canonical
"$PY" src/build_fig1_pixel_vs_action.py --psnr submitted

echo "== Fig. 2 (per-dimension radar; DIFF panel = Table 1 L2 variant)"
"$PY" src/build_fig2_per_dim_radar.py

echo "== Fig. 3 + Table 9 (V-JEPA 2 per-block probe)"
"$PY" src/build_fig3_per_layer.py

echo "== CAIP table (Appendix L)"
"$PY" src/build_caip_table.py

echo "== Fig. 6 (Appendix N closed-loop method)"
"$PY" src/build_fig6_closed_loop_method.py

echo "done -> paper_figures/out/"
