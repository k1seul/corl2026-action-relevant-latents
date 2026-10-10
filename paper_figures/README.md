# Paper figures and tables

`bash paper_figures/build_all.sh` checks the input checksums and rebuilds everything into `paper_figures/out/` (PDF at 5.5 in width, 300-dpi PNG, the plotted values as CSV, and a provenance JSON per figure). Needs numpy and matplotlib; set `PY=/path/to/python` to choose the interpreter.

| Output | Script | Inputs |
|---|---|---|
| Figure 1: pixel fidelity vs. action probe R² | `src/build_fig1_pixel_vs_action.py` | `data/results.jsonl` (probe R², rollout PSNR), `data/decoder_results.jsonl` (17M-decoder PSNR) |
| Figure 2: per-dimension R² by backbone | `src/build_fig2_per_dim_radar.py` | `data/results.jsonl` |
| Figure 3 and Table 9: V-JEPA 2 per-block probe | `src/build_fig3_per_layer.py` | `data/vjepa_per_layer.jsonl`, `data/vjepa_per_layer_frozen.jsonl` |
| Figure 6: closed-loop setup | `src/build_fig6_closed_loop_method.py` | `src/assets/` (LIBERO frames) |
| CAIP and frozen-coverage table (Appendix M) | `src/build_caip_table.py` | `data/caip_exact_protocol.csv` |

`data/results.jsonl` has one line per evaluated run: `variant` (run name), `test_cell.action_probe_r2`, `test_cell.per_dim_r2`, and `test_cell.psnr` for pixel-producing models; runs with `_pert_` in the name are perturbation evaluations. `data/submitted_fig*_*.csv` hold the values plotted in the submitted version's figures, extracted from its PDF, for comparison with the records.

`build_fig1_pixel_vs_action.py --with-diff-canonical` adds the DIFF + ID checkpoint, and `--psnr submitted` uses the PSNR values plotted in the submitted version.
