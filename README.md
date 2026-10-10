# What Makes Video World Model Latents Action-Relevant: Prediction over Reconstruction

Jewon Yeom\*, Hanseul Kim\*, Jeongjae Park, Sungmok Jung, Jaejin Lee, Taesup Kim†
Graduate School of Data Science, Seoul National University
*CoRL 2026* · \*equal contribution · †corresponding author

Code for the paper. We compare image-only, video-pretrained, reconstruction, diffusion, and dynamics encoders with a common inverse-dynamics (ID) action probe on LIBERO. We also test whether ID supervision needs correct transition–action pairing (matched controls), how probe readouts hold up under visual corruption, and whether an action-relevant target helps a π0.5 policy in closed loop.

**Main results: [RESULTS.md](RESULTS.md)** (tables and figures from the paper, with the files they come from).

![Pixel fidelity vs. action recoverability](assets/fig1_pixel_vs_action.png)

## Repository layout

| Path | Contents |
|---|---|
| [`src/`](src) | Library: frozen-feature extractors for every backbone (`comparison/extractors`), action probe and metrics, LIBERO clip datasets (`data/`), ID and other auxiliary losses (`training/`), matched-control conditions (`rebuttal/`), and the DIFF world model (`models/block_causal_transformer.py`) |
| [`scripts/`](scripts) | Training (`train.py` for DIFF, `train_*_aux*.py` for +ID fine-tuning of pretrained encoders, Dreamer 4 launchers), probe evaluation (`full_results_pipeline.py`, `probe_*.py`), matched controls and CAIP (`rebuttal/`), figure scripts |
| [`configs/`](configs) | DIFF and DIFF + ID training configs |
| [`data/`](data) | LIBERO episode splits ([`data/SPLITS.md`](data/SPLITS.md)) |
| [`runs/`](runs) | Evaluation manifests (`full_results/`) and the matched-control launchers, manifest, splits, and unit tests (`corl2026_rebuttal/`) |
| [`closed_loop/`](closed_loop) | π0.5 target-swap study: training, LIBERO / LIBERO-PRO evaluation, launchers ([`closed_loop/README.md`](closed_loop/README.md)) |
| [`paper_figures/`](paper_figures) | Scripts and recorded results that rebuild Figures 1–3 and 6 and the per-layer and CAIP tables |
| [`results/`](results) | Closed-loop success counts per suite and per task; matched-control R² per training seed × probe seed |
| [`env/`](env) | Package lists of the two environments used for the probe study |

## Setup

Probe study (Python 3.10+, one GPU):

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m pytest runs/corl2026_rebuttal/tests      # 44 pass; 3 need the LIBERO HDF5 files
```

`env/` lists the exact package versions (an older cluster with torch 2.4 and a GCP VM). The closed-loop study needs openpi and LIBERO; see [`closed_loop/README.md`](closed_loop/README.md).

**Data.** Download the LIBERO HDF5 demonstrations (`benchmark_scripts/download_libero_datasets.py` in the [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO)) to `datasets/libero/libero_{spatial,object,goal,10,90}/`. The split files in `data/` refer to episodes by suite, file and demo name.

**External models.** Hugging Face backbones download automatically (Web-DINO `facebook/webssl-dino300m-full2b-224`, SigLIP 2 `google/siglip2-large-patch16-256`, VideoMAE `MCG-NJU/videomae-large`, SDXL VAE `stabilityai/sdxl-vae`, Cosmos tokenizer `nvidia/Cosmos-Tokenizer-*`). The others are read from `external_models/` (or `$EXTERNAL_MODELS_DIR`): the [V-JEPA 2](https://github.com/facebookresearch/vjepa2) code and checkpoints (`vjepa2/`, `vjepa2_ckpts/vitl.pt`, `vjepa2_ckpts/vjepa2_1_vitb_dist_vitG_384.pt`), [LAPA](https://github.com/LatentActionPretraining/LAPA) (`LAPA/`, `lapa_ckpts/laq_openx.pt`), CAIP (`caip_ckpts/`), and the [unofficial Dreamer 4 implementation](https://github.com/nicklashansen/dreamer4) (`dreamer4/`). The extractor files in `src/comparison/extractors/` document each one.

**Released checkpoints** (Hugging Face, `Dreamer-VLA` organization):

| Checkpoint | Paper row |
|---|---|
| [`Dreamer-VLA/dreamer-vla-best-libero-vjepa2-aux`](https://huggingface.co/Dreamer-VLA/dreamer-vla-best-libero-vjepa2-aux) | V-JEPA 2 ViT-L + ID (Table 1); token-coupling target (Table 4) |
| [`Dreamer-VLA/dreamer-vla-lapa-aux-libero`](https://huggingface.co/Dreamer-VLA/dreamer-vla-lapa-aux-libero) | LAPA-LAQ-OpenX + ID |
| [`Dreamer-VLA/dreamer-vla-extra-backbones-libero`](https://huggingface.co/Dreamer-VLA/dreamer-vla-extra-backbones-libero) | Other +ID backbones |
| [`Dreamer-VLA/dreamer-vla-dreamer4-libero`](https://huggingface.co/Dreamer-VLA/dreamer-vla-dreamer4-libero) | Dreamer 4 reproduction |

## Reproducing the paper

Run commands from the repository root with `PYTHONPATH=src`.

| Paper item | Code |
|---|---|
| Table 1, robustness tables (frozen and +ID probes on LIBERO) | `scripts/build_full_results_manifest.py` → `scripts/full_results_pipeline.py --manifest runs/full_results/manifest.yaml --probe-train-clips 400 --probe-eval-clips 200` → `scripts/render_full_results.py`; perturbations: `scripts/run_perturbation_full_sweep.sh` |
| +ID fine-tuning of pretrained encoders | `scripts/train_{vjepa2,vjepa2_1,videomae,webdino,siglip2,lapa,sd_vae,cosmos_tokenizer}_aux*.py`; ID loss in `src/training/aux_loss.py` |
| DIFF and DIFF + ID | `python scripts/train.py --config configs/diff_id.yaml` (`configs/diff.yaml` without ID) |
| Dreamer 4 rows | `scripts/preprocess_libero_for_dreamer4.py`, `scripts/train_dreamer4_*.sh` (needs the Dreamer 4 repository and our auxiliary-loss patch, see Notes) |
| Figure 1 PSNR | `scripts/train_pixel_decoder.py` (17M decoder on +ID encoders), `scripts/eval_pixel_rollout.py` (rollout PSNR) |
| Figures 1–3, 6, Table 9, CAIP table | `bash paper_figures/build_all.sh` (from the recorded results in `paper_figures/data/`) |
| Table 2, matched controls (Appendix M) | `scripts/train_matched_control.py`, `src/rebuttal/`, `scripts/rebuttal/eval_matched_control.py`, `scripts/rebuttal/task_cluster_bootstrap.py`; launchers in `runs/corl2026_rebuttal/launchers/`; per-cell results in `results/matched_controls_per_cell.csv` |
| CAIP and R3M/VIP comparisons | `scripts/rebuttal/caip_exact_protocol.py`, `scripts/rebuttal/common_ridge_comparison.py`, `scripts/rebuttal/validate_baselines.py` |
| Appendix A–B (DIFF λ, loss form, capacity) | `scripts/train.py`, `scripts/run_sweep.py`, `scripts/probe_aux_ablations.sh`, `scripts/make_aux_ablation_summary.py` |
| Appendix C (temporal cosine) | `scripts/temporal_cosine_sweep.py` |
| Appendix D (CALVIN, MetaWorld) | `scripts/build_{calvin,metaworld}_{splits,manifest}.py`, `src/comparison/datasets/`, `scripts/build_three_benchmark_figure.py` |
| Appendix E (per-layer probe) | `scripts/probe_vjepa_per_layer.py` |
| Appendix F (probe data budget) | `scripts/probe_data_efficiency.py` |
| Appendix J (ID supervision fraction) | `--id-mask-p` of the +ID training scripts, `scripts/probe_sparsity_single.py`, `scripts/build_sparsity_figure.py` |
| Appendix K (probe subspace) | `scripts/cca_trajectory_figure.py` |
| Table 4, Appendix N (closed loop) | [`closed_loop/`](closed_loop) |

## Notes

- **Probe coverage.** The absolute probe results (Table 1, Figures 1–3, robustness) fit the probe on the first 400 training episodes (10 of 104 training tasks) and evaluate on the first 200 held-out episodes (4 of 26 held-out tasks). The matched controls use all 26 held-out tasks.
- **Sampling depends on the data loader.** Clip start frames are drawn inside DataLoader workers, so probe values can change with the number of workers, the encode batch size (8), or the environment. Re-running the frozen V-JEPA 2 probe on another machine gave 0.455 instead of 0.400.
- **Matched controls.** The final Table 2 numbers come from a re-run on a GCP VM with the submitted probe evaluator, raw predictions saved, and gradient accumulation (effective batch 16). This repository contains the original-cluster implementation of the same conditions and `scripts/camera_ready/verify_gcp_primary_contrasts.py`, which recomputes the reported contrasts from the recorded per-cell results.
- **Dreamer 4.** The Dreamer 4 rows build on [nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4) with a small auxiliary-loss patch; the repository and the patch are not included. We treat these rows as exploratory.
- **Hard-coded values.** `scripts/build_probe_efficiency_figure.py` contains the three reference probe values it plots.
- `scripts/build_pareto_figure.py` draws an earlier version of Figure 1; the camera-ready Figure 1 is `paper_figures/src/build_fig1_pixel_vs_action.py`.

## Citation

```bibtex
@inproceedings{yeom2026actionrelevant,
  title     = {What Makes Video World Model Latents Action-Relevant: Prediction over Reconstruction},
  author    = {Yeom, Jewon and Kim, Hanseul and Park, Jeongjae and Jung, Sungmok and Lee, Jaejin and Kim, Taesup},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```

## License

MIT ([LICENSE](LICENSE)). `closed_loop/perturbation.py` and `closed_loop/libero_ood/` are adapted from [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) (MIT). `closed_loop/scripts/convert_pi0_weights.py` wraps the checkpoint conversion of [openpi](https://github.com/Physical-Intelligence/openpi) (Apache-2.0).
