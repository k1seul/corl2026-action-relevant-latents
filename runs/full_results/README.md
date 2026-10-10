# Probe evaluation pipeline (Table 1, robustness and per-variant appendix tables)

Per-variant action-probe (and, for pixel models, rollout) metrics on the LIBERO task-held-out split.

## Files

- `manifest.yaml` (LIBERO), `manifest_calvin.yaml`, `manifest_metaworld.yaml`: every
  (backbone x variant) evaluated, with its checkpoint path and training settings.
- The pipeline writes `results.jsonl` here (one line per variant and split) and the
  rendered tables. The recorded results used for the paper are in
  `paper_figures/data/results.jsonl`.

## Pipeline

```bash
uv run python scripts/build_full_results_manifest.py
CUDA_VISIBLE_DEVICES=N uv run python scripts/full_results_pipeline.py
uv run python scripts/render_full_results.py
```

Pipeline resumes from `results.jsonl` if interrupted.

## Probe split protocol

To avoid leakage when reporting the "train" cell:

- **probe-train**: `splits_taskheld/train.json` episodes `[0 : 400]`
- **probe-eval-train** (= "train" cell): same file, episodes `[400 : 600]`
- **probe-eval-test_ood** (= "test" cell): `splits_taskheld/test_ood.json`
  episodes `[0 : 200]`

No episode overlap between probe-train and probe-eval-train, so the
train cell measures fit, not memorization.

## Training hyperparams (also in each CSV)

| Column | Source | Notes |
|---|---|---|
| `batch_size` | `cfg.train.batch_size` (Stage A) / `args.batch_size` in ckpt (Stage B) | per-GPU; multiply by world_size for effective |
| `max_steps` | `cfg.train.max_steps` | target step count when run launched |
| `lr` | `cfg.train.lr` | peak LR after warmup |
| `training_step_at_ckpt` | ckpt's `step` field | actual step when `ckpt_last.pt` was saved; should equal `max_steps` for completed runs |

Frozen baselines (`vjepa2_vit_l_frozen`, `vjepa2_1_vit_b_frozen`, `lapa_frozen`)
leave these blank — they are not trained in this study.

## Metrics

| Column | Definition | Available for |
|---|---|---|
| `recon_mse` | pixel MSE in [0,1] on the predicted span of a prefix→post rollout | DIFF / TOKEN |
| `psnr` | −10·log₁₀(recon_mse) | DIFF / TOKEN |
| `lpips` | AlexNet LPIPS, predicted span vs GT (inputs scaled to [-1,1]) | DIFF / TOKEN |
| `action_probe_mse` | mean of per-dim MSE | all |
| `action_probe_r2` | mean of per-dim R² | all |
| `dx, dy, dz` | per-dim translation MSE | all |
| `drx, dry, drz` | per-dim rotation MSE | all |
| `gripper` | gripper-dim MSE | all |
| `r2_<dim>` | per-dim R² | all |

Action probe is an MLP `[in → 256 → 128 → 7]` trained 3000 steps, batch
512, AdamW lr=3e-4 wd=1e-4. Reported numbers are the mean over 3 seeds
on the held-out evaluation features.

Rollout protocol: first 8 frames as prefix, rest predicted; 50-step
DDIM for DIFF, auto-regressive argmax for TOKEN, single-shot block-
causal for MAE.

## Caveats

- **LPIPS only for pixel models.** V-JEPA / LAPA / frozen-ViT variants
  produce latent features only — their `recon_mse`, `psnr`, `lpips` are
  reported as `NA`. Their action-probe columns are the only meaningful
  cells.
- **"Train cell" ≠ training-time train metric.** It is a probe evaluated
  on a held-out *episode subset of the same train data split*. Measures
  fit to the train distribution, not generalization, and not the loss
  the network was actually minimizing during training.
- **Per-dim MSE uses raw action values** (no normalization). The dims
  have different natural ranges (translation ≈ 0.1, gripper ≈ ±1), so
  absolute MSE values are not directly comparable across dims. R² is
  the unit-free comparison.
- **LIBERO 7-D action mapping** `[dx, dy, dz, drx, dry, drz, gripper]`.
  See `src/comparison/metrics/action_probe.py` for the canonical order.
  For Robomimic the mapping is identical; CALVIN is also 7-D abs-action
  in the same order; MetaWorld is 4-D natively (xyz + gripper) zero-
  padded to 7-D, so its rotation columns are exactly zero by
  construction.
- **All metrics in feature-pooled-mean space.** Features are mean-
  pooled over the spatial token grid before the action probe — spatial
  patterns are not directly evaluated here.
- **2-camera DIFF variants** (`A_diff_v6_action_2cam_100k`, the aux
  λ-sweep) use `(agentview, eye_in_hand)`; everything else uses
  `agentview` only. Recon cell reports per-frame MSE averaged over
  cameras, weighted equally.
- **Action-conditioned models** are evaluated with `actions=None` for
  the probe (no information leak). For the **recon rollout** they
  receive GT actions from the same clip — action-conditioned models
  assert when given `actions=None`, and skipping them entirely produces
  NA cells that disagree with their non-action peers. Passing GT actions
  matches the standard Dreamer/IRIS protocol for rollout fidelity
  evaluation. Non-action-conditioned variants ignore the actions arg
  regardless.
- **Short clips**: when `clip_len ≤ rollout_prefix` (only `q1_clip_8`
  in this table — clip_len=8, default prefix=8), the prefix is capped
  at `clip_len − 1` so at least one post frame is predicted. Recon
  cells for this row reflect a 1-frame rollout, which is harder than
  the default 24-frame span and shows lower PSNR.
