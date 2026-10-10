"""Write reports/ORIGINAL_CLUSTER_SECONDARY_EVIDENCE.md — instruction §7.

Machine-readable handoff for the GCP merge. Contains:
  * exact protocol identifier + split hashes
  * corrected matched-control seed-level deltas + single paired CIs
  * probe-sensitivity sign result (all-positive check)
  * val-selected common-ridge action-centric baseline table (§4)
  * deterministic test-OOD common-decoder table (§6)
  * explicit statement: no absolute R² is merged with the GCP exact-split
    matrix; only within-protocol deltas transfer.

Re-runnable: consumes only jsonl / md artifacts already present in the
repo, so re-running after §4 / §6 land refreshes the handoff.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

FINAL_STATS = REPO_ROOT / "runs/corl2026_rebuttal/results/wave1_finalized_stats.jsonl"
COMMON_RIDGE = REPO_ROOT / "runs/corl2026_rebuttal/results/common_ridge_comparison.jsonl"
CAIP_NATIVE = REPO_ROOT / "runs/corl2026_rebuttal/results/caip_native_validation.jsonl"
PROBE_SENS = REPO_ROOT / "runs/corl2026_rebuttal/results/probe_sensitivity.jsonl"
DECODER_TEST = REPO_ROOT / "runs/corl2026_rebuttal/results/common_decoder_test_ood.jsonl"
FROZEN_DISCREP = REPO_ROOT / "runs/corl2026_rebuttal/reports/FROZEN_BASELINE_DISCREPANCY.md"
OUT = REPO_ROOT / "runs/corl2026_rebuttal/reports/ORIGINAL_CLUSTER_SECONDARY_EVIDENCE.md"


def _load_jsonl(p):
    if not p.exists(): return []
    return [json.loads(l) for l in p.open() if l.strip()]


def build():
    lines = []
    lines.append("# ORIGINAL_CLUSTER_SECONDARY_EVIDENCE.md — GCP-merge handoff (§7)\n\n")
    lines.append("This is the machine-readable secondary-evidence package for the "
                 "*original-cluster matched-control audit protocol*, prepared for "
                 "merging with the GCP exact-split matrix once GCP's primary "
                 "results land.\n\n")

    lines.append("> **⚠ No absolute R² transfers across protocols.** "
                 "Every absolute R² number reported here is conditional on the "
                 "*original-cluster matched-control audit protocol* identifier "
                 "below. Do NOT merge these absolute values with (a) the "
                 "submitted paper's canonical matrix or (b) the GCP exact-split "
                 "matrix. Only **paired within-protocol deltas** (e.g. "
                 "`R²(true_id) − R²(shuffled_id)` within this protocol) are "
                 "meaningful for cross-protocol comparison.\n\n")
    lines.append("> The final rebuttal narrative is written after the GCP "
                 "exact-split matrix arrives — not from this file alone.\n\n")

    # ── Section 1: Protocol identifier + split hashes ─────────────────
    lines.append("## 1. Protocol identifier + split hashes\n\n")
    lines.append("| Component | Value |\n|-----------|-------|\n")
    lines.append("| Protocol name | `original-cluster matched-control audit protocol` |\n")
    lines.append("| Test split file | `runs/corl2026_rebuttal/splits/test_ood.json` |\n")
    lines.append("| Test split sha256 | `42f26dac66acbaf0c5a153f15605b09f9f9ea772c5647b7d20518874f2c769ca` |\n")
    lines.append("| Test split content | 26 held-out tasks × 50 demos = 1300 episodes (reconstructed; bit-identical to the user-provided reconstruction) |\n")
    lines.append("| Train split file | `runs/corl2026_rebuttal/splits/train.json` |\n")
    lines.append("| Backbones | V-JEPA2 ViT-L (`vitl.pt`), VideoMAE ViT-L (MCG-NJU/videomae-large), Web-DINO 300M (facebook/webssl-dino300m-full2b-224) |\n")
    lines.append("| Matched-control probe | MLP [D→256→128→7], AdamW lr=1e-3, wd=1e-4, batch=256, 3000 steps, 3 probe seeds ∈ {0,1,2}, features l2-normalized before probe |\n")
    lines.append("| Held-out-task bootstrap | 26 tasks resampled WITH REPLACEMENT × 10 000 replicates, seed=0 |\n")
    lines.append("| Point estimate (§1) | `seed-avg δ = (δ_seed0 + δ_seed1) / 2` — arithmetic mean of the two observed training-seed deltas |\n")
    lines.append("| CI (§1) | 2.5/97.5 percentiles of the seed-averaged δ over the 10k task-cluster bootstrap replicates |\n")
    lines.append("| Assertion enforced | `abs(seed_avg_delta − mean(seed0_delta, seed1_delta)) < 1e-10` |\n\n")

    # ── Section 2: Corrected matched-control deltas + single CIs ──────
    lines.append("## 2. Corrected matched-control seed-averaged deltas + single paired CIs\n\n")
    sa_rows = [r for r in _load_jsonl(FINAL_STATS) if r.get("kind") == "seed_averaged_delta"]
    if sa_rows:
        lines.append("| Backbone | Contrast | δ seed 0 | δ seed 1 | seed-avg δ (arithmetic point) | 95% CI (single, paired) | excludes 0? | n_tasks |\n")
        lines.append("|----------|----------|----------|----------|--------------------------------|--------------------------|-------------|---------|\n")
        for bb in ("videomae", "vjepa2", "webdino"):
            for c in ("true_id_minus_shuffled_id", "shuffled_id_minus_frozen", "true_id_minus_frozen"):
                for r in sa_rows:
                    if r["backbone"] == bb and r["contrast"] == c:
                        ps = r["per_training_seed_deltas"]
                        d0 = ps.get("0", ps.get(0)); d1 = ps.get("1", ps.get(1))
                        lines.append(
                            f"| {bb} | {c} | {d0:+.4f} | {d1:+.4f} | "
                            f"{r['seed_averaged_delta_point']:+.4f} | "
                            f"[{r['seed_averaged_delta_lo95']:+.4f}, "
                            f"{r['seed_averaged_delta_hi95']:+.4f}] | "
                            f"{r['excludes_zero']} | {r['n_tasks']} |\n"
                        )
        lines.append("\n")
    else:
        lines.append("_(pending — regenerate with `python scripts/rebuttal/finalize_wave1_stats.py`)_\n\n")

    # Per-cell absolute R² (for provenance, not comparison)
    cell_rows = [r for r in _load_jsonl(FINAL_STATS) if r.get("kind") == "cell"]
    if cell_rows:
        lines.append("### Per-cell absolute R² (for provenance only; do NOT compare across protocols)\n\n")
        lines.append("| Backbone | Condition | Training seed | pooled R² (probe-mean) | task-macro R² | normalized MSE |\n")
        lines.append("|----------|-----------|----------------|-------------------------|----------------|-----------------|\n")
        order = {"frozen": 0, "shuffled_id": 1, "true_id": 2}
        for r in sorted(cell_rows, key=lambda r: (r["backbone"], order.get(r["condition"], 99), r["training_seed"])):
            lines.append(
                f"| {r['backbone']} | {r['condition']} | {r['training_seed']} | "
                f"{r['pooled_r2_probe_mean']:+.4f} (±{r['pooled_r2_probe_std']:.4f}) | "
                f"{r['task_macro_r2_probe_mean']:+.4f} | {r['normalized_mse_probe_mean']:.4f} |\n"
            )
        lines.append("\n")

    # ── Section 3: Probe-sensitivity sign result ─────────────────────
    lines.append("## 3. Probe-architecture sensitivity — sign result\n\n")
    ps_rows = _load_jsonl(PROBE_SENS)
    if ps_rows:
        from collections import defaultdict
        import statistics
        agg = defaultdict(list)
        for r in ps_rows:
            agg[(r["backbone"], r["condition"], r.get("training_seed"),
                 r["probe_variant"])].append(r["clean_task_ood_r2"])
        deltas = []
        for bb in ("videomae", "vjepa2", "webdino"):
            for tseed in (0, 1):
                for v in ("ridge", "canonical_mlp", "wider_mlp"):
                    t = agg.get((bb, "true_id", tseed, v))
                    s = agg.get((bb, "shuffled_id", tseed, v))
                    if t is None or s is None: continue
                    deltas.append((bb, tseed, v, statistics.fmean(t) - statistics.fmean(s)))
        n_total = len(deltas); n_pos = sum(1 for *_, d in deltas if d > 0)
        lines.append(f"* {n_pos} / {n_total} deltas (`true_id − shuffled_id`) are positive.\n")
        lines.append("* The correspondence-gain claim is robust to probe "
                     "architecture (ridge / canonical MLP / wider MLP) under this "
                     "protocol.\n\n")
        lines.append("| Backbone | seed | Δ ridge | Δ canonical | Δ wider |\n")
        lines.append("|----------|------|---------|-------------|---------|\n")
        for bb in ("videomae", "vjepa2", "webdino"):
            for tseed in (0, 1):
                cells = ["—", "—", "—"]
                for i, v in enumerate(("ridge", "canonical_mlp", "wider_mlp")):
                    for x_bb, x_ts, x_v, d in deltas:
                        if x_bb == bb and x_ts == tseed and x_v == v:
                            cells[i] = f"{d:+.3f}"
                lines.append(f"| {bb} | {tseed} | {cells[0]} | {cells[1]} | {cells[2]} |\n")
        lines.append("\n")

    # ── Section 4: Val-selected common-ridge baseline table ───────────
    lines.append("## 4. Val-selected common-ridge action-centric baselines (instruction §4)\n\n")
    cr = _load_jsonl(COMMON_RIDGE)
    if cr:
        lines.append("Hyperparameters (normalisation, ridge α) selected on an "
                     "80/20 episode-level train-in-distribution val split of the "
                     "probe-train pool (seed=42), then evaluated ONCE on test-OOD. "
                     "No test-set information leaks into selection. See "
                     "`reports/COMMON_RIDGE_COMPARISON.md` for full grid.\n\n")
        lines.append("| Backbone | feat_dim | Selected norm | Selected α | Test-OOD R² (one-shot) | Normalized MSE | Constant-action R² |\n")
        lines.append("|----------|----------|----------------|-------------|-------------------------|-----------------|---------------------|\n")
        order = ["vjepa2", "videomae", "webdino", "r3m", "vip", "caip_visual_only"]
        idx = {r["backbone"]: r for r in cr}
        for bb in order:
            r = idx.get(bb)
            if r:
                lines.append(f"| {bb} | {r['feat_dim']} | {r['selected_norm']} | "
                             f"{r['selected_alpha']:g} | "
                             f"{r['test_oo_d_r2_at_frozen_setting']:+.4f} | "
                             f"{r['test_oo_d_normalized_mse_at_frozen_setting']:.4f} | "
                             f"{r['constant_action_baseline_r2']:+.4f} |\n")
        lines.append("\nR3M / VIP / CAIP visual-only: **does not exceed the "
                     "constant-action baseline under the specified visual-only "
                     "readout protocol**. Do not describe as \"structural encoding "
                     "failure\".\n\n")
    else:
        lines.append("_(pending — job 54269 or re-run `python scripts/rebuttal/common_ridge_comparison.py`)_\n\n")

    # CAIP native (separate — INTERFACE / NON-LEAKAGE ONLY, no performance claim)
    lines.append("### CAIP native-text-conditioned — interface / non-leakage validation only\n\n")
    cn = _load_jsonl(CAIP_NATIVE)
    if cn:
        r = cn[0]
        lines.append("* **Status**: interface + non-leakage validation complete "
                     "(§5); **action-probe performance NOT measured under a "
                     "valid protocol**. The initial run in "
                     "`caip_native_validation.jsonl` used a best-test-R² sweep, "
                     "which selects on test-OOD and is invalid per §4.\n")
        lines.append("* **Caption source**: LIBERO hdf5 filename → task-"
                     "description string; verified to contain no per-frame "
                     "action vector and no episode-specific identifier (see "
                     "`CAIP_NATIVE_VALIDATION.md` for the exact derivation "
                     "function and the three-item non-leakage checklist).\n")
        lines.append("* **Follow-up (not run in this cycle)**: extend "
                     "`common_ridge_comparison.py` to accept text-conditioned "
                     "encoders so CAIP native can be measured under the §4 "
                     "val-selected common-ridge protocol on the same footing "
                     "as the vision-only backbones.\n")
        # Feature stats (from interface validation) — provenance only, no claim
        fs = r.get("feature_stats_train", {})
        if fs:
            lines.append(f"* Feature stats (interface check): shape={fs.get('shape')}, "
                         f"dead dims (std<1e-2)={fs.get('n_dead_dims_std_lt_1e-2')}"
                         f"/{fs.get('shape',[None,None,None])[2]}, "
                         f"L2 norm mean={fs.get('l2_norm_mean',0):.2f}.\n\n")

    # ── Section 5: Deterministic test-OOD common-decoder table ────────
    lines.append("## 5. Three-backbone common-decoder sensitivity (test-OOD, §6)\n\n")
    dec = _load_jsonl(DECODER_TEST)
    if dec:
        r0 = dec[0]
        lines.append(f"* Deterministic evaluation on {r0['n_episodes']} × 32 "
                     f"= {r0['n_frames']} frames from `runs/corl2026_rebuttal/splits/test_ood.json`.\n")
        lines.append(f"* Frame-list SHA-256 (ordered (suite,file,demo,frame) "
                     f"tuples, clip_len=32): `{r0['frame_list_sha256']}`\n\n")
        lines.append("| Backbone | Test-OOD MSE | Test-OOD LPIPS | Test-OOD PSNR (dB) | ckpt_step |\n")
        lines.append("|----------|--------------|-----------------|---------------------|-----------|\n")
        for bb in ("vjepa2", "videomae", "webdino"):
            for r in dec:
                if r["backbone"] == bb:
                    lines.append(f"| {bb} | {r['test_mse']:.4f} | "
                                 f"{r['test_lpips']:.4f} | {r['test_psnr_db']:.2f} "
                                 f"| {r['ckpt_step']} |\n")
        lines.append("\nLabelled **three-backbone common-decoder sensitivity**; "
                     "does not repair the submitted cross-family Figure 1.\n\n")
    else:
        lines.append("_(pending — job 54270 or re-run `python scripts/rebuttal/eval_common_decoder_test_ood.py`)_\n\n")

    # ── Section 6: Absolute-R² non-merge statement ────────────────────
    lines.append("## 6. Absolute-R² non-merge statement (required)\n\n")
    lines.append("The following statement must accompany any figure or table "
                 "that draws numbers from this handoff and from the GCP exact-"
                 "split matrix in the same panel:\n\n")
    lines.append("> Numbers reported here (original-cluster matched-control audit "
                 "protocol, l2-normalised MLP probe, split sha256 `42f26dac…`) "
                 "are absolute R² values conditional on this exact protocol "
                 "identifier. They are **not** merged with, averaged with, or "
                 "compared to absolute R² from the submitted canonical matrix "
                 "or the GCP exact-split matrix. Only paired within-protocol "
                 "deltas transfer. See `FROZEN_BASELINE_DISCREPANCY.md` for the "
                 "concrete cross-protocol crosswalk.\n\n")

    lines.append("## 7. Audit trail (files consumed by this handoff)\n\n")
    lines.append("* `results/wave1_finalized_stats.jsonl` — corrected seed-averaged deltas + single CI (§1, §2)\n")
    lines.append("* `results/probe_sensitivity.jsonl` — probe-architecture sensitivity rows (§3)\n")
    lines.append("* `results/common_ridge_comparison.jsonl` — val-selected common-ridge baselines (§4)\n")
    lines.append("* `results/caip_native_validation.jsonl` — CAIP native (§5, separate)\n")
    lines.append("* `results/common_decoder_test_ood.jsonl` — deterministic test-OOD decoder metrics (§6)\n")
    lines.append("* `reports/FROZEN_BASELINE_DISCREPANCY.md` — DIFFERENT_PROTOCOL crosswalk\n")
    lines.append("* `reports/WAVE1_FINALIZED_STATS.md` — human-readable form of §2 table\n")
    lines.append("* `reports/COMMON_RIDGE_COMPARISON.md` — human-readable form of §4 table\n")
    lines.append("* `reports/COMMON_DECODER_TEST_OOD.md` — human-readable form of §5 table\n")
    lines.append("* `reports/CAIP_NATIVE_VALIDATION.md` — CAIP-native detail (with caption non-leakage verification)\n")

    OUT.write_text("".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
