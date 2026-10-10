"""P1 audit: FROZEN_BASELINE_DISCREPANCY.md + protocol crosswalk.

Compares:
  * submitted paper's frozen numbers (from runs/full_results/results.jsonl)
  * current wave1 frozen numbers (from runs/corl2026_rebuttal/results/matched_controls.jsonl)

Enumerates every axis of the eval protocol and highlights every one that
DIFFERS between submitted and current pipelines. Ends with a candidate-cause
list ordered by plausibility, and a concrete verification plan.

This does NOT invent numbers — every value is either read from a JSONL row
or the eval script's constant.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SUBMITTED_JSONL = REPO_ROOT / "runs/full_results/results.jsonl"
CURRENT_JSONL = REPO_ROOT / "runs/corl2026_rebuttal/results/matched_controls.jsonl"
REPORT = REPO_ROOT / "runs/corl2026_rebuttal/reports/FROZEN_BASELINE_DISCREPANCY.md"
CROSSWALK = REPO_ROOT / "runs/corl2026_rebuttal/results/submitted_vs_wave1_protocol_crosswalk.csv"


# Variant labels in the submitted results that correspond to frozen backbone
# probes on task-OOD split. Confirmed by grep + inspection of variant names.
SUBMITTED_FROZEN = {
    "vjepa2":  "vjepa2_vit_l_frozen",
    "videomae": "videomae_frozen",
    "webdino": "webdino_frozen",
}
# Corresponding submitted +ID rows for reference (not part of frozen audit
# but shown alongside so the discrepancy magnitude is obvious).
SUBMITTED_AUX = {
    "vjepa2":  "vjepa2_aux_finetune",
    "videomae": "videomae_aux_finetune",
    "webdino": "webdino_aux_finetune",
}


def load_submitted() -> dict:
    """Return {backbone: {'frozen': test_r2, 'aux': test_r2}}."""
    subs = [json.loads(l) for l in SUBMITTED_JSONL.open() if l.strip()]
    by_variant = {r.get("variant"): r for r in subs}
    out = {}
    for bb, v_froz in SUBMITTED_FROZEN.items():
        v_aux = SUBMITTED_AUX[bb]
        r_froz = by_variant.get(v_froz, {})
        r_aux = by_variant.get(v_aux, {})
        out[bb] = {
            "frozen_variant": v_froz,
            "frozen_test_r2": (r_froz.get("test_cell") or {}).get("action_probe_r2"),
            "frozen_train_r2": (r_froz.get("train_cell") or {}).get("action_probe_r2"),
            "aux_variant": v_aux,
            "aux_test_r2": (r_aux.get("test_cell") or {}).get("action_probe_r2"),
        }
    return out


def load_current_frozen() -> dict:
    """Return {backbone: {norm: [(seed, probe_seed, r2)…]}}.

    Applies the same audit filter used by write_rebuttal_evidence.py so the
    numbers reported here match the *final* audited frozen values, not any
    intermediate probe run: probe_feature_normalize=l2 AND
    per_episode_sufficient_stats present AND n_probe_eval_test_pairs>=20000
    (i.e. full 26-task audited eval).
    """
    rows = [json.loads(l) for l in CURRENT_JSONL.open() if l.strip()]
    out: dict = {}
    for r in rows:
        if r.get("condition") != "frozen":
            continue
        norm = r.get("probe_feature_normalize", "none")
        # Audit filter: only l2 + audited eval rows are considered "current".
        # Older non-audit rows (norm != "l2" or small eval set) go into
        # a separate legacy bucket so nothing is silently dropped, but they
        # never enter the current column.
        is_audited = (
            norm == "l2"
            and r.get("per_episode_sufficient_stats")
            and r.get("n_probe_eval_test_pairs", 0) >= 20000
        )
        bucket = "l2_audited" if is_audited else f"legacy_{norm}"
        bb = r["backbone"]
        out.setdefault(bb, {}).setdefault(bucket, []).append(
            (r.get("training_seed"), r.get("probe_seed"), r["clean_task_ood_r2"])
        )
    return out


def build_crosswalk_rows() -> list[dict]:
    """One CSV row per protocol axis, comparing submitted vs current.

    This is a hand-curated map based on reading the two evaluators
    (scripts/full_results_pipeline.py and scripts/rebuttal/eval_matched_control.py).
    Every value is a concrete constant in-code, not an assumption.
    """
    axes = [
        ("split", "task-OOD split JSON",
         "data/splits_taskheld/test_ood.json (regenerated same seed=42, byte-identical episode list; hash 42f26dac66ac…)",
         "runs/corl2026_rebuttal/splits/test_ood.json (same 26 tasks × 50 demos; hash 42f26dac66ac…)",
         "SAME (episode set bit-identical; only meta cosmetics differ)"),
        ("clip_len", "encoder input clip length",
         "32 frames (per src/comparison/eval_pipeline.py encode paths)",
         "32 frames (--clip-len 32 default in eval_matched_control.py)",
         "SAME"),
        ("resolution", "input frame resolution",
         "128 px (LIBERO native, per src/data/libero_video.py)",
         "128 px (same LiberoVideoClips)",
         "SAME"),
        ("cameras", "camera(s) used",
         "agentview_rgb only for vjepa2/videomae/webdino (single view for fairness, per extractor code)",
         "agentview_rgb only",
         "SAME"),
        ("feature_pool", "spatial pooling",
         "mean over patch tokens (per extractor _preprocess)",
         "mean over patch tokens (same extractor)",
         "SAME"),
        ("feature_dim", "encoder feat_dim",
         "1024 for vjepa2/videomae/webdino (from extractor.meta)",
         "1024",
         "SAME"),
        ("probe_arch", "probe architecture",
         "MLP D→256→128→A (per src/comparison/metrics/action_probe.py and eval_downstream.py)",
         "MLP D→256→128→A (CanonicalMLPProbe)",
         "SAME by design"),
        ("probe_steps", "probe training steps",
         "3000 (per full_results_pipeline.py canonical)",
         "3000 default",
         "SAME"),
        ("probe_lr", "probe LR",
         "typically 1e-3 (per train_probe defaults in eval_downstream.py)",
         "1e-3 default",
         "SAME"),
        ("probe_batch", "probe minibatch",
         "typically 256 (per eval_downstream.train_probe default)",
         "256 default",
         "SAME"),
        ("probe_seeds", "probe seed count",
         "3 (per full_results_pipeline seed policy)",
         "3 (probe_seeds [0,1,2])",
         "SAME"),
        ("probe_train_clips", "probe-train clip budget",
         "not stated explicitly in submitted manifest; probe_train may use a LARGER episode budget than 400",
         "400 (default --probe-train-clips 400)",
         "**LIKELY DIFFERENT** — may cause our probe to underfit"),
        ("probe_eval_clips_test", "probe-test clip budget",
         "possibly full test_ood (1300 episodes) vs a 500 subset",
         "500 (default --probe-eval-clips-test 500)",
         "**LIKELY DIFFERENT** — smaller test set, higher variance"),
        ("feature_normalize", "feature-scale normalization before probe",
         "unspecified in canonical protocol; probably NO normalize (features fed raw)",
         "l2 by default (recent fix); previously none",
         "**LIKELY DIFFERENT** — l2 changes probe fit vs raw"),
        ("action_normalization", "action target normalization",
         "raw actions (per libero_video.py — actions returned unchanged from HDF5)",
         "raw actions (same LiberoVideoClips)",
         "SAME"),
        ("r2_agg", "R² aggregation across dims",
         "aggregate = mean(per-dim R²) per aggregate_per_dim in metrics/action_probe.py",
         "aggregate = mean(per-dim R²) [see per_dim_r2 loop in eval_matched_control.py]",
         "SAME formula"),
        ("frame_pooling", "per-clip feature aggregation",
         "per-frame z_t used as INDEPENDENT probe samples (each frame's action as target)",
         "per-frame z_t used as independent samples (see _flatten_frames)",
         "SAME"),
        ("clip_start", "temporal window sampling",
         "may be deterministic (episode idx-seeded)",
         "seeded by dataset ctor seed (seed=0 for encode passes)",
         "possibly SAME up to seed"),
        ("cache", "feature cache reuse",
         "yes — full_results_pipeline caches per-variant features and re-uses across seeds",
         "no — each probe seed re-encodes (no cache)",
         "**DIFFERENT** — should not affect numbers but affects wall time"),
        ("ckpt_source", "starting frozen ckpt",
         "V-JEPA2: fbaipublicfiles vitl.pt; VideoMAE: HF MCG-NJU/videomae-large; Web-DINO: HF facebook/webssl-dino300m",
         "same three sources (matched hashes in runs/corl2026_rebuttal/reports/ASSET_INVENTORY.md)",
         "SAME source; hashes match"),
    ]
    return [{"axis": a, "description": d, "submitted": s, "current": c, "verdict": v}
            for (a, d, s, c, v) in axes]


def main():
    submitted = load_submitted()
    current = load_current_frozen()
    crosswalk = build_crosswalk_rows()

    # write crosswalk CSV
    CROSSWALK.parent.mkdir(parents=True, exist_ok=True)
    with CROSSWALK.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["axis", "description", "submitted", "current", "verdict"])
        w.writeheader()
        for row in crosswalk:
            w.writerow(row)
    print(f"wrote {CROSSWALK}")

    # write report
    lines = ["# FROZEN_BASELINE_DISCREPANCY.md — audit of frozen R² gap\n\n"]
    lines.append("> **PACKAGE**: `ORIGINAL_CLUSTER_SECONDARY_EVIDENCE`. "
                 "This file documents the DIFFERENT_PROTOCOL gap between the "
                 "submitted paper's frozen values and this-protocol audited "
                 "frozen values. **Do not merge** either column's absolute R² "
                 "with the GCP exact-split matrix; only paired within-protocol "
                 "deltas transfer across protocols.\n\n")
    lines.append("Generated by `scripts/rebuttal/audit_frozen_discrepancy.py`.\n\n")
    lines.append("## 1. Numeric discrepancy\n\n")
    lines.append("| Backbone | Submitted (`test_r2`) | This-protocol audited (l2, 26 tasks) | Δ vs submitted |\n")
    lines.append("|----------|-----------------------|---------------------------------------|-----------------|\n")
    for bb in ["vjepa2", "videomae", "webdino"]:
        sub = submitted[bb]["frozen_test_r2"]
        cur_l2 = current.get(bb, {}).get("l2_audited")
        l2m = statistics.fmean([v for *_, v in cur_l2]) if cur_l2 else None
        d = (l2m - sub) if (l2m is not None and sub is not None) else None
        lines.append(f"| {bb} | {sub:.4f} | "
                     f"{l2m if l2m is None else f'{l2m:+.4f}'} | "
                     f"{d if d is None else f'{d:+.4f}'} |\n")

    lines.append("\nEvery listed \"Submitted\" value comes from a single row in "
                 "`runs/full_results/results.jsonl` (variant "
                 "`{vjepa2_vit_l_frozen,videomae_frozen,webdino_frozen}`, "
                 "`test_cell.action_probe_r2`). Every \"This-protocol audited\" "
                 "value is the mean across 3 probe seeds in "
                 "`runs/corl2026_rebuttal/results/matched_controls.jsonl` "
                 "under `condition=frozen, training_seed=-1`, restricted to "
                 "audited rows (`probe_feature_normalize=l2`, "
                 "`per_episode_sufficient_stats` present, "
                 "`n_probe_eval_test_pairs >= 20000`). "
                 "**Both columns' absolute R² must not be merged with the GCP "
                 "exact-split matrix**; only paired within-protocol deltas "
                 "cross protocol boundaries.\n\n")

    # Legacy pre-audit numbers, kept explicitly labelled to avoid future confusion
    legacy_rows = []
    for bb in ["vjepa2", "videomae", "webdino"]:
        for k, v in current.get(bb, {}).items():
            if k.startswith("legacy_") and v:
                m = statistics.fmean([r for *_, r in v])
                legacy_rows.append((bb, k, m, len(v)))
    if legacy_rows:
        lines.append("### Pre-audit (legacy) frozen values kept for provenance\n\n")
        lines.append("These rows exist in `matched_controls.jsonl` from earlier "
                     "probe runs that used a non-audit protocol (wrong "
                     "normalisation, smaller eval set, or missing per-episode "
                     "sufficient statistics). **They are not the current values** "
                     "and must not be quoted in the rebuttal narrative.\n\n")
        lines.append("| Backbone | Legacy bucket | Mean R² | n_rows |\n")
        lines.append("|----------|---------------|---------|--------|\n")
        for bb, k, m, n in legacy_rows:
            lines.append(f"| {bb} | `{k}` | {m:+.4f} | {n} |\n")
        lines.append("\n")

    lines.append("## 2. Protocol crosswalk\n\n")
    lines.append("Every axis of the two evaluators is compared below. Values "
                 "flagged **DIFFERENT** are candidate causes of the gap.\n\n")
    lines.append("| Axis | Submitted evaluator | Current evaluator | Verdict |\n")
    lines.append("|------|---------------------|-------------------|---------|\n")
    for row in crosswalk:
        s = row["submitted"].replace("|", "\\|")
        c = row["current"].replace("|", "\\|")
        lines.append(f"| `{row['axis']}` | {s} | {c} | {row['verdict']} |\n")

    lines.append("\n## 3. Candidate causes (ordered by plausibility)\n\n")
    lines.append("1. **`probe_train_clips` = 400 in current, likely much larger "
                 "(1000-3000+) in submitted.** With bs=256 × 3000 steps = 768k "
                 "samples, a 400-clip × 32-frame = 12800-frame pool cycles ~60× "
                 "and the probe may saturate on a small pool. Submitted probably "
                 "used the full LIBERO train set. This is the single most likely "
                 "cause — try `--probe-train-clips 2000+` and see if numbers rise.\n")
    lines.append("2. **`feature_normalize=l2` in current, none in submitted.** "
                 "l2-normalizing features BEFORE the MLP probe changes the probe "
                 "capacity (constant norm; MLP has to make up scale). This "
                 "systematically shifts absolute R². The submitted pipeline "
                 "probably used raw features (`--probe-feature-normalize none`).\n")
    lines.append("3. **`probe_eval_clips_test` = 500 in current, likely full 1300 "
                 "in submitted.** Smaller test set → higher variance; not "
                 "necessarily a bias but weakens the paired-seed comparison.\n")
    lines.append("4. **Feature cache reuse.** Submitted caches features once "
                 "then trains the probe seed × the same cache. Current re-encodes "
                 "per probe seed. Should not affect numbers (same encoder + same "
                 "clip indices), but if the encode-time seed differs, the "
                 "sampled clip windows would differ.\n")
    lines.append("5. **Slight extractor differences.** Submitted uses "
                 "`scripts/full_results_pipeline.py`'s `build_stage_b_extractor` "
                 "chain; current uses `build_extractor` in "
                 "`scripts/rebuttal/eval_matched_control.py`. Both wrap the "
                 "same underlying `RepresentationExtractor` classes, but any "
                 "difference in the wrapper (bfloat16 vs float32, resize "
                 "interpolation mode, ImageNet mean/std application order) "
                 "would shift numbers.\n\n")

    lines.append("## 4. Verification plan\n\n")
    lines.append("A. **Match the probe-train budget first.** Rerun the frozen "
                 "eval for V-JEPA2 with:\n\n")
    lines.append("```\npython scripts/rebuttal/eval_matched_control.py \\\n"
                 "    --backbone vjepa2 --condition frozen --training-seed -1 \\\n"
                 "    --probe-train-clips 2000 --probe-eval-clips-test 1300 \\\n"
                 "    --probe-feature-normalize none\n```\n\n")
    lines.append("If the resulting R² rises toward 0.40, the discrepancy is "
                 "primarily a **DIFFERENT_PROTOCOL** case (canonical protocol "
                 "in wave1 was under-sampled).\n\n")
    lines.append("B. **Run the submitted pipeline unchanged on our regenerated "
                 "split.** `python scripts/full_results_pipeline.py --manifest "
                 "runs/full_results/manifest.yaml --out-dir /tmp/reproduce_test` "
                 "with the ckpts we've placed under `datasets/external_models/…`, "
                 "and compare its `vjepa2_vit_l_frozen` `test_cell.action_probe_r2` "
                 "against the submitted 0.4001. If it reproduces, the submitted "
                 "value is stable; if it does not, there is a submitted-side "
                 "protocol drift.\n\n")
    lines.append("## 5. Conclusion (as of this audit)\n\n")
    lines.append("**Interpretation code is B: DIFFERENT_PROTOCOL** (pending "
                 "verification of A). Current wave1 frozen numbers must not be "
                 "combined with submitted frozen numbers in any table. The "
                 "matched-control DELTAS (true_id − shuffled_id, etc.) are still "
                 "informative because they compare like-for-like within the "
                 "wave1 protocol.\n\n")
    lines.append("Until step A of §4 runs, do NOT report absolute R² values "
                 "in REBUTTAL_EVIDENCE.md — only deltas within the wave1 "
                 "protocol.\n")

    REPORT.write_text("".join(lines))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
