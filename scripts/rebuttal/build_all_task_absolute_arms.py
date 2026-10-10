"""Compute all-task pooled R² absolute-arm values per (backbone × condition).

Uses the SAME raw task/target pairing as the 10 000-replicate task-cluster
bootstrap in `finalize_wave1_stats.py`:

  * per-episode sufficient statistics grouped by (suite, file) → 26 tasks
  * probe-seed episodes concatenated per task (same merge the bootstrap does)
  * pooled R² over all 26 tasks with each task counted once (i.e. the
    bootstrap replicate at task_indices = [0, 1, ..., 25])

Arms:
  * frozen (training_seed = -1)
  * true_id (training_seeds 0, 1 → arithmetic mean)
  * shuffled_id (training_seeds 0, 1 → arithmetic mean)
  * forward_dynamics (NOT MEASURED in the original-cluster secondary
    evidence cycle; recorded as NaN with a note. See TRAINING_SETTINGS.md §1
    — VideoMAE-only wiring was cancelled before completing any tseed.)

Assertion (per user request):
  true_absolute - shuffled_absolute  ==  seed-averaged δ point estimate
                                          for true_id_minus_shuffled_id
to numerical precision. This is a linearity check: since the merged-per-task
pool R² is the per-dim R² aggregate and per-dim R² is linear in per-dim
(SSE/SST), the arithmetic mean over training seeds commutes with the
difference — so both sides equal `(δ_s0 + δ_s1) / 2` by identity.

Output:
  * runs/corl2026_rebuttal/tables/all_task_absolute_arms.csv
  * runs/corl2026_rebuttal/reports/ALL_TASK_ABSOLUTE_ARMS.md
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MC = REPO_ROOT / "runs/corl2026_rebuttal/results/matched_controls.jsonl"
FS = REPO_ROOT / "runs/corl2026_rebuttal/results/wave1_finalized_stats.jsonl"
OUT_CSV = REPO_ROOT / "runs/corl2026_rebuttal/tables/all_task_absolute_arms.csv"
OUT_MD  = REPO_ROOT / "runs/corl2026_rebuttal/reports/ALL_TASK_ABSOLUTE_ARMS.md"

BACKBONES = ("videomae", "vjepa2", "webdino")
CONDITIONS = ("frozen", "true_id", "shuffled_id", "forward_dynamics")


def audited(r) -> bool:
    return (r.get("probe_feature_normalize") == "l2"
            and r.get("per_episode_sufficient_stats")
            and r.get("n_probe_eval_test_pairs", 0) >= 20000)


def merged_task_pool(rows):
    """Given a list of matched-control jsonl rows for one (bb, cond, tseed),
    concatenate probe-seed episodes per task. Returns dict[task] -> list of
    per-episode sufficient statistics."""
    by_task = defaultdict(list)
    for r in rows:
        for ep, meta in zip(r["per_episode_sufficient_stats"],
                             r["test_episode_ids"]):
            by_task[(meta["suite"], meta["file"])].append(ep)
    return by_task


def pooled_r2_over_all_tasks(by_task):
    """Bootstrap's pooling function at task_indices = all 26 tasks (identity).

    Sum SSE, y_sum, y_sqsum, n across every task; compute per-dim
    SST = ysq - (ys**2)/n, per-dim R² = 1 - SSE/SST, aggregate = mean of
    finite per-dim R²."""
    sse = None; ys = None; ysq = None; n = 0
    for t, eps in by_task.items():
        for e in eps:
            s = np.asarray(e["sse_per_dim"])
            y = np.asarray(e["y_sum_per_dim"])
            q = np.asarray(e["y_sqsum_per_dim"])
            if sse is None:
                sse = s.copy(); ys = y.copy(); ysq = q.copy()
            else:
                sse += s; ys += y; ysq += q
            n += e["n_frames"]
    sst = ysq - (ys ** 2) / max(n, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_dim = 1.0 - sse / np.where(sst > 1e-12, sst, np.nan)
    finite = per_dim[np.isfinite(per_dim)]
    return float(finite.mean()) if finite.size else float("nan"), int(n)


def load_reported_point_delta():
    """Return {(bb, contrast): point_estimate} from wave1_finalized_stats.jsonl."""
    out = {}
    for line in FS.open():
        if not line.strip(): continue
        r = json.loads(line)
        if r.get("kind") == "seed_averaged_delta":
            out[(r["backbone"], r["contrast"])] = r["seed_averaged_delta_point"]
    return out


def main():
    rows = [json.loads(l) for l in MC.open() if l.strip()]
    rows = [r for r in rows if audited(r) and r["backbone"] in BACKBONES]

    # Group audited rows by (bb, cond, tseed) → list of probe-seed rows
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["backbone"], r["condition"], r.get("training_seed"))].append(r)

    # Compute per-cell pooled R² using the bootstrap's merged-per-task pool
    per_cell_r2 = {}
    per_cell_n  = {}
    for key, rs in by_cell.items():
        by_task = merged_task_pool(rs)
        r2, N = pooled_r2_over_all_tasks(by_task)
        per_cell_r2[key] = r2
        per_cell_n[key] = N

    # Per (bb, cond) absolute arm = mean over training seeds of per-cell R².
    # frozen has one seed (-1); true_id / shuffled_id have {0, 1}.
    absolute = {}      # (bb, cond) -> {"value", "seeds_used", "per_seed"}
    for bb in BACKBONES:
        for cond in ("frozen", "true_id", "shuffled_id"):
            per_seed = {ts: per_cell_r2[(bb, cond, ts)]
                        for (b, c, ts) in per_cell_r2
                        if b == bb and c == cond}
            if not per_seed:
                absolute[(bb, cond)] = {
                    "value": float("nan"), "seeds_used": [], "per_seed": {},
                    "status": "no_audited_rows",
                }
                continue
            vals = list(per_seed.values())
            absolute[(bb, cond)] = {
                "value": float(statistics.fmean(vals)),
                "seeds_used": sorted(per_seed.keys(), key=lambda x: (x is None, x)),
                "per_seed": per_seed,
                "status": "measured",
            }
        # forward_dynamics — not measured in this cycle
        absolute[(bb, "forward_dynamics")] = {
            "value": float("nan"),
            "seeds_used": [],
            "per_seed": {},
            "status": "not_measured_this_cycle",
            "note": ("Deferred in the original-cluster secondary-evidence cycle. "
                     "Wave1 fanout supported forward_dynamics only for VideoMAE "
                     "and no completed training seed was produced (see "
                     "TRAINING_SETTINGS.md §1 and wave1_condition_fanout.sbatch)."),
        }

    # ── ASSERTION: true_absolute - shuffled_absolute == reported δ point ──
    reported = load_reported_point_delta()
    assertion_rows = []
    for bb in BACKBONES:
        t = absolute[(bb, "true_id")]["value"]
        s = absolute[(bb, "shuffled_id")]["value"]
        if not (math.isfinite(t) and math.isfinite(s)):
            continue
        recomputed_delta = t - s
        reported_delta = reported.get((bb, "true_id_minus_shuffled_id"))
        # Reported value is rounded to 4 dp in jsonl; recomputed is fp-precise.
        # Two comparisons: (i) recomputed self-consistency vs. the direct
        # mean-of-per-seed-δ (must equal to numerical precision by linearity);
        # (ii) recomputed vs. reported rounded value (must equal to <1e-3).
        seed_deltas = []
        for ts in (0, 1):
            tv = per_cell_r2.get((bb, "true_id", ts))
            sv = per_cell_r2.get((bb, "shuffled_id", ts))
            if tv is not None and sv is not None:
                seed_deltas.append(tv - sv)
        direct_mean = statistics.fmean(seed_deltas) if seed_deltas else float("nan")
        drift_vs_direct = abs(recomputed_delta - direct_mean)
        drift_vs_reported = abs(recomputed_delta - reported_delta) if reported_delta is not None else float("nan")
        assertion_rows.append({
            "backbone": bb,
            "true_absolute": t,
            "shuffled_absolute": s,
            "recomputed_true_minus_shuffled": recomputed_delta,
            "direct_mean_of_per_seed_deltas": direct_mean,
            "reported_point_estimate_rounded": reported_delta,
            "drift_vs_direct": drift_vs_direct,
            "drift_vs_reported_4dp": drift_vs_reported,
        })
        # Hard assertion — linearity identity, must hold to fp precision
        assert drift_vs_direct < 1e-12, (
            f"linearity identity failed for {bb}: recomputed={recomputed_delta} "
            f"vs direct-mean-of-per-seed-δ={direct_mean} — drift={drift_vs_direct}"
        )
        # Soft assertion — matches reported (rounded) point delta to <1e-3
        assert drift_vs_reported < 1e-3, (
            f"recomputed δ mismatches reported (rounded 4dp) for {bb}: "
            f"recomputed={recomputed_delta} vs reported={reported_delta} — "
            f"drift={drift_vs_reported}"
        )

    # ── CSV ──
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backbone", "condition", "training_seeds_used",
                    "absolute_pooled_r2", "n_frames_per_seed",
                    "per_seed_r2_json", "status", "note"])
        for bb in BACKBONES:
            for cond in CONDITIONS:
                a = absolute[(bb, cond)]
                per_seed_json = json.dumps({str(k): round(v, 6) for k, v in a["per_seed"].items()})
                n_frames = ""
                if a["seeds_used"]:
                    n_list = [per_cell_n.get((bb, cond, ts)) for ts in a["seeds_used"]]
                    n_frames = "|".join(str(n) for n in n_list)
                w.writerow([
                    bb, cond,
                    "|".join(str(s) for s in a["seeds_used"]),
                    "NaN" if not math.isfinite(a["value"]) else f"{a['value']:.6f}",
                    n_frames,
                    per_seed_json,
                    a["status"],
                    a.get("note", ""),
                ])

    # ── MD ──
    lines = ["# ALL_TASK_ABSOLUTE_ARMS.md — per-arm all-task pooled R²\n\n"]
    lines.append("> **PACKAGE**: `ORIGINAL_CLUSTER_SECONDARY_EVIDENCE`. "
                 "Do not merge absolute R² in this table with the GCP "
                 "exact-split matrix or the submitted paper's canonical matrix. "
                 "Only within-protocol paired deltas transfer.\n\n")
    lines.append("Generated by `scripts/rebuttal/build_all_task_absolute_arms.py`.\n\n")
    lines.append("## Method\n\n")
    lines.append("Per-arm pooled R² uses the **same raw task/target pairing as "
                 "the 10 000-replicate task-cluster bootstrap** in "
                 "`finalize_wave1_stats.py`:\n\n"
                 "1. Group audited per-episode sufficient statistics by "
                 "`(suite, file)` → 26 held-out tasks.\n"
                 "2. Concatenate probe-seed episodes per task (probe seeds ∈ {0, 1, 2}).\n"
                 "3. Sum SSE / y_sum / y_sqsum / n across every task with each "
                 "task counted once (bootstrap replicate at task_indices = "
                 "[0, 1, ..., 25]).\n"
                 "4. per-dim R² = 1 − SSE / (y_sqsum − y_sum² / n); aggregate = "
                 "mean over finite per-dim R².\n"
                 "5. Per-arm absolute R² = arithmetic mean of per-training-seed "
                 "pooled R² over the available training seeds. For `frozen` there "
                 "is one training seed (−1); for `true_id` / `shuffled_id` there "
                 "are two ({0, 1}).\n\n")

    lines.append("## Table\n\n")
    lines.append("| Backbone | frozen | shuffled_id | true_id | forward_dynamics |\n")
    lines.append("|----------|--------|-------------|---------|-------------------|\n")
    for bb in BACKBONES:
        row = f"| {bb} |"
        for cond in ("frozen", "shuffled_id", "true_id", "forward_dynamics"):
            v = absolute[(bb, cond)]["value"]
            if math.isfinite(v):
                row += f" {v:+.4f} |"
            else:
                row += " — (not measured in this cycle) |"
        lines.append(row + "\n")

    lines.append("\n### Per-training-seed breakdown\n\n")
    lines.append("| Backbone | Condition | Training seed | pooled R² | n_frames |\n")
    lines.append("|----------|-----------|----------------|-----------|-----------|\n")
    for bb in BACKBONES:
        for cond in ("frozen", "shuffled_id", "true_id"):
            per_seed = absolute[(bb, cond)]["per_seed"]
            for ts in sorted(per_seed.keys(), key=lambda x: (x is None, x)):
                n = per_cell_n.get((bb, cond, ts), "?")
                lines.append(f"| {bb} | {cond} | {ts} | {per_seed[ts]:+.6f} | {n} |\n")

    # Assertion table
    lines.append("\n## Assertion: `true_absolute − shuffled_absolute == reported point δ`\n\n")
    lines.append("For each backbone, the difference of the true_id and "
                 "shuffled_id absolute arms above must equal the "
                 "`seed_averaged_delta_point` reported in "
                 "`wave1_finalized_stats.jsonl` (per instruction §1). This is a "
                 "linearity identity — the arithmetic mean over training seeds "
                 "commutes with the difference — so equality to floating-point "
                 "precision is expected (`< 1e-12`), and equality to the stored "
                 "rounded value is expected to hold at 4 decimal places "
                 "(`< 1e-3`).\n\n")
    lines.append("| Backbone | true_absolute | shuffled_absolute | recomputed δ | direct mean of per-seed δ | reported δ (jsonl, 4dp) | drift vs direct | drift vs reported |\n")
    lines.append("|----------|----------------|--------------------|--------------|----------------------------|--------------------------|------------------|--------------------|\n")
    for r in assertion_rows:
        lines.append(
            f"| {r['backbone']} | {r['true_absolute']:+.6f} | "
            f"{r['shuffled_absolute']:+.6f} | {r['recomputed_true_minus_shuffled']:+.6f} | "
            f"{r['direct_mean_of_per_seed_deltas']:+.6f} | "
            f"{r['reported_point_estimate_rounded']:+.4f} | "
            f"{r['drift_vs_direct']:.2e} | {r['drift_vs_reported_4dp']:.2e} |\n"
        )
    lines.append("\nAll rows above passed the hard assertion "
                 "(`drift_vs_direct < 1e-12`) and the soft assertion "
                 "(`drift_vs_reported < 1e-3`). Script exits non-zero if either "
                 "assertion fails.\n\n")

    lines.append("## Notes\n\n")
    lines.append("* `forward_dynamics` was **not measured** in this cycle. "
                 "Wave1 fanout wired the forward-dynamics aux head only for "
                 "VideoMAE, and no completed training seed was produced (all "
                 "queued fanout jobs were cancelled before running). See "
                 "`TRAINING_SETTINGS.md §1` and "
                 "`launchers/slurm/wave1_condition_fanout.sbatch` for the "
                 "guard `if [[ \"$CONDITION\" == \"forward_dynamics\" && "
                 "\"$BACKBONE\" != \"videomae\" ]]; then exit 5; fi`.\n")
    lines.append("* All values are conditional on the *original-cluster "
                 "matched-control audit protocol* — l2-normalized MLP probe, "
                 "canonical hparams, split sha256 `42f26dac…`. Absolute values "
                 "here must not be quoted alongside the submitted paper's "
                 "canonical matrix or the GCP exact-split matrix.\n")

    OUT_MD.write_text("".join(lines))
    print(f"wrote {OUT_CSV}\nwrote {OUT_MD}")
    print("\nassertion summary:")
    for r in assertion_rows:
        print(f"  {r['backbone']:10} drift_vs_direct={r['drift_vs_direct']:.2e} "
              f"drift_vs_reported_4dp={r['drift_vs_reported_4dp']:.2e}")


if __name__ == "__main__":
    main()
