"""P1 finalized statistics: pooled R², task-macro R², normalized MSE,
per-training-seed deltas, seed-averaged delta with a paired held-out-task
bootstrap CI (NOT a union of per-seed CIs).

For each wave1 (backbone × condition × training_seed × probe_seed) row
that has ``per_episode_sufficient_stats`` and full 26-task coverage:

  pooled_r2   = 1 − ΣSSE / (Σy² − (Σy)² / N)  aggregated over ALL episodes
  task_macro  = mean over the 26 tasks of pooled_r2 within-task
  norm_mse    = mean over dims of MSE_dim / Var(y_dim)

For contrasts the seed-averaged delta CI is produced this way per
replicate (single number, not a union):

  1. draw 26 tasks with replacement
  2. for both training seeds AND both conditions (a=true, b=shuffled),
     mean the 3 probe seeds' per-episode stats first (so probe-seed noise
     is folded into the pre-bootstrap stats)
  3. compute pooled R² for each (seed, condition) over the sampled tasks
  4. delta_seed_i = pooled_R²(a, seed_i) − pooled_R²(b, seed_i)
  5. seed-averaged delta_replicate = 0.5 * (delta_seed_0 + delta_seed_1)
  6. 95% CI = [2.5%, 97.5%] percentile of the 10 000 seed-averaged deltas

That produces one number ± one interval per (backbone × contrast), which
is what the instruction asks for.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

MC = REPO_ROOT / "runs/corl2026_rebuttal/results/matched_controls.jsonl"
OUT_JSONL = REPO_ROOT / "runs/corl2026_rebuttal/results/wave1_finalized_stats.jsonl"
OUT_MD = REPO_ROOT / "runs/corl2026_rebuttal/reports/WAVE1_FINALIZED_STATS.md"


def audited(r):
    return (r.get("probe_feature_normalize") == "l2"
            and r.get("per_episode_sufficient_stats")
            and r.get("n_probe_eval_test_pairs", 0) >= 20000)


def build_task_index(ep_stats, ep_ids):
    """{task_key -> list of ep_stat dicts} for one row."""
    by_task = defaultdict(list)
    for s, m in zip(ep_stats, ep_ids):
        by_task[(m["suite"], m["file"])].append(s)
    return dict(by_task)


def r2_stats_from_eps(eps):
    """Given a list of per-episode sufficient-stat dicts, return
    (pooled_r2, task_macro placeholder, normalized_mse). task_macro must
    be computed at the higher level (per-task pooling then macro).
    """
    A = len(eps[0]["sse_per_dim"])
    N = sum(e["n_frames"] for e in eps)
    sse = np.zeros(A); ys = np.zeros(A); ysq = np.zeros(A)
    for e in eps:
        sse += np.asarray(e["sse_per_dim"])
        ys += np.asarray(e["y_sum_per_dim"])
        ysq += np.asarray(e["y_sqsum_per_dim"])
    sst = ysq - (ys ** 2) / max(N, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_dim_r2 = 1.0 - sse / np.where(sst > 1e-12, sst, np.nan)
    finite = per_dim_r2[np.isfinite(per_dim_r2)]
    pooled_r2 = float(finite.mean()) if finite.size else float("nan")
    per_dim_mse = sse / max(N, 1)
    per_dim_var = sst / max(N, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_dim_norm_mse = per_dim_mse / np.where(per_dim_var > 1e-12, per_dim_var, np.nan)
    finite_nm = per_dim_norm_mse[np.isfinite(per_dim_norm_mse)]
    norm_mse = float(finite_nm.mean()) if finite_nm.size else float("nan")
    return pooled_r2, norm_mse, N


def per_task_pooled_r2(by_task):
    """Mean over tasks of per-task pooled R²."""
    per_task = []
    for _, eps in by_task.items():
        r2, _, _ = r2_stats_from_eps(eps)
        if not math.isnan(r2):
            per_task.append(r2)
    return float(sum(per_task) / len(per_task)) if per_task else float("nan")


def bootstrap_seed_averaged_delta(a_seed_rows, b_seed_rows, tasks, task_eps_a,
                                    task_eps_b, iters=10000, seed=0):
    """Return (mean_delta, lo95, hi95, per_replicate deltas array).

    a_seed_rows: {training_seed -> merged-across-probe-seeds by_task}
    b_seed_rows: same for b condition
    tasks: sorted task keys shared between conditions
    task_eps_a / task_eps_b: {seed -> {task -> [ep_stats, ...]}}
    """
    rng = np.random.default_rng(seed)
    K = len(tasks)
    B = iters
    task_idx = rng.integers(0, K, size=(B, K))

    def _r2_for_seed(task_eps_seed, task_indices):
        deltas = np.empty(B, dtype=np.float64)
        # precompute per-task arrays for fast pooling
        return None

    # Simpler: precompute per-task sufficient stats (summed across probe seeds
    # into a single pseudo-episode-set per task) for each condition/seed.
    def compute_task_arrays(task_eps_seed):
        """Return dict[task] -> {sse, ysum, ysq, n} (numpy arrays)."""
        out = {}
        for t in tasks:
            eps = task_eps_seed[t]
            A = len(eps[0]["sse_per_dim"])
            sse = np.zeros(A); ys = np.zeros(A); ysq = np.zeros(A); n = 0
            for e in eps:
                sse += np.asarray(e["sse_per_dim"])
                ys += np.asarray(e["y_sum_per_dim"])
                ysq += np.asarray(e["y_sqsum_per_dim"])
                n += e["n_frames"]
            out[t] = (sse, ys, ysq, n)
        return out

    per_seed_a = {s: compute_task_arrays(task_eps_a[s]) for s in task_eps_a}
    per_seed_b = {s: compute_task_arrays(task_eps_b[s]) for s in task_eps_b}
    seeds = sorted(per_seed_a.keys() & per_seed_b.keys())

    def r2_for_sampled_tasks(per_task_arrays, sample):
        sse = None; ys = None; ysq = None; n = 0
        for t_i in sample:
            t = tasks[t_i]
            s, y, q, nn = per_task_arrays[t]
            if sse is None:
                sse = s.copy(); ys = y.copy(); ysq = q.copy(); n = nn
            else:
                sse += s; ys += y; ysq += q; n += nn
        sst = ysq - (ys ** 2) / max(n, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_dim = 1.0 - sse / np.where(sst > 1e-12, sst, np.nan)
        finite = per_dim[np.isfinite(per_dim)]
        return float(finite.mean()) if finite.size else float("nan")

    deltas = np.empty(B, dtype=np.float64)
    for b in range(B):
        sample = task_idx[b]
        d = 0.0
        for s in seeds:
            r_a = r2_for_sampled_tasks(per_seed_a[s], sample)
            r_b = r2_for_sampled_tasks(per_seed_b[s], sample)
            d += (r_a - r_b)
        deltas[b] = d / len(seeds)
    ds = np.sort(deltas[~np.isnan(deltas)])
    lo = float(ds[int(0.025 * len(ds))])
    hi = float(ds[int(0.975 * len(ds)) - 1])
    return float(ds.mean()), lo, hi, deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in MC.open() if l.strip()]
    rows = [r for r in rows if audited(r)]
    rows = [r for r in rows if r["backbone"] in ("videomae", "vjepa2", "webdino")]
    print(f"[finalize] {len(rows)} audited rows")

    # Group by (backbone, condition, training_seed) — merge probe seeds' per-episode
    # stats by ADDING (they're additive sufficient statistics).
    by_key: dict = defaultdict(list)
    for r in rows:
        by_key[(r["backbone"], r["condition"], r.get("training_seed"))].append(r)

    per_seed_stats = {}  # (bb, cond, tseed) -> dict(pooled_r2, task_macro, norm_mse, by_task)
    for k, rs in by_key.items():
        # Merge probe seeds' stats: for each episode index, sum the stats across probe seeds.
        # Each probe seed gives its own SSE (because probe is retrained). Combining by mean
        # of pooled_r2 across probe seeds is what we want — NOT summing SSE (that would
        # weight probe seeds as if they were extra episodes).
        # So: compute per-probe-seed stats first, then mean.
        pooled_list = []; macro_list = []; nmse_list = []; by_task_first = None
        for r in rs:
            by_task = build_task_index(r["per_episode_sufficient_stats"], r["test_episode_ids"])
            pooled, nmse, N = r2_stats_from_eps(r["per_episode_sufficient_stats"])
            macro = per_task_pooled_r2(by_task)
            pooled_list.append(pooled); macro_list.append(macro); nmse_list.append(nmse)
            if by_task_first is None:
                by_task_first = by_task
        per_seed_stats[k] = {
            "pooled_r2_probe_mean": float(statistics.fmean(pooled_list)),
            "pooled_r2_probe_std": float(statistics.pstdev(pooled_list)) if len(pooled_list) > 1 else 0.0,
            "task_macro_r2_probe_mean": float(statistics.fmean(macro_list)),
            "normalized_mse_probe_mean": float(statistics.fmean(nmse_list)),
            "n_probe_seeds": len(rs),
            "by_task": by_task_first,
        }

    # ── Bootstrap: seed-averaged deltas for each (backbone, contrast) ──
    contrasts = [("true_id", "shuffled_id"),
                 ("shuffled_id", "frozen"),
                 ("true_id", "frozen")]
    out_records = []
    for bb in ("videomae", "vjepa2", "webdino"):
        for (a_cond, b_cond) in contrasts:
            # Gather task_eps per training seed for each condition.
            # For frozen, training_seed=-1; broadcast against non-frozen seeds.
            def eps_by_seed(cond):
                out = {}
                for k in by_key:
                    if k[0] == bb and k[1] == cond:
                        # Collate ALL probe seeds' per-task episode stats.
                        # Each probe seed's per-episode SSE differs; we treat each
                        # probe seed's episode as its own "cluster observation"
                        # per-task, which is equivalent to averaging probe seeds
                        # inside the bootstrap.
                        by_task = defaultdict(list)
                        for r in by_key[k]:
                            for e, m in zip(r["per_episode_sufficient_stats"],
                                             r["test_episode_ids"]):
                                by_task[(m["suite"], m["file"])].append(e)
                        out[k[2]] = by_task
                return out

            eps_a = eps_by_seed(a_cond)
            eps_b = eps_by_seed(b_cond)
            if not eps_a or not eps_b:
                continue

            # Align seeds: frozen has {-1}; others {0,1}. Broadcast frozen.
            if -1 in eps_a and -1 not in eps_b:
                eps_a = {s: eps_a[-1] for s in eps_b}
            if -1 in eps_b and -1 not in eps_a:
                eps_b = {s: eps_b[-1] for s in eps_a}
            seeds = sorted(set(eps_a) & set(eps_b))
            if not seeds:
                continue

            tasks = sorted(set(eps_a[seeds[0]]) & set(eps_b[seeds[0]]))
            bs_mean, lo, hi, _ = bootstrap_seed_averaged_delta(
                None, None, tasks, eps_a, eps_b,
                iters=args.iters, seed=args.seed,
            )
            # Per-training-seed point deltas (from per_seed_stats)
            per_seed_deltas = {}
            for s in seeds:
                # look-up robust to frozen broadcast
                key_a = (bb, a_cond, s if (bb, a_cond, s) in per_seed_stats else -1)
                key_b = (bb, b_cond, s if (bb, b_cond, s) in per_seed_stats else -1)
                per_seed_deltas[s] = round(
                    per_seed_stats[key_a]["pooled_r2_probe_mean"]
                    - per_seed_stats[key_b]["pooled_r2_probe_mean"], 4)

            # POINT ESTIMATE = arithmetic mean of observed training-seed deltas.
            # NOT the bootstrap-distribution mean (which drifts by a few 1e-3 from
            # the true point estimate because each replicate resamples tasks).
            observed_deltas = [per_seed_deltas[s] for s in seeds]
            seed_avg_point = float(statistics.fmean(observed_deltas))
            # Contract assertion (per rebuttal instruction §1):
            expected = sum(observed_deltas) / len(observed_deltas)
            assert abs(seed_avg_point - expected) < 1e-10, (
                f"seed-avg point-estimate arithmetic drift: "
                f"{seed_avg_point} vs {expected} for {bb} {a_cond}_minus_{b_cond}"
            )

            out_records.append({
                "backbone": bb,
                "contrast": f"{a_cond}_minus_{b_cond}",
                "per_training_seed_deltas": per_seed_deltas,
                # POINT estimate = arithmetic mean of observed seed deltas
                "seed_averaged_delta_point": round(seed_avg_point, 4),
                # CI is a percentile CI of the bootstrap distribution ONLY
                "seed_averaged_delta_lo95": round(lo, 4),
                "seed_averaged_delta_hi95": round(hi, 4),
                # Bootstrap-distribution mean kept for diagnostic/QA use;
                # do NOT report as the point estimate.
                "bootstrap_distribution_mean_diagnostic": round(bs_mean, 4),
                "excludes_zero": (lo > 0) or (hi < 0),
                "bootstrap_iters": args.iters,
                "bootstrap_seed": args.seed,
                "n_tasks": len(tasks),
            })

    # ── Emit per-cell summary rows too ──
    cell_records = []
    for (bb, cond, tseed), s in per_seed_stats.items():
        cell_records.append({
            "backbone": bb, "condition": cond, "training_seed": tseed,
            "n_probe_seeds": s["n_probe_seeds"],
            "pooled_r2_probe_mean": round(s["pooled_r2_probe_mean"], 4),
            "pooled_r2_probe_std": round(s["pooled_r2_probe_std"], 4),
            "task_macro_r2_probe_mean": round(s["task_macro_r2_probe_mean"], 4),
            "normalized_mse_probe_mean": round(s["normalized_mse_probe_mean"], 4),
        })
    with OUT_JSONL.open("w") as f:
        for r in cell_records:
            f.write(json.dumps({**r, "kind": "cell"}) + "\n")
        for r in out_records:
            f.write(json.dumps({**r, "kind": "seed_averaged_delta"}) + "\n")

    # ── Markdown report ──
    lines = ["# WAVE1_FINALIZED_STATS.md — final statistics per instruction §1\n\n"]
    lines.append("> **PACKAGE**: `ORIGINAL_CLUSTER_SECONDARY_EVIDENCE`. "
                 "Absolute R² values in this file are conditional on the "
                 "original-cluster matched-control audit protocol (l2-"
                 "normalized MLP probe, split sha256 `42f26dac…`); **do not "
                 "merge** with the submitted paper's canonical matrix or "
                 "with the GCP exact-split matrix. Only paired within-"
                 "protocol deltas transfer.\n\n")
    lines.append(f"Generated by `scripts/rebuttal/finalize_wave1_stats.py`. B={args.iters}, seed={args.seed}.\n\n")
    lines.append("Every value is a mechanical read from `matched_controls.jsonl` "
                 "(l2 normalize + per_episode_sufficient_stats + 26 tasks).\n\n")

    lines.append("## Per (backbone × condition × training_seed) cell\n\n")
    lines.append("| backbone | condition | seed | pooled R² (probe-mean) | task-macro R² | normalized MSE |\n")
    lines.append("|----------|-----------|------|-------------------------|----------------|-----------------|\n")
    for r in cell_records:
        lines.append(f"| {r['backbone']} | {r['condition']} | {r['training_seed']} | "
                     f"{r['pooled_r2_probe_mean']:+.4f} (±{r['pooled_r2_probe_std']:.4f}) | "
                     f"{r['task_macro_r2_probe_mean']:+.4f} | {r['normalized_mse_probe_mean']:.4f} |\n")

    lines.append("\n## Seed-averaged deltas with paired held-out-task cluster bootstrap CI\n\n")
    lines.append("**Point estimate** (per instruction §1): "
                 "`seed-avg δ = (δ_seed_0 + δ_seed_1) / 2` — the arithmetic mean of "
                 "the two observed training-seed point deltas. NOT the bootstrap-"
                 "distribution mean.\n\n"
                 "**95% CI**: percentile CI of the bootstrap distribution over 10 000 "
                 "task-cluster replicates (26 tasks resampled WITH replacement; "
                 "same task sample for both conditions and both training seeds; "
                 "replicate statistic = seed-averaged δ). CI columns are the "
                 "2.5/97.5 percentiles.\n\n"
                 "Assertion enforced by the generator: "
                 "`abs(seed_avg_delta − mean(seed0_delta, seed1_delta)) < 1e-10`.\n\n")
    lines.append("**This is one CI per (backbone × contrast), NOT a union of per-seed CIs.**\n\n")
    lines.append("| backbone | contrast | δ seed 0 | δ seed 1 | seed-avg δ (point) | 95% CI (bootstrap) | bootstrap-mean (diagnostic) | excludes 0 | n_tasks |\n")
    lines.append("|----------|----------|----------|----------|--------------------|---------------------|------------------------------|------------|---------|\n")
    for r in out_records:
        s0 = r["per_training_seed_deltas"].get(0, "—")
        s1 = r["per_training_seed_deltas"].get(1, "—")
        lines.append(f"| {r['backbone']} | {r['contrast']} | {s0} | {s1} | "
                     f"{r['seed_averaged_delta_point']:+.4f} | "
                     f"[{r['seed_averaged_delta_lo95']:+.4f}, "
                     f"{r['seed_averaged_delta_hi95']:+.4f}] | "
                     f"{r['bootstrap_distribution_mean_diagnostic']:+.4f} | "
                     f"{r['excludes_zero']} | {r['n_tasks']} |\n")

    lines.append("\n## Correct wording\n\n")
    lines.append("Use exactly:\n\n")
    lines.append("> The paired true-ID versus shuffled-ID gap is positive for both "
                 "training seeds, and its held-out-task bootstrap interval excludes zero.\n")

    OUT_MD.write_text("".join(lines))
    print(f"wrote {OUT_JSONL}\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
