"""P2: Task-cluster bootstrap using per-episode sufficient statistics.

Consumes rows in ``runs/corl2026_rebuttal/results/matched_controls.jsonl``
that carry the new ``per_episode_sufficient_stats`` field (added after the
audit patch to eval_matched_control.py).

For each row we can compute pooled R² over any subset of episodes without
re-running the probe: SSE = sum of per-episode `sse_per_dim`; SS_total =
sum of per-episode `y_sqsum_per_dim` minus (sum of `y_sum_per_dim`)²/N.
Both are correct across any episode subset because they are additive
sufficient statistics of the underlying regression.

Bootstrap protocol (matches instruction P2 wording):
  1. group episodes by task = (suite, file) — 26 tasks in the OOD split
  2. resample K=26 tasks WITH REPLACEMENT
  3. include every episode of each sampled task
  4. compute pooled aggregate R² from summed statistics
  5. repeat B=10000 times per contrast

For contrasts we pair CONDITIONS ROW-BY-ROW ONLY WHEN they have identical:
  backbone, probe_seed, training_seed, and (for the two trained conditions)
  the same set of test episodes.

CI: [2.5%, 97.5%] percentile of the bootstrap-delta distribution.
Report the per-training-seed contrast separately AND the mean-over-seeds.

Runs on stdlib + numpy only. No torch, no GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

MC_JSONL = REPO_ROOT / "runs/corl2026_rebuttal/results/matched_controls.jsonl"
OUT_DIR = REPO_ROOT / "runs/corl2026_rebuttal/results"
REPORT = REPO_ROOT / "runs/corl2026_rebuttal/reports/WAVE1_TASK_BOOTSTRAP.md"


# ---- pooled R² over an episode subset from sufficient stats ---------------


def pooled_r2(episode_stats: list[dict]) -> float:
    """Aggregate action-probe R² across the given episodes.

    Returns the mean over per-dim R², matching aggregate_per_dim used
    throughout the pipeline. Uses per-dim SSE + SST computed from the
    additive stats.
    """
    if not episode_stats:
        return float("nan")
    A = len(episode_stats[0]["sse_per_dim"])
    N = sum(e["n_frames"] for e in episode_stats)
    sse = [0.0] * A
    ysum = [0.0] * A
    ysqsum = [0.0] * A
    for e in episode_stats:
        for d in range(A):
            sse[d] += e["sse_per_dim"][d]
            ysum[d] += e["y_sum_per_dim"][d]
            ysqsum[d] += e["y_sqsum_per_dim"][d]
    per_dim_r2 = []
    for d in range(A):
        sst = ysqsum[d] - (ysum[d] ** 2) / max(N, 1)
        if sst <= 1e-12:
            per_dim_r2.append(float("nan"))
        else:
            per_dim_r2.append(1.0 - sse[d] / sst)
    finite = [x for x in per_dim_r2 if not math.isnan(x)]
    return sum(finite) / len(finite) if finite else float("nan")


# ---- deterministic bootstrap sampler (stdlib RNG for reproducibility) ----


def bootstrap_task_r2(row: dict, tasks: list[str], task_to_eps: dict,
                       B: int, seed: int) -> list[float]:
    """Draw B bootstrap replicates: each samples len(tasks) tasks WITH
    REPLACEMENT and returns pooled R² over the concatenated episodes.
    """
    rng = random.Random(seed)
    K = len(tasks)
    reps = []
    for _ in range(B):
        picks = [tasks[rng.randrange(K)] for _ in range(K)]
        eps = []
        for t in picks:
            eps.extend(task_to_eps[t])
        reps.append(pooled_r2(eps))
    return reps


def index_row_episodes(row: dict) -> tuple[list[str], dict, list[dict]]:
    """Return (unique_tasks, {task: [ep_stats, …]}, episode_stats_by_index)."""
    stats = row.get("per_episode_sufficient_stats")
    ep_ids = row.get("test_episode_ids")
    if not stats or not ep_ids or len(stats) != len(ep_ids):
        return [], {}, []
    per_task: dict = defaultdict(list)
    for e, meta in zip(stats, ep_ids):
        task = f"{meta['suite']}/{meta['file']}"
        per_task[task].append(e)
    return sorted(per_task), dict(per_task), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=str(MC_JSONL))
    ap.add_argument("--out-jsonl", default=str(OUT_DIR / "wave1_task_bootstrap.jsonl"))
    ap.add_argument("--out-md", default=str(REPORT))
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--norm", default="l2",
                    help="probe_feature_normalize to filter on")
    ap.add_argument("--with-self", action="store_true",
                    help="also bootstrap self R² per condition (much slower)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.src) if l.strip()]
    rows = [r for r in rows if r.get("probe_feature_normalize") == args.norm]
    with_stats = [r for r in rows if r.get("per_episode_sufficient_stats")]
    print(f"[bootstrap] filtered rows: total={len(rows)} "
          f"with_per_episode_stats={len(with_stats)}")

    if not with_stats:
        print("[bootstrap] NO rows have per_episode_sufficient_stats yet — "
              "run the patched eval_matched_control.py first "
              "(rerun 15 wave1 cells).")
        # still emit an empty output so downstream doesn't break on read
        Path(args.out_jsonl).write_text("")
        Path(args.out_md).write_text(
            "# WAVE1_TASK_BOOTSTRAP.md\n\n"
            "No `per_episode_sufficient_stats` yet — rerun the 15 wave1 cells "
            "with the patched `eval_matched_control.py` before this file is "
            "populated.\n"
        )
        return 0

    # Group rows by (backbone, training_seed, probe_seed) so we can pair
    # true_id / shuffled_id / frozen for the SAME probe-seed run.
    by_key: dict = defaultdict(dict)
    for r in with_stats:
        key = (r["backbone"], r.get("training_seed"), r.get("probe_seed"))
        by_key[key][r["condition"]] = r

    out_lines = []
    # For each (backbone, probe_seed), also collect the frozen row (which
    # has training_seed=-1) so it can be paired against the trained seeds.
    frozen_by = {}
    for (bb, tseed, pseed), rows_ in by_key.items():
        if tseed == -1 and "frozen" in rows_:
            frozen_by[(bb, pseed)] = rows_["frozen"]

    # Self-bootstrap of per-condition R² is skipped by default because the
    # paired contrast (below) is what actually informs the scientific claim
    # and is far cheaper. Enable via --with-self.
    for key, cond_rows in sorted(by_key.items()):
        bb, tseed, pseed = key
        # Inject the matching frozen row (same probe_seed) so downstream
        # pair-loop can consume it just like the other two conditions.
        if tseed != -1 and (bb, pseed) in frozen_by:
            cond_rows = {**cond_rows, "frozen": frozen_by[(bb, pseed)]}
        if args.with_self:
            for cond, r in cond_rows.items():
                tasks, task_to_eps, _ = index_row_episodes(r)
                if not tasks:
                    continue
                reps = bootstrap_task_r2(r, tasks, task_to_eps,
                                          B=args.iters, seed=args.bootstrap_seed)
                reps_sorted = sorted(reps)
                lo = reps_sorted[int(0.025 * args.iters)]
                hi = reps_sorted[int(0.975 * args.iters) - 1]
                out_lines.append({
                    "backbone": bb, "condition": cond,
                    "training_seed": tseed, "probe_seed": pseed,
                    "point_r2": r.get("clean_task_ood_r2"),
                    "bootstrap_r2_lo95": round(lo, 4),
                    "bootstrap_r2_hi95": round(hi, 4),
                    "bootstrap_iters": args.iters,
                    "bootstrap_seed": args.bootstrap_seed,
                    "n_tasks": len(tasks),
                    "n_episodes": sum(len(v) for v in task_to_eps.values()),
                    "kind": "self",
                })

        # Paired contrasts (same probe_seed, same training_seed, same tasks).
        # numpy-vectorized: precompute per-episode sufficient-stat arrays,
        # then bootstrap 10k replicates in a few seconds.
        for (a_cond, b_cond) in [("true_id", "shuffled_id"),
                                  ("true_id", "frozen"),
                                  ("shuffled_id", "frozen")]:
            if a_cond not in cond_rows or b_cond not in cond_rows:
                continue
            a = cond_rows[a_cond]; b = cond_rows[b_cond]
            a_ep_map = {(m["suite"], m["file"], m["demo"]): s
                        for s, m in zip(a["per_episode_sufficient_stats"],
                                         a["test_episode_ids"])}
            b_ep_map = {(m["suite"], m["file"], m["demo"]): s
                        for s, m in zip(b["per_episode_sufficient_stats"],
                                         b["test_episode_ids"])}
            common = sorted(set(a_ep_map) & set(b_ep_map))
            if not common:
                continue
            tasks_of = defaultdict(list)   # task -> list of episode indices in `common`
            for i, kid in enumerate(common):
                tasks_of[f"{kid[0]}/{kid[1]}"].append(i)
            tasks = sorted(tasks_of)
            K = len(tasks)
            E = len(common)
            A = len(a_ep_map[common[0]]["sse_per_dim"])

            def to_arrays(ep_map):
                sse = np.zeros((E, A), dtype=np.float64)
                ysum = np.zeros((E, A), dtype=np.float64)
                ysq = np.zeros((E, A), dtype=np.float64)
                nf = np.zeros(E, dtype=np.float64)
                for i, kid in enumerate(common):
                    e = ep_map[kid]
                    sse[i] = e["sse_per_dim"]
                    ysum[i] = e["y_sum_per_dim"]
                    ysq[i] = e["y_sqsum_per_dim"]
                    nf[i] = e["n_frames"]
                return sse, ysum, ysq, nf
            a_sse, a_ys, a_ysq, a_n = to_arrays(a_ep_map)
            b_sse, b_ys, b_ysq, b_n = to_arrays(b_ep_map)

            # Bootstrap: draw B x K task indices, expand to episode indices per rep.
            rng = np.random.default_rng(args.bootstrap_seed)
            task_indices = rng.integers(0, K, size=(args.iters, K))
            # Build a lookup: task_idx -> list of episode idx
            task_eps = [np.asarray(tasks_of[tasks[t]], dtype=np.int64) for t in range(K)]

            def _rep_r2(sse, ys, ysq, nf, ep_lists):
                # For each of B replicates, gather episodes from sampled tasks,
                # sum stats, compute per-dim R², average.
                out = np.empty(args.iters, dtype=np.float64)
                for r in range(args.iters):
                    idxs = np.concatenate([task_eps[t] for t in task_indices[r]])
                    N = nf[idxs].sum()
                    sum_sse = sse[idxs].sum(0)
                    sum_ys = ys[idxs].sum(0)
                    sum_ysq = ysq[idxs].sum(0)
                    sst = sum_ysq - (sum_ys ** 2) / max(N, 1.0)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        per_dim_r2 = 1.0 - sum_sse / np.where(sst > 1e-12, sst, np.nan)
                    finite = per_dim_r2[np.isfinite(per_dim_r2)]
                    out[r] = float(finite.mean()) if finite.size else float("nan")
                return out

            reps_a = _rep_r2(a_sse, a_ys, a_ysq, a_n, task_eps)
            reps_b = _rep_r2(b_sse, b_ys, b_ysq, b_n, task_eps)
            deltas = reps_a - reps_b
            deltas_sorted = np.sort(deltas[~np.isnan(deltas)])
            lo = float(deltas_sorted[int(0.025 * len(deltas_sorted))])
            hi = float(deltas_sorted[int(0.975 * len(deltas_sorted)) - 1])
            mean = float(deltas_sorted.mean())
            out_lines.append({
                "backbone": bb,
                "contrast": f"{a_cond}_minus_{b_cond}",
                "training_seed": tseed, "probe_seed": pseed,
                "point_delta": round(a.get("clean_task_ood_r2") - b.get("clean_task_ood_r2"), 4),
                "bootstrap_mean_delta": round(mean, 4),
                "bootstrap_delta_lo95": round(lo, 4),
                "bootstrap_delta_hi95": round(hi, 4),
                "excludes_zero": (lo > 0) or (hi < 0),
                "bootstrap_iters": args.iters,
                "bootstrap_seed": args.bootstrap_seed,
                "n_tasks": K,
                "n_episodes_common": len(common),
                "kind": "paired_task_cluster",
            })

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in out_lines:
            f.write(json.dumps(r) + "\n")

    # summary MD
    lines = ["# WAVE1_TASK_BOOTSTRAP.md — held-out-task cluster bootstrap\n\n"]
    lines.append(f"Generated by `scripts/rebuttal/task_cluster_bootstrap.py`.  "
                 f"B={args.iters}, seed={args.bootstrap_seed}, filter "
                 f"`probe_feature_normalize={args.norm}`.\n\n")
    lines.append("## Method\n\n")
    lines.append("For each row with `per_episode_sufficient_stats`:\n")
    lines.append("1. Episodes are grouped by task (`suite/file`; 26 tasks in "
                 "the held-out split).\n")
    lines.append("2. Bootstrap draws K=26 tasks WITH REPLACEMENT per replicate.\n")
    lines.append("3. All episodes of each sampled task are included.\n")
    lines.append("4. Aggregate R² is recomputed from POOLED sufficient "
                 "statistics — NEVER by averaging pre-computed R² values.\n")
    lines.append("5. Paired contrasts share the sampled tasks between the two "
                 "conditions and use the same probe seed and training seed.\n\n")
    lines.append("## Paired contrasts\n\n")
    lines.append("| backbone | contrast | tseed | pseed | point Δ | boot mean Δ | 95% CI | excludes 0 | n_tasks |\n")
    lines.append("|----------|----------|-------|-------|---------|-------------|--------|------------|---------|\n")
    for r in out_lines:
        if r.get("kind") == "paired_task_cluster":
            lines.append(
                f"| {r['backbone']} | {r['contrast']} | {r['training_seed']} "
                f"| {r['probe_seed']} | {r['point_delta']} "
                f"| {r['bootstrap_mean_delta']} | [{r['bootstrap_delta_lo95']}, "
                f"{r['bootstrap_delta_hi95']}] | {r['excludes_zero']} "
                f"| {r['n_tasks']} |\n"
            )
    Path(args.out_md).write_text("".join(lines))
    print(f"wrote {args.out_jsonl}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
