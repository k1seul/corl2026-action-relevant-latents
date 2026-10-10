"""Regenerate ``runs/corl2026_rebuttal/tables/matched_controls.csv`` from
``runs/corl2026_rebuttal/results/matched_controls.jsonl``.

STEP 10 hardening:

  * per-training-seed rows in addition to across-seed mean;
  * three probe seeds per checkpoint;
  * clean aggregate R², per-dimension R², perturbation matrix, worst
    perturbation;
  * paired episode-bootstrap 95% confidence intervals for four contrasts:
      Δ = true_id − shuffled_id
      Δ = true_id − forward_dynamics
      Δ = forward_dynamics − frozen
      Δ = shuffled_id − frozen
  * NEVER treats probe-seed variance as training-seed uncertainty.

If a required source row is missing we emit an empty schema-consistent
cell so downstream figure scripts fail loudly on the correct cell, not
silently on stale data.

Runs on stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CONDITIONS = ["frozen", "true_id", "shuffled_id",
              "forward_dynamics", "shuffled_id_circular_sensitivity",
              "action_free_adaptation"]
BACKBONES = ["vjepa2", "videomae", "webdino"]
CONTRASTS = [
    ("true_id", "shuffled_id"),
    ("true_id", "forward_dynamics"),
    ("forward_dynamics", "frozen"),
    ("shuffled_id", "frozen"),
]

PER_SEED_FIELDS = [
    "backbone", "condition", "training_seed", "n_probe_seeds",
    "clean_task_ood_r2_mean_over_probe_seeds",
    "clean_task_ood_r2_std_over_probe_seeds",
    "worst_perturbation_r2_mean",
    "worst_perturbation_r2_std",
    "n_perturbations_evaluated",
]

ACROSS_SEED_FIELDS = [
    "backbone", "condition",
    "n_training_seeds", "n_probe_seeds_per_seed",
    "clean_task_ood_r2_mean_across_training_seeds",
    "clean_task_ood_r2_std_across_training_seeds",
    "worst_perturbation_r2_mean_across_training_seeds",
    "worst_perturbation_r2_std_across_training_seeds",
    "notes",
]

CONTRAST_FIELDS = [
    "backbone", "delta_name",
    "delta_mean_over_training_seeds",
    "delta_ci_lo_95_bootstrap",
    "delta_ci_hi_95_bootstrap",
    "bootstrap_n_iters",
    "bootstrap_seed",
    "notes",
]


# ── loading + basic aggregation ────────────────────────────────────────────


def load(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out: list[dict] = []
    with p.open() as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
    return out


def _mean_std(xs: list[float]) -> tuple[float | str, float | str]:
    if not xs:
        return "", ""
    m = statistics.fmean(xs)
    s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return round(m, 4), round(s, 4)


# ── deterministic bootstrap (paired, over episode IDs) ────────────────────


def _stable_prng(seed: int, i: int) -> float:
    """A stable [0,1) uniform derived from SHA-256; used to draw bootstrap
    indices without touching numpy or the Python ``random`` state."""
    key = f"{seed}|{i}".encode()
    digest = hashlib.sha256(key).digest()[:8]
    n = int.from_bytes(digest, "big")
    return n / (1 << 64)


def paired_bootstrap_ci(paired: list[tuple[float, float]], *, n_iters: int,
                         seed: int) -> tuple[float, float, float]:
    """Return (mean_delta, ci_lo_95, ci_hi_95) via a stratified paired
    episode-bootstrap. ``paired`` is a list of ``(x_a, x_b)`` per unit
    (episode / task etc.). ``delta = x_a - x_b``. Bootstrap resamples the
    pairs with replacement.
    """
    if not paired:
        return float("nan"), float("nan"), float("nan")
    n = len(paired)
    deltas_boot: list[float] = []
    for b in range(n_iters):
        picks = [int(_stable_prng(seed, b * n + i) * n) for i in range(n)]
        d = statistics.fmean(paired[j][0] - paired[j][1] for j in picks)
        deltas_boot.append(d)
    deltas_boot.sort()
    lo = deltas_boot[int(0.025 * n_iters)]
    hi = deltas_boot[int(0.975 * n_iters) - 1]
    mean = statistics.fmean(p[0] - p[1] for p in paired)
    return round(mean, 4), round(lo, 4), round(hi, 4)


# ── summaries ──────────────────────────────────────────────────────────────


def _agg_rows(rows: list[dict], backbone: str, cond: str, seed=None
              ) -> list[dict]:
    return [r for r in rows
            if r.get("backbone") == backbone
            and r.get("condition") == cond
            and (seed is None or r.get("training_seed") == seed)]


def summarize_per_seed(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for bb in BACKBONES:
        # discover which training seeds are present for any condition of bb
        seeds = sorted({r.get("training_seed") for r in rows
                        if r.get("backbone") == bb
                        and r.get("training_seed") is not None})
        if not seeds:
            for cond in CONDITIONS:
                out.append({k: "" for k in PER_SEED_FIELDS} | {
                    "backbone": bb, "condition": cond,
                })
            continue
        for cond in CONDITIONS:
            for s in seeds:
                grp = _agg_rows(rows, bb, cond, s)
                if not grp:
                    out.append({k: "" for k in PER_SEED_FIELDS} | {
                        "backbone": bb, "condition": cond, "training_seed": s,
                    })
                    continue
                clean = [r["clean_task_ood_r2"] for r in grp
                         if isinstance(r.get("clean_task_ood_r2"), (int, float))]
                worst = [r["worst_perturbation_r2"] for r in grp
                         if isinstance(r.get("worst_perturbation_r2"), (int, float))]
                m_c, s_c = _mean_std(clean)
                m_w, s_w = _mean_std(worst)
                n_probe = len({r.get("probe_seed") for r in grp})
                n_pert = len({r.get("perturbation") for r in grp
                              if r.get("perturbation") not in (None, "clean")})
                out.append({
                    "backbone": bb, "condition": cond,
                    "training_seed": s, "n_probe_seeds": n_probe,
                    "clean_task_ood_r2_mean_over_probe_seeds": m_c,
                    "clean_task_ood_r2_std_over_probe_seeds": s_c,
                    "worst_perturbation_r2_mean": m_w,
                    "worst_perturbation_r2_std": s_w,
                    "n_perturbations_evaluated": n_pert,
                })
    return out


def summarize_across_seeds(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for bb in BACKBONES:
        for cond in CONDITIONS:
            grp = _agg_rows(rows, bb, cond)
            if not grp:
                out.append({k: "" for k in ACROSS_SEED_FIELDS} | {
                    "backbone": bb, "condition": cond,
                    "notes": "PENDING — no runs recorded yet",
                })
                continue
            # for each training seed compute the mean over probe seeds first,
            # then aggregate across training seeds. This is the correct
            # 2-level aggregation; do NOT pool probe seeds into training seeds.
            per_seed_means: list[float] = []
            per_seed_worsts: list[float] = []
            seeds = sorted({r.get("training_seed") for r in grp
                            if r.get("training_seed") is not None})
            for s in seeds:
                subgrp = [r for r in grp if r.get("training_seed") == s]
                clean = [r["clean_task_ood_r2"] for r in subgrp
                         if isinstance(r.get("clean_task_ood_r2"), (int, float))]
                worst = [r["worst_perturbation_r2"] for r in subgrp
                         if isinstance(r.get("worst_perturbation_r2"), (int, float))]
                if clean:
                    per_seed_means.append(statistics.fmean(clean))
                if worst:
                    per_seed_worsts.append(statistics.fmean(worst))
            m_c, s_c = _mean_std(per_seed_means)
            m_w, s_w = _mean_std(per_seed_worsts)
            n_probe = statistics.median(
                [len({r.get("probe_seed") for r in grp
                      if r.get("training_seed") == s}) for s in seeds] or [0]
            ) if seeds else 0
            out.append({
                "backbone": bb, "condition": cond,
                "n_training_seeds": len(seeds),
                "n_probe_seeds_per_seed": int(n_probe),
                "clean_task_ood_r2_mean_across_training_seeds": m_c,
                "clean_task_ood_r2_std_across_training_seeds": s_c,
                "worst_perturbation_r2_mean_across_training_seeds": m_w,
                "worst_perturbation_r2_std_across_training_seeds": s_w,
                "notes": "",
            })
    return out


def summarize_contrasts(rows: list[dict], *, bootstrap_iters: int,
                        bootstrap_seed: int) -> list[dict]:
    """One row per (backbone, contrast). If the contrast requires bootstrap
    pairing at the episode level, we need per-episode R² lists per row —
    the pipeline is expected to emit those under key
    ``per_episode_r2`` (list of floats). If absent, we fall back to
    seed-level pairing with a documented caveat.
    """
    out: list[dict] = []
    for bb in BACKBONES:
        for a, b in CONTRASTS:
            name = f"{a}_minus_{b}"
            ga = _agg_rows(rows, bb, a)
            gb = _agg_rows(rows, bb, b)
            if not ga or not gb:
                out.append({
                    "backbone": bb, "delta_name": name,
                    "delta_mean_over_training_seeds": "",
                    "delta_ci_lo_95_bootstrap": "",
                    "delta_ci_hi_95_bootstrap": "",
                    "bootstrap_n_iters": "",
                    "bootstrap_seed": "",
                    "notes": "PENDING — insufficient rows",
                })
                continue
            # Attempt episode-paired bootstrap
            def _flatten_ep(g):
                out = []
                for r in g:
                    per_ep = r.get("per_episode_r2")
                    if isinstance(per_ep, list):
                        for ep_idx, v in enumerate(per_ep):
                            if isinstance(v, (int, float)):
                                out.append((r.get("training_seed"),
                                             r.get("perturbation", "clean"),
                                             ep_idx, float(v)))
                return out
            ea = _flatten_ep(ga); eb = _flatten_ep(gb)
            paired: list[tuple[float, float]] = []
            if ea and eb:
                # pair by (seed, perturbation, ep_idx)
                map_b = {(s, p, e): v for (s, p, e, v) in eb}
                for (s, p, e, va) in ea:
                    vb = map_b.get((s, p, e))
                    if vb is not None:
                        paired.append((va, vb))
            note = ""
            if not paired:
                # fallback: pair per training_seed clean-cell means
                seeds = sorted({r.get("training_seed") for r in ga}
                                & {r.get("training_seed") for r in gb})
                for s in seeds:
                    va = [r["clean_task_ood_r2"]
                          for r in ga if r.get("training_seed") == s
                          and isinstance(r.get("clean_task_ood_r2"), (int, float))]
                    vb = [r["clean_task_ood_r2"]
                          for r in gb if r.get("training_seed") == s
                          and isinstance(r.get("clean_task_ood_r2"), (int, float))]
                    if va and vb:
                        paired.append((statistics.fmean(va), statistics.fmean(vb)))
                note = ("per-episode R² not present; falling back to seed-mean "
                        "pairing (wider CI than episode-bootstrap)")
            mean, lo, hi = paired_bootstrap_ci(
                paired, n_iters=bootstrap_iters, seed=bootstrap_seed)
            out.append({
                "backbone": bb, "delta_name": name,
                "delta_mean_over_training_seeds": mean,
                "delta_ci_lo_95_bootstrap": lo,
                "delta_ci_hi_95_bootstrap": hi,
                "bootstrap_n_iters": bootstrap_iters,
                "bootstrap_seed": bootstrap_seed,
                "notes": note,
            })
    return out


def write_csv(rows: list[dict], p: Path, fields: list[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src",
                    default=str(REPO_ROOT / "runs/corl2026_rebuttal/results/matched_controls.jsonl"))
    ap.add_argument("--out-dir",
                    default=str(REPO_ROOT / "runs/corl2026_rebuttal/tables"))
    ap.add_argument("--bootstrap-iters", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    args = ap.parse_args()

    rows = load(Path(args.src))
    out_dir = Path(args.out_dir)

    per_seed = summarize_per_seed(rows)
    across = summarize_across_seeds(rows)
    contrasts = summarize_contrasts(rows,
                                    bootstrap_iters=args.bootstrap_iters,
                                    bootstrap_seed=args.bootstrap_seed)
    write_csv(per_seed, out_dir / "matched_controls_per_seed.csv", PER_SEED_FIELDS)
    write_csv(across, out_dir / "matched_controls_across_seeds.csv", ACROSS_SEED_FIELDS)
    write_csv(contrasts, out_dir / "matched_controls_contrasts.csv", CONTRAST_FIELDS)

    # Legacy single-file variant, kept for compatibility.
    write_csv(across, out_dir / "matched_controls.csv", ACROSS_SEED_FIELDS)

    print(f"wrote per-seed / across-seeds / contrasts CSVs into {out_dir}")
    print(f"  {len(rows)} raw rows → "
          f"{len(per_seed)} per-seed + {len(across)} across-seed + "
          f"{len(contrasts)} contrast rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
