"""Independently recompute the final GCP matched-control contrasts from the 21 raw-prediction archives.

Inputs (from corl2026_rebuttal_artifacts.tar.gz, extracted under runs/gcp_primary_export/):
  corl2026_export/results/raw_predictions/*.npz      predictions (probe, perturbation, frame, 7), targets, task_ids
  corl2026_export/tables/rebuttal_decomposition.csv  the GCP-reported estimates and CIs
  corl2026_export/results/submitted_compatible_clean_probe.jsonl  the 200-episode point-estimate rows

Definitions follow REBUTTAL_PRIMARY_EVIDENCE.md: per (checkpoint, probe seed, perturbation) the pooled
per-dimension R² over all 26 held-out tasks (mean of 7 dims); contrasts are paired within each
training-seed x probe-seed cell (frozen is paired by probe seed) and averaged over the cells; the CI
resamples the 26 tasks with replacement (10,000 replicates). "worst_nonclean" is the contrast of each
cell's minimum R² over the seven non-clean perturbations. The bootstrap RNG of the original run is
not recorded, so point estimates should match exactly and CI bounds only approximately.

Output: runs/camera_ready/evidence/gcp_primary_recomputed_contrasts.csv
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT = REPO_ROOT / "runs/gcp_primary_export/corl2026_export"
RAW = EXPORT / "results/raw_predictions"
OUT = REPO_ROOT / "runs/camera_ready/evidence/gcp_primary_recomputed_contrasts.csv"
BACKBONES = ["vjepa2", "videomae", "webdino"]
CONTRASTS = [("true_id", "shuffled_id"), ("true_id", "forward_dynamics"), ("shuffled_id", "frozen")]
N_BOOT, SEED = 10_000, 0


def task_stats(path: Path):
    """Per-task sufficient statistics: sse (P, Q, T, 7), sy (T, 7), syy (T, 7), n (T,)."""
    z = np.load(path, allow_pickle=False)
    pred = z["predictions"].astype(np.float64)
    y = z["targets"].astype(np.float64)
    tid = z["task_ids"].astype(np.int64)
    T = int(tid.max()) + 1
    n = np.bincount(tid, minlength=T).astype(np.float64)
    sy = np.stack([np.bincount(tid, y[:, d], T) for d in range(7)], 1)
    syy = np.stack([np.bincount(tid, y[:, d] ** 2, T) for d in range(7)], 1)
    P, Q = pred.shape[:2]
    sse = np.empty((P, Q, T, 7))
    for p in range(P):
        for q in range(Q):
            e2 = (pred[p, q] - y) ** 2
            sse[p, q] = np.stack([np.bincount(tid, e2[:, d], T) for d in range(7)], 1)
    return {"sse": sse, "sy": sy, "syy": syy, "n": n, "tid": tid, "y": y,
            "perts": [str(s) for s in z["perturbations"]]}


def r2_from(sse, sy, syy, n):
    """Pooled R² (mean over 7 dims). Leading axes broadcast; task axis already summed."""
    var = syy - sy ** 2 / n[..., None]
    return (1.0 - sse / var).mean(-1)


def main() -> None:
    stats, ref_y = {}, None
    for f in sorted(RAW.glob("*.npz")):
        name = f.stem
        if name.startswith("submitted_") and name.endswith("_frozen"):
            key = (name.split("_")[1], "frozen", -1)
        else:
            _, bb, *cond, s = name.split("_")
            key = (bb, "_".join(cond), int(s[1:]))
        stats[key] = task_stats(f)
        if ref_y is None:
            ref_y, ref_tid = stats[key]["y"], stats[key]["tid"]
        elif not (np.array_equal(ref_y, stats[key]["y"]) and np.array_equal(ref_tid, stats[key]["tid"])):
            raise ValueError(f"{name}: targets/task pairing differ from the other archives")
    perts = next(iter(stats.values()))["perts"]
    clean, nonclean = perts.index("clean"), [i for i, p in enumerate(perts) if p != "clean"]
    sy, syy, n = (next(iter(stats.values()))[k] for k in ("sy", "syy", "n"))
    T = len(n)
    rng = np.random.default_rng(SEED)
    counts = rng.multinomial(T, np.full(T, 1 / T), size=N_BOOT).astype(np.float64)  # (B, T)

    def cell_r2(key, weights):
        """R² per (probe, perturbation) for task weights (W, T) -> (W, P, Q)."""
        s = stats[key]
        sse = np.einsum("wt,pqtd->wpqd", weights, s["sse"])
        return r2_from(sse, np.einsum("wt,td->wd", weights, sy)[:, None, None],
                       np.einsum("wt,td->wd", weights, syy)[:, None, None], (weights @ n)[:, None, None])

    full = np.ones((1, T))
    ref = {(r["backbone"], r["contrast"], r["perturbation"]): r
           for r in csv.DictReader((EXPORT / "tables/rebuttal_decomposition.csv").open())}
    rows = []
    for bb in BACKBONES:
        cache = {}

        def r2(cond, seed, weights, tag):
            k = (cond, seed, tag)
            if k not in cache:
                cache[k] = cell_r2((bb, cond, seed), weights)
            return cache[k]

        for a, b in CONTRASTS:
            for summary in ("clean", "worst_nonclean"):
                if summary == "worst_nonclean" and (a, b) != ("true_id", "shuffled_id"):
                    continue
                vals = {}
                for tag, w in (("full", full), ("boot", counts)):
                    diffs = []
                    for seed in (0, 1):
                        ra = r2(a, seed, w, tag)
                        rb = r2(b, -1 if b == "frozen" else seed, w, tag)
                        if summary == "clean":
                            diffs.append(ra[:, :, clean] - rb[:, :, clean])
                        else:
                            diffs.append(ra[:, :, nonclean].min(-1) - rb[:, :, nonclean].min(-1))
                    vals[tag] = np.mean(np.stack(diffs, 1), axis=(1, 2))  # average 2 seeds x 3 probes
                est = float(vals["full"][0])
                lo, hi = np.percentile(vals["boot"], [2.5, 97.5])
                g = ref.get((bb, f"{a} - {b}", summary))
                rows.append({"backbone": bb, "contrast": f"{a} - {b}", "summary": summary,
                             "recomputed_estimate": f"{est:.6f}", "gcp_estimate": g["estimate"] if g else "",
                             "estimate_abs_diff": f"{abs(est - float(g['estimate'])):.2e}" if g else "",
                             "recomputed_ci": f"[{lo:.3f}, {hi:.3f}]",
                             "gcp_ci": f"[{float(g['ci_low']):.3f}, {float(g['ci_high']):.3f}]" if g else ""})
        for cond, seeds in (("frozen", (-1,)), ("true_id", (0, 1)), ("shuffled_id", (0, 1)),
                            ("forward_dynamics", (0, 1))):
            v = np.mean([r2(cond, s, full, "full")[0, :, clean].mean() for s in seeds])
            rows.append({"backbone": bb, "contrast": f"absolute {cond} (26-task, seed/probe mean)",
                         "summary": "clean", "recomputed_estimate": f"{v:.6f}"})

    # 200-episode submitted-compatible point estimates: mean over training seeds x probe seeds
    sc = defaultdict(list)
    for line in (EXPORT / "results/submitted_compatible_clean_probe.jsonl").read_text().splitlines():
        r = json.loads(line)
        sc[(r["backbone"], r["condition"])].append(float(r["clean_task_ood_r2"]))
    for (bb, cond), v in sorted(sc.items()):
        rows.append({"backbone": bb, "contrast": f"absolute {cond} (200 ep, mean of {len(v)} rows)",
                     "summary": "clean", "recomputed_estimate": f"{np.mean(v):.6f}"})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["backbone", "contrast", "summary", "recomputed_estimate", "gcp_estimate", "estimate_abs_diff",
              "recomputed_ci", "gcp_ci"]
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
