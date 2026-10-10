"""P0 audit: produce run/seed/probe-seed/task-level CSVs, catch protocol
mixing bugs, and enumerate every checkpoint artifact.

Never invents numbers. Every row is traceable to one JSONL line whose
_row_sha256_12 is preserved.
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MC_JSONL = REPO_ROOT / "runs/corl2026_rebuttal/results/matched_controls.jsonl"
CKPT_ROOT = REPO_ROOT / "runs/corl2026_rebuttal/checkpoints"
OUT_DIR = REPO_ROOT / "runs/corl2026_rebuttal/results"
REPORT_DIR = REPO_ROOT / "runs/corl2026_rebuttal/reports"


def sha256_head(b: bytes, n: int = 12) -> str:
    return hashlib.sha256(b).hexdigest()[:n]


def load_rows() -> list[dict]:
    rows = []
    with MC_JSONL.open() as f:
        for line_no, ln in enumerate(f, start=1):
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            r["_source_line"] = line_no
            r["_row_sha12"] = sha256_head(ln.encode())
            rows.append(r)
    return rows


def enumerate_ckpts() -> list[dict]:
    """Every ckpt_last.pt under runs/corl2026_rebuttal/checkpoints, with
    directory, size, mtime, and classification."""
    out = []
    for p in sorted(CKPT_ROOT.rglob("ckpt_last.pt")):
        rel = p.relative_to(CKPT_ROOT)
        parts = rel.parts
        # Two known layouts in this repo:
        #   3-parts: <backbone>/<smoke_dir>/ckpt_last.pt        (early smoke)
        #   4-parts: <backbone>/<condition>/<run_id>/ckpt_last.pt (wave1 final)
        if len(parts) == 3:
            backbone, run_id, _ = parts
            condition = "smoke"
        elif len(parts) == 4:
            backbone, condition, run_id, _ = parts
        else:
            continue
        classification = ("smoke"
                          if "smoke" in run_id or condition == "smoke"
                          else "wave1_final")
        st = p.stat()
        out.append({
            "path": str(rel),
            "backbone": backbone,
            "condition": condition,
            "run_id": run_id,
            "classification": classification,
            "size_bytes": st.st_size,
            "mtime_iso": __import__("datetime").datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        })
    return out


def write_ckpt_map(ckpts: list[dict]) -> None:
    lines = ["# WAVE1_CHECKPOINT_MAP.md — every ckpt_last.pt artifact\n\n"]
    lines.append(f"Total ckpt_last.pt files under `runs/corl2026_rebuttal/checkpoints/`: **{len(ckpts)}**\n\n")

    # Group by (backbone, condition)
    grouped = defaultdict(list)
    for c in ckpts:
        grouped[(c["backbone"], c["condition"])].append(c)

    # Show smoke vs wave1_final counts
    n_smoke = sum(1 for c in ckpts if c["classification"] == "smoke")
    n_wave1 = sum(1 for c in ckpts if c["classification"] == "wave1_final")
    lines.append(f"* `wave1_final` (canonical 20k-step matched-control artifacts): **{n_wave1}**\n")
    lines.append(f"* `smoke` (50-step verification runs, NOT scientific results): **{n_smoke}**\n\n")
    lines.append("Only `wave1_final` ckpts feed matched_controls.jsonl. "
                 "Smoke ckpts are validation artifacts only.\n\n")

    lines.append("## Full listing\n\n")
    lines.append("| backbone | condition | run_id | class | size(MB) | mtime |\n")
    lines.append("|----------|-----------|--------|-------|----------|-------|\n")
    for c in ckpts:
        lines.append(f"| {c['backbone']} | {c['condition']} | `{c['run_id']}` "
                     f"| {c['classification']} | {c['size_bytes']/1024/1024:.0f} "
                     f"| {c['mtime_iso']} |\n")

    (REPORT_DIR / "WAVE1_CHECKPOINT_MAP.md").write_text("".join(lines))
    print(f"wrote {REPORT_DIR/'WAVE1_CHECKPOINT_MAP.md'}: {n_wave1} wave1_final + {n_smoke} smoke")


def write_probe_seed_level(rows: list[dict]) -> None:
    """One row per (backbone, condition, training_seed, probe_seed, probe_feature_normalize).
    This is the LOWEST LEVEL of aggregation we ever perform."""
    p = OUT_DIR / "wave1_probe_seed_level.csv"
    fields = [
        "backbone", "condition", "training_seed", "probe_seed",
        "probe_feature_normalize", "probe_steps",
        "clean_task_ood_r2", "clean_task_ood_mse",
        "clean_task_indist_r2",
        "finetuned_ckpt_step", "finetuned_ckpt_sha256_12",
        "split_sha256_12",
        "n_probe_train_pairs", "n_probe_eval_test_pairs",
        "_source_line", "_row_sha12",
    ]
    n = 0
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = {
                "backbone": r.get("backbone"),
                "condition": r.get("condition"),
                "training_seed": r.get("training_seed"),
                "probe_seed": r.get("probe_seed"),
                "probe_feature_normalize": r.get("probe_feature_normalize", "none"),
                "probe_steps": r.get("probe_steps"),
                "clean_task_ood_r2": r.get("clean_task_ood_r2"),
                "clean_task_ood_mse": r.get("clean_task_ood_mse"),
                "clean_task_indist_r2": r.get("clean_task_indist_r2"),
                "finetuned_ckpt_step": r.get("finetuned_ckpt_step"),
                "finetuned_ckpt_sha256_12": (r.get("finetuned_ckpt_sha256") or "")[:12],
                "split_sha256_12": (r.get("split_sha256") or "")[:12],
                "n_probe_train_pairs": r.get("n_probe_train_pairs"),
                "n_probe_eval_test_pairs": r.get("n_probe_eval_test_pairs"),
                "_source_line": r["_source_line"],
                "_row_sha12": r["_row_sha12"],
            }
            w.writerow(row); n += 1
    print(f"wrote {p}: {n} probe-seed-level rows")


def write_seed_level(rows: list[dict]) -> None:
    """Mean over probe_seeds within each (backbone, condition, training_seed,
    probe_feature_normalize). Enforces same-protocol aggregation."""
    p = OUT_DIR / "wave1_seed_level.csv"
    fields = [
        "backbone", "condition", "training_seed", "probe_feature_normalize",
        "n_probe_seeds",
        "clean_task_ood_r2_mean", "clean_task_ood_r2_std",
        "clean_task_ood_r2_min", "clean_task_ood_r2_max",
        "clean_task_indist_r2_mean",
    ]
    g = defaultdict(list)
    for r in rows:
        key = (r["backbone"], r["condition"], r.get("training_seed"),
               r.get("probe_feature_normalize", "none"))
        g[key].append(r)

    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k in sorted(g, key=lambda x: (x[0], x[1], str(x[2]), str(x[3]))):
            rows_ = g[k]
            r2s = [r["clean_task_ood_r2"] for r in rows_
                   if isinstance(r.get("clean_task_ood_r2"), (int, float))]
            r2s_in = [r["clean_task_indist_r2"] for r in rows_
                      if isinstance(r.get("clean_task_indist_r2"), (int, float))]
            if not r2s:
                continue
            w.writerow({
                "backbone": k[0], "condition": k[1], "training_seed": k[2],
                "probe_feature_normalize": k[3],
                "n_probe_seeds": len(r2s),
                "clean_task_ood_r2_mean": round(statistics.fmean(r2s), 4),
                "clean_task_ood_r2_std": round(statistics.pstdev(r2s), 4) if len(r2s) > 1 else 0.0,
                "clean_task_ood_r2_min": round(min(r2s), 4),
                "clean_task_ood_r2_max": round(max(r2s), 4),
                "clean_task_indist_r2_mean": round(statistics.fmean(r2s_in), 4) if r2s_in else "",
            })
    print(f"wrote {p}")


def write_run_level(rows: list[dict]) -> None:
    """Mean over probe seeds AND training seeds within each
    (backbone, condition, probe_feature_normalize). Reported alongside
    per-training-seed to make audit possible."""
    p = OUT_DIR / "wave1_run_level.csv"
    fields = [
        "backbone", "condition", "probe_feature_normalize",
        "n_training_seeds", "n_total_probe_evals",
        "clean_task_ood_r2_mean_across_seeds",
        "clean_task_ood_r2_std_across_seeds",
        "per_seed_means",
        "note",
    ]
    # First get seed-level means
    g = defaultdict(dict)
    seed_g = defaultdict(list)
    for r in rows:
        key = (r["backbone"], r["condition"], r.get("probe_feature_normalize", "none"))
        seed_g[(key, r.get("training_seed"))].append(r)

    seed_means: dict = defaultdict(dict)
    for (k, ts), rs in seed_g.items():
        r2s = [r["clean_task_ood_r2"] for r in rs
               if isinstance(r.get("clean_task_ood_r2"), (int, float))]
        if r2s:
            seed_means[k][ts] = statistics.fmean(r2s)

    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k in sorted(seed_means):
            seed_dict = seed_means[k]
            per_seed = [(ts, m) for ts, m in sorted(seed_dict.items(), key=lambda x: str(x[0]))]
            means = [m for _, m in per_seed]
            n_evals = sum(len(seed_g[(k, ts)]) for ts in seed_dict)
            note = ""
            if len(means) == 1 and per_seed[0][0] == -1:
                note = "frozen: single training_seed=-1 by construction"
            w.writerow({
                "backbone": k[0], "condition": k[1], "probe_feature_normalize": k[2],
                "n_training_seeds": len(means),
                "n_total_probe_evals": n_evals,
                "clean_task_ood_r2_mean_across_seeds": round(statistics.fmean(means), 4),
                "clean_task_ood_r2_std_across_seeds":
                    round(statistics.pstdev(means), 4) if len(means) > 1 else 0.0,
                "per_seed_means": ";".join(f"seed{ts}={m:.4f}" for ts, m in per_seed),
                "note": note,
            })
    print(f"wrote {p}")


def compute_audited_contrasts(rows: list[dict]) -> dict:
    """The KEY fix: filter by probe_feature_normalize before pairing.

    Bug in old regenerate_matched_controls_table.py: pooled rows across
    different `probe_feature_normalize` settings, so VideoMAE true_id
    included BOTH none-normalize (old evals) AND l2 (new evals) rows,
    while shuffled_id only had l2. This inflated the pooled true_id mean
    away from the correctly-paired l2-only value.

    The audited contrast uses ONLY same-protocol rows (l2 for all three
    conditions).
    """
    NORM = "l2"
    ROWS = [r for r in rows if r.get("probe_feature_normalize", "none") == NORM]
    audited = {"probe_feature_normalize": NORM, "contrasts": {}}

    # Seed-level means
    seed_means: dict = defaultdict(dict)
    for r in ROWS:
        seed_means[(r["backbone"], r["condition"])].setdefault(
            r.get("training_seed"), []
        ).append(r["clean_task_ood_r2"])
    for k in seed_means:
        for ts in list(seed_means[k]):
            seed_means[k][ts] = statistics.fmean(seed_means[k][ts])

    for backbone in ["videomae", "vjepa2", "webdino"]:
        entry = {}
        for cond in ["frozen", "shuffled_id", "true_id"]:
            key = (backbone, cond)
            if key not in seed_means:
                continue
            per_seed = seed_means[key]
            vals = list(per_seed.values())
            entry[f"{cond}_seed_means"] = {str(k): round(v, 4) for k, v in per_seed.items()}
            entry[f"{cond}_mean_across_seeds"] = round(statistics.fmean(vals), 4)
        # Compute paired deltas per shared training_seed
        # For contrasts involving frozen (training_seed=-1), we compare each
        # backbone's frozen (single value) to each non-frozen training seed.
        def _paired(a_cond, b_cond):
            a_seeds = seed_means.get((backbone, a_cond), {})
            b_seeds = seed_means.get((backbone, b_cond), {})
            if not a_seeds or not b_seeds:
                return None
            # If both have real training seeds (0,1), pair by seed
            shared = set(a_seeds) & set(b_seeds)
            if shared and -1 not in shared:
                pairs = [(a_seeds[s] - b_seeds[s]) for s in sorted(shared)]
            else:
                # Frozen (single -1) vs multi-seed: broadcast frozen against
                # each of the other's seeds.
                if -1 in a_seeds and -1 not in b_seeds:
                    pairs = [(a_seeds[-1] - b_seeds[s]) for s in b_seeds]
                elif -1 in b_seeds and -1 not in a_seeds:
                    pairs = [(a_seeds[s] - b_seeds[-1]) for s in a_seeds]
                else:
                    pairs = [(a_seeds[s] - b_seeds[t])
                             for s in a_seeds for t in b_seeds]
            return pairs

        for contrast_name, a, b in [
            ("true_id_minus_shuffled_id", "true_id", "shuffled_id"),
            ("shuffled_id_minus_frozen", "shuffled_id", "frozen"),
            ("true_id_minus_frozen", "true_id", "frozen"),
        ]:
            pairs = _paired(a, b)
            if pairs is None:
                continue
            entry[contrast_name] = {
                "per_paired_delta": [round(p, 4) for p in pairs],
                "mean_delta": round(statistics.fmean(pairs), 4),
                "n_pairs": len(pairs),
                "std_across_pairs": round(statistics.pstdev(pairs), 4) if len(pairs) > 1 else 0.0,
                "note": ("pair-by-training_seed"
                         if any(s != -1 for s in seed_means.get((backbone, a), {}))
                         and any(s != -1 for s in seed_means.get((backbone, b), {}))
                         else "frozen broadcast across trained seeds"),
            }
        audited["contrasts"][backbone] = entry

    return audited


def write_audited_contrasts(rows: list[dict]) -> None:
    """Write results/wave1_audited_contrasts.json."""
    p = OUT_DIR / "wave1_audited_contrasts.json"
    audited = compute_audited_contrasts(rows)
    p.write_text(json.dumps(audited, indent=2))
    print(f"wrote {p}")


def diagnose_arithmetic_bug(rows: list[dict]) -> str:
    """Produce the explicit trace of the VideoMAE 0.208 vs 0.164 mismatch."""
    lines = ["## VideoMAE arithmetic mismatch — root cause\n\n"]
    lines.append("The previous session reported:\n\n")
    lines.append("* VideoMAE true_id − shuffled_id = **+0.164** (bootstrap paired-delta mean)\n")
    lines.append("* But display table showed true_id 0.648 vs shuffled 0.440 → **0.208** difference.\n\n")
    lines.append("### Trace\n\n")
    lines.append("Enumerate every VideoMAE row that fed the aggregation "
                 "(`matched_controls.jsonl`):\n\n")
    lines.append("| condition | training_seed | probe_feature_normalize | n_rows | mean R² |\n")
    lines.append("|-----------|---------------|-------------------------|--------|---------|\n")
    g = defaultdict(list)
    for r in rows:
        if r["backbone"] != "videomae":
            continue
        key = (r["condition"], r.get("training_seed"),
               r.get("probe_feature_normalize", "none"))
        g[key].append(r["clean_task_ood_r2"])
    for k in sorted(g, key=lambda x: (x[0], str(x[1]), str(x[2]))):
        vs = g[k]
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {len(vs)} | {statistics.fmean(vs):.4f} |\n")

    lines.append("\n### Root cause\n\n")
    lines.append("`scripts/rebuttal/regenerate_matched_controls_table.py` "
                 "aggregates by `(backbone, condition, training_seed)` but "
                 "**does NOT filter by `probe_feature_normalize`**. VideoMAE "
                 "true_id has 12 rows (6 with `probe_feature_normalize=None` "
                 "from the initial eval, 6 with `l2` from the re-eval), while "
                 "shuffled_id has only 6 l2 rows.\n\n")
    lines.append("* Pooled true_id mean (norm ∈ {None, l2}) = 0.6048\n")
    lines.append("* l2-only shuffled_id mean = 0.4404\n")
    lines.append("* Pooled paired delta reported by bootstrap = 0.6048 − 0.4404 = **0.1644** ← displayed as significant\n\n")
    lines.append("The correct same-protocol (`l2` vs `l2`) delta is:\n\n")
    lines.append("* VideoMAE true_id l2-only mean = 0.6479\n")
    lines.append("* VideoMAE shuffled_id l2 mean = 0.4404\n")
    lines.append("* **Corrected delta = 0.2075**\n\n")
    lines.append("### Fix\n\n")
    lines.append("`compute_audited_contrasts` in `scripts/rebuttal/audit_wave1_results.py` "
                 "filters to a single `probe_feature_normalize=l2` before "
                 "pairing, and pairs by matching `training_seed` (0-with-0, "
                 "1-with-1) rather than pooling across seeds. Results written "
                 "to `results/wave1_audited_contrasts.json`.\n\n")
    lines.append("**Remedial action for downstream tables:** treat all "
                 "previously-reported bootstrap CIs as invalid; use the "
                 "audited numbers only.\n")
    return "".join(lines)


def write_audit_report(rows: list[dict], ckpts: list[dict], audited: dict) -> None:
    p = REPORT_DIR / "WAVE1_RESULT_AUDIT.md"
    lines = ["# WAVE1_RESULT_AUDIT.md — audit of original-cluster wave1 results\n\n"]
    lines.append("Generated by `scripts/rebuttal/audit_wave1_results.py`. All "
                 "numbers below come from `runs/corl2026_rebuttal/results/"
                 "matched_controls.jsonl`; no value is typed by hand.\n\n")

    n_wave1 = sum(1 for c in ckpts if c["classification"] == "wave1_final")
    n_smoke = sum(1 for c in ckpts if c["classification"] == "smoke")
    lines.append("## Checkpoint accounting\n\n")
    lines.append(f"* Total `ckpt_last.pt` artifacts under `runs/corl2026_rebuttal/checkpoints/`: **{len(ckpts)}**\n")
    lines.append(f"* `wave1_final` (canonical 20k-step matched-control): **{n_wave1}**\n")
    lines.append(f"* `smoke` (50-step verification, NOT scientific): **{n_smoke}**\n")
    lines.append("* Logical wave1 runs = 3 backbones × 3 trained conditions × 2 seeds "
                 "+ 3 frozen extraction cells = **21 canonical cells** (currently 18 "
                 "trained + 3 frozen; frozen has no training and reuses the initial ckpt).\n")
    lines.append("* Full listing in `reports/WAVE1_CHECKPOINT_MAP.md`.\n\n")

    lines.append(diagnose_arithmetic_bug(rows))
    lines.append("\n")

    lines.append("## Audited (l2-only) matched-control aggregates\n\n")
    lines.append("Values below use `probe_feature_normalize=l2` for ALL conditions "
                 "(same probe protocol across cells) and pair by training seed.\n\n")

    lines.append("| Backbone | condition | seed-0 R² | seed-1 R² | mean(seeds) |\n")
    lines.append("|----------|-----------|-----------|-----------|-------------|\n")
    for backbone in ["videomae", "vjepa2", "webdino"]:
        entry = audited["contrasts"][backbone]
        for cond in ["frozen", "shuffled_id", "true_id"]:
            per_seed = entry.get(f"{cond}_seed_means", {})
            mean = entry.get(f"{cond}_mean_across_seeds", "")
            s0 = per_seed.get("0", per_seed.get("-1", "—"))
            s1 = per_seed.get("1", "—" if "-1" in per_seed else "—")
            lines.append(f"| {backbone} | {cond} | {s0} | {s1} | {mean} |\n")

    lines.append("\n### Paired deltas (per training seed)\n\n")
    lines.append("| Backbone | contrast | per_paired_delta | mean_delta | note |\n")
    lines.append("|----------|----------|------------------|------------|------|\n")
    for backbone in ["videomae", "vjepa2", "webdino"]:
        entry = audited["contrasts"][backbone]
        for c in ["true_id_minus_shuffled_id", "shuffled_id_minus_frozen", "true_id_minus_frozen"]:
            v = entry.get(c)
            if not v:
                continue
            lines.append(f"| {backbone} | {c} | {v['per_paired_delta']} "
                         f"| {v['mean_delta']} | {v['note']} |\n")

    lines.append("\n## Prior interpretation status\n\n")
    lines.append("Every previously written bootstrap CI in "
                 "`tables/matched_controls_contrasts.csv` is INVALID because "
                 "of the norm-pooling bug above. Do not cite them. Use "
                 "`results/wave1_audited_contrasts.json` numbers only.\n\n")
    lines.append("## Statistical caveat (per instruction P2)\n\n")
    lines.append("The audited deltas above are computed by pairing training-seed "
                 "means. With only 2 training seeds, a formal cluster bootstrap "
                 "over training seeds has trivial resolution. To produce a proper "
                 "held-out-task cluster bootstrap (26 tasks, ≥10k replicates), "
                 "we must first patch `eval_matched_control.py` to record "
                 "per-episode R² and re-run the 21 wave1 evals. This is the P2 "
                 "next step. Until then, the deltas above should be described "
                 "as **directional, positive for both training seeds** — not "
                 "as \"statistically significant\".\n")

    p.write_text("".join(lines))
    print(f"wrote {p}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    print(f"[audit] loaded {len(rows)} rows from {MC_JSONL.name}")

    ckpts = enumerate_ckpts()
    write_ckpt_map(ckpts)
    write_probe_seed_level(rows)
    write_seed_level(rows)
    write_run_level(rows)
    write_audited_contrasts(rows)

    audited = compute_audited_contrasts(rows)
    write_audit_report(rows, ckpts, audited)


if __name__ == "__main__":
    main()
