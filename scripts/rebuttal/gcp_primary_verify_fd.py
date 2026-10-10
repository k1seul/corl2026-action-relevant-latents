"""Independent verification of forward-dynamics absolute R² arms from the
GCP primary import (`~/gcp_primary_import/raw_predictions/*.npz`).

**Boundary rule (user directive)**:
  * GCP primary evidence — imported tables (when they arrive) are the
    source of truth for the FD arms and contrasts.
  * These NPZ files are used **only for independent verification** of the
    absolute FD arms / contrasts computed from those tables.
  * All derived outputs live under
    `runs/corl2026_rebuttal/gcp_primary_imported_evidence/` — a namespace
    kept STRICTLY SEPARATE from the original-cluster secondary ledger.
  * Do NOT merge absolute R² across the secondary (original-cluster) and
    primary (GCP) protocols in any table or figure.

NPZ file layout (recorded by GCP pipeline):
  * predictions:  (n_probe_seeds=3, n_perturbations=8, n_frames=41600, 7)
  * targets:      (41600, 7)
  * task_ids:     (41600,) int16 — 0..25
  * episode_ids:  (41600,) int32 — 0..1299
  * perturbations: 8 str labels; index 0 = 'clean'
  * probe_seeds:  [0, 1, 2]
  * task_keys:    26 task strings (aligned with task_ids)
  * protocol_json:            submitted_compatible_clean_probe_v1 spec
  * cluster_protocol_json / sha256: task-cluster bootstrap protocol id

Absolute-arm computation (per backbone × training seed):
  1. For each probe seed, compute pooled R² over all 26 tasks × 50 episodes
     using the CLEAN perturbation predictions (index 0). Aggregate =
     mean over 7 per-dimension R². This matches the
     `submitted_compatible_clean_probe_v1` protocol declared in the NPZ.
  2. Per-cell arm = mean over the 3 probe seeds.
  3. Per-backbone arm (seed-averaged) = arithmetic mean over training
     seeds (s0, s1).

Outputs:
  * results/gcp_fd_absolute_arms_indep_verify.jsonl
  * tables/gcp_fd_absolute_arms_indep_verify.csv
  * reports/GCP_FD_ABSOLUTE_ARMS_INDEP_VERIFY.md
all under `runs/corl2026_rebuttal/gcp_primary_imported_evidence/`.
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
NAMESPACE = REPO_ROOT / "runs/corl2026_rebuttal/gcp_primary_imported_evidence"
NPZ_DIR = Path.home() / "gcp_primary_import/raw_predictions"

OUT_JSONL = NAMESPACE / "results/gcp_fd_absolute_arms_indep_verify.jsonl"
OUT_CSV   = NAMESPACE / "tables/gcp_fd_absolute_arms_indep_verify.csv"
OUT_MD    = NAMESPACE / "reports/GCP_FD_ABSOLUTE_ARMS_INDEP_VERIFY.md"

BACKBONES = ("vjepa2", "videomae", "webdino")
SEEDS = (0, 1)


def pooled_r2(pred_flat: np.ndarray, target_flat: np.ndarray) -> float:
    """Mean over 7 per-dim R² on flattened (N, 7) arrays."""
    var = target_flat.var(axis=0).clip(min=1e-12)
    per_dim_mse = ((pred_flat - target_flat) ** 2).mean(axis=0)
    per_dim_r2 = 1.0 - per_dim_mse / var
    finite = per_dim_r2[np.isfinite(per_dim_r2)]
    return float(finite.mean()) if finite.size else float("nan")


def per_task_pooled_r2(pred, target, task_ids, task_keys):
    """Task-macro R²: per-task pooled R², mean over tasks."""
    per_task = []
    for ti in range(len(task_keys)):
        m = task_ids == ti
        if m.sum() == 0: continue
        r2 = pooled_r2(pred[m], target[m])
        if np.isfinite(r2):
            per_task.append(r2)
    return float(np.mean(per_task)) if per_task else float("nan")


def load_npz_and_compute(path: Path):
    """Return per-cell record + per-seed diagnostics."""
    d = np.load(path, allow_pickle=True)
    preds = d["predictions"]      # (3, 8, 41600, 7)
    tgts  = d["targets"]          # (41600, 7)
    tids  = d["task_ids"]         # (41600,)
    eids  = d["episode_ids"]      # (41600,)
    perturbs = [str(p) for p in d["perturbations"].tolist()]
    probe_seeds = [int(s) for s in d["probe_seeds"].tolist()]
    task_keys = [str(t) for t in d["task_keys"].tolist()]

    assert perturbs[0].lower() == "clean", f"expected first perturbation = 'clean', got {perturbs[0]}"
    clean_idx = 0

    # Per-probe-seed pooled R² over all 26 tasks (clean perturbation)
    per_seed_pooled = []
    per_seed_taskmacro = []
    for ps_i, ps in enumerate(probe_seeds):
        pred_flat = preds[ps_i, clean_idx]   # (41600, 7)
        pooled = pooled_r2(pred_flat, tgts)
        taskmacro = per_task_pooled_r2(pred_flat, tgts, tids, task_keys)
        per_seed_pooled.append(pooled)
        per_seed_taskmacro.append(taskmacro)

    return {
        "npz_file": str(path.name),
        "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n_frames": int(tgts.shape[0]),
        "n_tasks": len(task_keys),
        "n_episodes": int(eids.max() + 1),
        "perturbation_used": perturbs[0],
        "probe_seeds": probe_seeds,
        "per_probe_seed_pooled_r2":   [round(v, 6) for v in per_seed_pooled],
        "per_probe_seed_task_macro_r2": [round(v, 6) for v in per_seed_taskmacro],
        "cell_pooled_r2_probe_mean":   float(statistics.fmean(per_seed_pooled)),
        "cell_task_macro_r2_probe_mean": float(statistics.fmean(per_seed_taskmacro)),
        "protocol_id": json.loads(str(d["protocol_json"].item()) if hasattr(d["protocol_json"], "item")
                                    else str(d["protocol_json"]))["protocol_id"],
        "cluster_protocol_sha256": str(d["cluster_protocol_sha256"]),
    }


def main():
    NAMESPACE.mkdir(parents=True, exist_ok=True)
    for sub in ("reports", "tables", "results"):
        (NAMESPACE / sub).mkdir(exist_ok=True)

    per_cell = {}   # (backbone, tseed) -> record
    for bb in BACKBONES:
        for s in SEEDS:
            fname = f"wave1_{bb}_forward_dynamics_s{s}.npz"
            p = NPZ_DIR / fname
            if not p.exists():
                print(f"[gcp-verify-fd] MISSING: {p}")
                continue
            print(f"[gcp-verify-fd] loading {fname}")
            per_cell[(bb, s)] = load_npz_and_compute(p)

    # Seed-averaged per-backbone arm
    per_backbone = {}
    for bb in BACKBONES:
        cells = [per_cell[(bb, s)] for s in SEEDS if (bb, s) in per_cell]
        if len(cells) < 2:
            per_backbone[bb] = {
                "seed_averaged_pooled_r2": float("nan"),
                "seed_averaged_task_macro_r2": float("nan"),
                "n_seeds_used": len(cells),
                "note": "seed-average requires both s0 and s1",
            }
            continue
        pooled_s = [c["cell_pooled_r2_probe_mean"] for c in cells]
        macro_s  = [c["cell_task_macro_r2_probe_mean"] for c in cells]
        per_backbone[bb] = {
            "seed_averaged_pooled_r2": float(statistics.fmean(pooled_s)),
            "seed_averaged_task_macro_r2": float(statistics.fmean(macro_s)),
            "per_training_seed_pooled_r2": {SEEDS[i]: pooled_s[i] for i in range(len(SEEDS))},
            "per_training_seed_task_macro_r2": {SEEDS[i]: macro_s[i] for i in range(len(SEEDS))},
            "n_seeds_used": len(cells),
        }

    # ── Write JSONL (one record per NPZ + one aggregate row per backbone) ──
    with OUT_JSONL.open("w") as f:
        for (bb, s), rec in sorted(per_cell.items()):
            f.write(json.dumps({"kind": "cell", "backbone": bb,
                                "training_seed": s, **rec}) + "\n")
        for bb, agg in per_backbone.items():
            f.write(json.dumps({"kind": "seed_averaged_arm",
                                "backbone": bb, "condition": "forward_dynamics",
                                **agg}) + "\n")

    # ── CSV ──
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "backbone", "condition", "seed_averaged_pooled_r2",
            "seed_averaged_task_macro_r2",
            "per_seed_pooled_r2_json", "per_seed_task_macro_r2_json",
            "n_seeds_used", "source_npz_files", "npz_sha256_head_s0", "npz_sha256_head_s1",
        ])
        for bb in BACKBONES:
            agg = per_backbone[bb]
            per_seed = agg.get("per_training_seed_pooled_r2", {})
            per_seed_m = agg.get("per_training_seed_task_macro_r2", {})
            files = [f"wave1_{bb}_forward_dynamics_s{s}.npz" for s in SEEDS]
            shas = [per_cell.get((bb, s), {}).get("npz_sha256", "")[:16] for s in SEEDS]
            w.writerow([
                bb, "forward_dynamics",
                f"{agg['seed_averaged_pooled_r2']:.6f}",
                f"{agg['seed_averaged_task_macro_r2']:.6f}",
                json.dumps({str(k): round(v, 6) for k, v in per_seed.items()}),
                json.dumps({str(k): round(v, 6) for k, v in per_seed_m.items()}),
                agg["n_seeds_used"],
                "|".join(files),
                shas[0], shas[1],
            ])

    # ── MD report ──
    lines = ["# GCP_FD_ABSOLUTE_ARMS_INDEP_VERIFY.md — independent FD arm computation from imported NPZ\n\n"]
    lines.append("> **NAMESPACE**: `GCP_PRIMARY_IMPORTED_EVIDENCE`. This file "
                 "lives strictly under `runs/corl2026_rebuttal/gcp_primary_imported_evidence/`. "
                 "It is a derived output of the GCP primary evidence import "
                 "and must **not** be merged with the original-cluster "
                 "secondary ledger (`ORIGINAL_CLUSTER_SECONDARY_EVIDENCE`). "
                 "Absolute R² here comes from the GCP `submitted_compatible_clean_probe_v1` "
                 "protocol and must not be quoted alongside secondary R² in "
                 "any single table, plot, or unqualified sentence.\n\n")
    lines.append("## Role\n\n")
    lines.append("The GCP primary tables (when they arrive under "
                 "`~/gcp_primary_import/tables/`) are the **source of truth** "
                 "for the FD arms and contrasts. This file is only an "
                 "**independent verification**: we load the six FD raw-"
                 "predictions NPZ files and recompute the absolute pooled R² "
                 "for each `(backbone × training_seed × probe_seed)` cell "
                 "directly from `predictions[probe_seed, clean, :, :]` and "
                 "`targets`. Once the GCP tables land, this file should "
                 "cross-check them to floating-point precision on shared "
                 "quantities.\n\n")

    lines.append("## NPZ provenance (SHA-256 of every file consumed)\n\n")
    lines.append("| Backbone | Seed | NPZ file | SHA-256 |\n|---|---|---|---|\n")
    for bb in BACKBONES:
        for s in SEEDS:
            rec = per_cell.get((bb, s))
            if rec:
                lines.append(f"| {bb} | {s} | `{rec['npz_file']}` | `{rec['npz_sha256']}` |\n")

    lines.append("\n## Protocol identifier (as recorded in every NPZ)\n\n")
    sample = next(iter(per_cell.values()))
    lines.append(f"* Protocol id: `{sample['protocol_id']}`\n")
    lines.append(f"* Cluster protocol SHA-256: `{sample['cluster_protocol_sha256']}`\n")
    lines.append(f"* Perturbation used for absolute arm: `{sample['perturbation_used']}`\n")
    lines.append(f"* n_frames per NPZ: {sample['n_frames']:,} = "
                 f"{sample['n_episodes']:,} episodes × 32 frames × 1 clip\n")
    lines.append(f"* n_tasks per NPZ: {sample['n_tasks']}\n\n")

    lines.append("## Independent FD absolute arms (this file)\n\n")
    lines.append("Per-cell = mean over the 3 probe seeds of "
                 "`pooled R² (clean perturbation, all 26 tasks × 50 episodes)`. "
                 "Seed-averaged = arithmetic mean over training seeds "
                 "`(cell_s0 + cell_s1) / 2`.\n\n")
    lines.append("| Backbone | Cell s0 (probe-mean) | Cell s1 (probe-mean) | Seed-averaged FD arm | Task-macro (probe-mean s0 / s1) |\n")
    lines.append("|----------|-----------------------|-----------------------|-----------------------|---------------------------------|\n")
    for bb in BACKBONES:
        s0 = per_cell.get((bb, 0), {})
        s1 = per_cell.get((bb, 1), {})
        agg = per_backbone[bb]
        p0 = s0.get("cell_pooled_r2_probe_mean")
        p1 = s1.get("cell_pooled_r2_probe_mean")
        m0 = s0.get("cell_task_macro_r2_probe_mean")
        m1 = s1.get("cell_task_macro_r2_probe_mean")
        lines.append(
            f"| {bb} | {p0:+.4f} | {p1:+.4f} | "
            f"**{agg['seed_averaged_pooled_r2']:+.4f}** | "
            f"{m0:+.4f} / {m1:+.4f} |\n"
        )

    lines.append("\n### Per-probe-seed diagnostic (for cross-check with GCP tables)\n\n")
    lines.append("| Backbone | Seed | Probe seed 0 | Probe seed 1 | Probe seed 2 |\n")
    lines.append("|----------|------|---------------|---------------|---------------|\n")
    for bb in BACKBONES:
        for s in SEEDS:
            rec = per_cell.get((bb, s))
            if not rec: continue
            r = rec["per_probe_seed_pooled_r2"]
            lines.append(f"| {bb} | {s} | {r[0]:+.4f} | {r[1]:+.4f} | {r[2]:+.4f} |\n")

    lines.append("\n## Cross-verification against GCP tables\n\n")
    lines.append("* GCP tables directory: `~/gcp_primary_import/tables/` — currently empty at time of writing.\n")
    lines.append("* When the GCP tables arrive, extend this file with a comparison row per (backbone, training_seed) between the FD absolute R² this script computes and the value reported in the GCP tables. Expected: agreement to floating-point precision if the NPZ predictions were produced by the same pipeline that emitted the tables. Any drift larger than 1e-5 warrants investigation.\n")
    lines.append("* Do NOT quote the numbers above as the final GCP FD arms until the GCP tables are cross-checked; they are an **independent** computation for verification, not the primary artifact.\n\n")

    lines.append("## What NOT to do\n\n")
    lines.append("* Do NOT merge the numbers here with the original-cluster secondary matched-control R² values in `runs/corl2026_rebuttal/reports/REBUTTAL_EVIDENCE.md`. The two protocols differ (different probe hparams, different feature normalisation, different train/eval slicing) and their absolute R² regimes are not comparable.\n")
    lines.append("* Do NOT use these numbers to fill in the `forward_dynamics` column of `ALL_TASK_ABSOLUTE_ARMS.md` — that table is under the secondary namespace and its FD column stays `not_measured_this_cycle` per the boundary rule.\n")

    OUT_MD.write_text("".join(lines))
    print(f"\nwrote {OUT_JSONL}\nwrote {OUT_CSV}\nwrote {OUT_MD}")
    print("\nseed-averaged FD arms (independent from NPZ):")
    for bb in BACKBONES:
        agg = per_backbone[bb]
        print(f"  {bb:10} seed-avg pooled R² (clean, 26-task) = "
              f"{agg['seed_averaged_pooled_r2']:+.4f}")


if __name__ == "__main__":
    main()
