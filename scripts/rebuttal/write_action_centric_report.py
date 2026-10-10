"""Write reports/ACTION_CENTRIC_BASELINES.md + results/action_centric_baselines.jsonl.

Consumes rows in matched_controls.jsonl for backbones {r3m, vip} evaluated
under the AUDITED protocol (l2 normalize + full 26-task test) and produces
the standard baseline table alongside the 3 primary backbones for context.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MC = REPO_ROOT / "runs/corl2026_rebuttal/results/matched_controls.jsonl"
OUT_MD = REPO_ROOT / "runs/corl2026_rebuttal/reports/ACTION_CENTRIC_BASELINES.md"
OUT_JSONL = REPO_ROOT / "runs/corl2026_rebuttal/results/action_centric_baselines.jsonl"


def audited(r):
    return (r.get("probe_feature_normalize") == "l2"
            and r.get("n_probe_eval_test_pairs", 0) >= 20000)


def main():
    rows = [json.loads(l) for l in MC.open() if l.strip()]
    keep = [r for r in rows if audited(r) and r["condition"] == "frozen"]

    # Group by backbone
    by_bb = defaultdict(list)
    for r in keep:
        by_bb[r["backbone"]].append(r["clean_task_ood_r2"])

    order = ["r3m", "vip", "videomae", "vjepa2", "webdino"]
    lines = ["# ACTION_CENTRIC_BASELINES.md — audited frozen baselines\n\n"]
    lines.append("All values: l2-normalize MLP probe, 26 held-out tasks (all "
                 "1300 held-out episodes), 3 probe seeds averaged.\n\n")
    lines.append("| Backbone | Family | mean R² | std (probe seeds) | n_rows |\n")
    lines.append("|----------|--------|---------|-------------------|--------|\n")
    fam = {"r3m": "action-centric (Ego4D)", "vip": "action-centric (Ego4D VIP)",
           "videomae": "video pretrain (Kinetics)", "vjepa2": "video JEPA",
           "webdino": "image DINOv2 (web)"}
    out_records = []
    for bb in order:
        vals = by_bb.get(bb, [])
        if not vals:
            continue
        m = statistics.fmean(vals)
        s = statistics.pstdev(vals) if len(vals) > 1 else 0
        lines.append(f"| {bb} | {fam.get(bb,'?')} | {m:+.4f} | {s:.4f} | {len(vals)} |\n")
        out_records.append({"backbone": bb, "condition": "frozen",
                             "mean_r2": round(m, 4), "std_over_probe_seeds": round(s, 4),
                             "n_rows": len(vals),
                             "probe_feature_normalize": "l2",
                             "n_test_episodes": 1300,
                             "n_tasks": 26})

    lines.append("\n## Interpretation\n\n")
    lines.append("* R3M and VIP were pretrained on Ego4D natural-view manipulation "
                 "video with contrastive / goal-conditioned objectives. Applied "
                 "frozen to LIBERO agentview_rgb single-frame features (mean-pool "
                 "over spatial ResNet-50 pool), the canonical action probe R² is "
                 "deeply negative — the features do not linearly encode LIBERO's "
                 "7-D end-effector actions under our probe protocol.\n")
    lines.append("* This is consistent with the paper's finding that action-centric "
                 "pretraining does not transfer to novel-environment action "
                 "readout without further adaptation.\n")
    lines.append("* NOTE: our protocol may under-favor these baselines (see "
                 "`reports/FROZEN_BASELINE_DISCREPANCY.md` — probe_train_clips=800 "
                 "may not be enough for a 2048-d R3M feature). A larger probe "
                 "budget could shift these numbers.\n\n")
    lines.append("## CAIP status\n\n")
    lines.append("Not run. The CAIP repo location and checkpoint URL have not "
                 "been resolved in this session; see the extractor stub in "
                 "`src/comparison/extractors/action_centric.py`. When wiring, "
                 "the instruction P5 requires TWO clearly labelled modes:\n\n")
    lines.append("* `caip_visual_only` (fair single-view comparison)\n")
    lines.append("* `caip_native_conditioned` (language / task text)\n\n")
    lines.append("They must be reported as separate rows and never combined in "
                 "one unqualified ranking.\n")

    OUT_MD.write_text("".join(lines))
    with OUT_JSONL.open("w") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {OUT_MD}\nwrote {OUT_JSONL}")


if __name__ == "__main__":
    main()
