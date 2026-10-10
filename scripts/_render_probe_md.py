"""Re-render probe_tier1_sweep.md from the existing jsonl.
Avoids re-running the sweep just because of the md-formatting bug fix."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import probe_sweep  # for MODELS, format_phase_table, mean_std, best_by_mean_r2

OUT = REPO_ROOT / "runs/downstream"
rows = [json.loads(l) for l in open(OUT / "probe_tier1_sweep.jsonl")]


def best_per_model(rows, phase, group_keys, kind="action"):
    out = {}
    for name in probe_sweep.MODELS:
        rs = [r for r in rows if r.get("phase") == phase and r["model"] == name and r["kind"] == kind]
        if not rs:
            continue
        b = probe_sweep.best_by_mean_r2(rs, group_keys)
        out[name] = b
    return out


# headline summary per model
summaries = {m: {} for m in probe_sweep.MODELS}
sanity = best_per_model(rows, "sanity_baseline", ["stack"])
for m, b in sanity.items():
    summaries[m]["sanity_baseline_r2"] = b["mean_r2"]
A_best = best_per_model(rows, "A", ["stack"])
for m, b in A_best.items():
    summaries[m]["best_stack"] = b["key"][0]
    summaries[m]["A_best_r2"] = b["mean_r2"]
B_best = best_per_model(rows, "B", ["arch"])
for m, b in B_best.items():
    summaries[m]["best_arch"] = b["key"][0]
    summaries[m]["B_best_r2"] = b["mean_r2"]
C_best = best_per_model(rows, "C", ["steps", "wd", "dropout"])
for m, b in C_best.items():
    summaries[m]["best_C"] = b["key"]
    summaries[m]["C_best_r2"] = b["mean_r2"]

# extras
for m in probe_sweep.MODELS:
    for kind in ("long_invdyn_h5", "multiframe_invdyn"):
        rs = [r for r in rows if r.get("phase") == "extras" and r.get("kind") == kind and r["model"] == m]
        if rs:
            v = [r["test_r2"] for r in rs]
            mu, sd = probe_sweep.mean_std(v)
            summaries[m][f"{kind}_r2"] = (mu, sd)

md = []
md.append("# Probe Tier-1 sweep — comprehensive probe-only OOD R² exploration\n")
md.append("Goal: push the v6_large_noact_taskheld task-OOD R² (baseline 0.717)")
md.append("via probe-side improvements only — no model retraining.\n")
md.append("**Models**: " + ", ".join(probe_sweep.MODELS.keys()) + ".\n")
md.append("> Caveat: `v_mix_token_noact_200k` was trained on the original 130-task split,")
md.append("> so the splits_taskheld/test_ood demos overlap with its training data.")
md.append("> Its 'OOD' R² is included for completeness but is *not* a true held-out generalization.\n")

md.append("## Headline — best probe per model (action H=1)\n")
md.append("| model | sanity baseline R² | best-stack R² (A) | best-arch R² (B) | best-hparam R² (C) |")
md.append("|---|---|---|---|---|")
for m in probe_sweep.MODELS:
    s = summaries[m]
    def f(x): return f"{x:+.4f}" if x is not None else "—"
    md.append(f"| {m} | {f(s.get('sanity_baseline_r2'))} | {f(s.get('A_best_r2'))} | {f(s.get('B_best_r2'))} | {f(s.get('C_best_r2'))} |")
md.append("")

md.append("## Phase A — frame stacking (action R²)\n")
for m in probe_sweep.MODELS:
    md.append(f"### {m}")
    md.append("**Action R² (H=1)**")
    md.append(probe_sweep.format_phase_table(
        [r for r in rows if r.get("phase") == "A" and r["model"] == m],
        ["stack"], "action",
    ))
    md.append("\n**Inverse dynamics R²**")
    md.append(probe_sweep.format_phase_table(
        [r for r in rows if r.get("phase") == "A" and r["model"] == m],
        ["stack"], "invdyn",
    ))
    md.append("")

md.append("## Phase B — MLP architecture (at best stack from A)\n")
for m in probe_sweep.MODELS:
    md.append(f"### {m} (stack={summaries[m].get('best_stack', '?')})")
    md.append(probe_sweep.format_phase_table(
        [r for r in rows if r.get("phase") == "B" and r["model"] == m],
        ["arch"], "action",
    ))
    md.append("")

md.append("## Phase C — training hparams (at best stack & arch)\n")
for m in probe_sweep.MODELS:
    md.append(f"### {m} (stack={summaries[m].get('best_stack', '?')}, "
              f"arch={summaries[m].get('best_arch', '?')})")
    md.append(probe_sweep.format_phase_table(
        [r for r in rows if r.get("phase") == "C" and r["model"] == m],
        ["steps", "wd", "dropout"], "action",
    ))
    md.append("")

md.append("## Extras — long & multi-frame inverse dynamics\n")
md.append("| model | long inv-dyn (H=5) R² | multi-frame inv-dyn (3 past + 3 future) R² |")
md.append("|---|---|---|")
for m in probe_sweep.MODELS:
    s = summaries[m]
    li = s.get("long_invdyn_h5_r2")
    mf = s.get("multiframe_invdyn_r2")
    li_s = f"{li[0]:+.4f}±{li[1]:.4f}" if li else "—"
    mf_s = f"{mf[0]:+.4f}±{mf[1]:.4f}" if mf else "—"
    md.append(f"| {m} | {li_s} | {mf_s} |")
md.append("")

md.append("## Production probe config recommendation\n")
for m in probe_sweep.MODELS:
    s = summaries[m]
    if "best_C" not in s:
        continue
    md.append(f"**{m}**: stack={s['best_stack']}, arch={s.get('best_arch')}, "
              f"hparams (steps,wd,dropout)={s['best_C']} → action R²={s.get('C_best_r2'):+.4f}")

(OUT / "probe_tier1_sweep.md").write_text("\n".join(md) + "\n")
print(f"wrote {OUT/'probe_tier1_sweep.md'}")
