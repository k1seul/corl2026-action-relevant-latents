"""Generate runs/sweep/aux_ablations/summary.md from probe results.

Run after probe_aux_ablations.sh completes:
    uv run python scripts/make_aux_ablation_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DS_ROOT = REPO_ROOT / "runs" / "downstream" / "aux_ablations"
OUT_MD = REPO_ROOT / "runs" / "sweep" / "aux_ablations" / "summary.md"


def load_txt(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def flt(d: dict, key: str, default: str = "?") -> str:
    v = d.get(key, default)
    try:
        return f"{float(v):.4f}"
    except (ValueError, TypeError):
        return str(v)


def main() -> None:
    baseline_r2 = 0.882
    baseline_ivd = 0.927

    def delta(val_str: str, ref: float) -> str:
        try:
            return f"{float(val_str) - ref:+.3f}"
        except (ValueError, TypeError):
            return "?"

    lines = ["# Aux ablation results\n"]

    # ── Table 1: head architecture ────────────────────────────────────────────
    A_runs = [
        ("Inverse dynamics L2 (current)", None, baseline_r2, baseline_ivd),
        ("Forward dynamics (A1)",         "aux_abl_A1_forward_dyn_ood",      None, None),
        ("Contrastive InfoNCE (A2)",       "aux_abl_A2_contrastive_ood",      None, None),
        ("Action classification 1-frame (A3)", "aux_abl_A3_single_frame_ood", None, None),
        ("Inverse dynamics L1 (A4)",       "aux_abl_A4_invdyn_l1_ood",        None, None),
        ("InvDyn L2 repro (A0)",           "aux_abl_A0_invdyn_l2_repro_ood",  None, None),
    ]

    lines.append("## Table 1 — head architecture variant\n")
    lines.append("| Variant | Action R² | Inv-dyn R² | Δ action R² vs baseline |")
    lines.append("|---|---|---|---|")
    for label, run_name, pre_r2, pre_ivd in A_runs:
        if run_name is None:
            r2_s, ivd_s = f"{pre_r2:.4f}", f"{pre_ivd:.4f}"
        else:
            m = load_txt(DS_ROOT / run_name / "metrics.txt")
            r2_s  = flt(m, "action_h1_r2_mean")
            ivd_s = flt(m, "invdyn_r2_mean")
        dv = delta(r2_s, baseline_r2)
        lines.append(f"| {label} | {r2_s} | {ivd_s} | {dv} |")

    lines.append("")

    # ── Table 2: multi-step ───────────────────────────────────────────────────
    B_runs = [
        (1,  None,                        baseline_r2, None),
        (2,  "aux_abl_B1_multistep_k2_ood", None,     None),
        (4,  "aux_abl_B2_multistep_k4_ood", None,     None),
        (8,  "aux_abl_B3_multistep_k8_ood", None,     None),
    ]

    lines.append("## Table 2 — multi-step inverse dynamics\n")
    lines.append("| k | Action R² (H=1 probe) | Action chunk R² (H=k probe) | Δ vs k=1 |")
    lines.append("|---|---|---|---|")
    for k, run_name, pre_r2, _ in B_runs:
        if run_name is None:
            r2_s = f"{pre_r2:.4f}"
            chunk_s = "—"
        else:
            m = load_txt(DS_ROOT / run_name / "metrics.txt")
            r2_s    = flt(m, "action_h1_r2_mean")
            chunk_s = flt(m, f"action_h{k}_r2_mean")
        dv = delta(r2_s, baseline_r2)
        lines.append(f"| {k} | {r2_s} | {chunk_s} | {dv} |")

    lines.append("")

    # ── Table 3: C1 vs C2 ────────────────────────────────────────────────────
    lines.append("## Table 3 — test-time aux head retain (C2)\n")
    lines.append("| Method | Action R² | Inv-dyn R² |")
    lines.append("|---|---|---|")
    lines.append(f"| Probe MLP trained from scratch (C1) | {baseline_r2:.4f} | {baseline_ivd:.4f} |")

    c2_m = load_txt(DS_ROOT / "C2_aux_direct" / "metrics.txt")
    c2_r2  = flt(c2_m, "action_r2")
    c2_ivd = flt(c2_m, "invdyn_r2")
    lines.append(f"| Frozen aux head applied directly (C2) | {c2_r2} | {c2_ivd} |")

    lines.append("")

    # ── Takeaways (auto-generated stubs) ─────────────────────────────────────
    lines.append("## Takeaways\n")
    lines.append("**Table 1**: _fill in after results_")
    lines.append("")
    lines.append("**Table 2**: _fill in after results_")
    lines.append("")
    lines.append("**Table 3**: _fill in after results_")
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
