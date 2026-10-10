"""R² vs ID-mask-p curve for V-JEPA + VideoMAE sparsity sweep."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    summary_path = REPO_ROOT / "runs/id_sparsity_sweep/summary.jsonl"
    rows = [json.loads(l) for l in summary_path.read_text().splitlines()]
    # dedup by variant: keep last
    by_var = {}
    for r in rows:
        by_var[r['variant']] = r

    # Endpoints from main matrix
    vjepa_frozen_r2 = 0.400  # vjepa2_vit_l_frozen test_cell
    videomae_frozen_r2 = 0.457  # videomae_frozen test_cell

    vjepa_pts = [(0.0, vjepa_frozen_r2)]
    videomae_pts = [(0.0, videomae_frozen_r2)]
    for tag, p in [("0p02", 0.02), ("0p05", 0.05), ("0p10", 0.10), ("0p25", 0.25), ("0p50", 0.50), ("1p00", 1.00)]:
        v = by_var.get(f"vjepa2_aux_p{tag}")
        if v: vjepa_pts.append((p, v['test_r2_aggregate']))
        m = by_var.get(f"videomae_aux_p{tag}")
        if m: videomae_pts.append((p, m['test_r2_aggregate']))
    # VideoMAE p=1.00 reuse main matrix (20k)
    videomae_pts.append((1.0, 0.748))

    vjepa_pts.sort(); videomae_pts.sort()

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    vx = [p for p, _ in vjepa_pts]; vy = [r for _, r in vjepa_pts]
    mx = [p for p, _ in videomae_pts]; my = [r for _, r in videomae_pts]
    ax.plot(vx, vy, "-o", color="tab:red",    linewidth=2, markersize=9,
              label="V-JEPA 2 ViT-L + ID")
    ax.plot(mx, my, "-s", color="tab:purple", linewidth=2, markersize=9,
              label="VideoMAE V1 ViT-L + ID")

    # Frozen reference dots labeled
    ax.scatter([0], [vjepa_frozen_r2], marker="o", s=140, facecolor="white",
                  edgecolor="tab:red", linewidth=2, zorder=5)
    ax.scatter([0], [videomae_frozen_r2], marker="s", s=140, facecolor="white",
                  edgecolor="tab:purple", linewidth=2, zorder=5)
    ax.text(0.005, vjepa_frozen_r2 - 0.03, "frozen", color="tab:red", fontsize=8)

    # Annotation: V-JEPA saturating
    ax.annotate("", xy=(0.10, 0.69), xytext=(0.02, 0.60),
                  arrowprops=dict(arrowstyle="->", color="tab:red", alpha=0.6, lw=1))
    ax.text(0.025, 0.55, "+0.20 R² lift\nfrom 2% ID", color="tab:red",
              fontsize=9, alpha=0.85)

    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xticks([0, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0])
    ax.set_xticklabels(["0", "0.02", "0.05", "0.10", "0.25", "0.50", "1.0"])
    ax.set_xlim(-0.005, 1.2)
    ax.set_ylim(0.30, 0.92)
    ax.set_xlabel("ID supervision fraction $p$ (per-sample mask probability)")
    ax.set_ylabel("Action probe R² on LIBERO task-OOD")
    ax.set_title("ID supervision sample-budget sweep:\nV-JEPA saturates at small $p$; VideoMAE scales more linearly")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.25, which="both")

    out_png = REPO_ROOT / "runs/id_sparsity_sweep/r2_vs_p.png"
    out_pdf = REPO_ROOT / "runs/id_sparsity_sweep/r2_vs_p.pdf"
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    fig.savefig(out_pdf)
    print(f"saved -> {out_png}")
    print(f"saved -> {out_pdf}")
    print(f"\nV-JEPA points: {vjepa_pts}")
    print(f"VideoMAE points: {videomae_pts}")


if __name__ == "__main__":
    main()
