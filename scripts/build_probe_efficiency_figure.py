"""R² vs probe-train size curve for Web-DINO/SigLIP 2 (frozen + aux)."""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    rows = []
    for line in (REPO_ROOT / "runs/full_results/libero_probe_data_efficiency.jsonl").read_text().splitlines():
        rows.append(json.loads(line))
    by_variant = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append((r["probe_train_size"], r["r2_test_mean"], r["r2_test_seeds"]))
    for k in by_variant:
        by_variant[k].sort()

    # Reference horizontals: V-JEPA 2 ViT-L +aux on LIBERO @ probe-train=400
    vjepa_ref = 0.845
    # CALVIN +aux of same backbones (at probe-train 400 default)
    calvin_ref = {"webdino_aux": 0.805, "siglip2_aux": 0.768}

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    colors = {"webdino_frozen": "tab:blue", "webdino_aux": "tab:blue",
              "siglip2_frozen": "tab:cyan", "siglip2_aux": "tab:cyan"}
    styles = {"webdino_frozen": ":o", "webdino_aux": "-o",
              "siglip2_frozen": ":s", "siglip2_aux": "-s"}
    labels = {"webdino_frozen": "Web-DINO frozen",
              "webdino_aux":    "Web-DINO +aux",
              "siglip2_frozen": "SigLIP 2 frozen",
              "siglip2_aux":    "SigLIP 2 +aux"}

    for variant in ["webdino_frozen", "webdino_aux", "siglip2_frozen", "siglip2_aux"]:
        pts = by_variant[variant]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, styles[variant], color=colors[variant],
                linewidth=2, markersize=8, label=labels[variant])

    # CALVIN reference (probe-train ~3200)
    for variant, r in calvin_ref.items():
        ax.scatter([4000], [r], marker="*", s=180,
                     color=colors[variant], edgecolor="black", zorder=5,
                     label=f"{labels[variant]} on CALVIN")

    ax.axhline(vjepa_ref, color="tab:red", linestyle="--", linewidth=1.5,
                  label=f"V-JEPA 2 +aux on LIBERO (R²={vjepa_ref:.2f})")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("Probe-train episodes (log)")
    ax.set_ylabel("Action R² on LIBERO task-OOD test")
    ax.set_title("LIBERO probe budget: image-SSL backbones do not catch up by 3200 episodes")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xticks([400, 800, 1600, 3200])
    ax.set_xticklabels(["400", "800", "1600", "3200"])
    ax.set_ylim(-0.1, 1.0)

    out_png = REPO_ROOT / "runs/full_results/probe_data_efficiency.png"
    out_pdf = REPO_ROOT / "runs/full_results/probe_data_efficiency.pdf"
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    fig.savefig(out_pdf)
    print(f"saved -> {out_png}")
    print(f"saved -> {out_pdf}")


if __name__ == "__main__":
    main()
