"""V-JEPA 2 ViT-L per-layer R² profile."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path):
    layers, r2 = [], []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        layers.append(r["layer"]); r2.append(r["r2_test_mean"])
    return layers, r2


def main():
    p_aux = REPO_ROOT / "runs/full_results/vjepa_per_layer.jsonl"
    layers_aux, r2_aux = _load(p_aux)
    p_frozen = REPO_ROOT / "runs/full_results/vjepa_per_layer_frozen.jsonl"
    has_frozen = p_frozen.exists()

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(layers_aux, r2_aux, "-o", color="tab:red", linewidth=2, markersize=7,
              label="V-JEPA 2 ViT-L +aux ckpt")
    if has_frozen:
        l, r = _load(p_frozen)
        ax.plot(l, r, "-s", color="tab:blue", linewidth=2, markersize=7,
                  label="V-JEPA 2 ViT-L frozen (no aux finetune)")

    peak = int(np.argmax(r2_aux))
    ax.scatter([peak], [r2_aux[peak]], marker="*", s=300, color="gold",
                  edgecolor="black", zorder=5, label=f"peak @ layer {peak}: R²={r2_aux[peak]:.3f}")
    ax.scatter([23], [r2_aux[23]], marker="X", s=150, color="tab:red",
                  edgecolor="black", zorder=5,
                  label=f"final layer 23: R²={r2_aux[23]:.3f}")

    ax.set_xlabel("V-JEPA 2 ViT-L transformer block index")
    ax.set_ylabel("Probe R² on LIBERO task-OOD (3 seeds mean)")
    ax.set_title("Per-layer action probe: features at the trunk's final layer\nare less action-relevant than mid-layer features")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xticks(range(0, 24, 2))
    ax.set_ylim(0, 0.85)

    out_png = REPO_ROOT / "runs/full_results/vjepa_per_layer.png"
    out_pdf = REPO_ROOT / "runs/full_results/vjepa_per_layer.pdf"
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    fig.savefig(out_pdf)
    print(f"saved -> {out_png}")


if __name__ == "__main__":
    main()
