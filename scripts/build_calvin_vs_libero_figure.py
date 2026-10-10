"""Compare LIBERO vs CALVIN action-probe R² for the 8 headline backbones."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_r2(path: Path) -> dict:
    r2 = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if "_pert_" in r["variant"]:
            continue
        tc = r.get("test_cell", {})
        if tc.get("action_probe_r2") is not None:
            r2[r["variant"]] = tc["action_probe_r2"]
    return r2


def main() -> None:
    lib = _load_r2(REPO_ROOT / "runs/full_results/results.jsonl")
    cal = _load_r2(REPO_ROOT / "runs/full_results_calvin/results.jsonl")

    rows = [
        ("V-JEPA 2 ViT-L",     "vjepa2_aux_finetune",          "vjepa2_aux_calvin",          "tab:red"),
        ("V-JEPA 2.1 ViT-B",   "vjepa2_1_aux_finetune",        "vjepa2_1_aux_calvin",        "tab:red"),
        ("VideoMAE ViT-L",     "videomae_aux_finetune",        "videomae_aux_calvin",        "tab:purple"),
        ("Web-DINO ViT-L",     "webdino_aux_finetune",         "webdino_aux_calvin",         "tab:blue"),
        ("SigLIP 2 ViT-L",     "siglip2_aux_finetune",         "siglip2_aux_calvin",         "tab:cyan"),
        ("LAPA",               "lapa_aux_finetune",            "lapa_aux_calvin",            "tab:olive"),
        ("SDXL VAE",           "sdvae_aux_finetune",           "sdvae_aux_calvin",           "tab:green"),
        ("Cosmos-1 tok",       "cosmos_tokenizer_aux_finetune","cosmos_tokenizer_aux_calvin", "tab:brown"),
    ]

    labels = [r[0] for r in rows]
    libv  = [lib.get(r[1], 0.0) for r in rows]
    calv  = [cal.get(r[2], 0.0) for r in rows]
    colors = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(rows))
    w = 0.36
    b1 = ax.bar(x - w/2, libv, w, label="LIBERO +aux R²", color="#777", edgecolor="black")
    b2 = ax.bar(x + w/2, calv, w, label="CALVIN +aux R²", color=colors, edgecolor="black")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Test action R² on held-out tasks")
    ax.set_title("LIBERO vs CALVIN: image-SSL backbones close the gap on CALVIN")
    ax.set_ylim(-0.5, 1.0)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # Annotate the V-JEPA gap on LIBERO vs CALVIN
    for i, (label, l, c) in enumerate(zip(labels, libv, calv)):
        ax.text(i + w/2, c + 0.02, f"{c:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i - w/2, l + 0.02 if l >= 0 else l - 0.07, f"{l:.2f}",
                 ha="center", va="bottom" if l >= 0 else "top", fontsize=8)

    out_png = REPO_ROOT / "runs/full_results_calvin/libero_vs_calvin_aux_r2.png"
    out_pdf = REPO_ROOT / "runs/full_results_calvin/libero_vs_calvin_aux_r2.pdf"
    out_jsonl = REPO_ROOT / "runs/full_results_calvin/libero_vs_calvin_summary.jsonl"
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    fig.savefig(out_pdf)
    with out_jsonl.open("w") as f:
        for label, l, c in zip(labels, libv, calv):
            f.write(json.dumps({"backbone": label, "libero_aux_r2": l, "calvin_aux_r2": c,
                                  "delta": c - l}) + "\n")
    print(f"saved -> {out_png}")
    print(f"saved -> {out_jsonl}")


if __name__ == "__main__":
    main()
