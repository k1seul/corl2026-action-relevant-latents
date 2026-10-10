"""LIBERO + CALVIN + MetaWorld three-benchmark +aux R^2 comparison."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path) -> dict:
    r2 = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if "_pert_" in r["variant"]:
            continue
        tc = r.get("test_cell", {})
        if tc.get("action_probe_r2") is not None:
            r2[r["variant"]] = tc["action_probe_r2"]
    return r2


def main():
    lib = _load(REPO_ROOT / "runs/full_results/results.jsonl")
    cal = _load(REPO_ROOT / "runs/full_results_calvin/results.jsonl")
    mw  = _load(REPO_ROOT / "runs/full_results_metaworld/results.jsonl")

    rows = [
        ("V-JEPA 2 ViT-L",   "vjepa2_aux_finetune",         "vjepa2_aux_calvin",         "vjepa2_aux_metaworld",         "tab:red"),
        ("V-JEPA 2.1 ViT-B", "vjepa2_1_aux_finetune",       "vjepa2_1_aux_calvin",       "vjepa2_1_aux_metaworld",       "tab:red"),
        ("VideoMAE ViT-L",   "videomae_aux_finetune",       "videomae_aux_calvin",       "videomae_aux_metaworld",       "tab:purple"),
        ("Web-DINO ViT-L",   "webdino_aux_finetune",        "webdino_aux_calvin",        "webdino_aux_metaworld",        "tab:blue"),
        ("SigLIP 2 ViT-L",   "siglip2_aux_finetune",        "siglip2_aux_calvin",        "siglip2_aux_metaworld",        "tab:cyan"),
        ("LAPA",             "lapa_aux_finetune",           "lapa_aux_calvin",           "lapa_aux_metaworld",           "tab:olive"),
        ("SDXL VAE",         "sdvae_aux_finetune",          "sdvae_aux_calvin",          "sdvae_aux_metaworld",          "tab:green"),
        ("Cosmos-1 tok",     "cosmos_tokenizer_aux_finetune","cosmos_tokenizer_aux_calvin","cosmos_tokenizer_aux_metaworld","tab:brown"),
    ]
    labels = [r[0] for r in rows]
    libv = [lib.get(r[1], 0.0) for r in rows]
    calv = [cal.get(r[2], 0.0) for r in rows]
    mwv  = [mw.get(r[3], 0.0) for r in rows]

    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    x = np.arange(len(rows))
    w = 0.26
    ax.bar(x - w, libv, w, label="LIBERO", color="#444", edgecolor="black")
    ax.bar(x,     calv, w, label="CALVIN", color="#bfa040", edgecolor="black")
    ax.bar(x + w, mwv,  w, label="MetaWorld", color="#3a7", edgecolor="black")

    for xi, label in zip(x, labels):
        for j, vec in enumerate((libv, calv, mwv)):
            xoff = (-w, 0, w)[j]
            v = vec[xi]
            yoff = 0.015 if v >= 0 else -0.06
            va = "bottom" if v >= 0 else "top"
            ax.text(xi + xoff, v + yoff, f"{v:.2f}", ha="center", va=va, fontsize=7.5)

    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("+aux action probe R² on task-OOD test split")
    ax.set_title("Three-benchmark cross-architecture comparison:\nV-JEPA dominates LIBERO + MetaWorld, but image-SSL catches up on CALVIN")
    ax.set_ylim(-0.65, 1.0)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")

    out_png = REPO_ROOT / "runs/full_results/three_benchmark_aux_r2.png"
    out_pdf = REPO_ROOT / "runs/full_results/three_benchmark_aux_r2.pdf"
    out_jsonl = REPO_ROOT / "runs/full_results/three_benchmark_summary.jsonl"
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    fig.savefig(out_pdf)
    with out_jsonl.open("w") as f:
        for label, l, c, m in zip(labels, libv, calv, mwv):
            f.write(json.dumps({"backbone": label, "libero": l, "calvin": c, "metaworld": m}) + "\n")
    print(f"saved -> {out_png}")
    print(f"saved -> {out_jsonl}")


if __name__ == "__main__":
    main()
