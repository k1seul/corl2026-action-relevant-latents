"""Build (Test PSNR, Action R^2) Pareto figure for the 8 encoder-only +aux backbones."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_psnr() -> dict:
    psnr = {}
    for line in (REPO_ROOT / "runs/full_results/decoder_results.jsonl").read_text().splitlines():
        r = json.loads(line)
        psnr[r["variant"]] = r["psnr_test_mean"]
    return psnr


def load_r2() -> dict:
    r2 = {}
    for line in (REPO_ROOT / "runs/full_results/results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "_pert_" in r["variant"]:
            continue
        tc = r.get("test_cell", {})
        if tc.get("action_probe_r2") is not None:
            r2[r["variant"]] = tc["action_probe_r2"]
    return r2


def main() -> None:
    psnr = load_psnr()
    r2 = load_r2()
    table = [
        ("V-JEPA 2 ViT-L +aux",   "vjepa2_aux",     "vjepa2_aux_finetune",         "tab:red"),
        ("V-JEPA 2.1 ViT-B +aux", "vjepa2_1_aux",   "vjepa2_1_aux_finetune",       "tab:red"),
        ("VideoMAE ViT-L +aux",   "videomae_aux",   "videomae_aux_finetune",       "tab:purple"),
        ("Web-DINO ViT-L +aux",   "webdino_aux",    "webdino_aux_finetune",        "tab:blue"),
        ("SigLIP 2 ViT-L +aux",   "siglip2_aux",    "siglip2_aux_finetune",        "tab:cyan"),
        ("LAPA +aux",             "lapa_aux",       "lapa_aux_finetune",           "tab:olive"),
        ("SDXL VAE +aux",         "sd3vae_aux",     "sdvae_aux_finetune",          "tab:green"),
        ("Cosmos-1 tok +aux",     "cosmos_aux",     "cosmos_tokenizer_aux_finetune","tab:brown"),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    for label, dvar, pvar, color in table:
        x = psnr[dvar]
        y = r2[pvar]
        ax.scatter(x, y, s=140, color=color, edgecolor="black", zorder=3)
        offset = (6, 5)
        if "VideoMAE" in label: offset = (-5, -16)
        if "V-JEPA 2.1" in label: offset = (-90, -10)
        if "SigLIP" in label: offset = (-90, -5)
        if "Web-DINO" in label: offset = (6, -15)
        if "Cosmos" in label: offset = (-95, -3)
        if "SDXL" in label: offset = (-50, 8)
        ax.annotate(label, (x, y), textcoords="offset points",
                       xytext=offset, fontsize=9)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Test PSNR (dB)  —  pixel recon fidelity on held-out tasks")
    ax.set_ylabel("Action probe R²  —  dynamics in latent (held-out tasks)")
    ax.set_title("Pixel fidelity vs action signal: LIBERO task-OOD")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(-0.6, 1.0)
    ax.set_xlim(16.5, 25.0)
    handles = [
        mpatches.Patch(color="tab:red", label="Video-JEPA family"),
        mpatches.Patch(color="tab:purple", label="VideoMAE (video MAE)"),
        mpatches.Patch(color="tab:blue", label="Web-DINO (image SSL)"),
        mpatches.Patch(color="tab:cyan", label="SigLIP 2 (vision-language)"),
        mpatches.Patch(color="tab:olive", label="LAPA (latent action)"),
        mpatches.Patch(color="tab:green", label="SDXL VAE (recon)"),
        mpatches.Patch(color="tab:brown", label="Cosmos-1 (recon)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)

    out_png = REPO_ROOT / "runs/full_results/pareto_psnr_vs_r2.png"
    out_pdf = REPO_ROOT / "runs/full_results/pareto_psnr_vs_r2.pdf"
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    fig.savefig(out_pdf)
    print(f"saved -> {out_png}")
    print(f"saved -> {out_pdf}")

    out_jsonl = REPO_ROOT / "runs/full_results/pareto_summary.jsonl"
    with out_jsonl.open("w") as f:
        for label, dvar, pvar, _ in table:
            f.write(json.dumps({"label": label, "psnr_test": psnr[dvar],
                                  "action_r2_test": r2[pvar]}) + "\n")
    print(f"saved -> {out_jsonl}")


if __name__ == "__main__":
    main()
