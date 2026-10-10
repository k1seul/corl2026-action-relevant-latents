"""R² vs aux-lambda curve for Web-DINO + SigLIP 2 sweep."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent


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
    r2 = load_r2()
    lambdas = [0.01, 0.025, 0.1, 0.3, 1.0]
    web = [r2[f"webdino_aux_lam{tag}"] for tag in ("0p01", "0p025", "0p1", "0p3")] + [r2["webdino_aux_finetune"]]
    sig = [r2[f"siglip2_aux_lam{tag}"] for tag in ("0p01", "0p025", "0p1", "0p3")] + [r2["siglip2_aux_finetune"]]
    vjepa = r2["vjepa2_aux_finetune"]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(lambdas, web, "-o", color="tab:blue", label="Web-DINO ViT-L +aux", linewidth=2)
    ax.plot(lambdas, sig, "-s", color="tab:cyan", label="SigLIP 2 ViT-L +aux", linewidth=2)
    ax.axhline(vjepa, color="tab:red", linestyle="--", linewidth=1.5,
                  label=f"V-JEPA 2 +aux (λ=1.0) — R²={vjepa:.2f}")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("aux-lambda  (inverse-dynamics loss weight)")
    ax.set_ylabel("Action probe R² on held-out LIBERO tasks")
    ax.set_title("Web-DINO & SigLIP 2: λ sweep cannot close gap to V-JEPA")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(loc="center right", fontsize=9)
    ax.set_ylim(-0.05, 1.0)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([f"{x}" for x in lambdas])

    out_png = REPO_ROOT / "runs/full_results/lambda_curve_image_ssl.png"
    out_pdf = REPO_ROOT / "runs/full_results/lambda_curve_image_ssl.pdf"
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    fig.savefig(out_pdf)
    print(f"saved -> {out_png}")
    print(f"saved -> {out_pdf}")

    # raw data dump
    out_jsonl = REPO_ROOT / "runs/full_results/lambda_curve.jsonl"
    with out_jsonl.open("w") as f:
        for lam, w, s in zip(lambdas, web, sig):
            f.write(json.dumps({"lambda": lam, "webdino_r2": w, "siglip2_r2": s}) + "\n")
    print(f"saved -> {out_jsonl}")


if __name__ == "__main__":
    main()
