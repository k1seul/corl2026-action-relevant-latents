"""Fig. 1 (camera-ready, option A): pixel fidelity vs. action probe R^2 of the same checkpoint.

The submitted Fig. 1 plotted +ID action R^2 under a "frozen" axis label, and four of its
PSNR values (SDXL 30.61, Cosmos 28.94, LAPA 19.84, SigLIP 20.55) are not in any record
(see figures/README.md). This script redraws the same ten checkpoints from records only:

  * y: action probe R^2 of the plotted checkpoint, from data/results.jsonl (Table 1 values);
       every point is a +ID checkpoint except the action-conditioned DIFF, which has no ID loss.
  * x: PSNR on the 200 held-out clips. Encoder-only backbones use the 17M pixel decoder
       trained on the +ID encoder (data/decoder_results.jsonl); pixel-producing world models
       use rollout PSNR (results.jsonl test_cell.psnr).

Options:
  --psnr submitted        plot the PSNR values printed in the submitted figure instead
                          (for side-by-side comparison only; four of them are unverified)
  --with-diff-canonical   also plot the Table 1 DIFF + ID row (the closed-loop target)

Outputs go to figures/out/fig1/: PDF (5.5 in wide, fonts at nominal size), 300-dpi PNG,
the plotted values as CSV, and a provenance JSON with input SHA-256 hashes.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FIG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FIG_ROOT.parent
RESULTS = FIG_ROOT / "data/results.jsonl"
DECODER = FIG_ROOT / "data/decoder_results.jsonl"
SUBMITTED = FIG_ROOT / "data/submitted_fig1_points.csv"
OUT_DIR = FIG_ROOT / "out/fig1"

# (label, group, R^2 variant in results.jsonl, PSNR source, PSNR key, has ID loss)
# PSNR source "decoder": decoder_results.jsonl variant; "rollout": results.jsonl test_cell.psnr.
POINTS = [
    ("V-JEPA 2 ViT-L", "video", "vjepa2_aux_finetune", "decoder", "vjepa2_aux", True),
    ("V-JEPA 2.1 ViT-B", "video", "vjepa2_1_aux_finetune", "decoder", "vjepa2_1_aux", True),
    ("VideoMAE V1 ViT-L", "video", "videomae_aux_finetune", "decoder", "videomae_aux", True),
    ("Web-DINO ViT-L", "image", "webdino_aux_finetune", "decoder", "webdino_aux", True),
    ("SigLIP 2 ViT-L", "image", "siglip2_aux_finetune", "decoder", "siglip2_aux", True),
    ("LAPA-LAQ-OpenX", "other", "lapa_aux_finetune", "decoder", "lapa_aux", True),
    ("SDXL VAE", "pixel", "sdvae_aux_finetune", "decoder", "sd3vae_aux", True),
    ("Cosmos-1 tokenizer", "pixel", "cosmos_tokenizer_aux_finetune", "decoder", "cosmos_aux", True),
    ("DIFF, action-cond. (no ID)", "pixel", "A_diff_v10_action_large_200k", "rollout", None, False),
    ("Dreamer 4 (276M)", "other", "dreamer4_xl_aux_agent", "rollout", None, True),
]
DIFF_CANONICAL = ("DIFF (Table 1)", "pixel", "aux_v6_large_noact_taskheld_lam0p05", "rollout", None, True)

# Row in data/submitted_fig1_points.csv for each label (used by --psnr submitted).
SUBMITTED_NAME = {
    "DIFF, action-cond. (no ID)": "DIFF-FULL (action-cond. 200k)",
    "Dreamer 4 (276M)": "Dreamer 4 276M + ID (agent)",
}

GROUPS = {  # Okabe-Ito hues; shapes repeat the submitted figure's grouping
    "video": ("Video-pretrained SSL", "#0072B2", "o"),
    "image": ("Image-only SSL", "#E69F00", "s"),
    "pixel": ("Pixel reconstruction / diffusion", "#D55E00", "^"),
    "other": ("Other (LAPA, Dreamer 4)", "#6f6e69", "D"),
}
INK, INK_2, MUTED, GRID, REF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# Label offsets in points (dx, dy) and alignment, tuned for the record-PSNR layout.
LABEL_POS = {
    "V-JEPA 2 ViT-L": (6, 3, "left"),
    "V-JEPA 2.1 ViT-B": (-6, 2, "right"),
    "VideoMAE V1 ViT-L": (6, -6, "left"),
    "Web-DINO ViT-L": (6, 4, "left"),
    "SigLIP 2 ViT-L": (6, -7, "left"),
    "LAPA-LAQ-OpenX": (-6, 0, "right"),
    "SDXL VAE": (6, -2, "left"),
    "Cosmos-1 tokenizer": (6, 2, "left"),
    "DIFF, action-cond. (no ID)": (-6, 6, "right"),
    "Dreamer 4 (276M)": (6, 5, "left"),
    "DIFF (Table 1)": (6, -1, "left"),
}

# The submitted PSNRs push SDXL/Cosmos to the right edge; label those on the left instead.
LABEL_POS_SUBMITTED = {
    "SDXL VAE": (-6, -2, "right"),
    "Cosmos-1 tokenizer": (-6, 2, "right"),
    "DIFF, action-cond. (no ID)": (6, 6, "left"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def load() -> tuple[dict, dict, dict]:
    results = {}
    for line in RESULTS.read_text().splitlines():
        r = json.loads(line)
        results[r["variant"]] = r
    decoder = {json.loads(l)["variant"]: json.loads(l) for l in DECODER.read_text().splitlines()}
    submitted = {row["matched_backbone"]: row for row in csv.DictReader(SUBMITTED.open())}
    return results, decoder, submitted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--psnr", choices=["record", "submitted"], default="record")
    ap.add_argument("--with-diff-canonical", action="store_true")
    args = ap.parse_args()

    results, decoder, submitted = load()
    points = POINTS + ([DIFF_CANONICAL] if args.with_diff_canonical else [])

    rows = []
    for label, group, r2_var, src, dec_var, has_id in points:
        r2 = results[r2_var]["test_cell"]["action_probe_r2"]
        if src == "decoder":
            psnr_record = decoder[dec_var]["psnr_test_mean"]
        else:
            psnr_record = results[r2_var]["test_cell"]["psnr"]
        psnr = psnr_record
        if args.psnr == "submitted":
            sub = submitted.get(SUBMITTED_NAME.get(label, label))
            if sub is None:
                continue  # not in the submitted figure (e.g. the canonical DIFF row)
            psnr = float(sub["plotted_psnr_db"])
        rows.append(dict(label=label, group=group, r2=r2, psnr=psnr, psnr_record=psnr_record,
                         psnr_source=src, has_id=has_id, r2_variant=r2_var, decoder_variant=dec_var))

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.edgecolor": REF, "axes.linewidth": 0.6,
                         "xtick.color": INK_2, "ytick.color": INK_2})
    fig, ax = plt.subplots(figsize=(5.5, 2.45))
    ax.axhline(0, color=REF, lw=0.6, zorder=1)
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for p in rows:
        _, color, marker = GROUPS[p["group"]]
        rollout = p["psnr_source"] == "rollout"
        ax.scatter(p["psnr"], p["r2"], s=34, marker=marker, zorder=4,
                   facecolor="white" if rollout else color, edgecolor=color,
                   linewidth=1.3 if rollout else 0.6)
        pos = {**LABEL_POS, **(LABEL_POS_SUBMITTED if args.psnr == "submitted" else {})}
        dx, dy, ha = pos.get(p["label"], (6, 3, "left"))
        ax.annotate(p["label"], (p["psnr"], p["r2"]), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=5.6, color=INK_2, zorder=5)

    xs = [p["psnr"] for p in rows]
    ax.set_xlim(min(xs) - 1.2, max(xs) + 1.6)
    ax.set_ylim(-0.6, 1.0)
    ax.tick_params(labelsize=6.3, length=2.5, width=0.6)
    ax.set_xlabel("Pixel fidelity: PSNR on 200 held-out clips (dB)", fontsize=6.8, color=INK)
    ax.set_ylabel("Action probe R²\n(plotted checkpoint)", fontsize=6.8, color=INK)

    handles = [Line2D([], [], ls="", marker=m, markersize=5, markerfacecolor=c, markeredgecolor=c,
                      label=name) for name, c, m in GROUPS.values()]
    handles += [Line2D([], [], ls="", marker="o", markersize=5, markerfacecolor=MUTED,
                       markeredgecolor=MUTED, label="PSNR: 17M decoder on +ID encoder"),
                Line2D([], [], ls="", marker="o", markersize=5, markerfacecolor="white",
                       markeredgecolor=MUTED, markeredgewidth=1.3, label="PSNR: world-model rollout")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=5.8,
               handletextpad=0.3, columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.115, right=0.98, top=0.97, bottom=0.33)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / ("fig1_pixel_vs_action" + ("" if args.psnr == "record" else "_submittedpsnr")
                      + ("_withdiff" if args.with_diff_canonical else ""))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)

    with stem.with_suffix(".csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "group", "action_r2", "psnr_plotted_db", "psnr_record_db", "psnr_source",
                    "has_id_loss", "r2_variant", "decoder_variant"])
        for p in rows:
            w.writerow([p["label"], p["group"], f"{p['r2']:.6f}", f"{p['psnr']:.4f}",
                        f"{p['psnr_record']:.4f}", p["psnr_source"], p["has_id"], p["r2_variant"],
                        p["decoder_variant"] or ""])
    prov = {
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "inputs": {str(p.relative_to(REPO_ROOT)): _sha256(p) for p in (RESULTS, DECODER, SUBMITTED)},
        "args": vars(args),
    }
    stem.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False))
    print(f"saved -> {stem}.{{pdf,png,csv,provenance.json}}")
    for p in rows:
        print(f"  {p['label']:28s} R2 {p['r2']:+.4f}  PSNR {p['psnr']:6.2f} ({p['psnr_source']})")


if __name__ == "__main__":
    main()
