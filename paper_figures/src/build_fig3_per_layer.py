"""Camera-ready Fig. 3 and Appendix Table 9: V-JEPA 2 ViT-L per-layer action R².

Reads ``runs/full_results/vjepa_per_layer_frozen.jsonl`` and
``runs/full_results/vjepa_per_layer.jsonl`` (24 blocks each, 3 probe seeds) and
plots every block with a ±1 SD band over probe seeds. The submitted Fig. 3 and
Table 9 showed only blocks 7, 14, 18-23 and labelled block 14 as the frozen
peak, although block 18 was higher in the same table and block 15 is the
recorded maximum.

The frozen plateau is computed, not hand-picked: the contiguous run of blocks
around the frozen argmax whose mean lies within one probe-seed SD (of the
argmax block) of the maximum.

Caption facts that follow from ``scripts/probe_vjepa_per_layer.py``:
  * features are block outputs captured by forward hooks, i.e. before the
    encoder's final LayerNorm, so block 23 is not the main-text extractor feature;
  * the +ID checkpoint is ``runs/comparison/vjepa2_aux_finetune/ckpt_last.pt``
    (30k steps, per runs/full_results/manifest.yaml);
  * probe: 400 train / 200 test-OOD episodes, probe seeds 11, 12, 13.

Outputs go to ``runs/camera_ready/fig3/``.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# figures/ layout (adapted from the 2026-10-06 handoff; only these paths changed)
FIG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FIG_ROOT.parent
SRC_FROZEN = FIG_ROOT / "data/vjepa_per_layer_frozen.jsonl"
SRC_ID = FIG_ROOT / "data/vjepa_per_layer.jsonl"
OUT_DIR = FIG_ROOT / "out/fig3"

C_ID = "#2a78d6"
C_FROZEN = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
PLATEAU = "#f0efec"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def load(path: Path) -> dict[str, np.ndarray]:
    rows = sorted((json.loads(l) for l in path.read_text().splitlines()), key=lambda r: r["layer"])
    layers = np.array([r["layer"] for r in rows])
    if not np.array_equal(layers, np.arange(len(layers))):
        raise ValueError(f"{path}: expected contiguous layers 0..N-1, got {layers.tolist()}")
    seeds = np.array([r["r2_seeds"] for r in rows], dtype=float)
    mean = np.array([r["r2_test_mean"] for r in rows], dtype=float)
    if not np.allclose(mean, seeds.mean(axis=1), atol=1e-9):
        raise ValueError(f"{path}: r2_test_mean disagrees with the mean of r2_seeds")
    return {"layer": layers, "mean": mean, "sd": seeds.std(axis=1, ddof=1), "seeds": seeds}


def plateau(mean: np.ndarray, sd: np.ndarray) -> tuple[int, int, int]:
    k = int(np.argmax(mean))
    floor = mean[k] - sd[k]
    lo = hi = k
    while lo - 1 >= 0 and mean[lo - 1] >= floor:
        lo -= 1
    while hi + 1 < len(mean) and mean[hi + 1] >= floor:
        hi += 1
    return k, lo, hi


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.unicode_minus": True})
    fz, ft = load(SRC_FROZEN), load(SRC_ID)
    if len(fz["layer"]) != len(ft["layer"]):
        raise ValueError("frozen and +ID files cover different numbers of blocks")
    x = fz["layer"]
    k_max, p_lo, p_hi = plateau(fz["mean"], fz["sd"])
    k_min_post = p_hi + 1 + int(np.argmin(fz["mean"][p_hi + 1:]))
    k_id_max = int(np.argmax(ft["mean"]))
    k_last = int(x[-1])

    fig, ax = plt.subplots(figsize=(5.5, 2.45))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.86, bottom=0.2)

    ax.axvspan(p_lo - 0.45, p_hi + 0.45, color=PLATEAU, zorder=0, linewidth=0)
    for d, color, frozen in ((fz, C_FROZEN, True), (ft, C_ID, False)):
        ax.fill_between(x, d["mean"] - d["sd"], d["mean"] + d["sd"], color=color, alpha=0.12,
                        linewidth=0, zorder=1)
        if frozen:
            ax.plot(x, d["mean"], color=color, linewidth=1.4, linestyle=(0, (3.2, 1.6)), zorder=3)
            ax.plot(x, d["mean"], "o", markersize=3.6, markerfacecolor="white",
                    markeredgecolor=color, markeredgewidth=1.0, zorder=4)
        else:
            ax.plot(x, d["mean"], color=color, linewidth=1.6, zorder=3)
            ax.plot(x, d["mean"], "o", markersize=3.8, markerfacecolor=color,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=5)

    ann = dict(fontsize=6.3, color=INK_2, zorder=6)
    lead = dict(arrowstyle="-", color=MUTED, linewidth=0.6)
    # Plateau label sits low inside the shaded span, clear of both curves.
    ax.text((p_lo + p_hi) / 2, 0.19,
            f"frozen plateau\nblocks {p_lo}–{p_hi}\nmax {fz['mean'][k_max]:.3f} (block {k_max})",
            ha="center", va="center", linespacing=1.25, **ann)
    ax.annotate(f"{fz['mean'][k_min_post]:.3f} (block {k_min_post})",
                xy=(k_min_post, fz["mean"][k_min_post]), xytext=(k_min_post - 0.6, 0.25),
                ha="center", va="center", arrowprops=lead, **ann)
    ax.annotate(f"{ft['mean'][k_id_max]:.3f} (block {k_id_max})",
                xy=(k_id_max, ft["mean"][k_id_max]), xytext=(k_id_max - 0.4, 0.81),
                ha="center", va="center", arrowprops=lead, **ann)
    ax.annotate(f"{ft['mean'][k_last]:.3f} (block {k_last})",
                xy=(k_last, ft["mean"][k_last]), xytext=(k_last + 0.45, 0.60),
                ha="right", va="top", arrowprops=lead, **ann)

    ax.set_xlim(-0.6, k_last + 0.6)
    ax.set_ylim(0.0, 0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) if (i % 2 == 0 or i == k_last) else "" for i in x], fontsize=6.3,
                       color=INK_2)
    ax.set_yticks(np.arange(0.0, 0.81, 0.2))
    ax.tick_params(axis="y", labelsize=6.3, colors=INK_2, length=0)
    ax.tick_params(axis="x", length=2, color=AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_xlabel("V-JEPA 2 ViT-L block index (0 = after patch embedding)", fontsize=6.8, color=INK)
    ax.set_ylabel("Action probe R²\n(4-task LIBERO subset)", fontsize=6.8, color=INK)

    handles = [
        Line2D([0], [0], color=C_FROZEN, linewidth=1.4, linestyle=(0, (3.2, 1.6)), marker="o",
               markersize=3.6, markerfacecolor="white", markeredgecolor=C_FROZEN, label="Frozen"),
        Line2D([0], [0], color=C_ID, linewidth=1.6, marker="o", markersize=3.8,
               markerfacecolor=C_ID, markeredgecolor="white", label="+ID fine-tune (30k steps)"),
        Patch(facecolor=MUTED, alpha=0.25, linewidth=0, label="±1 SD, 3 probe seeds"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=6.2,
               bbox_to_anchor=(0.54, 0.995), handlelength=2.0, columnspacing=1.6)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "fig3_vjepa_per_layer"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)

    with stem.with_suffix(".csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block", "frozen_mean", "frozen_sd", "frozen_seeds", "id_mean", "id_sd", "id_seeds",
                    "delta", "frozen_plateau"])
        for i in x:
            w.writerow([int(i), f"{fz['mean'][i]:.6f}", f"{fz['sd'][i]:.6f}",
                        "/".join(f"{s:.6f}" for s in fz["seeds"][i]),
                        f"{ft['mean'][i]:.6f}", f"{ft['sd'][i]:.6f}",
                        "/".join(f"{s:.6f}" for s in ft["seeds"][i]),
                        f"{ft['mean'][i] - fz['mean'][i]:.6f}", int(p_lo <= i <= p_hi)])

    # Appendix Table 9, all 24 blocks; numbers formatted from the same arrays.
    def cell(v: float, bold: bool) -> str:
        s = f"{v:.3f}".replace("-", "$-$")
        return f"\\textbf{{{s}}}" if bold else s

    lines = [
        "% Auto-generated by scripts/camera_ready/build_fig3_per_layer.py -- do not edit by hand.",
        "\\begin{tabular}{rccr}",
        "\\toprule",
        "Block & Frozen $R^2$ & + ID $R^2$ & $\\Delta$ \\\\",
        "\\midrule",
    ]
    for i in x:
        mark = "$^\\dagger$" if p_lo <= i <= p_hi else ""
        delta = ft["mean"][i] - fz["mean"][i]
        lines.append(f"{int(i)}{mark} & {cell(fz['mean'][i], i == k_max)} & "
                     f"{cell(ft['mean'][i], i == k_id_max)} & ${delta:+.3f}$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    stem.with_name("table9_per_layer.tex").write_text("\n".join(lines) + "\n")

    prov = {
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "inputs": {str(p.relative_to(REPO_ROOT)): _sha256(p) for p in (SRC_FROZEN, SRC_ID)},
        "derived": {
            "frozen_argmax_block": k_max, "frozen_max": float(fz["mean"][k_max]),
            "frozen_plateau_blocks": [p_lo, p_hi],
            "frozen_post_plateau_min_block": k_min_post,
            "frozen_post_plateau_min": float(fz["mean"][k_min_post]),
            "id_argmax_block": k_id_max, "id_max": float(ft["mean"][k_id_max]),
            "id_final_block": k_last, "id_final": float(ft["mean"][k_last]),
        },
    }
    stem.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False))
    print(f"saved -> {stem}.{{pdf,png,csv,provenance.json}} and table9_per_layer.tex")
    print(json.dumps(prov["derived"], indent=2))


if __name__ == "__main__":
    main()
