"""Camera-ready Fig. 2: per-dimension action R² by backbone (frozen vs +ID).

Every plotted number is read from ``runs/full_results/results.jsonl``:

* radar vertices = per-group means of ``test_cell.per_dim_r2``
  (Trans = mean(dx, dy, dz), Rot = mean(drx, dry, drz), Grip = gripper);
* panel labels   = ``test_cell.action_probe_r2`` (mean of the 7 per-dim R²),
  i.e. the same aggregate as Table 1, with Δ computed before rounding.

The submitted Fig. 2 labelled each panel with the mean of the three rounded
vertices, which is why its numbers disagreed with Table 1 (e.g. V-JEPA 2
frozen 0.46 vs 0.40). Outputs go to ``runs/camera_ready/fig2/``.

Usage:
    python scripts/camera_ready/build_fig2_per_dim_radar.py [--diff-variant l2|l1]
"""
from __future__ import annotations

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
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# figures/ layout (adapted from the 2026-10-06 handoff; only these paths changed)
FIG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FIG_ROOT.parent
RESULTS = FIG_ROOT / "data/results.jsonl"
OUT_DIR = FIG_ROOT / "out/fig2"

DIMS = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]

# (panel title, Table 1 family, frozen variant, +ID variant) — names only, no values.
PANELS = [
    ("V-JEPA 2 ViT-L", "Video + JEPA pred.", "vjepa2_vit_l_frozen", "vjepa2_aux_finetune"),
    ("V-JEPA 2.1 ViT-B", "Video + JEPA pred.", "vjepa2_1_vit_b_frozen", "vjepa2_1_aux_finetune"),
    ("VideoMAE V1 ViT-L", "Video + Pixel MAE", "videomae_frozen", "videomae_aux_finetune"),
    ("Web-DINO ViT-L", "Image SSL", "webdino_frozen", "webdino_aux_finetune"),
    ("SigLIP 2 ViT-L", "Image SSL", "siglip2_frozen", "siglip2_aux_finetune"),
    ("LAPA-LAQ-OpenX", "Latent action quant.", "lapa_frozen", "lapa_aux_finetune"),
    ("DIFF", "Pixel diffusion", "v6_large_noact_taskheld", None),  # +ID set by --diff-variant
    ("SDXL VAE", "Pixel reconstruction", "sdvae_frozen", "sdvae_aux_finetune"),
    ("Cosmos-1 tokenizer", "Pixel reconstruction", "cosmos_tokenizer_frozen",
     "cosmos_tokenizer_aux_finetune"),
    ("Dreamer 4", "Shortcut-forcing dyn.", "dreamer4_dynamics_agent", "dreamer4_dynamics_aux_agent"),
]
DIFF_ID_VARIANT = {
    "l2": "aux_v6_large_noact_taskheld_lam0p05",  # canonical L2, λ=0.05 — the Table 1 row
    "l1": "aux_abl_A4_invdyn_l1",                 # L1 variant used by the submitted Fig. 2
}

# Per-family colours (Okabe-Ito, colour-blind safe), shared with Fig. 1 where families overlap:
# video = blues, image SSL = orange, pixel reconstruction = vermillion, Dreamer 4 = grey.
# Frozen vs +ID is encoded by line style (dashed/hollow vs solid/filled) in the family colour.
FAMILY_COLOR = {
    "Video + JEPA pred.": "#0072B2",
    "Video + Pixel MAE": "#56B4E9",
    "Image SSL": "#E69F00",
    "Latent action quant.": "#009E73",
    "Pixel diffusion": "#CC79A7",
    "Pixel reconstruction": "#D55E00",
    "Shortcut-forcing dyn.": "#6f6e69",
}
C_LEGEND = "#52514e"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
REF = "#c3c2b7"

R_MIN, R_MAX = -1.2, 1.0
RINGS = [-1.0, -0.5, 0.0, 0.5, 1.0]
RING_LABELS = {0.0: "0", 1.0: "1"}  # the −1 ring sits at the centre, where a label would cover data
ANGLES = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])  # Trans (top), Rot (lower-left), Grip (lower-right)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _fmt(x: float, signed: bool = False) -> str:
    if round(x, 2) == 0:
        return "0.00"
    s = f"{x:+.2f}" if signed else f"{x:.2f}"
    return s.replace("-", "−")


def load_rows() -> dict[str, dict]:
    rows = {}
    for line in RESULTS.read_text().splitlines():
        r = json.loads(line)
        if "_pert_" in r["variant"]:
            continue
        rows[r["variant"]] = r
    return rows


def cell_values(row: dict) -> dict:
    tc = row["test_cell"]
    pd_ = tc["per_dim_r2"]
    if isinstance(pd_, list):
        pd_ = dict(zip(DIMS, pd_))
    per_dim = {d: float(pd_[d]) for d in DIMS}
    return {
        "agg": float(tc["action_probe_r2"]),
        "trans": float(np.mean([per_dim[d] for d in ("dx", "dy", "dz")])),
        "rot": float(np.mean([per_dim[d] for d in ("drx", "dry", "drz")])),
        "grip": per_dim["gripper"],
        "per_dim": per_dim,
    }


def draw_panel(ax, frozen: dict, plus_id: dict, color: str) -> None:
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(1)
    ax.set_ylim(R_MIN, R_MAX)
    ax.spines["polar"].set_visible(False)  # the r = 1 ring is the outer boundary
    ax.set_thetagrids(np.degrees(ANGLES), labels=["Trans.", "Rot.", "Grip."],
                      fontsize=5.6, color=INK_2)
    ax.tick_params(axis="x", pad=-2.5)
    # Ring labels sit in the empty upper-right sector (between Trans. and Grip.).
    ax.set_rgrids(RINGS, labels=[RING_LABELS.get(v, "") for v in RINGS],
                  angle=300, fontsize=4.6, color=MUTED)
    for t in ax.yaxis.get_ticklabels():
        t.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.3, alpha=0.85))
    ax.grid(color=GRID, linewidth=0.5, linestyle="-")

    closed = np.append(ANGLES, ANGLES[0])
    ax.plot(closed, [0.0] * 4, color=REF, linewidth=0.7, zorder=1)  # R² = 0 reference

    def series(vals: dict, color: str, frozen_style: bool) -> None:
        r = [vals["trans"], vals["rot"], vals["grip"]]
        rc = r + [r[0]]
        if frozen_style:
            ax.plot(closed, rc, color=color, linewidth=1.1, linestyle=(0, (3, 1.6)), zorder=3)
            ax.plot(ANGLES, r, "o", markersize=3.6, markerfacecolor="white",
                    markeredgecolor=color, markeredgewidth=1.0, zorder=4)
        else:
            ax.fill(closed, rc, color=color, alpha=0.22, linewidth=0, zorder=2)
            ax.plot(closed, rc, color=color, linewidth=1.4, zorder=3)
            ax.plot(ANGLES, r, "o", markersize=3.8, markerfacecolor=color,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=5)

    series(frozen, color, frozen_style=True)
    series(plus_id, color, frozen_style=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff-variant", choices=sorted(DIFF_ID_VARIANT), default="l2",
                    help="DIFF +ID row: l2 = canonical Table 1 row (default), l1 = submitted-Fig.2 choice")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
    })
    rows = load_rows()
    panels = []
    for title, family, v_frozen, v_id in PANELS:
        if v_id is None:
            v_id = DIFF_ID_VARIANT[args.diff_variant]
        for v in (v_frozen, v_id):
            if v not in rows:
                raise KeyError(f"variant {v!r} missing from {RESULTS}")
        panels.append((title, family, v_frozen, v_id,
                       cell_values(rows[v_frozen]), cell_values(rows[v_id])))

    # 5.5 in = CoRL text width, so fonts print at their nominal size. Layout in inches.
    fig_w, fig_h = 5.5, 3.6
    fig = plt.figure(figsize=(fig_w, fig_h))
    n_cols, col_w, diam = 5, fig_w / 5, 0.80
    row_tops = [fig_h - 0.06, fig_h - 0.06 - 1.52]

    def fx(x_in: float) -> float:
        return x_in / fig_w

    def fy(y_in: float) -> float:
        return y_in / fig_h

    for k, (title, family, v_frozen, v_id, frz, pid) in enumerate(panels):
        row, col = divmod(k, n_cols)
        cx_in = (col + 0.5) * col_w
        top = row_tops[row]
        ax_top = top - 0.40
        ax = fig.add_axes([fx(cx_in - diam / 2), fy(ax_top - diam), fx(diam), fy(diam)],
                          projection="polar")
        color = FAMILY_COLOR[family]
        draw_panel(ax, frz, pid, color)
        fig.text(fx(cx_in), fy(top), title, ha="center", va="top", fontsize=6.8,
                 color=INK, fontweight="semibold")
        fig.text(fx(cx_in), fy(top - 0.135), family, ha="center", va="top", fontsize=5.6,
                 color=color, fontweight="semibold")
        delta = pid["agg"] - frz["agg"]
        fig.text(fx(cx_in), fy(ax_top - diam - 0.08),
                 f"{_fmt(frz['agg'])} → {_fmt(pid['agg'])} (Δ {_fmt(delta, signed=True)})",
                 ha="center", va="top", fontsize=5.4, color=INK)

    handles = [
        Line2D([0], [0], color=C_LEGEND, linewidth=1.1, linestyle=(0, (3, 1.6)), marker="o",
               markersize=3.6, markerfacecolor="white", markeredgecolor=C_LEGEND,
               label="Frozen (dashed, hollow)"),
        Patch(facecolor=C_LEGEND, alpha=0.3, edgecolor=C_LEGEND, linewidth=1.3,
              label="+ID fine-tune (solid, filled)"),
        Line2D([0], [0], color=REF, linewidth=0.7, label="R² = 0"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6,
               bbox_to_anchor=(0.5, fy(0.0)), handlelength=2.2, columnspacing=1.6)
    fig.text(0.5, fy(0.24),
             "Colour = pretraining family. Numbers: Table 1 aggregate R² (mean of 7 action dims), "
             "frozen → +ID (Δ). Vertices: group means (Trans., Rot., Grip.).",
             ha="center", va="bottom", fontsize=5.3, color=MUTED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"fig2_per_dim_radar_diff{args.diff_variant}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)

    csv_path = stem.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "family", "condition", "variant", "agg_r2_table1", "trans", "rot", "grip",
                    *[f"r2_{d}" for d in DIMS]])
        for title, family, v_frozen, v_id, frz, pid in panels:
            for cond, var, vals in (("frozen", v_frozen, frz), ("+ID", v_id, pid)):
                w.writerow([title, family, cond, var, f"{vals['agg']:.6f}", f"{vals['trans']:.6f}",
                            f"{vals['rot']:.6f}", f"{vals['grip']:.6f}",
                            *[f"{vals['per_dim'][d]:.6f}" for d in DIMS]])

    prov = {
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "input": {"path": str(RESULTS.relative_to(REPO_ROOT)), "sha256": _sha256(RESULTS)},
        "args": vars(args),
        "panels": [{"panel": p[0], "frozen_variant": p[2], "id_variant": p[3]} for p in panels],
    }
    stem.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False))
    print(f"saved -> {stem}.{{pdf,png,csv,provenance.json}}")
    for title, _, _, _, frz, pid in panels:
        print(f"  {title:20s} frozen {frz['agg']:+.4f}  +ID {pid['agg']:+.4f}  Δ {pid['agg'] - frz['agg']:+.4f}")


if __name__ == "__main__":
    main()
