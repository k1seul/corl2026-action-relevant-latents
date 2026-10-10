"""Figure 6 (Appendix N): closed-loop target-swap setup.

1200 px design canvas, STIX serif, pastel panels, rounded token chips (blue = vision
tokens, red = learned token, sand = targets). Drawn with matplotlib so it is a vector
PDF that rebuilds from this repo. The observation thumbnails are LIBERO frames
(src/assets/).

Content follows the runs behind Table 4 (closed_loop/):
  * pi_0.5 base, fine-tuned on LIBERO with flow matching plus one auxiliary target loss;
  * clip coupling: pooled main-camera prefix tokens -> MLP (hidden 2048) -> z_hat, target =
    frozen DIFF + ID encoding of the two-camera clip o_{t:t+16}, ||z - z_hat||^2/||z||^2, w = 0.1;
    controls: batch-shuffled target, random-init encoder;
  * token coupling: one learned prefix token -> readout -> z_hat, target = frozen encoder
    feature of o_{t+16}, cosine loss, w = 0.01; targets V-JEPA 2 + ID, Web-DINO;
  * the target encoder is used only during training.

Output: paper_figures/out/fig6/fig6_closed_loop_method.{pdf,png} (5.5 in wide).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

FIG_ROOT = Path(__file__).resolve().parents[1]
ASSETS = FIG_ROOT / "src/assets"
OUT_DIR = FIG_ROOT / "out/fig6"

W_PX, H_PX = 1200, 620          # design canvas (px)
FIG_W_IN = 5.5                  # CoRL text width
PX = FIG_W_IN * 72 / W_PX       # points per design px

# palette
INK, MUTED, ARROW = "#222222", "#6b6a66", "#6b6a66"
PANEL_FC, PANEL_EC = "#f7f6f3", "#e3e2de"
VLM_FC, VLM_EC = "#e6eef7", "#8fb0d4"
EXP_FC, EXP_EC = "#f7f6f3", "#9a9994"
ENC_FC, ENC_EC = "#ecebe7", "#9a9994"
BLUE, BLUE_FC = "#2E6FB3", "#dbe6f3"
RED, RED_FC = "#B34A2E", "#f4ded7"
SAND, SAND_FC = "#8a7a55", "#f1ead9"
GRAY_EC, GRAY_FC = "#9a9994", "#e9e8e4"
DARK = "#4a4946"
CLIP_BG, CLIP_EC = "#f0f5fb", "#c9d9ec"
TOK_BG, TOK_EC = "#fbf3f0", "#e8cfc6"


def fs(px: float) -> float:
    return px * PX


def main() -> None:
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(FIG_W_IN, FIG_W_IN * H_PX / W_PX))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_PX)
    ax.set_ylim(H_PX, 0)
    ax.axis("off")

    def box(x, y, w, h, fc, ec, r=12, ls="-", lw=1.5, z=1):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc=fc, ec=ec, lw=lw * PX * 1.6, ls=ls, zorder=z))

    def chip(x, y, fc, ec, s=26, ls="-", z=3):
        box(x, y, s, s, fc, ec, r=6, ls=ls, lw=1.5, z=z)

    def chips(x, y, n, fc, ec, s=26, gap=8, ls="-"):
        for i in range(n):
            chip(x + i * (s + gap), y, fc, ec, s, ls)

    def txt(x, y, s, size=20, color=INK, ha="center", va="center", weight="normal", style="normal", z=5):
        ax.text(x, y, s, fontsize=fs(size), color=color, ha=ha, va=va, fontweight=weight,
                fontstyle=style, zorder=z)

    def arrow(x1, y1, x2, y2, color=ARROW, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=fs(22),
                                     color=color, lw=1.8 * PX * 1.6, ls=ls, zorder=4,
                                     shrinkA=0, shrinkB=0))

    def trapezoid(pts, z=2):
        ax.add_patch(Polygon(pts, closed=True, fc=ENC_FC, ec=ENC_EC, lw=1.5 * PX * 1.6,
                             joinstyle="round", zorder=z))

    def brace(x1, x2, y, h=7):
        xm = (x1 + x2) / 2
        verts = [(x1, y + h), (x1, y), (x1 + 7, y), (xm - 7, y), (xm, y), (xm, y - h),
                 (xm, y), (xm + 7, y), (x2 - 7, y), (x2, y), (x2, y + h)]
        codes = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
                 MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3]
        ax.add_patch(PathPatch(MPath(verts, codes), fc="none", ec=INK, lw=1.6 * PX * 1.6, zorder=4))

    def image(path, x, y, s, grid=True):
        im = plt.imread(path)
        # interpolation="none" embeds the 224x224 photo as is instead of resampling it at savefig dpi
        ax.imshow(im, extent=(x, x + s, y + s, y), zorder=3, interpolation="none")
        ax.add_patch(FancyBboxPatch((x, y), s, s, boxstyle="square,pad=0", fc="none", ec=DARK,
                                    lw=1.2 * PX * 1.6, zorder=4))
        if grid:
            for k in (1, 2, 3):
                ax.plot([x + k * s / 4] * 2, [y, y + s], color="white", lw=0.6 * PX * 1.6, alpha=0.75, zorder=4)
                ax.plot([x, x + s], [y + k * s / 4] * 2, color="white", lw=0.6 * PX * 1.6, alpha=0.75, zorder=4)

    # ------------------------------------------------------------------ future target panel
    box(24, 16, 724, 196, PANEL_FC, PANEL_EC)
    txt(42, 42, "Future target (frozen encoder, used only for training)", 24, weight="bold", ha="left")
    image(ASSETS / "obs_future.jpg", 62, 70, 92)          # back frame: a clip of future frames
    image(ASSETS / "obs_future.jpg", 50, 80, 92)          # front frame
    txt(98, 192, r"$o_{t:t+16}$", 22)
    trapezoid([(180, 86), (264, 108), (264, 154), (180, 176)])
    txt(222, 131, r"$E_{\mathrm{tgt}}$", 26)
    txt(222, 192, "frozen", 19, MUTED)
    arrow(270, 131, 296, 131)
    chip(304, 118, SAND_FC, SAND)
    txt(317, 162, r"$z$", 24)
    txt(372, 80, "Target encoder", 20, weight="bold", ha="left")
    txt(372, 108, r"(i) clip: DIFF + ID on $o_{t:t+16}$ (2 cameras)", 19, ha="left")
    txt(372, 134, r"(ii) token: V-JEPA 2 + ID or Web-DINO", 19, ha="left")
    txt(372, 158, r"$\quad\;\;$on $o_{t+16}$", 19, ha="left")
    txt(372, 186, r"features mean-pooled to one vector $z$", 19, MUTED, ha="left")

    # ------------------------------------------------------------------ prefix outputs above VLM
    chips(60, 266, 4, BLUE_FC, BLUE)
    brace(60, 188, 254)
    txt(124, 228, r"$Z_t^{\mathrm{vis}}$", 26)
    chips(222, 266, 4, BLUE_FC, BLUE)
    txt(290, 242, "wrist", 19, MUTED)
    chips(384, 266, 2, GRAY_FC, GRAY_EC)
    chip(470, 266, RED_FC, RED, ls="--")
    txt(483, 242, r"$q$", 24, RED)

    # ------------------------------------------------------------------ VLM and action expert
    box(28, 306, 480, 92, VLM_FC, VLM_EC)
    txt(268, 352, r"$\pi_{0.5}$ pretrained VLM", 30, weight="bold")
    box(566, 306, 176, 92, EXP_FC, EXP_EC)
    txt(654, 339, "Action expert", 24, weight="bold")
    txt(654, 368, "flow matching", 20, MUTED)
    arrow(512, 352, 560, 352)
    chips(603, 266, 3, DARK, DARK)
    txt(654, 242, r"$\hat{a}_{t:t+16}$", 24)
    chips(603, 414, 3, GRAY_FC, GRAY_EC)
    txt(654, 464, "noisy actions", 19, MUTED)

    # ------------------------------------------------------------------ prefix inputs below VLM
    chips(60, 414, 4, BLUE_FC, BLUE)
    chips(222, 414, 4, BLUE_FC, BLUE)
    chips(384, 414, 2, GRAY_FC, GRAY_EC)
    chip(470, 414, RED_FC, RED, ls="--")
    trapezoid([(82, 452), (166, 452), (184, 494), (64, 494)])
    txt(124, 474, r"$E_{\theta_v}$", 24)
    trapezoid([(244, 452), (328, 452), (346, 494), (226, 494)])
    txt(286, 474, r"$E_{\theta_v}$", 24)
    image(ASSETS / "obs_t.jpg", 83, 502, 82)
    image(ASSETS / "obs_wrist.jpg", 245, 502, 82)
    txt(124, 604, r"$o_t$", 22)
    txt(286, 604, r"$o_t^{\mathrm{w}}$", 22)
    box(388, 458, 50, 44, "white", "#9a9994", r=8)
    txt(413, 480, r"$\ell$", 26)
    txt(413, 524, "instruction", 19, MUTED)
    txt(486, 462, r"learned $q$", 19, RED)
    txt(486, 486, "(token only)", 17, RED)

    # ------------------------------------------------------------------ separator
    ax.plot([770, 770], [24, 600], color="#d9d8d3", lw=1.2 * PX * 1.6, zorder=1)

    # ------------------------------------------------------------------ (i) clip coupling
    box(790, 16, 390, 292, CLIP_BG, CLIP_EC)
    chip(808, 32, BLUE, BLUE, s=16)
    txt(834, 40, "(i) Clip coupling", 24, weight="bold", ha="left")
    for k in range(3):
        chip(818, 90 + k * 38, BLUE_FC, BLUE)
    txt(831, 212, r"$Z_t^{\mathrm{vis}}$", 24)
    arrow(852, 141, 876, 141)
    box(882, 106, 118, 70, "white", "#9a9994", r=10)
    txt(941, 129, "mean pool", 19)
    txt(941, 153, "+ MLP", 19)
    arrow(1004, 141, 1028, 141)
    chip(1036, 128, SAND_FC, SAND, ls="--")
    txt(1049, 176, r"$\hat{z}$", 24)
    txt(1112, 141, r"$\frac{\|z-\hat{z}\|^2}{\|z\|^2}$", 24)
    txt(1112, 180, r"$w=0.1$", 19, MUTED)
    txt(985, 244, "target: DIFF + ID world model", 19)
    txt(985, 272, r"controls: batch-shuffled $z$, random-init encoder", 18, MUTED)

    # ------------------------------------------------------------------ (ii) token coupling
    box(790, 328, 390, 276, TOK_BG, TOK_EC)
    chip(808, 344, RED, RED, s=16)
    txt(834, 352, "(ii) Token coupling", 24, weight="bold", ha="left")
    chip(818, 404, RED_FC, RED, ls="--")
    txt(831, 452, r"$q$", 24, RED)
    arrow(852, 417, 876, 417)
    box(882, 382, 118, 70, "white", "#9a9994", r=10)
    txt(941, 417, "readout", 19)
    arrow(1004, 417, 1028, 417)
    chip(1036, 404, SAND_FC, SAND, ls="--")
    txt(1049, 452, r"$\hat{z}$", 24)
    txt(1122, 417, r"$1-\cos(\hat{z},z)$", 20)
    txt(1122, 452, r"$w=0.01$", 19, MUTED)
    txt(985, 500, "targets: V-JEPA 2 + ID, Web-DINO", 19)
    txt(985, 528, "one learned token appended to the prefix", 18, MUTED)
    txt(985, 556, r"(V-JEPA 2 also at $w=0.05$)", 18, MUTED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "fig6_closed_loop_method"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"saved -> {stem}.{{pdf,png}}")


if __name__ == "__main__":
    main()
