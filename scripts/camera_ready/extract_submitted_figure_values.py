"""Read the plotted values out of the submitted paper's vector figures and compare them with the records.

The submitted Figs. 1-3 are matplotlib PDFs embedded as Form XObjects, so every
marker position can be recovered exactly from the page content stream and
converted to data units with the figure's own tick marks. Each output row puts
the plotted value next to the matching record value.

Outputs (``runs/camera_ready/evidence/``):
  * submitted_fig1_points.csv   - Fig. 1 markers: plotted PSNR / R² vs records
  * submitted_fig2_vertices.csv - Fig. 2 radar vertices vs per-group record means
  * submitted_fig3_points.csv   - Fig. 3 polylines vs per-layer records

Usage:
    python scripts/camera_ready/extract_submitted_figure_values.py --pdf <submitted paper PDF>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import pypdf

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "runs/full_results/results.jsonl"
DECODER = REPO_ROOT / "runs/full_results/decoder_results.jsonl"
PER_LAYER = {"frozen": REPO_ROOT / "runs/full_results/vjepa_per_layer_frozen.jsonl",
             "+ID": REPO_ROOT / "runs/full_results/vjepa_per_layer.jsonl"}
OUT_DIR = REPO_ROOT / "runs/camera_ready/evidence"
DIMS = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
NUM = r"-?[\d.]+"


def form_stream(reader: pypdf.PdfReader, page_index: int) -> str:
    xobjects = reader.pages[page_index]["/Resources"]["/XObject"]
    forms = [v.get_object() for v in xobjects.values() if v.get_object().get("/Subtype") == "/Form"]
    if len(forms) != 1:
        raise ValueError(f"page {page_index + 1}: expected one figure XObject, found {len(forms)}")
    return forms[0].get_data().decode("latin-1")


def ticks(stream: str, vertical: bool) -> list[tuple[float, float]]:
    """(position, label) for tick marks followed by a plain numeric label."""
    pat = re.compile(rf"({NUM}) ({NUM}) m\n({NUM}) ({NUM}) l\n\nB\n1 w\nq\n1 0 -0 1 {NUM} {NUM} cm\n"
                     rf"BT\n/F\d+ [\d.]+ Tf\n0 0 Td\n\[ \(([\d.]+)\) \] TJ")
    out = []
    for m in pat.finditer(stream):
        x1, y1, x2, y2 = map(float, m.group(1, 2, 3, 4))
        if vertical and abs(x1 - x2) < 1e-6:
            out.append((x1, float(m.group(5))))
        if not vertical and abs(y1 - y2) < 1e-6:
            out.append((y1, float(m.group(5))))
    return out


def linear_map(pairs: list[tuple[float, float]]):
    pos, val = np.array(pairs).T
    slope, intercept = np.polyfit(pos, val, 1)
    if np.max(np.abs(slope * pos + intercept - val)) > 1e-6:
        raise ValueError("tick positions are not linear in their labels")
    return lambda p: slope * p + intercept


TOKEN = re.compile(r"\((?:\\.|[^\\)])*\)|\[[^\]]*\]|/[^\s/\[\]()<>]+|[^\s/\[\]()<>]+")
PAINT = {"B", "B*", "b", "b*", "f", "F", "f*", "S", "s", "n"}


def painted_paths(stream: str):
    """Yield (fill colour, paint operator, points) for every painted path.

    A small content-stream interpreter: operands are stacked until an operator,
    the fill colour follows q/Q save/restore, and points come from m/l/c/re.
    """
    fill, saved, pts, operands = None, [], [], []
    for tok in TOKEN.findall(stream):
        try:
            operands.append(float(tok))
            continue
        except ValueError:
            pass
        if tok[0] in "/[(":
            operands.append(tok)
            continue
        nums = [o for o in operands if isinstance(o, float)]
        if tok == "q":
            saved.append(fill)
        elif tok == "Q":
            fill = saved.pop() if saved else None
        elif tok == "rg" and len(nums) >= 3:
            fill = tuple(round(v, 3) for v in nums[-3:])
        elif tok == "g" and nums:
            fill = (round(nums[-1], 3),) * 3
        elif tok in ("m", "l") and len(nums) >= 2:
            pts.append((nums[-2], nums[-1]))
        elif tok == "c" and len(nums) >= 6:
            pts.append((nums[-2], nums[-1]))
        elif tok == "re" and len(nums) >= 4:
            x, y, w, h = nums[-4:]
            pts += [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        elif tok in PAINT:
            if pts:
                yield fill, tok, pts
            pts = []
        operands = []


def load_records():
    rows = {}
    for line in RESULTS.read_text().splitlines():
        r = json.loads(line)
        if "_pert_" not in r["variant"]:
            rows[r["variant"]] = r
    dec = {json.loads(l)["variant"]: json.loads(l) for l in DECODER.read_text().splitlines()}
    return rows, dec


def r2(rows, v):
    return float(rows[v]["test_cell"]["action_probe_r2"])


def groups(rows, v):
    pd_ = rows[v]["test_cell"]["per_dim_r2"]
    if isinstance(pd_, list):
        pd_ = dict(zip(DIMS, pd_))
    return {"Trans": float(np.mean([pd_[d] for d in DIMS[:3]])),
            "Rot": float(np.mean([pd_[d] for d in DIMS[3:6]])), "Grip": float(pd_["gripper"])}


def fig1(reader, rows, dec, writer):
    s = form_stream(reader, 3)
    fx, fy = linear_map(ticks(s, True)), linear_map(ticks(s, False))
    colour_group = {(0.18, 0.49, 0.196): "video SSL", (0.984, 0.549, 0.0): "image SSL",
                    (0.776, 0.157, 0.157): "pixel fidelity", (0.459, 0.459, 0.459): "other"}
    # (name, legend group, frozen variant or None, +ID variant, decoder_results variant or None=rollout PSNR)
    cands = [
        ("V-JEPA 2 ViT-L", "video SSL", "vjepa2_vit_l_frozen", "vjepa2_aux_finetune", "vjepa2_aux"),
        ("V-JEPA 2.1 ViT-B", "video SSL", "vjepa2_1_vit_b_frozen", "vjepa2_1_aux_finetune", "vjepa2_1_aux"),
        ("VideoMAE V1 ViT-L", "video SSL", "videomae_frozen", "videomae_aux_finetune", "videomae_aux"),
        ("Web-DINO ViT-L", "image SSL", "webdino_frozen", "webdino_aux_finetune", "webdino_aux"),
        ("SigLIP 2 ViT-L", "image SSL", "siglip2_frozen", "siglip2_aux_finetune", "siglip2_aux"),
        ("LAPA-LAQ-OpenX", "other", "lapa_frozen", "lapa_aux_finetune", "lapa_aux"),
        ("SDXL VAE", "pixel fidelity", "sdvae_frozen", "sdvae_aux_finetune", "sd3vae_aux"),
        ("Cosmos-1 tokenizer", "pixel fidelity", "cosmos_tokenizer_frozen", "cosmos_tokenizer_aux_finetune",
         "cosmos_aux"),
        ("DIFF-FULL (action-cond. 200k)", "pixel fidelity", None, "A_diff_v10_action_large_200k", None),
        ("Dreamer 4 276M + ID (agent)", "other", None, "dreamer4_xl_aux_agent", None),
    ]
    pts = []
    for fill, _, p in painted_paths(s):
        xs, ys = zip(*p)
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        if 8 <= w <= 20 and 8 <= h <= 20 and 54 < cx < 584 and 59 < cy < 234 and fill in colour_group:
            pts.append((colour_group[fill], float(fx(cx)), float(fy(cy))))
    if len(pts) != 10:
        raise ValueError(f"Fig. 1: expected 10 markers, found {len(pts)}")
    for group, psnr, val in sorted(pts, key=lambda t: -t[2]):
        pool = [c for c in cands if c[1] == group]
        name, _, v_frz, v_id, v_dec = min(pool, key=lambda c: abs(r2(rows, c[3]) - val))
        if v_dec:
            rec_psnr, src = float(dec[v_dec]["psnr_test_mean"]), f"decoder_results.jsonl:{v_dec} (decoder on +ID encoder)"
        else:
            rec_psnr, src = float(rows[v_id]["test_cell"]["psnr"]), f"results.jsonl:{v_id} test_cell.psnr (rollout)"
        writer.writerow(["Fig1", name, group, f"{psnr:.3f}", f"{val:.4f}",
                         f"{r2(rows, v_frz):.4f}" if v_frz else "", f"{r2(rows, v_id):.4f}",
                         f"{rec_psnr:.3f}", src])


def fig2(reader, rows, writer):
    s = form_stream(reader, 5)
    panels = [("V-JEPA 2 ViT-L", "vjepa2_vit_l_frozen", "vjepa2_aux_finetune"),
              ("V-JEPA 2.1 ViT-B", "vjepa2_1_vit_b_frozen", "vjepa2_1_aux_finetune"),
              ("VideoMAE V1 ViT-L", "videomae_frozen", "videomae_aux_finetune"),
              ("Web-DINO ViT-L", "webdino_frozen", "webdino_aux_finetune"),
              ("SigLIP 2 ViT-L", "siglip2_frozen", "siglip2_aux_finetune"),
              ("LAPA-LAQ-OpenX", "lapa_frozen", "lapa_aux_finetune"),
              ("DIFF (L1)", "v6_large_noact_taskheld", "aux_abl_A4_invdyn_l1"),
              ("SDXL VAE", "sdvae_frozen", "sdvae_aux_finetune"),
              ("Cosmos-1 tokenizer", "cosmos_tokenizer_frozen", "cosmos_tokenizer_aux_finetune"),
              ("Dreamer 4", "dreamer4_dynamics_agent", "dreamer4_dynamics_aux_agent")]
    grp = re.compile(rf"(1 g\s*)?\nq\n1 0 0 1 ({NUM}) ({NUM}) cm /M\d+ Do\n1 0 0 1 ({NUM}) ({NUM}) cm /M\d+ Do\n"
                     rf"1 0 0 1 ({NUM}) ({NUM}) cm /M\d+ Do\nQ")
    found = []
    for m in grp.finditer(s):
        hollow = m.group(1) is not None
        x, y = float(m.group(2)), float(m.group(3))
        p = [(x, y)]
        for k in (4, 6):
            x += float(m.group(k)); y += float(m.group(k + 1)); p.append((x, y))
        (xt, yt), (xr, yr), (xg, yg) = p
        cx = xt
        cy = yr + (cx - xr) / math.sqrt(3)  # Rot vertex sits at 210 degrees
        radii = sorted({round(float(a) - cx, 4) for a, b in re.findall(rf"({NUM}) ({NUM}) m\n", s)
                        if abs(float(b) - cy) < 1e-3 and float(a) - cx > 1})
        if len(radii) < 5:
            raise ValueError("Fig. 2: could not find the five radial grid rings")
        to_val = linear_map(list(zip(radii[:5], [-1.0, -0.5, 0.0, 0.5, 1.0])))
        vals = {"Trans": to_val(yt - cy), "Rot": to_val(math.hypot(xr - cx, yr - cy)),
                "Grip": to_val(math.hypot(xg - cx, yg - cy))}
        found.append((-round(cy, 1), cx, "frozen" if hollow else "+ID", vals))
    if len(found) != 20:
        raise ValueError(f"Fig. 2: expected 20 marker groups, found {len(found)}")
    found.sort(key=lambda t: (t[0], t[1]))
    for k, (_, _, cond, vals) in enumerate(found):
        name, v_frz, v_id = panels[k // 2]
        rec = groups(rows, v_frz if cond == "frozen" else v_id)
        for axis in ("Trans", "Rot", "Grip"):
            writer.writerow(["Fig2", name, cond, axis, f"{vals[axis]:.4f}", f"{rec[axis]:.4f}",
                             f"{vals[axis] - rec[axis]:+.4f}"])


def fig3(reader, writer):
    s = form_stream(reader, 6)
    fy = linear_map(ticks(s, False))
    xt = ticks(s, True)
    fx = linear_map(xt)
    lines = [p for _, op, p in painted_paths(s) if op == "S" and len(p) >= 5]
    if len(lines) != 2:
        raise ValueError(f"Fig. 3: expected 2 polylines, found {len(lines)}")
    rec = {k: {json.loads(l)["layer"]: json.loads(l)["r2_test_mean"] for l in p.read_text().splitlines()}
           for k, p in PER_LAYER.items()}
    # The +ID curve is the one that is higher at the last plotted block.
    lines.sort(key=lambda p: p[-1][1])
    for series, p in zip(("frozen", "+ID"), lines):
        for x, y in p:
            block = int(round(fx(x)))
            writer.writerow(["Fig3", series, block, f"{fy(y):.4f}", f"{rec[series][block]:.4f}",
                             f"{fy(y) - rec[series][block]:+.4f}"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="submitted paper PDF")
    args = ap.parse_args()
    reader = pypdf.PdfReader(args.pdf)
    rows, dec = load_records()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "submitted_fig1_points.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["figure", "matched_backbone", "legend_group", "plotted_psnr_db", "plotted_r2",
                    "record_frozen_r2", "record_id_r2", "record_psnr_db", "record_psnr_source"])
        fig1(reader, rows, dec, w)
    with (OUT_DIR / "submitted_fig2_vertices.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["figure", "panel", "condition", "axis", "plotted", "record_group_mean", "plotted_minus_record"])
        fig2(reader, rows, w)
    with (OUT_DIR / "submitted_fig3_points.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["figure", "series", "block", "plotted", "record_mean", "plotted_minus_record"])
        fig3(reader, w)
    print(f"saved -> {OUT_DIR}/submitted_fig{{1,2,3}}_*.csv")


if __name__ == "__main__":
    main()
