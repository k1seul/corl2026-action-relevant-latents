"""Build a cross-variant comparison table.

Joins rollout metrics (from runs/sweep/variants/<name>/eval/metrics.txt)
with downstream probe metrics (from runs/downstream/<name>/metrics.json)
and writes runs/downstream/summary.md sorted by composite score.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VARIANTS_DIR = REPO_ROOT / "runs/sweep/variants"
DOWNSTREAM_DIR = REPO_ROOT / "runs/downstream"
STRONG_V1_DIR = REPO_ROOT / "runs/sweep/strong_v1/strong_v1_clip32_medium_50k"


def parse_kv(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def main() -> None:
    rows = []
    for ds_dir in sorted(DOWNSTREAM_DIR.iterdir()):
        if not ds_dir.is_dir():
            continue
        name = ds_dir.name
        metrics_json = ds_dir / "metrics.json"
        if not metrics_json.exists():
            continue
        d = json.loads(metrics_json.read_text())
        # locate rollout metrics
        if name == "strong_v1_A_mae":
            ru_dir = STRONG_V1_DIR / "eval"
        else:
            ru_dir = VARIANTS_DIR / name / "eval"
        ru_metrics = parse_kv(ru_dir / "metrics.txt")
        rollout_psnr = num(ru_metrics.get("rollout/psnr_db_mean")) or num(
            ru_metrics.get("masked_patch_psnr_db")
        )
        rollout_ssim = num(ru_metrics.get("rollout/ssim_mean"))

        # action metrics: tolerate both old single-seed and new multi-seed formats
        action_section = d.get("action", {})
        if "test_r2" in action_section:                     # legacy single-seed H=1
            action_by_h = {"1": {"test_r2_mean": action_section.get("test_r2"),
                                  "test_r2_std": 0.0,
                                  "test_mse_mean": action_section.get("test_mse")}}
        else:
            action_by_h = {}
            for h, m in action_section.items():
                if "test_r2_mean" in m:
                    action_by_h[h] = m
                else:
                    action_by_h[h] = {
                        "test_r2_mean": m.get("test_r2"),
                        "test_r2_std": 0.0,
                        "test_mse_mean": m.get("test_mse"),
                    }
        task = d.get("task", {})
        if "test_top1_mean" in task:
            top1 = task["test_top1_mean"]
            top1_std = task["test_top1_std"]
            top5 = task["test_top5_mean"]
        else:
            top1 = task.get("test_top1")
            top1_std = 0.0
            top5 = task.get("test_top5")
        invdyn = d.get("inverse_dynamics", {})
        invdyn_r2 = invdyn.get("test_r2_mean")
        invdyn_r2_std = invdyn.get("test_r2_std", 0.0)
        fs = d.get("feature_stats", {})
        rows.append({
            "name": name,
            "objective": d.get("objective", "?"),
            "feat_dim": d.get("feat_dim", "?"),
            "rollout_psnr": rollout_psnr,
            "rollout_ssim": rollout_ssim,
            "action_r2_h1": action_by_h.get("1", {}).get("test_r2_mean"),
            "action_r2_h1_std": action_by_h.get("1", {}).get("test_r2_std", 0.0),
            "action_r2_h5": action_by_h.get("5", {}).get("test_r2_mean"),
            "action_r2_h5_std": action_by_h.get("5", {}).get("test_r2_std", 0.0),
            "invdyn_r2": invdyn_r2,
            "invdyn_r2_std": invdyn_r2_std,
            "task_top1": top1,
            "task_top1_std": top1_std,
            "task_top5": top5,
            "feat_separation": fs.get("task_separation"),
            "feat_norm": fs.get("norm_mean"),
        })

    # sort by rollout psnr descending (treat None as -inf)
    rows.sort(key=lambda r: (-(r["rollout_psnr"] or float("-inf"))))

    def fnum(x, fmt=".3f"):
        return ("—" if x is None else f"{x:{fmt}}")

    md = [
        "# Downstream evaluation",
        "",
        "Probes operate on frozen features pooled over the spatial token grid.",
        "**Action regression** uses a 2-layer MLP (256, 128) on per-frame",
        "features, predicting the next H actions (`H ∈ {1, 5}`). **Inverse",
        "dynamics** trains the same MLP on `concat(feat[t], feat[t+1])` ->",
        "`action[t]`. **Task classification** is a linear readout on the",
        "per-clip pooled feature, predicting one of the 130 LIBERO tasks.",
        "Each probe is retrained over 3 seeds; values are mean ± std.",
        "Action input is withheld during feature extraction so",
        "action-conditioned models can't shortcut.",
        "",
        "_`feat sep` = within-task cosine sim − across-task cosine sim_",
        "_(higher = features cluster better by task)._",
        "",
        "| variant | obj | feat | rollout PSNR | action R² (H=1) | R² (H=5) | inv-dyn R² | task top-1 | feat sep |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    def with_std(mean, std):
        if mean is None:
            return "—"
        return f"{mean:+.3f}±{std:.3f}" if std else f"{mean:+.3f}"
    for r in rows:
        md.append(
            f"| {r['name']} | {r['objective']} | {r['feat_dim']} | "
            f"{fnum(r['rollout_psnr'], '.2f')} | "
            f"{with_std(r['action_r2_h1'], r['action_r2_h1_std'])} | "
            f"{with_std(r['action_r2_h5'], r['action_r2_h5_std'])} | "
            f"{with_std(r['invdyn_r2'], r['invdyn_r2_std'])} | "
            f"{with_std(r['task_top1'], r['task_top1_std'])} | "
            f"{fnum(r.get('feat_separation'), '.4f')} |"
        )

    out_path = DOWNSTREAM_DIR / "summary.md"
    out_path.write_text("\n".join(md) + "\n")
    print(f"wrote {out_path}")
    for r in rows:
        h1 = fnum(r["action_r2_h1"], ".3f")
        h5 = fnum(r["action_r2_h5"], ".3f")
        t1 = fnum(r["task_top1"], ".3f")
        print(f"  {r['name']:<40s} R²[H=1]={h1} R²[H=5]={h5} top1={t1}")


if __name__ == "__main__":
    main()
