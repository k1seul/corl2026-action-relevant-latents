"""Render runs/full_results/results.jsonl into the three appendix CSVs.

Outputs:
  full_table_train.csv     train-cell metrics (probe trained on train, evaluated
                            on a held-out train-data subset).
  full_table_test.csv      test_ood-cell metrics.
  full_table_combined.csv  wide-form: every column from train and test_ood
                            side-by-side, suffixed _train / _test.

Sanity checks emitted to stderr:
  - per_dim_mse.mean() vs aggregate action_probe_mse (rounding tolerance)
  - psnr vs -10*log10(recon_mse) (rounding tolerance)
  - flag rows where train MSE > test MSE (suspicious)
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DIM_NAMES = ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")

# columns per the spec (Step 4) + training hyperparams pulled from manifest:
# base_model, variant, seq_len, model_size, params, batch_size, max_steps,
# lr, training_step_at_ckpt, pretrained, ckpt,
# recon_mse, lpips, psnr, action_mse, dx, dy, dz, drx, dry, drz, gripper
SHARED_HEAD = ["base_model", "variant", "family", "seq_len", "model_size",
                "n_params_million", "batch_size", "max_steps", "lr",
                "training_step_at_ckpt", "pretrained", "ckpt", "notes"]
METRIC_COLS = ["recon_mse", "lpips", "psnr",
                "action_probe_mse", "action_probe_r2"] + list(DIM_NAMES) + \
              [f"r2_{d}" for d in DIM_NAMES]


def _row_metrics(cell: dict) -> dict:
    if cell is None or "action_probe_mse" not in cell:
        return {c: float("nan") for c in METRIC_COLS}
    out = {
        "recon_mse": cell.get("recon_mse", float("nan")),
        "lpips": cell.get("lpips", float("nan")),
        "psnr": cell.get("psnr", float("nan")),
        "action_probe_mse": cell.get("action_probe_mse", float("nan")),
        "action_probe_r2": cell.get("action_probe_r2", float("nan")),
    }
    per_mse = cell.get("per_dim_mse", {}) or {}
    per_r2 = cell.get("per_dim_r2", {}) or {}
    for d in DIM_NAMES:
        out[d] = per_mse.get(d, float("nan"))
        out[f"r2_{d}"] = per_r2.get(d, float("nan"))
    return out


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return "NA"
        return f"{v:.6f}"
    return str(v)


def _shared(row: dict) -> dict:
    return {k: row.get(k, "") for k in SHARED_HEAD}


def _sanity(row: dict, cell_name: str, errors: list[str]):
    cell = row.get(cell_name)
    if not cell or "action_probe_mse" not in cell:
        return
    per = cell.get("per_dim_mse", {})
    if per:
        mean_pd = sum(per.values()) / len(per)
        agg = cell.get("action_probe_mse", float("nan"))
        if not math.isnan(agg) and abs(mean_pd - agg) > 1e-6:
            errors.append(
                f"  {row['variant']} {cell_name}: per_dim mean {mean_pd:.6g} "
                f"!= agg {agg:.6g}")
    recon = cell.get("recon_mse", float("nan"))
    psnr = cell.get("psnr", float("nan"))
    if not (math.isnan(recon) or math.isnan(psnr)):
        psnr_check = -10.0 * math.log10(max(1e-12, recon))
        if abs(psnr_check - psnr) > 0.01:
            errors.append(
                f"  {row['variant']} {cell_name}: psnr {psnr:.4f} "
                f"vs -10log10(mse) {psnr_check:.4f}")


def _load_manifest_index(out_dir: Path) -> dict[str, dict]:
    """Return {variant: manifest_entry}. Used to merge in batch_size/max_steps/lr
    columns that are NOT in the per-row results.jsonl (they're metadata)."""
    import yaml
    mf = out_dir / "manifest.yaml"
    if not mf.exists():
        print(f"WARN: manifest.yaml not at {mf}; batch_size/max_steps/lr cols will be blank",
              file=sys.stderr)
        return {}
    entries = yaml.safe_load(mf.read_text()) or []
    return {e["variant"]: e for e in entries}


def main():
    out_dir = REPO_ROOT / "runs" / "full_results"
    src = out_dir / "results.jsonl"
    if not src.exists():
        print(f"missing {src}", file=sys.stderr)
        sys.exit(1)
    rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]

    # merge training hyperparams + base_model label from manifest. The
    # manifest is authoritative for metadata; the JSONL only stores
    # per-row measurement output. Overwrite (not setdefault) so renaming
    # base_model takes effect without re-running the pipeline.
    manifest_idx = _load_manifest_index(out_dir)
    for row in rows:
        entry = manifest_idx.get(row.get("variant"), {})
        for k in ("base_model", "batch_size", "max_steps", "lr",
                   "training_step_at_ckpt", "pretrained"):
            if k in entry and entry[k] is not None:
                row[k] = entry[k]

    errors: list[str] = []

    # train-half CSV
    fields = SHARED_HEAD + METRIC_COLS
    with (out_dir / "full_table_train.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            if "error" in row:
                continue
            m = _row_metrics(row.get("train_cell"))
            line = {**_shared(row), **m}
            w.writerow({k: _fmt(v) for k, v in line.items()})
            _sanity(row, "train_cell", errors)

    # test-half CSV
    with (out_dir / "full_table_test.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            if "error" in row:
                continue
            m = _row_metrics(row.get("test_cell"))
            line = {**_shared(row), **m}
            w.writerow({k: _fmt(v) for k, v in line.items()})
            _sanity(row, "test_cell", errors)

    # combined wide-form
    combined_fields = (
        SHARED_HEAD
        + [f"{c}_train" for c in METRIC_COLS]
        + [f"{c}_test" for c in METRIC_COLS]
    )
    with (out_dir / "full_table_combined.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=combined_fields)
        w.writeheader()
        for row in rows:
            if "error" in row:
                continue
            mt = _row_metrics(row.get("train_cell"))
            me = _row_metrics(row.get("test_cell"))
            line = {**_shared(row)}
            for k, v in mt.items():
                line[f"{k}_train"] = v
            for k, v in me.items():
                line[f"{k}_test"] = v
            w.writerow({k: _fmt(v) for k, v in line.items()})

    # write error log
    err_path = out_dir / "sanity_check_log.txt"
    if errors:
        err_path.write_text("Sanity-check warnings:\n" + "\n".join(errors) + "\n")
        print(f"WARN: {len(errors)} sanity issues, see {err_path}", file=sys.stderr)
    elif err_path.exists():
        err_path.unlink()

    n_good = sum(1 for r in rows if "error" not in r)
    n_err = sum(1 for r in rows if "error" in r)
    print(f"wrote 3 CSVs to {out_dir}  ({n_good} rows, {n_err} errored)")


if __name__ == "__main__":
    main()
