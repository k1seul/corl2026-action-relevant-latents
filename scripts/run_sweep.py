"""Run a queue of train+eval configurations sequentially.

For each entry in the queue YAML, this orchestrator:
  1. Merges the entry's overrides on top of `--base-config` and writes
     the resulting full config under `<out-dir>/<run_name>/config.yaml`.
  2. Spawns `scripts/train.py` to train; logs to `train.log`.
  3. Spawns `scripts/eval.py` on the resulting `ckpt_last.pt`; logs to
     `eval.log` and writes per-run metrics + sample mp4s.
  4. Appends a JSON record to `<out-dir>/results.jsonl` and rewrites
     `<out-dir>/summary.md` (sorted by PSNR) after each run.

Existing runs (those whose `eval/metrics.txt` already exists) are
skipped, so the orchestrator is resumable. Failures are recorded with
status `train_failed` / `eval_failed` / `error` and do not abort the
queue.

Queue YAML format::

    runs:
      - name: q1_baseline
        overrides: {}
      - name: q1_mask_75
        overrides:
          train:
            mask_ratio: 0.75
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_metrics(metrics_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not metrics_path.exists():
        return out
    for line in metrics_path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def run_one(
    name: str,
    base_cfg_path: str,
    overrides: dict,
    out_root: Path,
) -> dict:
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "eval" / "metrics.txt"

    if metrics_path.exists():
        rec = {"name": name, "status": "ok", "elapsed_sec": 0.0,
               "overrides": overrides, "skipped": True}
        rec.update(parse_metrics(metrics_path))
        print(f"[skip ] {name}: already complete (PSNR={rec.get('masked_patch_psnr_db', '?')})")
        return rec

    # build merged config
    base = OmegaConf.load(base_cfg_path)
    overrides_node = OmegaConf.create(overrides) if overrides else OmegaConf.create({})
    merged = OmegaConf.merge(base, overrides_node)
    OmegaConf.update(merged, "train.ckpt_dir", str(out_dir), force_add=True)
    OmegaConf.update(merged, "eval.out_dir", str(out_dir / "eval"), force_add=True)
    # sweep-friendly defaults: no intermediate ckpts, less verbose logging
    OmegaConf.update(merged, "train.ckpt_interval", 9_999_999, force_add=True)
    if merged.train.get("log_interval", 0) < 200:
        OmegaConf.update(merged, "train.log_interval", 500, force_add=True)
    cfg_path = out_dir / "config.yaml"
    OmegaConf.save(merged, cfg_path)

    train_log = out_dir / "train.log"
    eval_log = out_dir / "eval.log"

    print(f"[start] {name}")
    t0 = time.time()

    cmd_train = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train.py"),
        "--config", str(cfg_path),
    ]
    with train_log.open("w") as f:
        ret = subprocess.run(
            cmd_train, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT)
        )
    if ret.returncode != 0:
        elapsed = time.time() - t0
        print(f"[FAIL ] {name}: train rc={ret.returncode} (see {train_log})")
        return {"name": name, "status": "train_failed",
                "elapsed_sec": elapsed, "overrides": overrides}

    ckpt = out_dir / "ckpt_last.pt"
    cmd_eval = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval.py"),
        "--ckpt", str(ckpt),
        "--out-dir", str(out_dir / "eval"),
    ]
    with eval_log.open("w") as f:
        ret = subprocess.run(
            cmd_eval, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT)
        )
    if ret.returncode != 0:
        elapsed = time.time() - t0
        print(f"[FAIL ] {name}: eval rc={ret.returncode} (see {eval_log})")
        return {"name": name, "status": "eval_failed",
                "elapsed_sec": elapsed, "overrides": overrides}

    elapsed = time.time() - t0
    rec = {"name": name, "status": "ok", "elapsed_sec": elapsed,
           "overrides": overrides}
    rec.update(parse_metrics(metrics_path))
    print(f"[done ] {name}: PSNR={rec.get('masked_patch_psnr_db','?')} dB ({elapsed/60:.1f} min)")
    return rec


def write_summary(out_root: Path) -> None:
    results_path = out_root / "results.jsonl"
    if not results_path.exists():
        return
    rows = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]

    def psnr_of(r: dict) -> float:
        try:
            return float(r.get("masked_patch_psnr_db", "-inf"))
        except (TypeError, ValueError):
            return float("-inf")

    rows.sort(key=lambda r: -psnr_of(r))

    md = ["# Sweep results", "", f"_{len(rows)} runs_", ""]
    md.append("| name | status | PSNR (dB) | MSE | clips | elapsed (min) | overrides |")
    md.append("|------|--------|-----------|-----|-------|---------------|-----------|")
    for r in rows:
        ovr = json.dumps(r.get("overrides", {}), separators=(",", ":"), ensure_ascii=False)
        md.append(
            f"| {r['name']} | {r['status']} | {r.get('masked_patch_psnr_db','?')} "
            f"| {r.get('masked_patch_mse','?')} | {r.get('clips_evaluated','?')} "
            f"| {(r.get('elapsed_sec', 0) / 60):.1f} | `{ovr}` |"
        )
    (out_root / "summary.md").write_text("\n".join(md) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True, help="path to queue YAML")
    ap.add_argument(
        "--base-config",
        default=str(REPO_ROOT / "configs" / "baseline.yaml"),
    )
    ap.add_argument("--out-dir", required=True, help="output root for this queue")
    args = ap.parse_args()

    queue = OmegaConf.load(args.queue)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    results_path = out_root / "results.jsonl"

    print(f"queue: {args.queue}")
    print(f"out  : {out_root}")
    print(f"runs : {len(queue.runs)}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    for entry in queue.runs:
        name = entry.name
        ov = entry.get("overrides")
        overrides = OmegaConf.to_container(ov, resolve=True) if ov is not None else {}
        try:
            result = run_one(name, args.base_config, overrides, out_root)
        except Exception as e:  # noqa: BLE001
            elapsed = 0
            print(f"[error] {name}: {e!r}")
            result = {"name": name, "status": "error",
                      "error": repr(e), "overrides": overrides,
                      "elapsed_sec": elapsed}
        with results_path.open("a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        write_summary(out_root)

    print(f"\nsweep complete. summary: {out_root / 'summary.md'}")


if __name__ == "__main__":
    main()
