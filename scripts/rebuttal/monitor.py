"""Summarize the status of every rebuttal training / eval job.

Prints one line per run directory:
  RUN_ID  GPU  step/target  last-loss  ETA  status

Sources:
  * provenance.json in each run dir (created by record_provenance)
  * <run_dir>/train.log or corresponding wave log for the last loss/step line
  * nvidia-smi to map PIDs to GPUs (best-effort)

Fails soft: any missing file just produces "?" in the corresponding column.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


LOSS_RE = re.compile(r"(?:step|iter)[\s=:]+(?P<step>\d+).*?(?:loss)[\s=:]+"
                     r"(?P<loss>[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)")


def parse_last_progress(log_path: Path) -> tuple[str, str]:
    if not log_path.exists():
        return "?", "?"
    try:
        # Read last 128 KB of the log
        sz = log_path.stat().st_size
        with log_path.open("rb") as f:
            f.seek(max(0, sz - 128 * 1024))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return "?", "?"
    step = loss = "?"
    for m in LOSS_RE.finditer(tail):
        step = m.group("step")
        loss = m.group("loss")
    return step, loss


def gpu_pid_map() -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    if not shutil.which("nvidia-smi"):
        return out
    try:
        p = subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=gpu_uuid,pid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        # This form doesn't include gpu index; fall back to XML if needed.
        # Simpler: use nvidia-smi -q -x and parse; skipped here for brevity.
        for ln in p.stdout.splitlines():
            _, pid_s = [x.strip() for x in ln.split(",", 1)]
            out.setdefault(-1, []).append(int(pid_s))
    except Exception:
        pass
    return out


def find_run_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob("provenance.json"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default=str(REPO_ROOT / "runs/corl2026_rebuttal/checkpoints"))
    ap.add_argument("--log-root", default=str(REPO_ROOT / "runs/corl2026_rebuttal/logs"))
    args = ap.parse_args()

    dirs = find_run_dirs(Path(args.ckpt_root))
    if not dirs:
        print("no runs found under", args.ckpt_root)
        return 0

    print(f"{'run_id':<40} {'cond':<20} {'bb':<10} {'step/target':<15} "
          f"{'loss':<12} {'elapsed_h':<10} status")
    print("-" * 120)

    now = time.time()
    for d in dirs:
        prov_path = d / "provenance.json"
        try:
            prov = json.loads(prov_path.read_text())
        except Exception:
            continue
        run_id = prov.get("run_id", d.name)
        cond = prov.get("condition", "?")
        bb = prov.get("backbone", "?")
        target = prov.get("target_updates", "?")
        start = prov.get("wall_clock_start_unix")
        end = prov.get("wall_clock_end_unix")
        final_step = prov.get("final_checkpoint_step")

        # log candidates
        log_candidates = [
            d / "train.log",
            Path(args.log_root) / f"{bb}_{cond}_s{prov.get('training_seed', '?')}.log",
        ]
        step = loss = "?"
        for c in log_candidates:
            s, l = parse_last_progress(c)
            if s != "?":
                step, loss = s, l
                break
        if final_step is not None:
            step = str(final_step)

        if end is not None:
            elapsed_h = f"{(end - start) / 3600.0:.2f}"
            status = "DONE"
        elif start is not None:
            elapsed_h = f"{(now - start) / 3600.0:.2f}"
            status = "RUNNING"
        else:
            elapsed_h = "?"
            status = "?"

        print(f"{run_id:<40} {cond:<20} {bb:<10} "
              f"{step}/{target:<10} {loss:<12} {elapsed_h:<10} {status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
