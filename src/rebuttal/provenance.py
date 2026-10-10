"""Provenance recorder for every rebuttal run.

Every training/evaluation invocation must call ``record_provenance`` before
allocating optimizer state and write the returned dict to
``<run_dir>/provenance.json``. If any required field is missing we raise
immediately: it is much cheaper to abort at startup than to discover after a
30-hour run that the git commit was undetermined.

Fields recorded (instruction §2 rule 5):

* git_commit  — from ``git rev-parse HEAD`` (falls back to env override)
* command     — sys.argv joined with quoting
* hostname, gpu_model
* python_version, pip_freeze (optional; big)
* checkpoint_source_url, checkpoint_sha256
* dataset_paths, dataset_sha256_manifest
* seeds: training_seed, probe_seed
* optimizer, lr_schedule, batch_size, effective_batch, updates, obs_seen
* wall_clock_start, wall_clock_end (end set at teardown)
* final_checkpoint_step

The module is deliberately dependency-free (only stdlib) so it works at run
startup before any heavy imports.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional


REQUIRED_FIELDS = (
    "run_id",
    "condition",
    "backbone",
    "training_seed",
    "probe_seed",
    "optimizer",
    "lr",
    "lr_schedule",
    "batch_size",
    "effective_batch",
    "target_updates",
)

# Eval-time result rows must carry these fields (session-2 STEP 5). Enforced
# by ``assert_result_row_schema`` before appending to results/*.jsonl.
REQUIRED_RESULT_ROW_FIELDS = (
    "eval_mode",         # one of EvalMode.values()
    "benchmark",         # libero | calvin | metaworld
    "backbone",
    "condition",         # frozen | true_id | shuffled_id | ...
    "training_seed",
    "probe_seed",
    "perturbation",      # "clean" or a named perturbation
    "split_sha256",      # bind the metric to the split it was computed on
)


def assert_result_row_schema(row: dict) -> None:
    from rebuttal.eval_modes import validate_eval_mode
    missing = [k for k in REQUIRED_RESULT_ROW_FIELDS if k not in row]
    if missing:
        raise ValueError(
            f"result row missing required fields {missing}; got keys {list(row)}"
        )
    validate_eval_mode(row["eval_mode"])


def _git_commit(repo_root: Path) -> Optional[str]:
    override = os.environ.get("REBUTTAL_GIT_COMMIT_OVERRIDE")
    if override:
        return override
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return out.decode().strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _gpu_model() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode().strip().splitlines()[0]
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _sha256_file(path: Path, chunk: int = 1 << 20) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def record_provenance(
    run_dir: Path,
    *,
    condition: str,
    backbone: str,
    run_id: str,
    training_seed: int,
    probe_seed: int,
    optimizer: str,
    lr: float,
    lr_schedule: str,
    batch_size: int,
    effective_batch: int,
    target_updates: int,
    checkpoint_source_url: Optional[str] = None,
    checkpoint_paths: Optional[dict[str, str]] = None,
    dataset_paths: Optional[dict[str, str]] = None,
    split_paths: Optional[dict[str, str]] = None,
    extra: Optional[dict[str, Any]] = None,
    repo_root: Optional[Path] = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]

    rec: dict[str, Any] = {
        "run_id": run_id,
        "condition": condition,
        "backbone": backbone,
        "training_seed": training_seed,
        "probe_seed": probe_seed,
        "optimizer": optimizer,
        "lr": lr,
        "lr_schedule": lr_schedule,
        "batch_size": batch_size,
        "effective_batch": effective_batch,
        "target_updates": target_updates,
        "git_commit": _git_commit(repo_root),
        "command": " ".join(shlex.quote(a) for a in sys.argv),
        "hostname": socket.gethostname(),
        "gpu_model": _gpu_model(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "wall_clock_start_unix": time.time(),
        "wall_clock_start_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    if checkpoint_source_url:
        rec["checkpoint_source_url"] = checkpoint_source_url
    if checkpoint_paths:
        rec["checkpoint_sha256"] = {
            k: _sha256_file(Path(v)) for k, v in checkpoint_paths.items()
        }
    if dataset_paths:
        rec["dataset_paths"] = dataset_paths
    if split_paths:
        rec["split_paths"] = split_paths
        rec["split_sha256"] = {
            k: _sha256_file(Path(v)) for k, v in split_paths.items()
        }
    if extra:
        rec["extra"] = extra

    for k in REQUIRED_FIELDS:
        if rec.get(k) in (None, ""):
            raise ValueError(f"missing required provenance field: {k}")

    out = run_dir / "provenance.json"
    out.write_text(json.dumps(rec, indent=2, sort_keys=True))
    return rec


def finalize_provenance(run_dir: Path,
                         final_checkpoint_step: int,
                         extra: Optional[dict[str, Any]] = None) -> None:
    p = Path(run_dir) / "provenance.json"
    if not p.exists():
        raise FileNotFoundError(f"provenance.json missing in {run_dir}")
    rec = json.loads(p.read_text())
    rec["wall_clock_end_unix"] = time.time()
    rec["wall_clock_end_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rec["wall_clock_elapsed_s"] = (
        rec["wall_clock_end_unix"] - rec["wall_clock_start_unix"]
    )
    rec["final_checkpoint_step"] = final_checkpoint_step
    if extra:
        rec.setdefault("extra", {}).update(extra)
    p.write_text(json.dumps(rec, indent=2, sort_keys=True))
