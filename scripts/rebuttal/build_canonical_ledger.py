"""Build ``runs/corl2026_rebuttal/results/canonical_ledger.jsonl``.

Consolidates the existing submitted-paper JSONLs into ONE stream of records,
tagging each with its source file, byte offset, and SHA-256 so every reported
number can be traced back to a raw artifact.

Runs on stdlib only. Does NOT modify or move the original JSONL files. Does
NOT invent rebuttal numbers — rebuttal records will be appended by
downstream evaluation scripts as they complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]

# (relative_path, source_tag) — one row per source file
SOURCES = [
    ("runs/full_results/results.jsonl",             "submitted_libero_main"),
    ("runs/full_results/decoder_results.jsonl",     "submitted_libero_decoder"),
    ("runs/full_results/lambda_curve.jsonl",        "submitted_lambda_sweep"),
    ("runs/full_results/pareto_summary.jsonl",      "submitted_pareto"),
    ("runs/full_results/three_benchmark_summary.jsonl", "submitted_three_benchmarks"),
    ("runs/full_results/vjepa_per_layer.jsonl",     "submitted_vjepa_per_layer_aux"),
    ("runs/full_results/vjepa_per_layer_frozen.jsonl", "submitted_vjepa_per_layer_frozen"),
    ("runs/full_results/libero_probe_data_efficiency.jsonl",
     "submitted_libero_probe_budget"),
    ("runs/full_results_calvin/results.jsonl",      "submitted_calvin_main"),
    ("runs/full_results_calvin/libero_vs_calvin_summary.jsonl",
     "submitted_libero_vs_calvin"),
    ("runs/full_results_metaworld/results.jsonl",   "submitted_metaworld_main"),
    ("runs/id_sparsity_sweep/summary.jsonl",        "submitted_id_sparsity_sweep"),
]


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stream_jsonl(p: Path, source_tag: str, file_sha: str) -> Iterator[dict]:
    line_no = 0
    with p.open("rb") as f:
        while True:
            raw = f.readline()
            if not raw:
                return
            line_no += 1
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                yield {
                    "_source_file": str(p.relative_to(REPO_ROOT)),
                    "_source_tag": source_tag,
                    "_source_file_sha256": file_sha,
                    "_source_line": line_no,
                    "_parse_error": str(e),
                }
                continue
            obj.setdefault("_source_file", str(p.relative_to(REPO_ROOT)))
            obj.setdefault("_source_tag", source_tag)
            obj.setdefault("_source_file_sha256", file_sha)
            obj.setdefault("_source_line", line_no)
            obj.setdefault("_row_sha256_12", _sha256_bytes(s)[:12])
            obj.setdefault("_rebuttal", False)
            yield obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    ap.add_argument("--out", default=str(
        REPO_ROOT / "runs/corl2026_rebuttal/results/canonical_ledger.jsonl"))
    ap.add_argument("--summary", default=str(
        REPO_ROOT / "runs/corl2026_rebuttal/results/canonical_ledger_summary.json"))
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    summary: dict = {"sources": [], "total_records": 0, "parse_errors": 0}

    with out.open("w") as f:
        for rel, tag in SOURCES:
            p = root / rel
            if not p.exists():
                summary["sources"].append(
                    dict(path=rel, tag=tag, status="MISSING", records=0))
                continue
            sha = _sha256_file(p)
            n = 0
            errs = 0
            for rec in _stream_jsonl(p, tag, sha):
                f.write(json.dumps(rec) + "\n")
                n += 1
                if "_parse_error" in rec:
                    errs += 1
            summary["sources"].append(
                dict(path=rel, tag=tag, status="OK",
                     records=n, parse_errors=errs, sha256=sha))
            summary["total_records"] += n
            summary["parse_errors"] += errs

    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    print(f"  {summary['total_records']} records "
          f"from {sum(1 for s in summary['sources'] if s['status']=='OK')} "
          f"sources ({summary['parse_errors']} parse errors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
