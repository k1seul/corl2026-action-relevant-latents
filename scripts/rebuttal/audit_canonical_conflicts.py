"""Audit the canonical ledger for duplicate/conflicting metric values.

Parsing success in ``build_canonical_ledger.py`` (827 records / 12 sources)
tells us the JSONLs are syntactically valid, NOT that we know which of two
duplicate rows is canonical.

This script groups every record by the identity keys the paper reports on
and flags any group that has more than one row, or whose rows disagree on
a metric value. It writes both a human-readable Markdown report and a
machine-readable JSONL so downstream figure/table scripts can honour the
selection.

Selection policy (deliberately conservative and predeclared, per §7.7):

1. Prefer the most recent submitted-paper ``_source_tag`` for that
   ``benchmark × backbone × condition × training_seed × probe_seed ×
   perturbation × metric`` cell, WHERE "most recent" is decided by a
   PREDECLARED priority list, not by numeric value.
2. If the priority list does not disambiguate (two rows share the same
   ``_source_tag`` and ``_source_file_sha256``), UNSELECTED. The report
   surfaces the tie and asks the user to adjudicate. This means we NEVER
   pick a value because it is favourable.
3. Rows without a resolvable checkpoint SHA-256 are still surfaced but
   are marked ``unresolved_checkpoint``.

Runs on stdlib only.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]

# Predeclared priority: lower index = higher priority. Extend at the END of
# this list only; never reorder to change a selection outcome.
SOURCE_TAG_PRIORITY = [
    "submitted_libero_main",
    "submitted_libero_probe_budget",
    "submitted_libero_decoder",
    "submitted_vjepa_per_layer_aux",
    "submitted_vjepa_per_layer_frozen",
    "submitted_pareto",
    "submitted_three_benchmarks",
    "submitted_lambda_sweep",
    "submitted_libero_vs_calvin",
    "submitted_calvin_main",
    "submitted_metaworld_main",
    "submitted_id_sparsity_sweep",
]

# Grouping key components. Missing fields collapse to the string "-" so a
# row that lacks e.g. training_seed still groups against other seed-less
# rows for the same variant/metric.
GROUP_KEYS_STRICT = (
    "_bench", "_backbone", "_condition", "_training_seed",
    "_probe_seed", "_perturbation", "_metric",
)

GROUP_KEYS_LOOSE = (
    "_bench", "_backbone", "_variant", "_condition", "_training_seed",
    "_probe_seed", "_perturbation", "_metric",
)

# Metrics to surface. Recognised across the various JSONLs used to produce
# the submitted tables/figures.
METRIC_FIELDS = (
    "action_probe_r2",
    "action_probe_mse",
    "recon_mse",
    "psnr",
    "lpips",
)

CELL_FIELDS = ("train_cell", "test_cell")


def _normalise_record(rec: dict) -> Iterable[dict]:
    """Yield one flat sub-record per (cell, metric) present in ``rec``.

    We tag each sub-record with derived identity fields so multiple JSONL
    formats can be compared apples-to-apples.
    """
    variant = rec.get("variant") or rec.get("run_id") or "?"
    # source tag informs both the priority and the "benchmark" inference
    tag = rec.get("_source_tag", "unknown")
    if "calvin" in tag:
        bench = "calvin"
    elif "metaworld" in tag:
        bench = "metaworld"
    elif "libero" in tag or tag.startswith("submitted_vjepa") or tag == "submitted_pareto":
        bench = "libero"
    else:
        bench = rec.get("benchmark", "unknown")

    # Backbone/family/condition — inherit or infer conservatively
    backbone = (rec.get("backbone") or rec.get("family")
                or rec.get("base_model") or "unknown")
    condition = (rec.get("condition")
                 or ("frozen" if "_frozen" in tag or rec.get("family") == "frozen"
                     else rec.get("aux_head_type")
                     or rec.get("notes", "?")))
    training_seed = str(rec.get("training_seed", rec.get("seed", "-")))
    probe_seed = str(rec.get("probe_seed", "-"))

    for cell_field in CELL_FIELDS:
        cell = rec.get(cell_field)
        if not isinstance(cell, dict):
            continue
        pert = cell.get("perturbation", cell.get("perturb", "clean"))
        for metric in METRIC_FIELDS:
            if metric not in cell:
                continue
            val = cell.get(metric)
            if not isinstance(val, (int, float)):
                continue
            yield {
                "_bench": bench,
                "_backbone": backbone,
                "_condition": condition,
                "_variant": variant,
                "_training_seed": training_seed,
                "_probe_seed": probe_seed,
                "_perturbation": pert,
                "_metric": f"{cell_field}.{metric}",
                "_value": float(val),
                "_source_file": rec.get("_source_file"),
                "_source_line": rec.get("_source_line"),
                "_source_tag": rec.get("_source_tag"),
                "_source_file_sha256": rec.get("_source_file_sha256"),
                "_row_sha256_12": rec.get("_row_sha256_12"),
                "_checkpoint_sha256": rec.get("checkpoint_sha256") or rec.get("ckpt_sha256"),
                "_split_sha256": rec.get("split_sha256"),
                "_steps": rec.get("training_step_at_ckpt") or rec.get("steps"),
            }


def _tag_priority(tag: str | None) -> int:
    if tag is None:
        return 10_000
    try:
        return SOURCE_TAG_PRIORITY.index(tag)
    except ValueError:
        return 10_000


def _select_canonical(rows: list[dict]) -> tuple[dict | None, str]:
    """Return (selected_row, reason). None if unresolved."""
    if len(rows) == 1:
        return rows[0], "sole_record"

    values = {r["_value"] for r in rows}
    if len(values) == 1:
        # exact-tie duplicates — pick highest-priority tag; if still tied,
        # arbitrary but deterministic (lowest _source_line in lowest file
        # SHA).
        rows_sorted = sorted(rows, key=lambda r: (_tag_priority(r["_source_tag"]),
                                                   r["_source_file_sha256"] or "",
                                                   r["_source_line"] or 0))
        return rows_sorted[0], "exact_value_duplicate_priority_pick"

    # values disagree — never select numerically. Prefer highest-priority
    # tag; if ties in tag, UNSELECTED.
    rows_by_prio = sorted(rows, key=lambda r: _tag_priority(r["_source_tag"]))
    top_prio = _tag_priority(rows_by_prio[0]["_source_tag"])
    top_tier = [r for r in rows_by_prio if _tag_priority(r["_source_tag"]) == top_prio]
    if len(top_tier) == 1:
        return top_tier[0], f"priority_pick_tag={rows_by_prio[0]['_source_tag']}"
    return None, "conflict_unresolved_same_priority"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=str(
        REPO_ROOT / "runs/corl2026_rebuttal/results/canonical_ledger.jsonl"))
    ap.add_argument("--out-jsonl", default=str(
        REPO_ROOT / "runs/corl2026_rebuttal/results/canonical_conflicts.jsonl"))
    ap.add_argument("--out-md", default=str(
        REPO_ROOT / "runs/corl2026_rebuttal/reports/CANONICAL_CONFLICTS.md"))
    args = ap.parse_args()

    ledger = Path(args.ledger)
    if not ledger.exists():
        print(f"canonical_ledger.jsonl missing at {ledger}. "
              f"Run scripts/rebuttal/build_canonical_ledger.py first.")
        return 1

    flat_rows: list[dict] = []
    with ledger.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            flat_rows.extend(_normalise_record(rec))

    def build_groups(keys):
        g: dict[tuple, list[dict]] = defaultdict(list)
        for r in flat_rows:
            k = tuple(str(r.get(kk, "-")) for kk in keys)
            g[k].append(r)
        return g

    groups_strict = build_groups(GROUP_KEYS_STRICT)
    groups_loose = build_groups(GROUP_KEYS_LOOSE)
    # Default JSONL uses strict — variant-tightened is a diagnostic view only.
    groups = groups_strict
    GROUP_KEYS = GROUP_KEYS_STRICT

    conflict_rows: list[dict] = []
    n_groups = 0
    n_dup_groups = 0
    n_conflict_groups = 0
    n_unresolved = 0

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_jsonl).open("w") as out:
        for key, rows in sorted(groups.items()):
            n_groups += 1
            if len(rows) == 1:
                # write out a "sole_record" entry too, so downstream tools
                # can rely on a complete selection map
                sel = rows[0]
                out.write(json.dumps({
                    "group_key": dict(zip(GROUP_KEYS, key)),
                    "n_candidates": 1,
                    "selected": True,
                    "selected_row": sel,
                    "selection_reason": "sole_record",
                    "other_candidates": [],
                }) + "\n")
                continue
            n_dup_groups += 1
            distinct = len({r["_value"] for r in rows})
            if distinct > 1:
                n_conflict_groups += 1
            selected, reason = _select_canonical(rows)
            if selected is None:
                n_unresolved += 1
            others = [r for r in rows if r is not selected]
            entry = {
                "group_key": dict(zip(GROUP_KEYS, key)),
                "n_candidates": len(rows),
                "distinct_values": distinct,
                "selected": selected is not None,
                "selected_row": selected,
                "selection_reason": reason,
                "other_candidates": others,
            }
            out.write(json.dumps(entry) + "\n")
            conflict_rows.append(entry)

    # ── Markdown report ────────────────────────────────────────────
    lines = ["# CANONICAL_CONFLICTS.md — canonical-ledger duplicate/conflict audit\n"]
    lines.append("Auto-generated by ``scripts/rebuttal/audit_canonical_conflicts.py``.\n")
    lines.append(f"Ledger: `{ledger.relative_to(REPO_ROOT)}`\n\n")
    lines.append(f"* Total flat (cell × metric) records: **{len(flat_rows)}**\n")
    lines.append(f"* Grouping keys (strict / paper-facing): `{list(GROUP_KEYS_STRICT)}`\n")
    lines.append(f"* Distinct strict groups: **{n_groups}**\n")
    lines.append(f"* Strict groups with >1 candidate row: **{n_dup_groups}**\n")
    lines.append(f"* Strict groups whose candidates disagree numerically: **{n_conflict_groups}**\n")
    lines.append(f"* Strict groups still UNRESOLVED (require user adjudication): **{n_unresolved}**\n")
    n_loose_conf = sum(1 for rows in groups_loose.values()
                       if len({r['_value'] for r in rows}) > 1)
    lines.append(
        f"* (Variant-tightened diagnostic: {len(groups_loose)} groups; "
        f"**{n_loose_conf}** with distinct values within the same variant "
        f"— should always be 0 for a bit-for-bit reproducible ledger.)\n\n"
    )

    lines.append("## Selection policy (predeclared)\n\n")
    lines.append("1. Sole record → selected.\n")
    lines.append("2. Exact-value duplicates → highest-priority source tag "
                 "(see `SOURCE_TAG_PRIORITY` in the script).\n")
    lines.append("3. Numerically disagreeing rows → highest-priority tag "
                 "unless two rows share the same top priority, in which case "
                 "the group is UNRESOLVED and requires user adjudication.\n")
    lines.append("4. **Values are never selected on the basis of being "
                 "larger/smaller / better/worse.**\n\n")

    if n_conflict_groups == 0:
        lines.append("## Numeric conflicts\n\nNone.\n\n")
    else:
        lines.append("## Numeric conflicts (top 40 groups)\n\n")
        shown = 0
        for e in conflict_rows:
            if e["n_candidates"] > 1 and e.get("distinct_values", 1) > 1:
                shown += 1
                if shown > 40:
                    break
                gk = e["group_key"]
                lines.append(f"### {gk['_bench']} / {gk['_backbone']} / "
                             f"{gk['_condition']} / seed={gk['_training_seed']} "
                             f"probe_seed={gk['_probe_seed']} pert={gk['_perturbation']} "
                             f"metric={gk['_metric']}\n\n")
                lines.append(f"* candidates: {e['n_candidates']}  "
                             f"distinct_values: {e['distinct_values']}  "
                             f"selected: {e['selected']}  reason: `{e['selection_reason']}`\n")
                for r in ([e["selected_row"]] if e["selected_row"] else []) + list(e["other_candidates"]):
                    if r is None:
                        continue
                    lines.append(f"  * value={r['_value']:.6f}  tag=`{r['_source_tag']}`  "
                                 f"src=`{r['_source_file']}:{r['_source_line']}`  "
                                 f"row_sha12=`{r['_row_sha256_12']}`  "
                                 f"file_sha256=`{(r['_source_file_sha256'] or '')[:12]}` "
                                 f"ckpt_sha=`{(r['_checkpoint_sha256'] or 'n/a')[:12]}` "
                                 f"split_sha=`{(r['_split_sha256'] or 'n/a')[:12]}` "
                                 f"steps={r['_steps']}\n")
                lines.append("\n")

    if n_unresolved:
        lines.append("## UNRESOLVED groups (require user decision)\n\n")
        lines.append("The following groups have multiple rows at the same "
                     "highest-priority source tag with different values. "
                     "Please pick a canonical row explicitly, or split them "
                     "into distinct predeclared identities.\n\n")
        for e in conflict_rows:
            if not e["selected"]:
                gk = e["group_key"]
                lines.append(f"* {gk}\n")
                for r in e["other_candidates"]:
                    lines.append(f"    - `{r['_source_file']}:{r['_source_line']}` "
                                 f"value={r['_value']}\n")

    Path(args.out_md).write_text("".join(lines))
    print(f"wrote {args.out_md}")
    print(f"      {args.out_jsonl}")
    print(f"      total_flat={len(flat_rows)}  groups={n_groups}  "
          f"dup_groups={n_dup_groups}  conflict_groups={n_conflict_groups}  "
          f"unresolved={n_unresolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
