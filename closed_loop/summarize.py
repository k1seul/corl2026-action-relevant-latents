"""Print closed-loop results in the layout of Table 4.

    python closed_loop/summarize.py runs/closed_loop_eval/<run>        # summary.json files from eval.sh
    python closed_loop/summarize.py --csv results/closed_loop_success_by_suite.csv   # the paper's runs

Columns: LIBERO success per suite (perturbation none) and their mean; the five LIBERO-PRO
shifts on LIBERO-10; "All" = mean over the five shifts and four suites (each suite x shift
has 100 episodes, so this equals the pooled success rate).
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import defaultdict

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SHIFTS = ("language", "position", "object", "task", "environment")


def table_row(sr: dict) -> list:
    """sr[(suite, perturbation)] = success rate in %."""
    lib = [sr.get((s, "none")) for s in SUITES]
    pro_l10 = [sr.get(("libero_10", k)) for k in SHIFTS]
    pro_all = [sr.get((s, k)) for s in SUITES for k in SHIFTS]
    mean = lambda xs: (sum(xs) / len(xs)) if xs and all(x is not None for x in xs) else None
    return lib + [mean(lib)] + pro_l10 + [mean(pro_all)]


def fmt(row: list) -> str:
    return " ".join(f"{'--':>6}" if v is None else f"{v:6.1f}" for v in row)


HEADER = " ".join(f"{h:>6}" for h in ["Spat", "Obj", "Goal", "L10", "Avg", "Lang", "Pos", "Obj", "Task", "Env", "All"])


def from_summaries(root: pathlib.Path) -> dict:
    sr = {}
    for p in root.rglob("summary.json"):
        d = json.loads(p.read_text())
        sr[(d["task_suite"], d["perturbation"])] = 100.0 * d["overall_success_rate"]
    return sr


def from_csv(path: pathlib.Path) -> dict:
    by_row = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        by_row[r["row"]][(r["suite"], r["perturbation"])] = 100.0 * int(r["successes"]) / int(r["episodes"])
    return by_row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", help="directory containing summary.json files")
    ap.add_argument("--csv", help="results/closed_loop_success_by_suite.csv")
    a = ap.parse_args()
    print(f"{'':28s}{HEADER}")
    if a.csv:
        for name, sr in from_csv(pathlib.Path(a.csv)).items():
            print(f"{name:28s}{fmt(table_row(sr))}")
    else:
        print(f"{pathlib.Path(a.root).name[:27]:28s}{fmt(table_row(from_summaries(pathlib.Path(a.root))))}")


if __name__ == "__main__":
    main()
