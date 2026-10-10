"""Stdlib-only tests for the within-task cross-episode derangement.

Covers guarantees D1..D7 from ``src/rebuttal/derangement.py``.
"""
from __future__ import annotations

import statistics
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rebuttal.derangement import (  # noqa: E402
    DerangementPlan,
    build_derangement_plan,
    mapping_hash,
    within_task_derangement,
)


class TestDerangement(unittest.TestCase):

    def _sample_task(self, n: int) -> list[str]:
        return [f"ep_{i:04d}" for i in range(n)]

    # D1
    def test_no_source_maps_to_itself(self) -> None:
        for n in (2, 3, 5, 20, 100):
            m = within_task_derangement(0, "taskA", self._sample_task(n))
            for src, tgt in m.items():
                self.assertNotEqual(src, tgt, msg=f"n={n} src={src} tgt={tgt}")

    # D2
    def test_bijection(self) -> None:
        for n in (2, 3, 5, 20, 100):
            eps = self._sample_task(n)
            m = within_task_derangement(0, "taskA", eps)
            self.assertEqual(sorted(m.keys()), sorted(eps))
            self.assertEqual(sorted(m.values()), sorted(eps))

    # D4 — action multiset preserved (task-level bijection)
    def test_action_multiset_preserved_per_task(self) -> None:
        # simulate deterministic per-episode "action" as a single float
        n = 30
        eps = self._sample_task(n)
        actions_by_ep = {e: hash(e) % 997 / 997.0 for e in eps}
        m = within_task_derangement(0, "taskA", eps)
        original = sorted(actions_by_ep.values())
        deranged = sorted(actions_by_ep[m[e]] for e in eps)
        self.assertEqual(original, deranged,
                         "task-level action multiset changed under derangement")
        # mean/std also identical
        self.assertAlmostEqual(statistics.fmean(original),
                                statistics.fmean(deranged), places=12)

    # D5 + D6
    def test_deterministic_and_seed_sensitive(self) -> None:
        eps = self._sample_task(50)
        m0a = within_task_derangement(7, "taskA", eps)
        m0b = within_task_derangement(7, "taskA", eps)
        self.assertEqual(m0a, m0b, "same seed produced different derangements")
        m1 = within_task_derangement(11, "taskA", eps)
        diffs = sum(1 for e in eps if m0a[e] != m1[e])
        self.assertGreater(diffs, 40,
                           f"a different seed should change most mappings; only {diffs}/50 differed")

    # D7 — alignment destroyed. We assess this via a proxy: build a
    # deterministic per-episode "action_at_step_0". Under derangement, the
    # ordering of action[0] across episodes ceases to match the ordering of
    # z[0] across episodes.
    def test_alignment_destroyed(self) -> None:
        n = 40
        eps = self._sample_task(n)
        m = within_task_derangement(0, "taskA", eps)
        # correlation of index positions under identity vs derangement
        original_positions = {e: i for i, e in enumerate(eps)}
        deranged_positions = {e: original_positions[m[e]] for e in eps}
        # count how many episodes see a positional shift ≥ 2
        shifted = sum(1 for e in eps
                      if abs(original_positions[e] - deranged_positions[e]) >= 2)
        self.assertGreater(shifted, n // 2,
                           f"only {shifted}/{n} episodes moved ≥2 positions — "
                           "derangement is not sufficiently mixing.")

    def test_singleton_task_returns_empty_mapping(self) -> None:
        self.assertEqual(within_task_derangement(0, "solo", ["only_ep"]), {})

    def test_pair_task_still_derangeable(self) -> None:
        m = within_task_derangement(0, "pair", ["e0", "e1"])
        self.assertEqual(m, {"e0": "e1", "e1": "e0"})

    def test_mapping_hash_is_stable_and_seed_sensitive(self) -> None:
        eps = self._sample_task(30)
        h_a = mapping_hash(within_task_derangement(0, "T", eps))
        h_b = mapping_hash(within_task_derangement(0, "T", eps))
        self.assertEqual(h_a, h_b)
        h_c = mapping_hash(within_task_derangement(1, "T", eps))
        self.assertNotEqual(h_a, h_c)


class TestPlan(unittest.TestCase):

    def test_plan_covers_multi_task_and_records_singletons(self) -> None:
        episodes_per_task = {
            "task_A": [f"a_{i}" for i in range(4)],
            "task_B": [f"b_{i}" for i in range(6)],
            "task_C": ["c_only"],   # singleton, must be skipped, not deranged into another task
        }
        plan = build_derangement_plan(run_seed=42, episodes_per_task=episodes_per_task)
        self.assertIsInstance(plan, DerangementPlan)
        self.assertIn("task_A", plan.per_task)
        self.assertIn("task_B", plan.per_task)
        self.assertNotIn("task_C", plan.per_task)
        self.assertEqual(plan.skipped_singleton_tasks, ["task_C"])
        # No cross-task leakage: task_A's mapping values are all in task_A's episodes
        for task, m in plan.per_task.items():
            src_pool = set(episodes_per_task[task])
            self.assertTrue(set(m.values()).issubset(src_pool),
                            f"task {task} mapping crossed task boundaries: {m}")
            self.assertTrue(set(m.keys()).issubset(src_pool))

    def test_plan_hash_changes_with_seed(self) -> None:
        eps = {"T": [f"e{i}" for i in range(10)]}
        p0 = build_derangement_plan(0, eps)
        p1 = build_derangement_plan(1, eps)
        self.assertNotEqual(p0.global_hash(), p1.global_hash())


if __name__ == "__main__":
    unittest.main()
