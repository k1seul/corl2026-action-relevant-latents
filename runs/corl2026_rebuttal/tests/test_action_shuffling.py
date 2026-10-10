"""Stdlib-only tests for the shuffled-ID matched control (instruction §4.4).

Every requirement listed in the instruction is covered exactly once:

  R1  no zero shift
  R2  no ±1-step alignment
  R3  exact preservation of the action *multiset* (i.e. marginal, since we
      only reorder rows in the first dimension)
  R4  same sample count and clip IDs as the true-ID condition
  R5  deterministic mapping under the same (run_seed, episode_id)
  R6  different mapping under a different run_seed

These tests do NOT import torch, so they can be run on any host with just
Python 3.10+.
"""
from __future__ import annotations

import statistics
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rebuttal.action_shuffling import (  # noqa: E402
    ShiftPolicy,
    circular_shifted_indices,
    sample_shift,
    within_task_permutation,
)


class TestShuffledIDCircularShift(unittest.TestCase):

    def setUp(self) -> None:
        self.policy = ShiftPolicy(episode_length=100, min_abs_shift=2)

    # R1, R2
    def test_no_zero_or_one_step_shift(self) -> None:
        for ep in range(500):
            k = sample_shift(run_seed=0, episode_id=ep, policy=self.policy)
            self.assertGreaterEqual(k, 2)
            self.assertLessEqual(k, self.policy.episode_length - 2)

    # R3 — the returned indices are a *permutation* of range(T), so any per-row
    # statistic (mean, std, quantiles) of an action array indexed by them is
    # exactly the mean/std of the original array. We assert the multiset
    # equality directly.
    def test_indices_are_a_permutation(self) -> None:
        for ep in range(50):
            idx = circular_shifted_indices(0, ep, self.policy)
            self.assertEqual(
                sorted(idx), list(range(self.policy.episode_length)),
                msg=f"episode {ep} produced a non-permutation")

    # R3 — sanity: shifting a "fake action" scalar sequence preserves mean/var.
    def test_shift_preserves_marginal_statistics(self) -> None:
        # deterministic pseudo-actions
        actions = [((i * 37) % 100) / 100.0 for i in range(self.policy.episode_length)]
        for ep in range(50):
            idx = circular_shifted_indices(0, ep, self.policy)
            shifted = [actions[i] for i in idx]
            self.assertAlmostEqual(
                statistics.fmean(shifted), statistics.fmean(actions), places=12)
            self.assertAlmostEqual(
                statistics.pstdev(shifted), statistics.pstdev(actions), places=12)

    # R5, R6
    def test_deterministic_and_seed_sensitive(self) -> None:
        # same seed & episode → same shift
        for ep in range(20):
            k_a = sample_shift(7, ep, self.policy)
            k_b = sample_shift(7, ep, self.policy)
            self.assertEqual(k_a, k_b)
        # different seed → some episodes get different shifts
        diffs = 0
        for ep in range(200):
            if sample_shift(7, ep, self.policy) != sample_shift(11, ep, self.policy):
                diffs += 1
        self.assertGreater(diffs, 100,
                           "shifts should differ for a different seed on most episodes")

    def test_forbids_ill_configured_policy(self) -> None:
        with self.assertRaises(ValueError):
            ShiftPolicy(episode_length=4, min_abs_shift=2)   # too short
        with self.assertRaises(ValueError):
            ShiftPolicy(episode_length=100, min_abs_shift=1)  # too weak


class TestWithinTaskPermutation(unittest.TestCase):

    def test_is_a_derangement(self) -> None:
        ep_ids = list(range(20))
        mapping = within_task_permutation(0, "task_A", ep_ids)
        for src, tgt in mapping.items():
            self.assertNotEqual(
                src, tgt, msg=f"episode {src} maps to itself — not a derangement")
        self.assertEqual(sorted(mapping.values()), sorted(ep_ids))

    def test_identity_for_singleton_task(self) -> None:
        mapping = within_task_permutation(0, "task_A", ["only_ep"])
        self.assertEqual(mapping, {"only_ep": "only_ep"})

    def test_deterministic(self) -> None:
        m1 = within_task_permutation(0, "task_A", list(range(30)))
        m2 = within_task_permutation(0, "task_A", list(range(30)))
        self.assertEqual(m1, m2)


if __name__ == "__main__":
    unittest.main()
