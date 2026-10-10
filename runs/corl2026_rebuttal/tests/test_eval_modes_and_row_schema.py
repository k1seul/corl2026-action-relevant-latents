"""Stdlib-only tests for STEP 5 — three eval-mode separation.

Covers:
  * EvalMode enum values are what downstream regeneration scripts expect;
  * validate_eval_mode rejects unknown strings;
  * separate_modes correctly partitions rows and surfaces unlabeled ones;
  * assert_result_row_schema fails on rows missing eval_mode / split_sha256
    (which would let a causal-context number be silently placed in a
    canonical-offline column).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rebuttal.eval_modes import EvalMode, separate_modes, validate_eval_mode  # noqa: E402
from rebuttal.provenance import assert_result_row_schema  # noqa: E402


class TestEvalModes(unittest.TestCase):

    def test_enum_values(self) -> None:
        self.assertEqual(EvalMode.CANONICAL_OFFLINE.value,
                         "canonical_offline_action_probe")
        self.assertEqual(EvalMode.CAUSAL_CONTEXT.value,
                         "causal_context_action_probe")
        self.assertEqual(EvalMode.CLOSED_LOOP.value,
                         "closed_loop_policy_success")

    def test_validate_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            validate_eval_mode("action_probe")     # legacy vague name

    def test_validate_accepts_all_three(self) -> None:
        for m in EvalMode:
            validate_eval_mode(m.value)   # must not raise

    def test_separate_modes_partitions_and_flags_unlabeled(self) -> None:
        rows = [
            {"eval_mode": "canonical_offline_action_probe", "x": 1},
            {"eval_mode": "causal_context_action_probe", "x": 2},
            {"eval_mode": "closed_loop_policy_success", "x": 3},
            {"x": 4},   # unlabeled
            {"eval_mode": "bogus", "x": 5},   # also unlabeled
        ]
        out = separate_modes(rows)
        self.assertEqual(len(out["canonical_offline_action_probe"]), 1)
        self.assertEqual(len(out["causal_context_action_probe"]), 1)
        self.assertEqual(len(out["closed_loop_policy_success"]), 1)
        self.assertEqual(len(out["__unlabeled__"]), 2)


class TestResultRowSchema(unittest.TestCase):

    def _good(self):
        return {
            "eval_mode": "canonical_offline_action_probe",
            "benchmark": "libero",
            "backbone": "vjepa2",
            "condition": "true_id",
            "training_seed": 0,
            "probe_seed": 0,
            "perturbation": "clean",
            "split_sha256": "42f26dac66ac",
            "action_probe_r2": 0.8532,
        }

    def test_passes_on_full_row(self) -> None:
        assert_result_row_schema(self._good())   # must not raise

    def test_fails_missing_eval_mode(self) -> None:
        r = self._good(); del r["eval_mode"]
        with self.assertRaisesRegex(ValueError, "eval_mode"):
            assert_result_row_schema(r)

    def test_fails_missing_split_sha(self) -> None:
        r = self._good(); del r["split_sha256"]
        with self.assertRaisesRegex(ValueError, "split_sha256"):
            assert_result_row_schema(r)

    def test_fails_on_unknown_eval_mode_value(self) -> None:
        r = self._good(); r["eval_mode"] = "action_probe"
        with self.assertRaises(ValueError):
            assert_result_row_schema(r)


if __name__ == "__main__":
    unittest.main()
