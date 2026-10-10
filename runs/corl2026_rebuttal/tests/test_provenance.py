"""Stdlib-only tests for the provenance recorder.

We do NOT test git detection or nvidia-smi (both environment-dependent);
we test the field-validation contract, which is what protects long runs
from starting without required metadata.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rebuttal import provenance  # noqa: E402


BASE_KWARGS = dict(
    condition="true_id",
    backbone="vjepa2_vitl",
    run_id="test_run_001",
    training_seed=0,
    probe_seed=0,
    optimizer="adamw",
    lr=5e-5,
    lr_schedule="cosine_warmup2000",
    batch_size=4,
    effective_batch=16,
    target_updates=20000,
)


class TestProvenance(unittest.TestCase):

    def test_writes_provenance_json_with_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rec = provenance.record_provenance(Path(td), **BASE_KWARGS)
            for k in provenance.REQUIRED_FIELDS:
                self.assertIn(k, rec)
            saved = json.loads((Path(td) / "provenance.json").read_text())
            self.assertEqual(saved["condition"], "true_id")
            self.assertEqual(saved["target_updates"], 20000)
            self.assertIn("wall_clock_start_unix", saved)

    def test_missing_field_raises(self) -> None:
        kwargs = dict(BASE_KWARGS)
        kwargs["optimizer"] = ""
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                provenance.record_provenance(Path(td), **kwargs)

    def test_finalize_appends_end_and_step(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            provenance.record_provenance(Path(td), **BASE_KWARGS)
            provenance.finalize_provenance(Path(td), final_checkpoint_step=20000)
            saved = json.loads((Path(td) / "provenance.json").read_text())
            self.assertEqual(saved["final_checkpoint_step"], 20000)
            self.assertIn("wall_clock_end_unix", saved)
            self.assertGreaterEqual(saved["wall_clock_elapsed_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
