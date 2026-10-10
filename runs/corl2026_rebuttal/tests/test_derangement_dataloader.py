"""End-to-end torch-guarded test for the derangement dataloader wiring.

Verifies:
  * enumerate_episodes_per_task matches user's test_ood split counts;
  * build_plan_and_action_source produces a callable whose output shape
    matches the requested window and whose values come from ANOTHER
    episode of the same task;
  * LiberoVideoClips with action_source_override yields actions from a
    different episode than its own.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import h5py  # noqa: F401
    import numpy as np
    HAVE_H5PY = True
except Exception:
    HAVE_H5PY = False

TEST_OOD = REPO_ROOT / "runs/corl2026_rebuttal/splits/test_ood.json"
DATASETS_DIR = REPO_ROOT / "datasets/libero_downloaded"

REQUIRES_DATA = TEST_OOD.exists() and DATASETS_DIR.exists()


@unittest.skipUnless(HAVE_H5PY and REQUIRES_DATA,
                     "requires h5py + downloaded LIBERO datasets")
class TestDerangementDataloader(unittest.TestCase):

    def test_enumerate_matches_split_counts(self) -> None:
        from rebuttal.derangement_dataloader import enumerate_episodes_per_task
        per_task = enumerate_episodes_per_task(str(TEST_OOD))
        # test_ood.json has 26 tasks with 50 demos each → 1300 total
        total_demos = sum(len(v) for v in per_task.values())
        self.assertEqual(total_demos, 1300)
        self.assertEqual(len(per_task), 26)
        for task, demos in per_task.items():
            self.assertEqual(len(demos), 50, f"task {task} has {len(demos)} demos, expected 50")

    def test_action_source_from_different_episode(self) -> None:
        from rebuttal.derangement_dataloader import build_plan_and_action_source
        plan, action_source, bank = build_plan_and_action_source(
            run_seed=0, split_path=str(TEST_OOD), datasets_dir=str(DATASETS_DIR),
        )
        # Pick the first non-empty task in the plan
        task_key = next(iter(plan.per_task))
        own_ep = next(iter(plan.per_task[task_key]))
        src_ep = plan.per_task[task_key][own_ep]
        self.assertNotEqual(own_ep, src_ep, "derangement produced identity map")

        # Sanity: fetch a window and confirm it matches the SOURCE episode,
        # not the OWN episode.
        read_indices = [0, 1, 2, 3, 4]
        got = action_source(task_key, own_ep, read_indices, pad_after=0)
        expected = bank[task_key][src_ep][read_indices]
        np.testing.assert_allclose(got, expected)
        # And it must DIFFER from the own episode's action window
        own = bank[task_key][own_ep][read_indices]
        # With very high probability the two action windows are not identical
        self.assertFalse(np.allclose(got, own),
                         "source-episode actions coincide with own — degenerate")

    def test_libero_video_yields_deranged_actions(self) -> None:
        from data.libero_video import LiberoVideoClips
        from rebuttal.derangement_dataloader import build_plan_and_action_source
        plan, action_source, bank = build_plan_and_action_source(
            run_seed=0, split_path=str(TEST_OOD), datasets_dir=str(DATASETS_DIR),
        )
        ds = LiberoVideoClips(
            split_path=str(TEST_OOD),
            datasets_dir=str(DATASETS_DIR),
            clip_len=8, cameras=["agentview_rgb"], resolution=64,
            seed=0, return_actions=True, return_ids=True,
            action_source_override=action_source,
        )
        # Grab first N samples and verify at least one has non-matching actions.
        n_diff = 0
        for i in range(min(20, len(ds))):
            clip, actions, suite, file, demo = ds[i]
            self.assertEqual(actions.shape[0], 8)
            self.assertEqual(actions.shape[1], 7)
            # own actions — fetch fresh
            with __import__("h5py").File(
                str(DATASETS_DIR / suite / file), "r") as f:
                own = np.asarray(f["data"][demo]["actions"][:8], dtype=np.float32)
                if own.shape[0] < 8:
                    own = np.concatenate([own,
                                          np.repeat(own[-1:], 8 - own.shape[0], axis=0)])
            if not np.allclose(actions.numpy(), own):
                n_diff += 1
        # At least half should differ (derangement guarantees per-task
        # bijection, so mismatch is the rule not the exception).
        self.assertGreater(n_diff, 10,
                           f"only {n_diff}/20 samples had deranged actions — control may be leaking")


if __name__ == "__main__":
    unittest.main()
