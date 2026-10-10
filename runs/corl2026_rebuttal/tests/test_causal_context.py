"""Stdlib-only tests for the past-only causal-context window construction.

Covers:
  * every window ends exactly at t;
  * no window contains a future frame;
  * left-padding is by repetition of index 0;
  * chunk-causal windows tile the episode without overlap;
  * the ``assert_no_future`` helper actually raises on a bad window.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rebuttal.causal_context import (  # noqa: E402
    assert_no_future,
    build_causal_windows,
    build_chunk_causal_windows,
)


class TestPerTimestepCausalWindows(unittest.TestCase):

    def test_window_ends_at_t(self) -> None:
        w = build_causal_windows(num_frames=10, clip_len=4)
        for t, window in enumerate(w):
            self.assertEqual(window[-1], t)

    def test_no_future_frames(self) -> None:
        w = build_causal_windows(num_frames=32, clip_len=8)
        assert_no_future(w)   # must not raise

    def test_left_padding_by_repeat_of_first_frame(self) -> None:
        w = build_causal_windows(num_frames=6, clip_len=4)
        self.assertEqual(w[0], [0, 0, 0, 0])
        self.assertEqual(w[1], [0, 0, 0, 1])
        self.assertEqual(w[2], [0, 0, 1, 2])
        self.assertEqual(w[3], [0, 1, 2, 3])
        self.assertEqual(w[4], [1, 2, 3, 4])
        self.assertEqual(w[5], [2, 3, 4, 5])

    def test_window_length_is_always_clip_len(self) -> None:
        clip_len = 8
        w = build_causal_windows(num_frames=32, clip_len=clip_len)
        for window in w:
            self.assertEqual(len(window), clip_len)

    def test_rejects_nonpositive_clip_len(self) -> None:
        with self.assertRaises(ValueError):
            build_causal_windows(num_frames=10, clip_len=0)


class TestChunkCausalWindows(unittest.TestCase):

    def test_windows_tile_episode(self) -> None:
        w = build_chunk_causal_windows(num_frames=16, clip_len=8)
        self.assertEqual(len(w), 2)
        self.assertEqual(w[0], list(range(8)))
        self.assertEqual(w[1], list(range(8, 16)))

    def test_final_short_window_is_left_padded(self) -> None:
        w = build_chunk_causal_windows(num_frames=10, clip_len=4)
        # windows: [0..3], [4..7], [8..9] left-padded to length 4
        self.assertEqual(w[-1], [8, 8, 8, 9])


class TestAssertNoFuture(unittest.TestCase):

    def test_raises_on_future_frame(self) -> None:
        bad = [[0, 1, 5]]   # t=0 but window includes frame 5
        with self.assertRaises(AssertionError):
            assert_no_future(bad)


if __name__ == "__main__":
    unittest.main()
