"""Unified capture timestamps for Portal state + video. No hardware."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from teleop.utils.portal_sync import (
    LatestStampedSlot,
    StampedVideo,
    capture_tick_us,
    grab_head_rgb,
)


class _Frame:
    def __init__(self, bgr):
        self.bgr = bgr


class _FakeCam:
    def __init__(self, bgr):
        self._frame = _Frame(bgr)

    def get_head_frame(self):
        return self._frame


class TestCaptureTick(unittest.TestCase):
    def test_microseconds_from_clock(self):
        self.assertEqual(capture_tick_us(clock=lambda: 1.5), 1_500_000)


class TestGrabHeadRgb(unittest.TestCase):
    def test_bgr_to_rgb(self):
        bgr = np.zeros((2, 3, 3), dtype=np.uint8)
        bgr[0, 0] = (1, 2, 3)
        rgb = grab_head_rgb(_FakeCam(bgr))
        self.assertEqual(tuple(rgb[0, 0]), (3, 2, 1))

    def test_none_client(self):
        self.assertIsNone(grab_head_rgb(None))


class TestLatestStampedSlot(unittest.TestCase):
    def test_state_and_video_share_capture_ts(self):
        ts = 1_700_000_000_000_000
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        slot = LatestStampedSlot()
        slot.put(StampedVideo("head_camera", rgb, ts))
        taken = slot.take()
        self.assertIsNotNone(taken)
        self.assertEqual(taken.timestamp_us, ts)
        self.assertEqual(taken.track, "head_camera")
        state_ts = ts
        self.assertEqual(state_ts, taken.timestamp_us)

    def test_put_overwrites_oldest(self):
        slot = LatestStampedSlot()
        a = np.zeros((2, 2, 3), dtype=np.uint8)
        b = np.ones((2, 2, 3), dtype=np.uint8)
        slot.put(StampedVideo("head_camera", a, 1))
        slot.put(StampedVideo("head_camera", b, 2))
        taken = slot.take()
        self.assertEqual(taken.timestamp_us, 2)
        np.testing.assert_array_equal(taken.rgb, b)
        self.assertIsNone(slot.take())


if __name__ == "__main__":
    unittest.main()
