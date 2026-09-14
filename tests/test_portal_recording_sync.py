"""Portal recording join: exact timestamp match, drop-oldest, Hz floor."""
from __future__ import annotations

import logging
import os
import sys
import unittest

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from teleop.robot_control.portal_mapping import UnpackedAction
from teleop.utils.portal_recording import (
    MIN_HZ,
    TARGET_HZ,
    MatchedObservation,
    PortalRecordingJoin,
    RecordingRateGuard,
    frames_to_colors,
    joined_to_episode,
)


class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def _obs(ts: int, arm=None) -> MatchedObservation:
    frame = np.zeros((4, 8, 3), dtype=np.uint8)
    frame[:, :4] = 10
    frame[:, 4:] = 20
    return MatchedObservation(
        timestamp_us=ts,
        frames_bgr={"head_camera": frame},
        arm_q=np.arange(14, dtype=np.float64) if arm is None else arm,
        hand_q=np.linspace(0.0, 1.0, 14),
        fsm_id=1,
    )


def _act(reply_ts: int, arm=None) -> UnpackedAction:
    return UnpackedAction(
        arm_q=np.ones(14) if arm is None else arm,
        hand_q=np.zeros(14),
        vx=0.1,
        vy=-0.2,
        vyaw=0.3,
        fsm_id=1,
        timestamp_us=reply_ts + 5,
        in_reply_to_ts_us=reply_ts,
    )


class TestPortalRecordingJoin(unittest.TestCase):
    def setUp(self):
        self.joined = []
        self.join = PortalRecordingJoin(on_joined=self.joined.append)
        self.join.set_enabled(True)

    def test_exact_timestamp_emits_once(self):
        self.join.on_observation(_obs(100))
        self.assertEqual(self.joined, [])
        self.join.on_action(_act(100))
        self.assertEqual(len(self.joined), 1)
        sample = self.joined[0]
        self.assertEqual(sample.timestamp_us, 100)
        self.assertEqual(sample.in_reply_to_ts_us, 100)
        self.assertEqual(self.join.consume_counts()["joined"], 1)

    def test_action_before_observation(self):
        self.join.on_action(_act(50))
        self.assertEqual(self.joined, [])
        self.join.on_observation(_obs(50))
        self.assertEqual(len(self.joined), 1)

    def test_no_emit_without_matching_action(self):
        self.join.on_observation(_obs(1))
        self.join.on_action(_act(2))
        self.assertEqual(self.joined, [])

    def test_disabled_drops_everything(self):
        self.join.set_enabled(False)
        self.join.on_observation(_obs(7))
        self.join.on_action(_act(7))
        self.assertEqual(self.joined, [])

    def test_newer_observation_drops_oldest(self):
        self.join.on_observation(_obs(10))
        self.join.on_observation(_obs(11))
        self.join.on_action(_act(10))
        self.assertEqual(self.joined, [])
        self.join.on_action(_act(11))
        self.assertEqual(len(self.joined), 1)
        self.assertEqual(self.joined[0].timestamp_us, 11)
        self.assertGreaterEqual(self.join.consume_counts()["join_miss"], 1)

    def test_no_arm_q_does_not_emit(self):
        obs = _obs(3)
        obs.arm_q = None
        self.join.on_observation(obs)
        self.join.on_action(_act(3))
        self.assertEqual(self.joined, [])


class TestEpisodePayload(unittest.TestCase):
    def test_timestamps_and_loco(self):
        joined = []
        join = PortalRecordingJoin(on_joined=joined.append)
        join.set_enabled(True)
        join.on_observation(_obs(42))
        join.on_action(_act(42))
        payload = joined_to_episode(
            joined[0],
            declared_videos=["head_camera"],
            binocular=True,
            include_loco=True,
        )
        self.assertEqual(payload["timestamp_us"], 42)
        self.assertEqual(payload["in_reply_to_ts_us"], 42)
        self.assertEqual(payload["actions"]["body"]["qpos"], [0.1, -0.2, 0.3])
        self.assertIn("color_0", payload["colors"])
        self.assertIn("color_1", payload["colors"])
        self.assertEqual(int(payload["colors"]["color_0"][0, 0, 0]), 10)
        self.assertEqual(int(payload["colors"]["color_1"][0, 0, 0]), 20)

    def test_frames_to_colors_mono_then_wrist(self):
        head = np.zeros((2, 2, 3), dtype=np.uint8)
        wrist = np.ones((2, 2, 3), dtype=np.uint8)
        colors = frames_to_colors(
            {"head_camera": head, "left_wrist": wrist},
            ["head_camera", "left_wrist"],
            binocular=False,
        )
        self.assertIn("color_0", colors)
        self.assertIn("color_1", colors)
        np.testing.assert_array_equal(colors["color_1"], wrist)


class TestRecordingRateGuard(unittest.TestCase):
    def test_below_min_sets_flag(self):
        clock = FakeClock(0.0)
        log = logging.getLogger("test.record.rate")
        guard = RecordingRateGuard(
            log, target_hz=TARGET_HZ, min_hz=MIN_HZ, window_s=1.0, clock=clock)
        for _ in range(5):
            self.assertIsNone(guard.mark())
        clock.t = 1.0
        hz = guard.mark()
        self.assertIsNotNone(hz)
        self.assertLess(hz, MIN_HZ)
        self.assertTrue(guard.below_min)

    def test_target_rate_is_ok(self):
        clock = FakeClock(0.0)
        log = logging.getLogger("test.record.rate")
        guard = RecordingRateGuard(
            log, target_hz=TARGET_HZ, min_hz=MIN_HZ, window_s=1.0, clock=clock)
        for _ in range(29):
            guard.mark()
        clock.t = 1.0
        hz = guard.mark()
        self.assertGreaterEqual(hz, MIN_HZ)
        self.assertFalse(guard.below_min)


if __name__ == "__main__":
    unittest.main()
