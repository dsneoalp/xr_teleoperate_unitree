"""Matched obs.frames and unmatched on_video_frame both fill get_head_frame()."""
from __future__ import annotations

import threading

import numpy as np

from teleop.robot_control.obs_record_buffer import RecordingObsBuffer
from teleop.robot_control.portal_mapping import PortalMapping
from teleop.robot_control.portal_operator import PortalTeleopBridge
from teleop.tests.paths import PORTAL_MAPPING, PORTAL_YAML

TRACK = "head_camera"


class _FakeFrame:
    def __init__(self, rgb: np.ndarray, timestamp_us: int = 0):
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w = rgb.shape[:2]
        self.data = rgb.tobytes()
        self.width = w
        self.height = h
        self.timestamp_us = timestamp_us


class _FakeObs:
    def __init__(self, timestamp_us: int, frames: dict | None = None):
        self.timestamp_us = timestamp_us
        self.frames = frames or {}
        self.raw_state = {}
        self.state = {}


def _solid_rgb(r: int, g: int, b: int, h: int = 4, w: int = 4) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = (r, g, b)
    return frame


def _video_bridge() -> PortalTeleopBridge:
    bridge = PortalTeleopBridge.__new__(PortalTeleopBridge)
    bridge._map = PortalMapping(PORTAL_MAPPING, PORTAL_YAML)
    arm_dof = bridge._map.arm_dof
    bridge._obs_lock = threading.Lock()
    bridge._declared_videos = [TRACK]
    bridge._xr_track = TRACK
    bridge._expected_hw = {}
    bridge._frames = {}
    bridge._decode_cache = {}
    bridge._frames_logged = set()
    bridge._last_obs_had_frames = False
    bridge._obs_ts_us = None
    bridge._pending_action_wall = None
    bridge._recording_enabled = False
    bridge._record_pair_cb = None
    bridge._rec_buf = RecordingObsBuffer()
    bridge._state_q = None
    bridge._state_dq = np.zeros(arm_dof)
    bridge._state_ts_wall = 0.0
    bridge._prev_state_q = None
    bridge._prev_state_ts_us = None
    bridge._rtt_cb = None
    bridge._dual_hand_state_array_out = None
    bridge._dual_hand_data_lock = None
    bridge._match_timing = None
    bridge._stamp_log_n = 0
    bridge._framed_obs_n = 0
    bridge._unmatched_idle_logged = False
    bridge._unmatched_seen = False
    bridge._unmatched_logged = False
    bridge._last_metrics_log = 0.0
    return bridge


def test_matched_observation_fills_get_head_frame():
    bridge = _video_bridge()
    rgb = _solid_rgb(10, 20, 30)
    bridge._on_observation(_FakeObs(1001, {TRACK: _FakeFrame(rgb, 1001)}))
    head = bridge.get_head_frame()
    assert head.bgr is not None
    np.testing.assert_array_equal(head.bgr[0, 0], (30, 20, 10))
    assert head.timestamp_us == 1001
    assert bridge.last_obs_had_frames() is True


def test_unmatched_video_frame_still_updates_display():
    bridge = _video_bridge()
    rgb = _solid_rgb(1, 2, 3)
    bridge._on_video_frame(TRACK, _FakeFrame(rgb, 50))
    head = bridge.get_head_frame()
    assert head.bgr is not None
    np.testing.assert_array_equal(head.bgr[0, 0], (3, 2, 1))
    assert head.timestamp_us == 50


def test_newer_unmatched_frame_not_replaced_by_older_observation():
    bridge = _video_bridge()
    bridge._on_video_frame(TRACK, _FakeFrame(_solid_rgb(0, 0, 255), 200))
    bridge._on_observation(_FakeObs(100, {TRACK: _FakeFrame(_solid_rgb(255, 0, 0), 100)}))
    head = bridge.get_head_frame()
    np.testing.assert_array_equal(head.bgr[0, 0], (255, 0, 0))
    assert head.timestamp_us == 200


def test_observation_replaces_older_unmatched_frame():
    bridge = _video_bridge()
    bridge._on_video_frame(TRACK, _FakeFrame(_solid_rgb(255, 0, 0), 100))
    bridge._on_observation(_FakeObs(200, {TRACK: _FakeFrame(_solid_rgb(0, 255, 0), 200)}))
    head = bridge.get_head_frame()
    np.testing.assert_array_equal(head.bgr[0, 0], (0, 255, 0))
    assert head.timestamp_us == 200


def test_same_timestamp_is_decoded_once():
    bridge = _video_bridge()
    timing = _CountTiming()
    bridge._match_timing = timing
    rgb = _solid_rgb(1, 2, 3)
    frame = _FakeFrame(rgb, 99)
    bridge._on_video_frame(TRACK, frame)
    bridge._on_observation(_FakeObs(99, {TRACK: frame}))
    assert timing.counts.get("decode_miss") == 1
    assert timing.counts.get("decode_hit") == 1
    assert len(timing.samples.get("decode_ms", [])) == 1
    head = bridge.get_head_frame()
    np.testing.assert_array_equal(head.bgr[0, 0], (3, 2, 1))
    assert head.timestamp_us == 99
    bridge = _video_bridge()
    bridge._recording_enabled = True
    rgb = _solid_rgb(9, 8, 7)
    bridge._on_observation(_FakeObs(42, {TRACK: _FakeFrame(rgb, 42)}))
    assert bridge.get_last_obs_ts_us() == 42
    head = bridge.get_head_frame()
    assert head.bgr is not None
    obs = bridge._rec_buf.take(42)
    assert obs is not None
    np.testing.assert_array_equal(obs.frames[TRACK][0, 0], head.bgr[0, 0])


def test_recording_skips_observation_without_frames():
    bridge = _video_bridge()
    bridge._recording_enabled = True
    bridge._on_observation(_FakeObs(7, {}))
    assert bridge.get_last_obs_ts_us() == 7
    assert bridge._rec_buf.take(7) is None


class _CountTiming:
    def __init__(self):
        self.counts: dict[str, int] = {}
        self.samples: dict[str, list[float]] = {}

    def count(self, name: str, n: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + n

    def add(self, name: str, value_ms: float) -> None:
        self.samples.setdefault(name, []).append(float(value_ms))


def test_match_timing_counts_obs_drops_and_unmatched_video():
    bridge = _video_bridge()
    timing = _CountTiming()
    bridge._match_timing = timing
    rgb = _solid_rgb(1, 2, 3)
    bridge._on_video_frame(TRACK, _FakeFrame(rgb, 1))
    bridge._on_observation(_FakeObs(10, {TRACK: _FakeFrame(rgb, 10)}))
    bridge._on_observation(_FakeObs(11, {}))
    bridge._on_drop([{"q": 0.0}, {"q": 1.0}])
    assert timing.counts["unmatched_video"] == 1
    assert timing.counts["obs"] == 2
    assert timing.counts["obs_framed"] == 1
    assert timing.counts["drop"] == 2
    assert timing.counts["ts_eq"] == 1
    assert timing.samples["match_delta_ms"] == [0.0]


def test_stamp_mismatch_and_zero_are_counted():
    bridge = _video_bridge()
    timing = _CountTiming()
    bridge._match_timing = timing
    rgb = _solid_rgb(1, 2, 3)
    bridge._on_observation(_FakeObs(1000, {TRACK: _FakeFrame(rgb, 0)}))
    bridge._on_observation(_FakeObs(2000, {TRACK: _FakeFrame(rgb, 2500)}))
    assert timing.counts["frame_ts_zero"] == 1
    assert timing.counts["ts_ne"] == 1
    assert timing.samples["match_abs_delta_ms"] == [0.5]


def test_unmatched_idle_after_framed_obs(monkeypatch):
    from teleop.robot_control import portal_operator as po
    monkeypatch.setattr(po, "UNMATCHED_IDLE_AFTER_FRAMED_OBS", 2)
    monkeypatch.setattr(po, "STAMP_LOG_FIRST", 0)
    bridge = _video_bridge()
    rgb = _solid_rgb(1, 2, 3)
    bridge._on_observation(_FakeObs(1, {TRACK: _FakeFrame(rgb, 1)}))
    assert bridge._unmatched_idle_logged is False
    bridge._on_observation(_FakeObs(2, {TRACK: _FakeFrame(rgb, 2)}))
    assert bridge._unmatched_idle_logged is True


def test_drop_n_accepts_list_and_int():
    assert PortalTeleopBridge._drop_n([1, 2, 3]) == 3
    assert PortalTeleopBridge._drop_n(4) == 4
    assert PortalTeleopBridge._drop_n(None) == 0
