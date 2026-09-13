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


def test_observation_fills_recording_and_display_from_one_decode():
    bridge = _video_bridge()
    bridge._recording_enabled = True
    rgb = _solid_rgb(9, 8, 7)
    bridge._on_observation(_FakeObs(42, {TRACK: _FakeFrame(rgb, 42)}))
    head = bridge.get_head_frame()
    assert head.bgr is not None
    obs = bridge._rec_buf.take(42)
    assert obs is not None
    np.testing.assert_array_equal(obs.frames[TRACK][0, 0], head.bgr[0, 0])
