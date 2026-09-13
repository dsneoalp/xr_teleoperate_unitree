"""send_targets publishes send_action on the caller thread (no asyncio hop)."""
from __future__ import annotations

import threading

import numpy as np

from teleop.robot_control.obs_record_buffer import (
    RecordingObsBuffer,
    RecordingObservation,
)
from teleop.robot_control.portal_mapping import PortalMapping
from teleop.robot_control.portal_operator import PortalTeleopBridge
from teleop.tests.paths import PORTAL_MAPPING, PORTAL_YAML


class _FakeOp:
    """Records send_action calls without LiveKit."""

    def __init__(self):
        self.calls = []
        self.fail = False

    def send_action(self, values, timestamp_us=None, in_reply_to_ts_us=None):
        if self.fail:
            raise RuntimeError("send_action failed")
        self.calls.append({
            "values": dict(values),
            "timestamp_us": timestamp_us,
            "in_reply_to_ts_us": in_reply_to_ts_us,
            "thread": threading.current_thread().ident,
        })


def _bridge(*, connected: bool = True) -> PortalTeleopBridge:
    bridge = PortalTeleopBridge.__new__(PortalTeleopBridge)
    bridge._map = PortalMapping(PORTAL_MAPPING, PORTAL_YAML)
    arm_dof = bridge._map.arm_dof
    hand_dof = bridge._map.hand_dof
    bridge._obs_lock = threading.Lock()
    bridge._arm_lock = threading.Lock()
    bridge._hand_lock = threading.Lock()
    bridge._obs_ts_us = None
    bridge._pending_action_wall = None
    bridge._recording_enabled = False
    bridge._record_pair_cb = None
    bridge._rec_buf = RecordingObsBuffer()
    bridge._last_sent_q = np.zeros(arm_dof)
    bridge._last_sent_dq = np.zeros(arm_dof)
    bridge._last_send_wall = 0.0
    bridge._hand_q = np.zeros(hand_dof)
    bridge._fsm_id = 0
    bridge._op = _FakeOp()
    bridge._connected_evt = threading.Event()
    if connected:
        bridge._connected_evt.set()
    return bridge


def test_send_targets_skipped_when_not_connected():
    bridge = _bridge(connected=False)
    bridge.send_targets(np.ones(bridge._map.arm_dof))
    assert bridge._op.calls == []


def test_send_targets_calls_send_action_on_caller_thread():
    bridge = _bridge()
    arm = np.linspace(0.1, 0.4, bridge._map.arm_dof)
    hand = np.linspace(0.5, 0.8, bridge._map.hand_dof)
    caller = threading.current_thread().ident
    bridge.send_targets(arm, hand_q=hand, vx=0.12, vy=-0.04, vyaw=0.08, fsm_id=1)
    assert len(bridge._op.calls) == 1
    call = bridge._op.calls[0]
    assert call["thread"] == caller
    unpacked = bridge._map.unpack_action(call["values"])
    np.testing.assert_allclose(unpacked.arm_q, arm)
    np.testing.assert_allclose(unpacked.hand_q, hand)
    assert unpacked.vx == 0.12
    assert unpacked.vy == -0.04
    assert unpacked.vyaw == 0.08
    assert unpacked.fsm_id == 1
    np.testing.assert_allclose(bridge.get_current_dual_arm_q(), arm)


def test_send_targets_does_not_coalesce_two_calls():
    bridge = _bridge()
    first = np.zeros(bridge._map.arm_dof)
    second = np.ones(bridge._map.arm_dof)
    bridge.send_targets(first, fsm_id=1)
    bridge.send_targets(second, fsm_id=1)
    assert len(bridge._op.calls) == 2
    np.testing.assert_allclose(
        bridge._map.unpack_action(bridge._op.calls[0]["values"]).arm_q, first)
    np.testing.assert_allclose(
        bridge._map.unpack_action(bridge._op.calls[1]["values"]).arm_q, second)


def test_send_targets_none_hand_uses_retarget_state():
    bridge = _bridge()
    retarget = np.linspace(0.2, 0.33, bridge._map.hand_dof)
    with bridge._hand_lock:
        bridge._hand_q[:] = retarget
    bridge.send_targets(np.zeros(bridge._map.arm_dof), hand_q=None, fsm_id=1)
    unpacked = bridge._map.unpack_action(bridge._op.calls[0]["values"])
    np.testing.assert_allclose(unpacked.hand_q, retarget)


def test_send_targets_pairs_recording_obs_by_in_reply_to():
    bridge = _bridge()
    obs_ts = 9001
    arm = np.arange(bridge._map.arm_dof, dtype=np.float64)
    hand = np.arange(bridge._map.hand_dof, dtype=np.float64)
    pairs = []
    bridge._obs_ts_us = obs_ts
    bridge._recording_enabled = True
    bridge._record_pair_cb = pairs.append
    bridge._rec_buf.push(RecordingObservation(
        timestamp_us=obs_ts, arm_q=arm.copy(), hand_q=hand.copy(), frames={}))
    bridge.send_targets(arm, hand_q=hand, vx=0.1, vy=0.2, vyaw=0.3, fsm_id=1)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.obs.timestamp_us == obs_ts
    assert bridge._op.calls[0]["in_reply_to_ts_us"] == obs_ts
    np.testing.assert_allclose(pair.arm_action, arm)
    np.testing.assert_allclose(pair.hand_action, hand)
    assert pair.vx == 0.1
    assert pair.vy == 0.2
    assert pair.vyaw == 0.3
    assert pair.fsm_id == 1


def test_send_targets_failed_send_does_not_update_last_sent():
    bridge = _bridge()
    before = bridge.get_current_dual_arm_q().copy()
    bridge._op.fail = True
    bridge.send_targets(np.ones(bridge._map.arm_dof), fsm_id=1)
    np.testing.assert_allclose(bridge.get_current_dual_arm_q(), before)
    assert bridge._op.calls == []
