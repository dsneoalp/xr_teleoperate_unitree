"""Unit tests for recording observation buffer keyed by in_reply_to."""
import numpy as np

from teleop.robot_control.obs_record_buffer import (
    OBS_RECORD_BUFFER_MAXLEN,
    RecordingObservation,
    RecordingObsBuffer,
    RecordingPair,
)
from teleop.utils.episode_item import colors_from_head, states_actions_from_pair


def _obs(ts, arm=None):
    return RecordingObservation(
        timestamp_us=ts,
        arm_q=None if arm is None else np.asarray(arm, dtype=np.float64),
        hand_q=np.arange(14, dtype=np.float64),
        frames={"head_camera": np.zeros((4, 4, 3), dtype=np.uint8)},
    )


def test_take_matches_in_reply_to_not_fifo():
    buf = RecordingObsBuffer(maxlen=2)
    buf.push(_obs(1, arm=np.zeros(14)))
    buf.push(_obs(2, arm=np.ones(14)))
    got = buf.take(2)
    assert got is not None
    assert got.timestamp_us == 2
    np.testing.assert_allclose(got.arm_q, np.ones(14))
    leftover = buf.take(1)
    assert leftover is not None
    assert leftover.timestamp_us == 1


def test_take_missing_returns_none():
    buf = RecordingObsBuffer()
    buf.push(_obs(5))
    assert buf.take(99) is None
    assert buf.take(None) is None
    assert buf.take(5) is not None


def test_drop_oldest_when_full():
    buf = RecordingObsBuffer(maxlen=2)
    buf.push(_obs(1))
    buf.push(_obs(2))
    buf.push(_obs(3))
    assert buf.drops == 1
    assert buf.take(1) is None
    assert buf.take(2) is not None
    assert buf.take(3) is not None


def test_clear_drops_pending():
    buf = RecordingObsBuffer()
    buf.push(_obs(8))
    buf.clear()
    assert buf.take(8) is None


def test_default_maxlen():
    assert OBS_RECORD_BUFFER_MAXLEN == 2


def test_colors_from_head_mono_and_binocular():
    head = np.zeros((2, 8, 3), dtype=np.uint8)
    head[:, :4] = 1
    head[:, 4:] = 2
    mono = colors_from_head(head, {"head_camera": {"binocular": False}})
    assert list(mono) == ["color_0"]
    bi = colors_from_head(
        head, {"head_camera": {"binocular": True, "image_shape": [2, 8]}})
    assert list(bi) == ["color_0", "color_1"]
    assert colors_from_head(None, {}) == {}


def test_states_actions_from_pair_splits_q():
    obs = _obs(9, arm=np.arange(14, dtype=np.float64))
    pair = RecordingPair(
        obs=obs,
        action_timestamp_us=100,
        arm_action=np.arange(14, dtype=np.float64) + 10,
        hand_action=np.arange(14, dtype=np.float64) + 20,
        vx=0.1, vy=0.2, vyaw=0.3, fsm_id=1,
    )
    states, actions = states_actions_from_pair(pair, include_body=True)
    assert states["left_arm"]["qpos"] == list(range(7))
    assert actions["right_arm"]["qpos"] == list(range(17, 24))
    assert actions["body"]["qpos"] == [0.1, 0.2, 0.3]
    _, actions_idle = states_actions_from_pair(pair, include_body=False)
    assert actions_idle["body"]["qpos"] == []
