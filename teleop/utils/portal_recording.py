"""Join Portal-matched observations with the action that replied to them.

Recording is observation-driven: emit only when
`action.in_reply_to_ts_us == obs.timestamp_us`. Each side keeps a single
latest-wins slot (drop-oldest). No queue is added on the Operator↔Robot path.

Target rate is Portal `fps` (30 Hz). Rolling rate below `min_hz` (20 Hz)
is an error: the dataset is no longer usable.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from teleop.robot_control.portal_mapping import UnpackedAction

TARGET_HZ = 30.0
MIN_HZ = 20.0
RATE_WINDOW_S = 1.0


@dataclass
class MatchedObservation:
    """Operator-side observation after Portal fused state with video frames."""

    timestamp_us: int
    frames_bgr: dict[str, np.ndarray]
    arm_q: np.ndarray | None
    hand_q: np.ndarray | None
    fsm_id: int


@dataclass
class JoinedSample:
    """One episode row: matched observation plus the action that replied to it."""

    timestamp_us: int
    in_reply_to_ts_us: int
    frames_bgr: dict[str, np.ndarray]
    arm_q: np.ndarray
    hand_q: np.ndarray | None
    fsm_id: int
    action: UnpackedAction


class RecordingRateGuard:
    """1 s rolling join-rate. Logs an error when the window falls below min_hz."""

    def __init__(
        self,
        logger,
        target_hz: float = TARGET_HZ,
        min_hz: float = MIN_HZ,
        window_s: float = RATE_WINDOW_S,
        clock=time.monotonic,
    ) -> None:
        if min_hz <= 0 or target_hz <= 0 or window_s <= 0:
            raise ValueError("rate guard hz/window must be > 0")
        self._log = logger
        self.target_hz = float(target_hz)
        self.min_hz = float(min_hz)
        self._window_s = float(window_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._marks: list[float] = []
        self._window_start = clock()
        self.last_hz: float | None = None
        self.below_min = False

    def mark(self) -> float | None:
        """Record one joined sample. Returns window hz when a window closes."""
        now = self._clock()
        with self._lock:
            self._marks.append(now)
            elapsed = now - self._window_start
            if elapsed < self._window_s:
                return None
            hz = len(self._marks) / elapsed if elapsed > 0 else 0.0
            self.last_hz = hz
            self.below_min = hz < self.min_hz
            self._marks = []
            self._window_start = now
        if self.below_min:
            self._log.error(
                f"[record] {hz:.1f} Hz over {elapsed:.2f}s "
                f"(min {self.min_hz:.0f} Hz, target {self.target_hz:.0f} Hz)"
            )
        else:
            self._log.info(
                f"[record] {hz:.1f} Hz over {elapsed:.2f}s "
                f"(target {self.target_hz:.0f} Hz)"
            )
        return hz


class PortalRecordingJoin:
    """Latest-wins join of matched observations and subscribed actions."""

    def __init__(
        self,
        on_joined: Callable[[JoinedSample], None],
        logger=None,
        target_hz: float = TARGET_HZ,
        min_hz: float = MIN_HZ,
        clock=time.monotonic,
    ) -> None:
        self._on_joined = on_joined
        self._lock = threading.Lock()
        self._enabled = False
        self._obs: MatchedObservation | None = None
        self._action: UnpackedAction | None = None
        self.join_miss = 0
        self.joined_n = 0
        self.rate = RecordingRateGuard(
            logger, target_hz=target_hz, min_hz=min_hz, clock=clock
        ) if logger is not None else None

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = bool(enabled)
            if not self._enabled:
                self._obs = None
                self._action = None

    def consume_counts(self) -> dict[str, int]:
        with self._lock:
            out = {"join_miss": self.join_miss, "joined": self.joined_n}
            self.join_miss = 0
            self.joined_n = 0
            return out

    def on_observation(self, obs: MatchedObservation) -> None:
        sample = None
        with self._lock:
            if not self._enabled:
                return
            if self._obs is not None:
                self.join_miss += 1
            self._obs = obs
            sample = self._try_join_locked()
        self._emit(sample)

    def on_action(self, action: UnpackedAction) -> None:
        sample = None
        with self._lock:
            if not self._enabled:
                return
            if self._action is not None:
                self.join_miss += 1
            self._action = action
            sample = self._try_join_locked()
        self._emit(sample)

    def _emit(self, sample: JoinedSample | None) -> None:
        if sample is None:
            return
        if self.rate is not None:
            self.rate.mark()
        self._on_joined(sample)

    @staticmethod
    def _matches(action: UnpackedAction | None, obs: MatchedObservation | None) -> bool:
        if action is None or obs is None:
            return False
        if action.in_reply_to_ts_us is None or obs.timestamp_us is None:
            return False
        if obs.arm_q is None:
            return False
        return int(action.in_reply_to_ts_us) == int(obs.timestamp_us)

    def _try_join_locked(self) -> JoinedSample | None:
        if not self._matches(self._action, self._obs):
            return None
        obs = self._obs
        action = self._action
        self._obs = None
        self._action = None
        self.joined_n += 1
        return JoinedSample(
            timestamp_us=obs.timestamp_us,
            in_reply_to_ts_us=int(action.in_reply_to_ts_us),
            frames_bgr=obs.frames_bgr,
            arm_q=obs.arm_q,
            hand_q=obs.hand_q,
            fsm_id=obs.fsm_id,
            action=action,
        )


def frames_to_colors(
    frames_bgr: dict[str, np.ndarray],
    declared_videos: list[str],
    binocular: bool,
) -> dict[str, np.ndarray]:
    """Map Portal track order onto EpisodeWriter color_N keys."""
    colors: dict[str, np.ndarray] = {}
    next_idx = 0
    for i, track in enumerate(declared_videos):
        img = frames_bgr.get(track)
        if img is None:
            if i == 0:
                next_idx = 2 if binocular else 1
            continue
        if i == 0 and binocular:
            width = img.shape[1] // 2
            colors["color_0"] = img[:, :width]
            colors["color_1"] = img[:, width:]
            next_idx = 2
            continue
        if i == 0:
            colors["color_0"] = img
            next_idx = 1
            continue
        colors[f"color_{next_idx}"] = img
        next_idx += 1
    return colors


def joined_to_episode(
    sample: JoinedSample,
    declared_videos: list[str],
    binocular: bool,
    include_loco: bool,
) -> dict:
    """Build an EpisodeWriter.add_item kwargs dict from a joined Portal sample."""
    arm_q = np.asarray(sample.arm_q, dtype=np.float64).reshape(-1)
    half = arm_q.size // 2
    left_arm_state, right_arm_state = arm_q[:half], arm_q[half:]
    act_arm = np.asarray(sample.action.arm_q, dtype=np.float64).reshape(-1)
    act_half = act_arm.size // 2
    left_arm_action, right_arm_action = act_arm[:act_half], act_arm[act_half:]

    left_ee_state: list = []
    right_ee_state: list = []
    if sample.hand_q is not None and sample.hand_q.size:
        hand = np.asarray(sample.hand_q, dtype=np.float64).reshape(-1)
        h = hand.size // 2
        left_ee_state = hand[:h].tolist()
        right_ee_state = hand[h:].tolist()

    left_hand_action: list = []
    right_hand_action: list = []
    if sample.action.hand_q is not None and sample.action.hand_q.size:
        ah = np.asarray(sample.action.hand_q, dtype=np.float64).reshape(-1)
        h = ah.size // 2
        left_hand_action = ah[:h].tolist()
        right_hand_action = ah[h:].tolist()

    body_action = (
        [sample.action.vx, sample.action.vy, sample.action.vyaw] if include_loco else []
    )
    return {
        "colors": frames_to_colors(sample.frames_bgr, declared_videos, binocular),
        "depths": {},
        "states": {
            "left_arm": {"qpos": left_arm_state.tolist(), "qvel": [], "torque": []},
            "right_arm": {"qpos": right_arm_state.tolist(), "qvel": [], "torque": []},
            "left_ee": {"qpos": left_ee_state, "qvel": [], "torque": []},
            "right_ee": {"qpos": right_ee_state, "qvel": [], "torque": []},
            "body": {"qpos": []},
        },
        "actions": {
            "left_arm": {"qpos": left_arm_action.tolist(), "qvel": [], "torque": []},
            "right_arm": {"qpos": right_arm_action.tolist(), "qvel": [], "torque": []},
            "left_ee": {"qpos": left_hand_action, "qvel": [], "torque": []},
            "right_ee": {"qpos": right_hand_action, "qvel": [], "torque": []},
            "body": {"qpos": body_action},
        },
        "timestamp_us": sample.timestamp_us,
        "in_reply_to_ts_us": sample.in_reply_to_ts_us,
    }
