"""Drop-oldest observation buffer for Portal recording pairs.

Stores matched observations (robot tick, joints, decoded frames). Take is
keyed by `in_reply_to_ts_us == obs.timestamp_us`, not FIFO, so an action
cannot bind to a different observation than the one it replied to.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock

import numpy as np

# Default matches portal.yaml slack=5: 2×slack, floor 10.
OBS_RECORD_BUFFER_MAXLEN = 10


def record_buffer_maxlen(slack: int) -> int:
    """Recording buffer depth from Portal slack (ticks). At least 10."""
    return max(10, int(slack) * 2)


@dataclass
class RecordingObservation:
    """One Portal observation snapshot for an episode item."""

    timestamp_us: int
    arm_q: np.ndarray | None
    hand_q: np.ndarray | None
    frames: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class RecordingPair:
    """Observation plus the action that replied to it on the wire."""

    obs: RecordingObservation
    action_timestamp_us: int
    arm_action: np.ndarray
    hand_action: np.ndarray
    vx: float
    vy: float
    vyaw: float
    fsm_id: int


class RecordingObsBuffer:
    """Thread-safe drop-oldest buffer. `take` matches `timestamp_us` exactly."""

    def __init__(self, maxlen: int = OBS_RECORD_BUFFER_MAXLEN) -> None:
        if maxlen < 1:
            raise ValueError(f"maxlen must be >= 1, got {maxlen}")
        self._items: deque[RecordingObservation] = deque(maxlen=maxlen)
        self._lock = Lock()
        self._drops = 0

    @property
    def drops(self) -> int:
        with self._lock:
            return self._drops

    def push(self, obs: RecordingObservation) -> None:
        with self._lock:
            if len(self._items) == self._items.maxlen:
                self._drops += 1
            self._items.append(obs)

    def take(self, in_reply_to_ts_us: int | None) -> RecordingObservation | None:
        """Remove and return the observation with this robot tick, else None."""
        if in_reply_to_ts_us is None:
            return None
        with self._lock:
            for i, item in enumerate(self._items):
                if item.timestamp_us == in_reply_to_ts_us:
                    del self._items[i]
                    return item
        return None

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
