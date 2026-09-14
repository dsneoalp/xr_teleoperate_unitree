"""Portal unified-sampling helpers for the robot publish path.

LiveKit Portal matches video to state only when both carry the same sender
`timestamp_us`. This module stamps one capture tick and holds at most one
pending video frame (drop-oldest) so encode stays off the control loop
without introducing a queue between Operator and Robot.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np


def capture_tick_us(clock=time.time) -> int:
    """Sender-clock timestamp in microseconds for one capture tick."""
    return int(clock() * 1_000_000)


def grab_head_rgb(img_client) -> np.ndarray | None:
    """Latest head frame as packed RGB24, or None if the camera has nothing."""
    if img_client is None:
        return None
    head = img_client.get_head_frame()
    if head is None or getattr(head, "bgr", None) is None:
        return None
    return np.ascontiguousarray(head.bgr[:, :, ::-1])


@dataclass(frozen=True)
class StampedVideo:
    """One RGB frame tagged with the capture timestamp shared with state."""

    track: str
    rgb: np.ndarray
    timestamp_us: int


class LatestStampedSlot:
    """Single-slot drop-oldest holder. Put overwrites; take returns the latest."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: StampedVideo | None = None

    def put(self, item: StampedVideo) -> None:
        with self._lock:
            self._item = item

    def take(self) -> StampedVideo | None:
        with self._lock:
            item = self._item
            self._item = None
            return item
