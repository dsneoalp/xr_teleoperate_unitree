"""One-shot tick timestamp shared by the robot control loop and video thread.

The control loop is the clock: it may store a tick only when the previous
tick has been consumed. The video thread takes that timestamp at most once
and never waits for the next control tick. `now_us` is wall-clock
microseconds, matching LiveKit Portal's `timestamp_us` convention.
"""
from __future__ import annotations

import threading
import time


def now_us() -> int:
    """Wall-clock microseconds for Portal `timestamp_us` fields."""
    return time.time_ns() // 1000


class OneShotTickSlot:
    """Holds at most one unpublished tick timestamp.

    `try_publish(ts)` stores `ts` and returns True only if the slot is empty.
    `take()` returns that timestamp once, then clears the slot. A consumed
    timestamp is never reused.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ts: int | None = None

    def try_publish(self, ts: int) -> bool:
        """Store `ts` if the previous tick was consumed. False if still pending."""
        with self._lock:
            if self._ts is not None:
                return False
            self._ts = int(ts)
            return True

    def take(self) -> int | None:
        """Consume the pending timestamp, or None if the slot is empty."""
        with self._lock:
            ts = self._ts
            self._ts = None
            return ts

    def pending(self) -> bool:
        with self._lock:
            return self._ts is not None
