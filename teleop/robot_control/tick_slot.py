"""Latest tick timestamp shared by the robot control loop and video thread.

The control loop is the clock: it always publishes `send_state(tick_ts)` and
offers that stamp to video. Video takes the stamp at most once and never
reuses it. A pending unused stamp is overwritten by the next tick so a slow
encoder does not block state. `now_us` is wall-clock microseconds, matching
LiveKit Portal's `timestamp_us` convention (unified sampling).
"""
from __future__ import annotations

import threading
import time


def now_us() -> int:
    """Wall-clock microseconds for Portal `timestamp_us` fields."""
    return time.time_ns() // 1000


class LatestTickSlot:
    """Holds at most one unpublished tick timestamp.

    `offer(ts)` always stores `ts` (replaces a pending stamp). `take()` /
    `wait_take()` return that timestamp once, then clear the slot. A consumed
    timestamp is never reused.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._ts: int | None = None

    def offer(self, ts: int) -> bool:
        """Store `ts` for video. True if the slot was empty, False if replaced."""
        with self._lock:
            replaced = self._ts is not None
            self._ts = int(ts)
            self._event.set()
            return not replaced

    def take(self) -> int | None:
        """Consume the pending timestamp, or None if the slot is empty."""
        with self._lock:
            ts = self._ts
            self._ts = None
            self._event.clear()
            return ts

    def wait_take(self, timeout: float | None = None) -> int | None:
        """Block until a tick is offered, then consume it. None on timeout."""
        if not self._event.wait(timeout):
            return None
        return self.take()

    def pending(self) -> bool:
        with self._lock:
            return self._ts is not None
