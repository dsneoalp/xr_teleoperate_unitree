"""Video-thread encode timing: grab vs blocking send_video_frame."""
from __future__ import annotations

import threading
import time
import numpy as np

from teleop.robot_control.tick_slot import LatestTickSlot
from teleop.robot_control.portal_robot import video_publish_loop


class _RecordingTiming:
    def __init__(self):
        self.samples: dict[str, list[float]] = {}
        self.counts: dict[str, int] = {}

    def add(self, name: str, value_ms: float) -> None:
        self.samples.setdefault(name, []).append(float(value_ms))

    def count(self, name: str, n: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + n


class _FakeSource:
    def __init__(self):
        self._seq = 0
        self._rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    def latest_rgb(self):
        self._seq += 1
        return self._rgb, self._seq


class _FakePortal:
    def __init__(self, encode_s: float):
        self.encode_s = encode_s
        self.sends = 0
        self.stamps = []

    def send_video_frame(self, track, frame, timestamp_us=None):
        time.sleep(self.encode_s)
        self.sends += 1
        self.stamps.append(timestamp_us)


def test_video_loop_records_encode_ms():
    encode_s = 0.02
    slot = LatestTickSlot()
    stop_evt = threading.Event()
    portal = _FakePortal(encode_s)
    timing = _RecordingTiming()
    thread = threading.Thread(
        target=video_publish_loop,
        args=(_FakeSource(), portal, "head", stop_evt, slot, timing),
        daemon=True,
    )
    thread.start()
    slot.offer(1001)
    deadline = time.monotonic() + 1.0
    while portal.sends < 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    slot.offer(1002)
    deadline = time.monotonic() + 1.0
    while portal.sends < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    stop_evt.set()
    thread.join(timeout=1.0)

    encode_xs = timing.samples.get("encode_ms", [])
    assert len(encode_xs) >= 2
    assert min(encode_xs) >= encode_s * 1000.0 * 0.8
    assert max(encode_xs) < encode_s * 1000.0 * 3.0
    assert len(timing.samples.get("grab_ms", [])) >= 2
    assert len(timing.samples.get("video_gap_ms", [])) >= 1
    assert portal.stamps[:2] == [1001, 1002]
