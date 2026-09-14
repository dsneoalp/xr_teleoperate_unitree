"""LoopTiming flushes count() as n and Hz once per interval."""
from __future__ import annotations

import time

from teleop.utils.loop_timing import LoopTiming


class _Log:
    def __init__(self):
        self.lines: list[str] = []

    def info(self, msg: str) -> None:
        self.lines.append(msg)


def test_counts_flush_as_hz():
    log = _Log()
    timing = LoopTiming(log, prefix="[timing-match]", interval_s=1.0)
    timing.count("obs", 21)
    timing.count("drop", 9)
    timing._last_flush = time.monotonic() - 1.0
    timing.count("unmatched_video", 30)
    assert len(log.lines) == 1
    line = log.lines[0]
    assert line.startswith("[timing-match] ")
    assert "drop n=9 hz=" in line
    assert "obs n=21 hz=" in line
    assert "unmatched_video n=30 hz=" in line


def test_samples_and_counts_share_one_line():
    log = _Log()
    timing = LoopTiming(log, prefix="[timing]", interval_s=1.0)
    timing.add("rtt_ms", 8.0)
    timing.count("obs", 4)
    timing._last_flush = time.monotonic() - 1.0
    timing.add("rtt_ms", 9.0)
    assert len(log.lines) == 1
    line = log.lines[0]
    assert "rtt_ms n=2" in line
    assert "obs n=4 hz=" in line
