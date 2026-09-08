"""1 Hz p50/p95/max logs for teleop loops. Same-clock metrics only."""
from __future__ import annotations

import time
import threading


def _percentile(sorted_xs: list[float], p: float) -> float:
    if not sorted_xs:
        return 0.0
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    k = (len(sorted_xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = k - lo
    return sorted_xs[lo] * (1.0 - frac) + sorted_xs[hi] * frac


class LoopTiming:
    def __init__(self, logger, prefix: str = "[timing]", interval_s: float = 1.0):
        self._log = logger
        self._prefix = prefix
        self._interval = interval_s
        self._lock = threading.Lock()
        self._samples: dict[str, list[float]] = {}
        self._counts: dict[str, int] = {}
        self._last_flush = time.monotonic()

    def add(self, name: str, value_ms: float) -> None:
        with self._lock:
            self._samples.setdefault(name, []).append(float(value_ms))
            self._flush_if_due()

    def count(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counts[name] = self._counts.get(name, 0) + n
            self._flush_if_due()

    def _flush_if_due(self) -> None:
        now = time.monotonic()
        if now - self._last_flush < self._interval:
            return
        elapsed = now - self._last_flush
        self._last_flush = now
        samples = self._samples
        counts = self._counts
        self._samples = {}
        self._counts = {}
        parts = []
        for name in sorted(samples):
            xs = samples[name]
            xs.sort()
            parts.append(
                f"{name} n={len(xs)} p50={_percentile(xs, 50):.1f} "
                f"p95={_percentile(xs, 95):.1f} max={xs[-1]:.1f}"
            )
        num = counts.get("stale", 0)
        den = counts.get("loops", 0)
        if den:
            hz = den / elapsed if elapsed > 0 else 0.0
            parts.append(f"stale_frac={num / den:.2f} hz={hz:.1f}")
        if parts:
            self._log.info(f"{self._prefix} " + " | ".join(parts))
