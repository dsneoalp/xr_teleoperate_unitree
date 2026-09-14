"""CLI: print effective Hz from an episode data.json.

Robot ticks use `timestamp_us`; operator sends use `action_timestamp_us`.
Accepts a data.json file or an episode directory that contains one.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

ROBOT_TS_KEY = "timestamp_us"
ACTION_TS_KEY = "action_timestamp_us"


def resolve_data_json(path: Path) -> Path:
    """Return the data.json path. Directories must contain data.json."""
    if path.is_dir():
        candidate = path / "data.json"
        if not candidate.is_file():
            raise FileNotFoundError(f"no data.json in directory: {path}")
        return candidate
    if not path.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    return path


def hz_from_timestamps(ts_us: list[int]) -> dict | None:
    """Mean Hz over the span, median Hz from gaps. None if fewer than two ticks."""
    if len(ts_us) < 2:
        return None
    dt_s = [(b - a) / 1e6 for a, b in zip(ts_us, ts_us[1:]) if b > a]
    if not dt_s:
        return None
    span_s = (ts_us[-1] - ts_us[0]) / 1e6
    return {
        "n": len(ts_us),
        "span_s": span_s,
        "mean_hz": (len(ts_us) - 1) / span_s if span_s > 0 else 0.0,
        "median_hz": 1.0 / statistics.median(dt_s),
        "max_gap_ms": max(dt_s) * 1000.0,
    }


def collect_timestamps(items: list[dict], key: str) -> list[int]:
    return [int(item[key]) for item in items if key in item]


def format_hz(name: str, stats: dict | None) -> str:
    if stats is None:
        return f"  {name}: too few timestamps"
    return (
        f"  {name}: n={stats['n']}  {stats['span_s']:.2f}s  "
        f"mean={stats['mean_hz']:.2f} Hz  median={stats['median_hz']:.2f} Hz  "
        f"max_gap={stats['max_gap_ms']:.1f} ms"
    )


def report_episode(path: Path) -> str:
    data_json = resolve_data_json(path)
    payload = json.loads(data_json.read_text(encoding="utf-8"))
    items = payload.get("data")
    if not isinstance(items, list):
        raise ValueError(f"{data_json} has no 'data' array")
    lines = [f"{data_json}  items={len(items)}"]
    lines.append(format_hz("robot  timestamp_us", hz_from_timestamps(
        collect_timestamps(items, ROBOT_TS_KEY))))
    lines.append(format_hz("action action_timestamp_us", hz_from_timestamps(
        collect_timestamps(items, ACTION_TS_KEY))))
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute episode Hz from data.json timestamps.")
    parser.add_argument(
        "path",
        type=Path,
        help="episode data.json, or a directory that contains data.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    try:
        logger.info(report_episode(args.path))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
