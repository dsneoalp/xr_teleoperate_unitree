"""Unit tests for episode Hz reporting from data.json timestamps."""
import json

import pytest

from teleop.utils.episode_hz import (
    collect_timestamps,
    hz_from_timestamps,
    report_episode,
    resolve_data_json,
)


def test_hz_from_30hz_span():
    ts = [i * 33_333 for i in range(31)]
    stats = hz_from_timestamps(ts)
    assert stats is not None
    assert stats["n"] == 31
    assert 29.5 < stats["mean_hz"] < 30.5
    assert 29.5 < stats["median_hz"] < 30.5


def test_hz_from_too_few():
    assert hz_from_timestamps([]) is None
    assert hz_from_timestamps([1]) is None


def test_resolve_directory_and_file(tmp_path):
    episode = tmp_path / "episode_0001"
    episode.mkdir()
    data = episode / "data.json"
    data.write_text(json.dumps({"data": []}), encoding="utf-8")
    assert resolve_data_json(data) == data
    assert resolve_data_json(episode) == data


def test_report_episode_counts_both_clocks(tmp_path):
    payload = {
        "data": [
            {"timestamp_us": 0, "action_timestamp_us": 1_000},
            {"timestamp_us": 1_000_000, "action_timestamp_us": 1_001_000},
        ]
    }
    path = tmp_path / "data.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    text = report_episode(path)
    assert "items=2" in text
    assert "mean=1.00 Hz" in text


def test_collect_skips_missing_keys():
    items = [{"timestamp_us": 1}, {"idx": 1}, {"timestamp_us": 2}]
    assert collect_timestamps(items, "timestamp_us") == [1, 2]


def test_resolve_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_data_json(tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError):
        resolve_data_json(tmp_path)
