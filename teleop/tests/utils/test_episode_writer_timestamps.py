"""EpisodeWriter keeps robot and operator timestamps on each item."""
import json
import os
import time

import numpy as np
import pytest

from teleop.utils.episode_writer import EpisodeWriter


@pytest.fixture
def writer(tmp_path):
    w = EpisodeWriter(task_dir=str(tmp_path), frequency=30, rerun_log=False)
    yield w
    w.close()


def test_timestamp_fields_survive_json(writer):
    assert writer.create_episode()
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    writer.add_item(
        colors={"color_0": frame},
        depths={},
        states={"left_arm": {"qpos": [0.1], "qvel": [], "torque": []}},
        actions={"left_arm": {"qpos": [0.2], "qvel": [], "torque": []}},
        timestamp_us=1_700_000_000_001,
        action_timestamp_us=1_700_000_000_050,
    )
    writer.save_episode()
    deadline = time.time() + 5.0
    while not writer.is_ready():
        if time.time() > deadline:
            pytest.fail("episode save timed out")
        time.sleep(0.01)
    json_path = writer.json_path
    assert os.path.isfile(json_path)
    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)
    item = payload["data"][0]
    assert item["timestamp_us"] == 1_700_000_000_001
    assert item["action_timestamp_us"] == 1_700_000_000_050
    assert item["idx"] == 0
