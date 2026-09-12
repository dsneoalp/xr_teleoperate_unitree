"""Assemble EpisodeWriter payloads from a Portal recording pair.

Keeps episode JSON shape (color_0 / left_arm / …) out of the operator
control loop and out of the LiveKit bridge.
"""
from __future__ import annotations

import numpy as np

from teleop.robot_control.obs_record_buffer import RecordingPair


def colors_from_head(head_bgr, camera_config: dict) -> dict:
    """Split or copy the matched head frame into episode `colors` keys."""
    colors = {}
    if head_bgr is None:
        return colors
    head_cfg = camera_config.get("head_camera") or {}
    if head_cfg.get("binocular"):
        shape = head_cfg.get("image_shape") or (None, None)
        width = int(shape[1]) if shape[1] is not None else int(head_bgr.shape[1])
        half = width // 2
        colors["color_0"] = head_bgr[:, :half]
        colors["color_1"] = head_bgr[:, half:]
        return colors
    colors["color_0"] = head_bgr
    return colors


def _split_q(q) -> tuple[list, list]:
    if q is None:
        return [], []
    arr = np.asarray(q).reshape(-1)
    if arr.size == 0:
        return [], []
    half = arr.size // 2
    return arr[:half].tolist(), arr[half:].tolist()


def states_actions_from_pair(pair: RecordingPair, *, include_body: bool) -> tuple[dict, dict]:
    """Joint state from the observation, actions from the wire send."""
    left_arm_s, right_arm_s = _split_q(pair.obs.arm_q)
    left_ee_s, right_ee_s = _split_q(pair.obs.hand_q)
    left_arm_a, right_arm_a = _split_q(pair.arm_action)
    left_ee_a, right_ee_a = _split_q(pair.hand_action)
    body = [float(pair.vx), float(pair.vy), float(pair.vyaw)] if include_body else []
    states = {
        "left_arm": {"qpos": left_arm_s, "qvel": [], "torque": []},
        "right_arm": {"qpos": right_arm_s, "qvel": [], "torque": []},
        "left_ee": {"qpos": left_ee_s, "qvel": [], "torque": []},
        "right_ee": {"qpos": right_ee_s, "qvel": [], "torque": []},
        "body": {"qpos": []},
    }
    actions = {
        "left_arm": {"qpos": left_arm_a, "qvel": [], "torque": []},
        "right_arm": {"qpos": right_arm_a, "qvel": [], "torque": []},
        "left_ee": {"qpos": left_ee_a, "qvel": [], "torque": []},
        "right_ee": {"qpos": right_ee_a, "qvel": [], "torque": []},
        "body": {"qpos": body},
    }
    return states, actions
