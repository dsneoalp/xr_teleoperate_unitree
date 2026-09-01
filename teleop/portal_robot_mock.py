"""Mock robot gateway for testing xr_teleoperate portal mode.

Loads the shared wire contract from teleop/portal.yaml and the
credentials from teleop/.env, then:

  * publishes a moving test pattern on the head_camera track (480x640@30)
  * publishes state: echoes the latest received arm action into j0..j13
    (sinus idle before the first action), hand action excerpt into
    j14..j19, fsm_id pass-through
  * prints every received action (arm + hand joint targets) and a
    once-per-second rate summary

Run (from the teleop directory):
    conda activate tv
    python portal_robot_mock.py                # runs until Ctrl+C
    python portal_robot_mock.py --duration 30  # exit after 30 s
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import pathlib
import sys
import time

import numpy as np

from dotenv import load_dotenv
from livekit.portal import Action, Robot, RobotConfig

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from teleop.robot_control.portal_operator import mint_portal_token

HERE = pathlib.Path(__file__).parent
CONFIG_PATH = HERE / "portal.yaml"
ENV_PATH = HERE / ".env"

WIDTH, HEIGHT = 640, 480
STATE_ARM_FIELDS = 14   # j0..j13
STATE_EXTRA = 6         # j14..j19


def _make_frame(width: int, height: int, phase: float) -> np.ndarray:
    """Moving sinusoidal test pattern (RGB uint8), full-frame entropy."""
    x = np.arange(width, dtype=np.float32) / width
    y = np.arange(height, dtype=np.float32)[:, None] / height
    two_pi = 2.0 * math.pi
    r = (0.5 + 0.5 * np.sin(two_pi * (x + phase))) * 255.0
    g = (0.5 + 0.5 * np.sin(two_pi * (y + phase * 0.7))) * 255.0
    b = (0.5 + 0.5 * np.sin(two_pi * (x * 0.5 + y * 0.5 + phase * 1.3))) * 255.0
    frame = np.stack([np.broadcast_to(r, (height, width)),
                      np.broadcast_to(g, (height, width)),
                      np.broadcast_to(b, (height, width))], axis=-1)
    return frame.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=0.0,
                        help="seconds to run, 0 = until Ctrl+C (default)")
    parser.add_argument("--identity", type=str, default="robot-mock")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    load_dotenv(ENV_PATH, override=False)
    url = os.environ.get("LIVEKIT_URL")
    room = os.environ.get("LIVEKIT_ROOM", "g1-portal")
    api_key = os.environ.get("LIVEKIT_API_KEY")
    api_secret = os.environ.get("LIVEKIT_API_SECRET")
    if not all([url, api_key, api_secret]):
        raise SystemExit("LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET missing "
                         f"(looked in env and {ENV_PATH})")
    token = mint_portal_token(api_key, api_secret, args.identity, room)

    cfg = RobotConfig.from_yaml_file(str(CONFIG_PATH), room)
    robot = Robot(cfg)

    actions_total = 0
    actions_window = 0
    window_start = time.monotonic()
    last_action_ts = None
    last_arm_q = None      # np.array(14) from latest action
    last_hand_q = None     # np.array(14)
    last_fsm_id = 0

    def on_action(action: Action) -> None:
        nonlocal actions_total, actions_window, last_action_ts
        nonlocal last_arm_q, last_hand_q, last_fsm_id
        actions_total += 1
        actions_window += 1
        last_action_ts = time.monotonic()
        v = action.raw_values
        try:
            last_arm_q = np.array([v[f"left_arm_q{i}"] for i in range(7)] +
                                  [v[f"right_arm_q{i}"] for i in range(7)])
            last_hand_q = np.array([v[f"left_hand_q{i}"] for i in range(7)] +
                                   [v[f"right_hand_q{i}"] for i in range(7)])
            last_fsm_id = int(v.get("fsm_id", 0))
        except KeyError as exc:
            print(f"[mock] action missing field: {exc}")
            return
        print(f"[action #{actions_total}] ts={action.timestamp_us} fsm={last_fsm_id} "
              f"Larm={np.round(last_arm_q[:7], 3).tolist()} "
              f"Rarm={np.round(last_arm_q[7:], 3).tolist()} "
              f"Lhand={np.round(last_hand_q[:7], 3).tolist()} "
              f"Rhand={np.round(last_hand_q[7:], 3).tolist()}")

    robot.on_action(on_action)

    async def run() -> None:
        nonlocal actions_total, actions_window, window_start, last_action_ts
        nonlocal last_arm_q, last_hand_q, last_fsm_id
        print(f"[mock] connecting as '{args.identity}' to room '{room}' at {url} ...")
        print(f"[mock] wire contract: {CONFIG_PATH}")
        await robot.connect(url, token)
        print("[mock] connected; streaming test pattern + state. Ctrl+C to stop.")
        interval = 1.0 / args.fps
        next_tick = time.monotonic()
        i = 0
        start = time.monotonic()
        try:
            while True:
                phase = i / args.fps
                ts_us = int(time.time() * 1_000_000)

                robot.send_video_frame("head_camera", _make_frame(WIDTH, HEIGHT, phase),
                                       timestamp_us=ts_us)

                state = {}
                for k in range(STATE_ARM_FIELDS + STATE_EXTRA):
                    if k < STATE_ARM_FIELDS and last_arm_q is not None:
                        state[f"j{k}"] = float(last_arm_q[k])
                    elif k >= STATE_ARM_FIELDS and last_hand_q is not None:
                        state[f"j{k}"] = float(last_hand_q[k - STATE_ARM_FIELDS])
                    else:
                        state[f"j{k}"] = 0.1 * math.sin(phase + 0.5 * k)
                state["fsm_id"] = last_fsm_id
                robot.send_state(state, timestamp_us=ts_us)

                now = time.monotonic()
                if now - window_start >= 1.0:
                    rate = actions_window / (now - window_start)
                    age = (now - last_action_ts) if last_action_ts is not None else None
                    age_str = f"{age:.2f}s ago" if age is not None else "never"
                    print(f"[mock] actions: {rate:5.1f} Hz (last {age_str}, total {actions_total})")
                    actions_window = 0
                    window_start = now

                i += 1
                next_tick += interval
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                else:
                    next_tick = time.monotonic()

                if args.duration > 0 and (time.monotonic() - start) >= args.duration:
                    break
        finally:
            print(f"[mock] total actions received: {actions_total}")
            print("[mock] disconnecting ...")
            await robot.disconnect()
            robot.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
