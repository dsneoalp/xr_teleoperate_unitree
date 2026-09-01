"""Operator-side round-trip test for the portal wire contract.

Connects with the same portal.yaml / .env as the teleop bridge, sends
sinusoidal arm actions at 30 Hz and reports observation / state / video
reception. Use it together with portal_robot_mock.py to verify the
schema and the action round trip without an XR headset:

    # terminal 1
    python portal_robot_mock.py
    # terminal 2
    python portal_operator_test.py --duration 15
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import pathlib
import sys
import time

from dotenv import load_dotenv
from livekit.portal import Observation, Operator, OperatorConfig

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from teleop.robot_control.portal_operator import mint_portal_token, claim_active_operator

HERE = pathlib.Path(__file__).parent
CONFIG_PATH = HERE / "portal.yaml"
ENV_PATH = HERE / ".env"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--identity", type=str, default="operator-test")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    load_dotenv(ENV_PATH, override=False)
    url = os.environ.get("LIVEKIT_URL")
    room = os.environ.get("LIVEKIT_ROOM", "g1-portal")
    api_key = os.environ.get("LIVEKIT_API_KEY")
    api_secret = os.environ.get("LIVEKIT_API_SECRET")
    if not all([url, api_key, api_secret]):
        raise SystemExit(f"LIVEKIT_URL / API_KEY / API_SECRET missing (looked in {ENV_PATH})")
    token = mint_portal_token(api_key, api_secret, args.identity, room)

    cfg = OperatorConfig.from_yaml_file(str(CONFIG_PATH), room)
    op = Operator(cfg)

    obs_count = 0
    frame_count = 0
    sent = 0
    last_report = time.monotonic()

    def on_observation(obs: Observation) -> None:
        nonlocal obs_count, frame_count
        obs_count += 1
        if obs.frames.get("head_camera") is not None:
            frame_count += 1

    op.on_observation(on_observation)

    async def run() -> None:
        nonlocal sent, last_report
        print(f"[test] connecting as '{args.identity}' to room '{room}' ...")
        await op.connect(url, token)
        await claim_active_operator(op)
        print(f"[test] connected and active; sending sinus actions @ {args.fps} Hz "
              f"for {args.duration:.0f}s")

        interval = 1.0 / args.fps
        next_tick = time.monotonic()
        start = time.monotonic()
        i = 0
        try:
            while (time.monotonic() - start) < args.duration:
                phase = i / args.fps
                values = {"fsm_id": 0}
                for k in range(7):
                    values[f"left_arm_q{k}"] = 0.2 * math.sin(phase + k)
                    values[f"right_arm_q{k}"] = 0.2 * math.cos(phase + k)
                for k in range(7):
                    values[f"left_hand_q{k}"] = 0.1 + 0.05 * math.sin(phase)
                    values[f"right_hand_q{k}"] = 0.1 + 0.05 * math.cos(phase)
                op.send_action(values, timestamp_us=int(time.time() * 1_000_000))
                sent += 1

                now = time.monotonic()
                if now - last_report >= 1.0:
                    got = op.get_observation()
                    state_desc = "n/a"
                    frame_desc = "none"
                    if got is not None:
                        j0 = got.state.get("j0")
                        state_desc = f"j0={j0:.3f}" if j0 is not None else "empty"
                        f = got.frames.get("head_camera")
                        if f is not None:
                            frame_desc = f"{f.width}x{f.height}"
                    print(f"[test] sent={sent} obs={obs_count} frames={frame_count} "
                          f"state={state_desc} video={frame_desc}")
                    last_report = now

                i += 1
                next_tick += interval
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                else:
                    next_tick = time.monotonic()
        finally:
            print(f"[test] done: sent={sent} observations={obs_count} frames={frame_count}")
            print("[test] disconnecting ...")
            await op.disconnect()
            op.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
