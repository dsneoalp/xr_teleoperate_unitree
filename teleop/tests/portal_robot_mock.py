"""Echo Portal robot for LiveKit tests (no Unitree SDK).

Receives actions, publishes them back as state, and streams a test pattern
on the first portal.yaml video track. State and video share one `tick_ts`.

    python teleop/tests/portal_robot_mock.py --duration 20
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import threading

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

from teleop.tests.paths import ENV_FILE, PORTAL_MAPPING, PORTAL_YAML, REPO_ROOT

from teleop.robot_control.portal_robot import PortalRobotTransport
from teleop.robot_control.portal_mapping import UnpackedAction
from teleop.robot_control.tick_slot import now_us

MOCK_FRAME_HEIGHT = 480
MOCK_FRAME_WIDTH = 640
MOCK_ACTION_LOG_PERIOD_S = 1.0


def _test_pattern(h: int, w: int, t: float) -> np.ndarray:
    """RGB24 uint8 frame with a moving bar."""
    y = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = x
    frame[:, :, 1] = y
    col = int((t * 80) % w)
    frame[:, max(0, col - 4):col + 4, 2] = 255
    return frame


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Echo Portal robot: actions as state plus test-pattern video.")
    parser.add_argument('--duration', type=float, default=0.0,
                        help='seconds then exit; 0 = until Ctrl-C')
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--portal-yaml', type=str, default=PORTAL_YAML)
    parser.add_argument('--portal-mapping', type=str, default=PORTAL_MAPPING)
    parser.add_argument('--env-file', type=str, default=ENV_FILE)
    parser.add_argument('--livekit-url', type=str, default=None)
    parser.add_argument('--livekit-room', type=str, default=None)
    parser.add_argument('--portal-identity', type=str, default='xr-robot-mock')
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    portal = PortalRobotTransport(
        portal_yaml=args.portal_yaml,
        mapping_yaml=args.portal_mapping,
        env_file=args.env_file,
        identity=args.portal_identity,
        room=args.livekit_room,
        url=args.livekit_url)
    portal.wait_until_connected()

    lock = threading.Lock()
    latest = {'action': None, 'count': 0}

    def on_action(action: UnpackedAction):
        with lock:
            latest['action'] = action
            latest['count'] += 1

    portal.on_unpacked_action(on_action)
    track = portal.video_tracks[0] if portal.video_tracks else None
    logger_mp.info(
        f"[mock-robot] echoing actions as state; video={track!r}; "
        "logging incoming actions at 1 Hz")

    t0 = time.time()
    last_log = 0.0
    last_count = 0
    try:
        while True:
            now = time.time()
            if args.duration > 0 and (now - t0) >= args.duration:
                break
            with lock:
                action = latest['action']
                n = latest['count']
            tick_ts = now_us()
            if action is not None:
                portal.send_state(arm_q=action.arm_q, hand_q=action.hand_q,
                                  fsm_id=action.fsm_id, timestamp_us=tick_ts)
            else:
                portal.send_state(fsm_id=0, timestamp_us=tick_ts)
            if track:
                rgb = _test_pattern(MOCK_FRAME_HEIGHT, MOCK_FRAME_WIDTH, now - t0)
                portal.send_video_frame(track, rgb, timestamp_us=tick_ts)
            if now - last_log >= MOCK_ACTION_LOG_PERIOD_S:
                rate = n - last_count
                last_count = n
                last_log = now
                if action is None:
                    logger_mp.info("[mock-robot] 1Hz: waiting for actions (count=0)")
                else:
                    arm = action.arm_q
                    hand = action.hand_q
                    logger_mp.info(
                        f"[mock-robot] 1Hz: count={n} (+{rate}/s) fsm={action.fsm_id} "
                        f"vx={action.vx:.3f} vy={action.vy:.3f} vyaw={action.vyaw:.3f} "
                        f"L_SHOULDER_PITCH={arm[0]:.4f} R_SHOULDER_PITCH={arm[7]:.4f} "
                        f"left_thumb_mcp={hand[0]:.4f} tick_ts={tick_ts}")
            elapsed = time.time() - now
            time.sleep(max(0.0, (1.0 / args.fps) - elapsed))
    except KeyboardInterrupt:
        logger_mp.info("[mock-robot] interrupt")
        return 130
    finally:
        portal.close()
        logger_mp.info("[mock-robot] exit")
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
