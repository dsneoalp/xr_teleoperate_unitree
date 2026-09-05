"""Echo Portal robot for LiveKit smoke tests (no Unitree SDK).

Receives actions, publishes them back as state, and streams a test pattern
on the first portal.yaml video track.

    python teleop/portal_robot_mock.py --duration 20
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import threading

import numpy as np
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.robot_control.portal_robot import PortalRobotTransport
from teleop.robot_control.portal_mapping import UnpackedAction


def _test_pattern(h: int, w: int, t: float) -> np.ndarray:
    """RGB24 uint8 frame, 480x640-style, moving bar."""
    y = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = x
    frame[:, :, 1] = y
    col = int((t * 80) % w)
    frame[:, max(0, col - 4):col + 4, 2] = 255
    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=0.0, help='seconds then exit; 0 = until Ctrl-C')
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--portal-yaml', type=str, default=os.path.join(current_dir, 'portal.yaml'))
    parser.add_argument('--portal-mapping', type=str, default=os.path.join(current_dir, 'portal_mapping.yaml'))
    parser.add_argument('--env-file', type=str, default=os.path.join(current_dir, '.env'))
    parser.add_argument('--livekit-url', type=str, default=None)
    parser.add_argument('--livekit-room', type=str, default=None)
    parser.add_argument('--portal-identity', type=str, default='xr-robot-mock')
    args = parser.parse_args()

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
    logger_mp.info(f"[mock] echoing actions as state; video={track!r}; logging incoming actions at 1 Hz")

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
            if action is not None:
                portal.send_state(arm_q=action.arm_q, hand_q=action.hand_q, fsm_id=action.fsm_id)
            else:
                portal.send_state(fsm_id=0)
            if track:
                rgb = _test_pattern(480, 640, now - t0)
                portal.send_video_frame(track, rgb, timestamp_us=int(now * 1_000_000))
            if now - last_log >= 1.0:
                rate = n - last_count
                last_count = n
                last_log = now
                if action is None:
                    print("[mock] 1Hz: waiting for actions (count=0)", flush=True)
                else:
                    arm = action.arm_q
                    hand = action.hand_q
                    print(
                        f"[mock] 1Hz: count={n} (+{rate}/s) fsm={action.fsm_id} "
                        f"vx={action.vx:.3f} vy={action.vy:.3f} vyaw={action.vyaw:.3f} "
                        f"L_SHOULDER_PITCH={arm[0]:.4f} R_SHOULDER_PITCH={arm[7]:.4f} "
                        f"left_thumb_mcp={hand[0]:.4f}",
                        flush=True,
                    )
            elapsed = time.time() - now
            time.sleep(max(0.0, (1.0 / args.fps) - elapsed))
    except KeyboardInterrupt:
        logger_mp.info("[mock] interrupt")
    finally:
        portal.close()
        logger_mp.info("[mock] exit")
