"""Mock Portal operator for LiveKit tests (no TeleVuer, no IK, no Unitree SDK).

Sends dummy arm/hand targets and logs matched observations. Use `--expect-echo`
to exit 0 only after the robot echoes those joints, an observation carries
a matched frame (same robot tick_ts), and get_head_frame() has a BGR image.

    python teleop/tests/portal_operator_mock.py --duration 20
    python teleop/tests/portal_operator_mock.py --expect-echo --duration 20
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

from teleop.tests.paths import ENV_FILE, PORTAL_MAPPING, PORTAL_YAML

from teleop.robot_control.portal_mapping import PortalMapping
from teleop.robot_control.portal_operator import PortalTeleopBridge

ECHO_ATOL = 1e-3
LOG_PERIOD_S = 1.0
SEND_VX = 0.12
SEND_VY = -0.04
SEND_VYAW = 0.08
SEND_FSM_ID = 1


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mock Portal operator: dummy actions, no XR/IK.")
    parser.add_argument('--duration', type=float, default=0.0,
                        help='seconds then exit; 0 = until Ctrl-C')
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--expect-echo', action='store_true',
                        help='exit 0 after echoed joints, a matched observation frame, and get_head_frame()')
    parser.add_argument('--portal-yaml', type=str, default=PORTAL_YAML)
    parser.add_argument('--portal-mapping', type=str, default=PORTAL_MAPPING)
    parser.add_argument('--env-file', type=str, default=ENV_FILE)
    parser.add_argument('--livekit-url', type=str, default=None)
    parser.add_argument('--livekit-room', type=str, default=None)
    parser.add_argument('--portal-identity', type=str, default='xr-operator-mock')
    return parser.parse_args(argv)


def _dummy_targets(mapping: PortalMapping):
    arm = np.linspace(0.05, 0.18, mapping.arm_dof)
    hand = np.linspace(0.2, 0.33, mapping.hand_dof)
    return arm, hand


def run(args: argparse.Namespace) -> int:
    mapping = PortalMapping(args.portal_mapping, args.portal_yaml)
    arm, hand = _dummy_targets(mapping)
    bridge = PortalTeleopBridge(
        portal_yaml=args.portal_yaml,
        mapping_yaml=args.portal_mapping,
        env_file=args.env_file,
        identity=args.portal_identity,
        room=args.livekit_room,
        url=args.livekit_url,
        ee=None,
    )
    try:
        bridge.wait_until_connected()
    except Exception as exc:
        logger_mp.error(f"[mock-op] connect failed: {exc}")
        bridge.close()
        return 1
    try:
        logger_mp.info(
            f"[mock-op] sending dummy arm/hand at {args.fps:.0f} Hz; "
            f"expect_echo={args.expect_echo}")
        t0 = time.time()
        last_log = 0.0
        echoed = False
        saw_frame = False
        interval = 1.0 / max(args.fps, 1.0)
        while True:
            now = time.time()
            if args.duration > 0 and (now - t0) >= args.duration:
                break
            loop_t0 = now
            bridge.send_targets(
                arm, hand_q=hand, vx=SEND_VX, vy=SEND_VY, vyaw=SEND_VYAW,
                fsm_id=SEND_FSM_ID)
            reported = bridge.get_reported_arm_q()
            if reported is not None and np.allclose(reported, arm, atol=ECHO_ATOL):
                echoed = True
            head = bridge.get_head_frame()
            xr_ok = head is not None and head.bgr is not None
            obs_frames = bridge.last_obs_had_frames()
            if xr_ok:
                saw_frame = True
            if now - last_log >= LOG_PERIOD_S:
                last_log = now
                q0 = None if reported is None else float(reported[0])
                logger_mp.info(
                    f"[mock-op] 1Hz: echoed={echoed} xr_frame={xr_ok} "
                    f"obs_frames={obs_frames} obs_ts={bridge.get_last_obs_ts_us()} "
                    f"arm[0]={q0}")
            if args.expect_echo and echoed and obs_frames and xr_ok:
                logger_mp.info(
                    f"[mock-op] echo+matched observation ts={bridge.get_last_obs_ts_us()} "
                    f"xr_frame=True")
                return 0
            sleep = interval - (time.time() - loop_t0)
            if sleep > 0:
                time.sleep(sleep)
        if args.expect_echo:
            logger_mp.error(
                f"[mock-op] timeout: echoed={echoed} xr_frame={saw_frame} "
                f"obs_frames={bridge.last_obs_had_frames()} "
                f"obs_ts={bridge.get_last_obs_ts_us()}")
            return 1
        return 0
    except KeyboardInterrupt:
        logger_mp.info("[mock-op] interrupt")
        return 130
    finally:
        bridge.close()
        logger_mp.info("[mock-op] exit")


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
