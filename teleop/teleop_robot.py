"""Portal robot control loop: apply joint targets via SDK, stream state.

No TeleVuer / IK. Pair with teleop_operator.py in the same LiveKit room.

    python teleop/teleop_robot.py --ee dex3 --arm G1_29
    python teleop/teleop_robot.py --ee dex3 --arm G1_29 --motion
    python teleop/teleop_robot.py --ee dex3 --arm G1_29 --sim
"""
import time
import math
import argparse
import threading
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.robot_control.portal_robot import PortalRobotTransport
from teleop.robot_control.portal_mapping import UnpackedAction
from teleop.utils.loop_timing import LoopTiming

FSM_IDLE = 0
FSM_TELEOP = 1
FSM_HOME = 2
FSM_HAND_SETUP = 3
ACTION_TIMEOUT = 0.1
IMAGE_CLIENT_RETRIES = 50
IMAGE_CLIENT_RETRY_S = 0.1


def interp_cmd(state, q_cmd, now, tau):
    """Linear segment q0→q1 over tau s. Restart on a new target; hold at s=1."""
    q_cmd = np.asarray(q_cmd, dtype=float)
    if tau <= 0.0:
        state['q'] = q_cmd.copy()
        return q_cmd
    if 'q' not in state:
        state['q'] = q_cmd.copy()
        state['q0'] = q_cmd.copy()
        state['q1'] = q_cmd.copy()
        state['t0'] = now
        return q_cmd.copy()
    prev = state.get('q1')
    if prev is None or prev.shape != q_cmd.shape or not np.allclose(q_cmd, prev):
        state['q0'] = np.asarray(state['q'], dtype=float).copy()
        state['q1'] = q_cmd.copy()
        state['t0'] = now
    s = min(1.0, (now - state['t0']) / tau)
    q = (1.0 - s) * state['q0'] + s * state['q1']
    state['q'] = q
    return q


def filter_cmd(state, q_cmd, dt, tau):
    """PT1 toward q_cmd. Continues catching up while the target is held."""
    q_cmd = np.asarray(q_cmd, dtype=float)
    if tau <= 0.0:
        state['q'] = q_cmd.copy()
        return q_cmd
    if 'q' not in state:
        state['q'] = q_cmd.copy()
        return q_cmd.copy()
    alpha = 1.0 - math.exp(-max(dt, 1e-6) / tau)
    q = state['q'] + alpha * (q_cmd - state['q'])
    state['q'] = q
    return q


def _connect_image_client(host: str):
    """Subscribe to the teleimager ZMQ server; retry while it starts."""
    from teleimager.image_client import ImageClient

    last_error = None
    for attempt in range(1, IMAGE_CLIENT_RETRIES + 1):
        try:
            client = ImageClient(host=host, request_bgr=True)
            logger_mp.info(f"ImageClient connected to {host} (attempt {attempt})")
            return client
        except Exception as exc:
            last_error = exc
            time.sleep(IMAGE_CLIENT_RETRY_S)
    logger_mp.warning(f"ImageClient failed after {IMAGE_CLIENT_RETRIES} tries: {last_error}")
    return None


def _video_publish_loop(img_client, portal, track, stop_evt, fps: float):
    """Grab latest ZMQ frame and publish off the control loop. Drop-oldest: no queue."""
    logged = False
    interval = 1.0 / max(fps, 1.0)
    while not stop_evt.is_set():
        t0 = time.time()
        try:
            head = img_client.get_head_frame()
            if head is not None and getattr(head, "bgr", None) is not None:
                rgb = np.ascontiguousarray(head.bgr[:, :, ::-1])
                portal.send_video_frame(
                    track, rgb, timestamp_us=int(time.time() * 1_000_000))
                if not logged:
                    h, w = rgb.shape[:2]
                    logger_mp.info(f"publishing '{track}' {w}x{h} (video thread)")
                    logged = True
        except Exception as exc:
            logger_mp.warning(f"video thread: {exc}")
        sleep = interval - (time.time() - t0)
        if sleep > 0:
            stop_evt.wait(sleep)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frequency', type=float, default=30.0, help='state publish rate')
    parser.add_argument('--arm', type=str, choices=['G1_29'], default='G1_29')
    parser.add_argument('--ee', type=str, choices=['dex3'], default=None)
    parser.add_argument('--motion', action='store_true')
    parser.add_argument('--network-interface', type=str, default=None)
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--no-img', action='store_true',
                        help='do not subscribe to teleimager / publish video')
    parser.add_argument('--img-server-ip', type=str, default='127.0.0.1',
                        help='teleimager ZMQ host (config port 60000)')
    parser.add_argument('--portal-yaml', type=str, default=os.path.join(current_dir, 'portal.yaml'))
    parser.add_argument('--portal-mapping', type=str, default=os.path.join(current_dir, 'portal_mapping.yaml'))
    parser.add_argument('--env-file', type=str, default=os.path.join(current_dir, '.env'))
    parser.add_argument('--livekit-url', type=str, default=None)
    parser.add_argument('--livekit-room', type=str, default=None)
    parser.add_argument('--portal-identity', type=str, default='xr-robot')
    parser.add_argument('--cmd-tau', type=float, default=0.3,
                        help='command smoothing time (s); 0 = off. Switch interp/filter in the TELEOP loop.')
    args = parser.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(1 if args.sim else 0, networkInterface=args.network_interface)

    from teleop.robot_control.robot_arm import G1_29_ArmController
    from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper

    loco_wrapper = None
    if args.motion:
        loco_wrapper = LocoClientWrapper(robot_type="G1")
    else:
        motion_switcher = MotionSwitcher()
        status, result = motion_switcher.Enter_Debug_Mode()
        logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

    arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)

    # DDS apply only. XR→Dex3 retargeting runs on the operator; actions arrive as hand_q.
    hand_ctrl = None
    if args.ee == "dex3":
        from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
        hand_ctrl = Dex3_1_Controller(apply_targets=True, simulation_mode=args.sim)

    portal = PortalRobotTransport(
        portal_yaml=args.portal_yaml,
        mapping_yaml=args.portal_mapping,
        env_file=args.env_file,
        identity=args.portal_identity,
        room=args.livekit_room,
        url=args.livekit_url)
    portal.wait_until_connected()

    img_client = None if args.no_img else _connect_image_client(args.img_server_ip)
    video_track = portal.video_tracks[0] if portal.video_tracks else None
    video_stop = threading.Event()
    video_thread = None
    if img_client is not None and video_track:
        video_thread = threading.Thread(
            target=_video_publish_loop,
            args=(img_client, portal, video_track, video_stop, args.frequency),
            daemon=True)
        video_thread.start()
        logger_mp.info("video publish thread started")

    lock = threading.Lock()
    latest = {'action': None, 'wall': 0.0}
    applied_fsm = FSM_IDLE
    timing = LoopTiming(logger_mp)
    last_action_wall = {'t': 0.0}
    arm_cmd = {}
    hand_cmd = {}
    last_tick = 0.0

    def on_action(action: UnpackedAction):
        now = time.time()
        with lock:
            prev = last_action_wall['t']
            latest['action'] = action
            latest['wall'] = now
            last_action_wall['t'] = now
        if prev:
            timing.add("action_gap_ms", (now - prev) * 1000.0)

    portal.on_unpacked_action(on_action)
    logger_mp.info("robot loop running; waiting for operator actions")

    try:
        while True:
            start = time.time()
            dt = (start - last_tick) if last_tick else (1.0 / args.frequency)
            last_tick = start
            with lock:
                action = latest['action']
                age = start - latest['wall'] if latest['wall'] else 1e9

            fsm = action.fsm_id if action is not None else FSM_IDLE

            if fsm == FSM_HOME:
                if applied_fsm != FSM_HOME:
                    arm_ctrl.ctrl_dual_arm_go_home()
                    applied_fsm = FSM_HOME
                arm_cmd.clear()
                hand_cmd.clear()
            elif fsm == FSM_HAND_SETUP and action:
                if applied_fsm != FSM_HAND_SETUP:
                    arm_cmd.clear()
                    hand_cmd.clear()
                    if hand_ctrl is not None:
                        hand_cmd['q'] = np.asarray(hand_ctrl.get_current_dual_hand_q(), dtype=float).copy()
                if hand_ctrl is not None and action.hand_q.size:
                    half = action.hand_q.size // 2
                    hand_q = action.hand_q
                    # hand_q = interp_cmd(hand_cmd, action.hand_q, start, args.cmd_tau)
                    hand_ctrl.ctrl_dual_hand(hand_q[:half], hand_q[half:])
                applied_fsm = FSM_HAND_SETUP
            elif fsm == FSM_TELEOP and action:
                if applied_fsm != FSM_TELEOP:
                    arm_ctrl.speed_gradual_max()
                    arm_cmd.clear()
                    arm_cmd['q'] = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=float).copy()
                    hand_cmd.clear()
                    if hand_ctrl is not None:
                        hand_cmd['q'] = np.asarray(hand_ctrl.get_current_dual_hand_q(), dtype=float).copy()
                tauff = np.zeros_like(action.arm_q)
                # pick one (arm + hand must match):
                arm_q = action.arm_q
                # arm_q = interp_cmd(arm_cmd, action.arm_q, start, args.cmd_tau)
                # arm_q = filter_cmd(arm_cmd, action.arm_q, dt, args.cmd_tau)
                arm_ctrl.ctrl_dual_arm(arm_q, tauff)
                if hand_ctrl is not None and action.hand_q.size:
                    half = action.hand_q.size // 2
                    hand_q = interp_cmd(hand_cmd, action.hand_q, start, args.cmd_tau)
                    # hand_q = filter_cmd(hand_cmd, action.hand_q, dt, args.cmd_tau)
                    hand_ctrl.ctrl_dual_hand(hand_q[:half], hand_q[half:])
                if loco_wrapper is not None:
                    loco_wrapper.Move(action.vx, action.vy, action.vyaw)
                applied_fsm = FSM_TELEOP
            else:
                applied_fsm = FSM_IDLE
                arm_cmd.clear()
                hand_cmd.clear()

            motor_q = arm_ctrl.get_current_motor_q()
            hand_q = hand_ctrl.get_current_dual_hand_q() if hand_ctrl is not None else None
            portal.send_state(motor_q=motor_q, hand_q=hand_q, fsm_id=fsm)

            sleep = max(0.0, (1.0 / args.frequency) - (time.time() - start))
            time.sleep(sleep)
    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt, exiting ...")
    finally:
        try:
            if motion_switcher is not None:
                motion_switcher.Exit_Debug_Mode()
                logger_mp.info(f"Exit debug / SelectMode('ai'): status={status} result={result}")
                _, mode = motion_switcher.msc.CheckMode()
                logger_mp.info(f"CheckMode after exit: {mode}")
                
        except Exception as e:
            logger_mp.error(f"Exit debug mode failed: {e}")
        try:
            portal.close()
        except Exception as e:
            logger_mp.error(f"portal close failed: {e}")
        video_stop.set()
        if video_thread is not None:
            video_thread.join(timeout=2.0)
        if img_client is not None:
            try:
                img_client.close()
            except Exception as e:
                logger_mp.error(f"ImageClient close failed: {e}")
        logger_mp.info("robot exited.")
