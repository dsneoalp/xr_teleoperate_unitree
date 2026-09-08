"""Portal robot control loop: apply joint targets via SDK, stream state.

No TeleVuer / IK. Pair with teleop_operator.py in the same LiveKit room.

    python teleop/teleop_robot.py --ee dex3 --arm G1_29
    python teleop/teleop_robot.py --ee dex3 --arm G1_29 --motion
    python teleop/teleop_robot.py --ee dex3 --arm G1_29 --sim
"""
import time
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
ACTION_TIMEOUT = 0.2
IMAGE_CLIENT_RETRIES = 50
IMAGE_CLIENT_RETRY_S = 0.1


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
    args = parser.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(1 if args.sim else 0, networkInterface=args.network_interface)

    from teleop.robot_control.robot_arm import G1_29_ArmController
    from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper

    loco_wrapper = None
    if args.motion:
        loco_wrapper = LocoClientWrapper()
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
            with lock:
                action = latest['action']
                age = start - latest['wall'] if latest['wall'] else 1e9

            fsm = action.fsm_id if action is not None else FSM_IDLE
            # fresh = action is not None and age < ACTION_TIMEOUT
            # timing.count("loops")
            # if action is not None:
            #     timing.add("age_ms", age * 1000.0)
            # if not fresh:
            #     timing.count("stale")

            if fsm == FSM_HOME:
                if applied_fsm != FSM_HOME:
                    arm_ctrl.ctrl_dual_arm_go_home()
                    applied_fsm = FSM_HOME
            elif fsm == FSM_TELEOP and action:
                if applied_fsm != FSM_TELEOP:
                    arm_ctrl.speed_gradual_max()
                tauff = np.zeros_like(action.arm_q)
                arm_ctrl.ctrl_dual_arm(action.arm_q, tauff)
                if hand_ctrl is not None and action.hand_q.size:
                    half = action.hand_q.size // 2
                    hand_ctrl.ctrl_dual_hand(action.hand_q[:half], action.hand_q[half:])
                if loco_wrapper is not None:
                    loco_wrapper.Move(action.vx, action.vy, action.vyaw)
                applied_fsm = FSM_TELEOP
            else:
                applied_fsm = FSM_IDLE

            motor_q = arm_ctrl.get_current_motor_q()
            hand_q = hand_ctrl.get_current_dual_hand_q() if hand_ctrl is not None else None
            portal.send_state(motor_q=motor_q, hand_q=hand_q, fsm_id=fsm)

            sleep = max(0.0, (1.0 / args.frequency) - (time.time() - start))
            time.sleep(sleep)
    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt, exiting ...")
    finally:

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
